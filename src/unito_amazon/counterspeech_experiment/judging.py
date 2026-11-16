"""Pairwise JudgeLM evaluation: every pair of strategies, both presentation orders.

Each message and pair is judged twice (A first, then B first). Per message the two
outcomes are combined from A's point of view (win 1, tie 0.5, loss 0): mean > 0.5 is
a win, < 0.5 a loss, 0.5 a tie. This removes JudgeLM's position bias from the result.
"""

import gc
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations

from scipy.stats import binomtest, wilcoxon

from unito_amazon.card_creation.io import append_jsonl, read_jsonl, write_json
from unito_amazon.evaluation.judge import LLMJudge
from unito_amazon.evaluation.schema import JUDGE_PROTOCOL_VERSION

from .config import ExperimentConfig


def build_comparisons(generations: list[dict], strategies: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(dict)
    for record in generations:
        key = (record["source"], record["id"])
        if record["generation_strategy"] in groups[key]:
            raise ValueError(f"Duplicate generation {key} {record['generation_strategy']}.")
        groups[key][record["generation_strategy"]] = record
    comparisons = []
    for (source, message_id), group in sorted(groups.items()):
        missing = set(strategies) - set(group)
        if missing:
            raise ValueError(f"Missing generations for {source} {message_id}: {sorted(missing)}.")
        for system_a, system_b in combinations(strategies, 2):
            for order, (first, second) in (("a_first", (system_a, system_b)), ("b_first", (system_b, system_a))):
                comparisons.append({
                    "id": message_id, "source": source, "system_a": system_a, "system_b": system_b,
                    "order": order, "hateful_message": group[first]["hateful_message"],
                    "response_1_strategy": first, "response_1": group[first]["generated_counter_speech"],
                    "response_2_strategy": second, "response_2": group[second]["generated_counter_speech"],
                })
    return comparisons


def comparison_key(record: dict) -> tuple:
    return (record["source"], record["id"], record["response_1_strategy"], record["response_2_strategy"],
            record["response_1"], record["response_2"])


def run_judge(config: ExperimentConfig) -> None:
    comparisons = build_comparisons(read_jsonl(config.generations_path), config.strategies)
    done = set()
    if config.judgments_path.exists():
        done = {comparison_key(record) for record in read_jsonl(config.judgments_path)
                if record["judgment"].get("winner") is not None}
    pending = [record for record in comparisons if comparison_key(record) not in done]
    print(f"{len(comparisons)} comparisons, {len(comparisons) - len(pending)} already judged.")
    if pending:
        from unito_amazon.llm.factory import LLMFactory

        llm = LLMFactory.create({"provider": "huggingface", "model_name": config.judge_model,
                                 "device": config.device, "dtype": config.judge_dtype})
        judge = LLMJudge(llm=llm, max_retries=config.judge_max_retries,
                         retry_temperature=config.judge_retry_temperature)
        try:
            for start in range(0, len(pending), config.judge_batch_size):
                batch = pending[start:start + config.judge_batch_size]
                for record, result in zip(batch, judge.judge_batch(batch), strict=True):
                    append_jsonl(config.judgments_path, {
                        **record, "judge_model": config.judge_model,
                        "judge_protocol": JUDGE_PROTOCOL_VERSION, "judgment": result.model_dump(),
                        "judged_at": datetime.now(timezone.utc).isoformat(),
                    })
                print(f"Judged {min(start + len(batch), len(pending))}/{len(pending)}")
        finally:
            llm.destroy()
            gc.collect()
    write_json(config.output_dir / "judge_summary.json", summarize(config))
    print(f"Summary written to {config.output_dir / 'judge_summary.json'}")


def _points_for_a(record: dict) -> float:
    winner = record["judgment"]["winner"]
    if winner == "tie":
        return 0.5
    winning_strategy = record["response_1_strategy"] if winner == "response_1" else record["response_2_strategy"]
    return 1.0 if winning_strategy == record["system_a"] else 0.0


def _score_difference(record: dict) -> float:
    """JudgeLM score of A minus score of B."""
    judgment = record["judgment"]
    first, second = judgment["response_1_score"], judgment["response_2_score"]
    return first - second if record["response_1_strategy"] == record["system_a"] else second - first


def summarize(config: ExperimentConfig) -> dict:
    latest = {}
    for record in read_jsonl(config.judgments_path):
        if record["judgment"].get("winner") is not None:
            latest[comparison_key(record)] = record
    current = {comparison_key(record) for record in build_comparisons(
        read_jsonl(config.generations_path), config.strategies)}
    records = [record for key, record in latest.items() if key in current]

    by_item = defaultdict(dict)
    for record in records:
        by_item[(record["source"], record["system_a"], record["system_b"], record["id"])][record["order"]] = record
    tables = {}
    first_position_wins = defaultdict(lambda: [0, 0])
    for source in ("all", *sorted({record["source"] for record in records})):
        for system_a, system_b in combinations(config.strategies, 2):
            points, differences, agree, incomplete = [], [], 0, 0
            for (item_source, a, b, _), orders in by_item.items():
                if (a, b) != (system_a, system_b) or source not in ("all", item_source):
                    continue
                if set(orders) != {"a_first", "b_first"}:
                    incomplete += 1
                    continue
                pair = [_points_for_a(orders["a_first"]), _points_for_a(orders["b_first"])]
                agree += pair[0] == pair[1]
                points.append(sum(pair) / 2)
                differences.append(sum(_score_difference(record) for record in orders.values()) / 2)
            wins, losses = sum(p > 0.5 for p in points), sum(p < 0.5 for p in points)
            nonzero = [d for d in differences if d != 0]
            tables[f"{source}|{system_a}|{system_b}"] = {
                "dataset": source, "system_a": system_a, "system_b": system_b,
                "messages": len(points), "incomplete_messages": incomplete,
                "a_wins": wins, "ties": len(points) - wins - losses, "b_wins": losses,
                "a_score": sum(points) / len(points) if points else None,
                "order_agreement": agree / len(points) if points else None,
                "mean_judgelm_score_difference_a_minus_b": sum(differences) / len(differences) if differences else None,
                "sign_test_p": binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else None,
                "wilcoxon_p": wilcoxon(nonzero).pvalue if len(nonzero) >= 10 else None,
            }
    for record in records:
        if record["judgment"]["winner"] != "tie":
            stats = first_position_wins[record["source"]]
            stats[0] += record["judgment"]["winner"] == "response_1"
            stats[1] += 1
    return {
        "judge_model": config.judge_model, "judge_protocol": JUDGE_PROTOCOL_VERSION,
        "valid_judgments": len(records), "expected_judgments": len(current),
        "method": ("per message, the outcomes of both presentation orders are averaged from A's "
                   "point of view (win 1, tie 0.5, loss 0); a_score is the mean of these values. "
                   "Sign test on wins vs losses; Wilcoxon on the per-message mean JudgeLM score difference."),
        "first_position_win_rate": {source: wins / total for source, (wins, total) in first_position_wins.items()},
        "pairs": tables,
    }
