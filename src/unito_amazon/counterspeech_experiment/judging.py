"""Pairwise JudgeLM evaluation: every pair of systems (config.judge_all_pairs) or the pairs in
config.JUDGED_PAIRS, both presentation orders.

Each message and pair is judged twice (A first, then B first). Per message the two
outcomes are combined from A's point of view (win 1, tie 0.5, loss 0): mean > 0.5 is
a win, < 0.5 a loss, 0.5 a tie. This removes JudgeLM's position bias from the result.
"""

import gc
from collections import defaultdict
from datetime import datetime, timezone

from scipy.stats import binomtest, wilcoxon

from unito_amazon.card_creation.io import append_jsonl, read_jsonl, write_json
from unito_amazon.evaluation.judge import LLMJudge
from unito_amazon.evaluation.prompts import CounterSpeechPairwisePrompt
from unito_amazon.evaluation.schema import JUDGE_PROTOCOL_VERSION

from .config import ExperimentConfig, judged_pairs, pairs_for, strategies_for


def build_comparisons(generations: list[dict], strategies: tuple[str, ...], all_pairs: bool = False) -> list[dict]:
    groups = defaultdict(dict)
    for record in generations:
        key = (record["source"], record["id"])
        if record["generation_strategy"] in groups[key]:
            raise ValueError(f"Duplicate generation {key} {record['generation_strategy']}.")
        groups[key][record["generation_strategy"]] = record
    comparisons = []
    for (source, message_id), group in sorted(groups.items()):
        applicable = strategies_for(source, strategies)
        missing = set(applicable) - set(group)
        if missing:
            raise ValueError(f"Missing generations for {source} {message_id}: {sorted(missing)}.")
        for system_a, system_b in pairs_for(source, strategies, all_pairs):
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
    comparisons = build_comparisons(read_jsonl(config.generations_path), config.strategies, config.judge_all_pairs)
    done = set()
    if config.judgments_path.exists():
        previous = list(read_jsonl(config.judgments_path))
        # Judgments without the field were made with the original "facts" criteria.
        criteria = {record.get("judge_criteria", "facts") for record in previous}
        if criteria - {config.judge_criteria}:
            raise ValueError(f"{config.judgments_path} has judgments with criteria {sorted(criteria)}, "
                             f"not {config.judge_criteria!r}; use another output_dir.")
        done = {comparison_key(record) for record in previous if record["judgment"].get("winner") is not None}
    pending = [record for record in comparisons if comparison_key(record) not in done]
    print(f"{len(comparisons)} comparisons, {len(comparisons) - len(pending)} already judged.")
    if pending:
        from unito_amazon.llm.factory import LLMFactory

        llm = LLMFactory.create({"provider": "huggingface", "model_name": config.judge_model,
                                 "device": config.device, "dtype": config.judge_dtype})
        judge = LLMJudge(llm=llm, prompt=CounterSpeechPairwisePrompt(config.judge_criteria),
                         max_retries=config.judge_max_retries,
                         retry_temperature=config.judge_retry_temperature)
        try:
            for start in range(0, len(pending), config.judge_batch_size):
                batch = pending[start:start + config.judge_batch_size]
                for record, result in zip(batch, judge.judge_batch(batch), strict=True):
                    append_jsonl(config.judgments_path, {
                        **record, "judge_model": config.judge_model,
                        "judge_protocol": JUDGE_PROTOCOL_VERSION, "judge_criteria": config.judge_criteria,
                        "judgment": result.model_dump(),
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


def _agreement(first: list[float], second: list[float], first_diff: list[float], second_diff: list[float]) -> dict:
    """Agreement of JudgeLM with itself across the two presentation orders.

    Outcomes from A's point of view, coded -1 (B wins), 0 (tie), 1 (A wins). Cohen's kappa
    from scikit-learn (nominal and linearly weighted), Krippendorff's ordinal alpha from
    krippendorff, Spearman correlation of the score differences A - B from scipy.
    """
    import krippendorff
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    def defined(value):
        return None if value is None or np.isnan(value) else float(value)

    n = len(first)
    decisive_both = [(x, y) for x, y in zip(first, second) if x != 0 and y != 0]
    return {
        "messages": n,
        "exact_agreement": sum(x == y for x, y in zip(first, second)) / n,
        "cohen_kappa": defined(cohen_kappa_score(first, second, labels=[-1, 0, 1])),
        "cohen_kappa_linear": defined(cohen_kappa_score(first, second, labels=[-1, 0, 1], weights="linear")),
        "krippendorff_alpha_ordinal": defined(krippendorff.alpha(
            reliability_data=np.array([first, second], dtype=float), level_of_measurement="ordinal",
            value_domain=[-1, 0, 1])) if len(set(first) | set(second)) > 1 else None,
        "spearman_score_difference": defined(spearmanr(first_diff, second_diff).statistic)
        if len(set(first_diff)) > 1 and len(set(second_diff)) > 1 else None,
        # A clear winner in both orders, but a different one: the decision follows the position.
        "opposite_winner_rate": sum(x != y for x, y in decisive_both) / n,
        "tie_in_one_order_rate": sum((x == 0) != (y == 0) for x, y in zip(first, second)) / n,
    }


def order_agreement(by_item: dict, pairs: list[tuple[str, str]]) -> dict:
    """Agreement between the a_first and b_first judgments, per dataset and pair and overall."""
    coded = {1.0: 1, 0.5: 0, 0.0: -1}
    groups = defaultdict(lambda: ([], [], [], []))
    for (source, a, b, _), orders in by_item.items():
        if set(orders) != {"a_first", "b_first"}:
            continue
        values = (coded[_points_for_a(orders["a_first"])], coded[_points_for_a(orders["b_first"])],
                  _score_difference(orders["a_first"]), _score_difference(orders["b_first"]))
        for key in (f"{source}|{a}|{b}", f"all|{a}|{b}", f"{source}|all pairs", "all|all pairs"):
            for column, value in zip(groups[key], values):
                column.append(value)
    order = {f"{a}|{b}": index for index, (a, b) in enumerate(pairs)}
    return {key: _agreement(*groups[key]) for key in sorted(
        groups, key=lambda k: (k.split("|", 1)[0] != "all", k.split("|", 1)[0], order.get(k.split("|", 1)[1], -1)))}


def summarize(config: ExperimentConfig) -> dict:
    latest = {}
    for record in read_jsonl(config.judgments_path):
        if record["judgment"].get("winner") is not None:
            latest[comparison_key(record)] = record
    current = {comparison_key(record) for record in build_comparisons(
        read_jsonl(config.generations_path), config.strategies, config.judge_all_pairs)}
    pairs = judged_pairs(config.strategies, config.judge_all_pairs)
    records = [record for key, record in latest.items() if key in current]

    by_item = defaultdict(dict)
    for record in records:
        by_item[(record["source"], record["system_a"], record["system_b"], record["id"])][record["order"]] = record
    tables = {}
    first_position_wins = defaultdict(lambda: [0, 0])
    for source in ("all", *sorted({record["source"] for record in records})):
        for system_a, system_b in pairs:
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
            if not points and not incomplete:
                continue  # pair not applicable to this dataset (gold labels exist only for EDOS)
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
        "judge_criteria": config.judge_criteria, "judge_all_pairs": config.judge_all_pairs,
        "valid_judgments": len(records), "expected_judgments": len(current),
        "method": ("per message, the outcomes of both presentation orders are averaged from A's "
                   "point of view (win 1, tie 0.5, loss 0); a_score is the mean of these values. "
                   "Sign test on wins vs losses; Wilcoxon on the per-message mean JudgeLM score difference."),
        "first_position_win_rate": {source: wins / total for source, (wins, total) in first_position_wins.items()},
        "pairs": tables,
        "order_agreement": order_agreement(by_item, pairs),
    }
