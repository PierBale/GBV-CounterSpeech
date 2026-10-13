"""All system-vs-system pairs for each message and decoder (no gold baseline)."""

from collections import defaultdict
from itertools import combinations


STRATEGY_ORDER = (
    "without_rag", "rag_bm25", "rag_qwen3_emb_0.6b", "rag_qwen3_emb_4b", "rag_cards",
)
CONTROLLED_PAIR = {"rag_qwen3_emb_4b", "rag_cards"}


def build_pairwise_comparisons(records: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    strategies_by_model = defaultdict(set)
    for record in records:
        strategy = record.get("generation_strategy")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("A generated response is missing generation_strategy.")
        for field in ("hateful_message", "generated_counter_speech", "generation_model", "source", "id"):
            if record.get(field) is None or not str(record[field]).strip():
                raise ValueError(f"Generated response is missing {field}.")
        key = (record["id"], record["source"], record.get("split"), record["generation_model"])
        if strategy in groups[key]:
            raise ValueError(f"Duplicate strategy {strategy!r} for {key}.")
        groups[key][strategy] = record
        strategies_by_model[record["generation_model"]].add(strategy)

    def order(strategy):
        return (STRATEGY_ORDER.index(strategy) if strategy in STRATEGY_ORDER else len(STRATEGY_ORDER), strategy)

    comparisons = []
    for key, group in groups.items():
        expected = strategies_by_model[key[-1]]
        if set(group) != expected:
            raise ValueError(f"Incomplete generation group {key}; missing strategies: {sorted(expected - set(group))}.")
        if len(group) < 2:
            raise ValueError(f"At least two generated strategies are required for {key}.")
        messages = {record["hateful_message"] for record in group.values()}
        if len(messages) != 1:
            raise ValueError(f"Mismatched hateful messages for comparison {key}.")
        for left, right in combinations(sorted(group, key=order), 2):
            first, second = group[left], group[right]
            comparisons.append({
                "id": first["id"], "source": first["source"], "split": first.get("split"),
                "target": first.get("target"), "edos_label": first.get("edos_label"),
                "generation_model": first["generation_model"],
                "comparison_kind": "system_pair",
                "primary_comparison": {left, right} == CONTROLLED_PAIR,
                # Retain these names so old zero-shot judgments can still be resumed.
                "baseline_strategy": left, "challenger_strategy": right,
                "hateful_message": first["hateful_message"],
                "response_1_strategy": left, "response_1": first["generated_counter_speech"],
                "response_2_strategy": right, "response_2": second["generated_counter_speech"],
                "response_1_context_length": first.get("context_length"),
                "response_2_context_length": second.get("context_length"),
            })
    if not comparisons:
        raise ValueError("No system-vs-system pairwise comparisons could be built.")
    return comparisons
