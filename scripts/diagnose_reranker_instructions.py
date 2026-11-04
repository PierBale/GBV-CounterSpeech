"""Diagnose how the Qwen3-Reranker instruction affects the final top-k.

Reuses the dense top-100 candidates already computed by the experiment
(work/retrieval/<embedder>/edos_passages_dense.jsonl), loads Qwen3-Reranker
once and re-scores them under several instructions. No human labels are used.

Metrics per (candidate pool, instruction):
- diversity of the top-k across the 11 labels: distinct chunks, chunks shared
  by >= 3 labels, the most shared chunk, distinct documents, top-document share;
- label specificity: every chunk selected in the top-k of label L is scored
  against the queries of all 11 labels. "own_label_rank" is the rank of L among
  them (1 = the passage fits L better than any other label; ~6 = chance).
  "own_label_top1" is the share of selections where L ranks first;
- overlap (mean Jaccard over labels) of the top-k with the BGE reranker of the
  same run (an independent model) and with the run's own Qwen ranking (sanity
  check: the "current" instruction must reproduce it, Jaccard ~1).

Run from the repository root on a GPU node (see diagnose_reranker_instructions.slurm):
    python -u scripts/diagnose_reranker_instructions.py \\
        --config src/unito_amazon/card_creation/config.edos-v2-words-clean-v4.json
"""

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INSTRUCTIONS = {
    # Default instruction of the Qwen3-Reranker model card.
    "qwen_default": "Given a web search query, retrieve relevant passages that answer the query",
    # Instruction of the edos-v2-words-clean-20261104 run.
    "old_verifiable_evidence": (
        "Given a type of sexism and its description, determine whether the passage "
        "contains specific, verifiable evidence relevant to the described type and useful "
        "to counter this type of sexist hate speech."
    ),
    # Instruction of the v4 run (reranking.RERANK_INSTRUCTION, filled at runtime).
    "current": "CURRENT",
    # Mentions no sexism at all: the category comes only from the query.
    "behaviour_specific": (
        "Judge whether the passage is about the specific behaviour described in the "
        "query and contains information that would help respond to it."
    ),
}
EMBEDDERS = ("Qwen/Qwen3-Embedding-8B", "BAAI/bge-m3")


def jaccard(first, second) -> float:
    first, second = set(first), set(second)
    return len(first & second) / len(first | second) if first | second else 1.0


def diversity(rankings: dict[str, list[str]], documents: dict[str, str], k: int) -> dict:
    selected = [chunk for ranked in rankings.values() for chunk in ranked[:k]]
    labels_per_chunk = defaultdict(set)
    for label, ranked in rankings.items():
        for chunk in ranked[:k]:
            labels_per_chunk[chunk].add(label)
    docs = Counter(documents[chunk] for chunk in selected)
    return {
        f"distinct_chunks@{k}": len(labels_per_chunk),
        f"chunks_in_3plus_labels@{k}": sum(len(v) >= 3 for v in labels_per_chunk.values()),
        f"max_labels_per_chunk@{k}": max(len(v) for v in labels_per_chunk.values()),
        f"distinct_documents@{k}": len(docs),
        f"top_document_share@{k}": round(docs.most_common(1)[0][1] / len(selected), 3),
        f"top_document@{k}": docs.most_common(1)[0][0],
    }


def specificity(rankings, queries, passages, score_fn, k: int) -> dict:
    """Rank of the selecting label among all labels, for every top-k selection."""
    labels = list(queries)
    selections = [(label, chunk) for label in labels for chunk in rankings[label][:k]]
    chunks = list(dict.fromkeys(chunk for _, chunk in selections))
    scores = {}
    for label in labels:
        for chunk, score in zip(chunks, score_fn(queries[label], [passages[c] for c in chunks])):
            scores[label, chunk] = score
    ranks = []
    for label, chunk in selections:
        ordered = sorted(labels, key=lambda other: -scores[other, chunk])
        ranks.append(ordered.index(label) + 1)
    return {
        f"own_label_rank@{k}": round(float(np.mean(ranks)), 2),
        f"own_label_top1@{k}": round(sum(rank == 1 for rank in ranks) / len(ranks), 3),
    }


def load_reference(path: Path, k: int) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    ranked = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["reranker_rank"]):
        ranked[row["label"]].append(row["chunk"]["chunk_id"])
    return {label: chunks[:k] for label, chunks in ranked.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--reranker", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--top-k", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--instructions", nargs="+", choices=list(INSTRUCTIONS), default=list(INSTRUCTIONS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    from unito_amazon.card_creation.config import load_config, model_slug
    from unito_amazon.card_creation.passage_text import format_chunk_text
    from unito_amazon.card_creation.reranking import RERANK_INSTRUCTION, QwenReranker
    from unito_amazon.card_creation.schema import Chunk

    INSTRUCTIONS["current"] = RERANK_INSTRUCTION
    config = load_config(args.config).model_copy(update={"model_scoped_artifacts": True})
    run_dir = config.output_dir / "runs" / config.run_id / "reranking"
    output_dir = args.output_dir or config.output_dir / "diagnostics" / "reranker_instructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_k = max(args.top_k)

    pools = {}
    for embedder in EMBEDDERS:
        current = config.model_copy(update={"embedding_model": embedder})
        rows = [json.loads(line) for line in current.dense_path.open(encoding="utf-8")]
        queries, candidates, passages, documents = {}, defaultdict(list), {}, {}
        for row in sorted(rows, key=lambda row: row["dense_rank"]):
            chunk = Chunk.model_validate(row["chunk"])
            queries[row["label"]] = row["query_text"]
            candidates[row["label"]].append(chunk.chunk_id)
            passages[chunk.chunk_id] = format_chunk_text(chunk, config.include_section_titles)
            documents[chunk.chunk_id] = chunk.source.markdown_file
        slug = model_slug(embedder)
        pools[slug] = {
            "queries": queries, "candidates": candidates, "passages": passages, "documents": documents,
            "bge_reference": load_reference(run_dir / f"{slug}__bge-reranker-v2-m3.jsonl", max_k),
            "qwen_run": load_reference(run_dir / f"{slug}__{model_slug(args.reranker)}.jsonl", max_k),
        }

    reranker = QwenReranker(args.reranker, config.reranker_batch_size, config.reranker_max_tokens, config.device)
    results, listings = [], []
    try:
        for (slug, pool), name in itertools.product(pools.items(), args.instructions):
            reranker.instruction = INSTRUCTIONS[name]
            rankings, scores_by_label = {}, {}
            for label, query in pool["queries"].items():
                ids = pool["candidates"][label]
                scores = reranker.score(query, [pool["passages"][chunk] for chunk in ids])
                order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))  # ties: dense rank
                rankings[label] = [ids[i] for i in order]
                scores_by_label[label] = [scores[i] for i in order]
            row = {"embedder": slug, "instruction": name}
            for k in args.top_k:
                row.update(diversity(rankings, pool["documents"], k))
            k = min(args.top_k)
            row.update(specificity(rankings, pool["queries"], pool["passages"], reranker.score, k))
            for reference in ("bge_reference", "qwen_run"):
                if pool[reference]:
                    row[f"jaccard_vs_{reference}@{k}"] = round(float(np.mean([
                        jaccard(rankings[label][:k], pool[reference].get(label, [])[:k]) for label in rankings
                    ])), 3)
            row["mean_top1_score"] = round(float(np.mean([s[0] for s in scores_by_label.values()])), 3)
            results.append(row)
            for label, ranked in rankings.items():
                listings.append({
                    "embedder": slug, "instruction": name, "label": label,
                    "top10": [{"rank": rank, "chunk_id": chunk, "score": round(score, 4),
                               "document": pool["documents"][chunk],
                               "start": pool["passages"][chunk][:200]}
                              for rank, (chunk, score) in enumerate(zip(ranked[:10], scores_by_label[label][:10]), start=1)],
                })
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        reranker.close()

    (output_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "top10_by_instruction.jsonl").open("w", encoding="utf-8") as handle:
        for item in listings:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "summary.md").write_text(render_markdown(results, args.top_k), encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


def render_markdown(results: list[dict], top_k: list[int]) -> str:
    k = min(top_k)
    columns = ["instruction"]
    for size in top_k:
        columns += [f"distinct_chunks@{size}", f"chunks_in_3plus_labels@{size}", f"max_labels_per_chunk@{size}"]
    columns += [f"distinct_documents@{k}", f"top_document_share@{k}", f"own_label_rank@{k}", f"own_label_top1@{k}",
                f"jaccard_vs_bge_reference@{k}", f"jaccard_vs_qwen_run@{k}", "mean_top1_score"]
    lines = [
        "# Qwen3-Reranker instruction diagnostic", "",
        "Re-scores the run's dense top-100 candidates; no human labels.",
        f"own_label_rank@{k}: rank (1-11) of the selecting label when the passage is scored against all 11 queries; "
        "1 = label-specific, ~6 = chance.",
        "jaccard_vs_qwen_run must be ~1 for 'current' (reproduces the run).", "",
    ]
    for embedder in dict.fromkeys(row["embedder"] for row in results):
        lines += [f"## Candidates from {embedder}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for row in (row for row in results if row["embedder"] == embedder):
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
