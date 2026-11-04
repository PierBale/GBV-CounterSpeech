"""Diagnose how query format and embedder instruction affect dense retrieval.

No human labels are used. For every embedder and every (query format,
instruction) variant, the 11 EDOS queries are encoded against the same chunk
index and the script measures:

- how similar the queries become to each other (mean/max pairwise cosine):
  a shared prefix that dominates the vector makes labels indistinguishable;
- hubness of the top-k lists (Radovanovic et al., JMLR 2010): distinct chunks
  across labels, chunks shared by >= 3 labels, the most shared chunk;
- diversity of sources and the share of table/number-heavy chunks in the
  top-k (tests whether words such as "prevalence" pull statistics tables);
- overlap (Jaccard) of every variant's top-k with a reference variant.

Document vectors never depend on the instruction (Qwen3-Embedding is
asymmetric), so one index per embedder serves all variants. The index is the
one the experiment uses: if it already exists it is loaded, otherwise it is
built and saved there, and the experiment will reuse it.

Run from the repository root on a GPU node (or: sbatch scripts/diagnose_query_instructions.slurm):
    python -u scripts/diagnose_query_instructions.py \
        --config src/unito_amazon/card_creation/config.edos-v2-words-clean-v4.json
"""

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUERY_FORMATS = {
    # Format of the edos-v2-words-clean-20261104 run.
    "old_format": lambda label, definition: f"Sexism type: {label}\nDescription: {definition}",
    # Current format (queries.default_query_text).
    "new_format": None,
}
INSTRUCTIONS = {
    "no_instruction": None,
    # Instruction of the edos-v2-words-clean-20261104 run.
    "old_verifiable_evidence": (
        "Given a type of sexism and its description, retrieve passages containing "
        "specific verifiable evidence useful to counter this type of sexist hate speech."
    ),
    # Instruction currently in vector_store.CHUNK_INSTRUCTION (filled at runtime).
    "current_long": "CURRENT",
    "short_task_only": (
        "Given a type of sexist speech and its definition, retrieve passages "
        "that help counter this specific type of sexism."
    ),
}
REFERENCE_VARIANT = ("new_format", "current_long")
TABLE_OR_NUMBERS = re.compile(r"<table\b", re.I)


def numeric_share(text: str) -> float:
    tokens = re.findall(r"\w+", text)
    return sum(token.isdigit() for token in tokens) / max(1, len(tokens))


def is_statistical(text: str) -> bool:
    return bool(TABLE_OR_NUMBERS.search(text)) or numeric_share(text) > 0.15


def query_similarity(vectors: np.ndarray) -> dict:
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(vectors), k=1)]
    return {"mean_query_cosine": float(upper.mean()), "max_query_cosine": float(upper.max()),
            "min_query_cosine": float(upper.min())}


def hubness(rankings: dict[str, list[str]], k: int) -> dict:
    """rankings: label -> ranked chunk ids. Counts how chunks are shared across labels."""
    occurrences = Counter(chunk for ranked in rankings.values() for chunk in ranked[:k])
    labels = list(rankings)
    jaccard = [
        len(set(rankings[a][:k]) & set(rankings[b][:k])) / len(set(rankings[a][:k]) | set(rankings[b][:k]))
        for a, b in itertools.combinations(labels, 2)
    ]
    return {
        f"distinct_chunks@{k}": len(occurrences),
        f"max_labels_per_chunk@{k}": max(occurrences.values()),
        f"chunks_in_3plus_labels@{k}": sum(count >= 3 for count in occurrences.values()),
        f"mean_jaccard_between_labels@{k}": float(np.mean(jaccard)),
    }


def composition(rankings: dict[str, list[str]], chunks: dict, k: int) -> dict:
    selected = [chunks[chunk] for ranked in rankings.values() for chunk in ranked[:k]]
    documents = Counter(chunk["source"]["markdown_file"] for chunk in selected)
    return {
        f"distinct_documents@{k}": len(documents),
        f"top_document_share@{k}": round(documents.most_common(1)[0][1] / len(selected), 3),
        f"top_document@{k}": documents.most_common(1)[0][0],
        f"statistical_share@{k}": round(sum(is_statistical(chunk["body_text"]) for chunk in selected) / len(selected), 3),
    }


def overlap_with(rankings: dict, reference: dict, k: int) -> float:
    return float(np.mean([
        len(set(rankings[label][:k]) & set(reference[label][:k])) / len(set(rankings[label][:k]) | set(reference[label][:k]))
        for label in rankings
    ]))


def format_query(instruction: str | None, text: str) -> str:
    # Same template as FaissEvidenceRetriever.format_query.
    return text if instruction is None else f"Instruct: {instruction}\nQuery: {text}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-Embedding-8B", "BAAI/bge-m3"])
    parser.add_argument("--top-k", type=int, nargs="+", default=[4, 10, 100])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    from unito_amazon.card_creation.chunking import create_chunks
    from unito_amazon.card_creation.config import load_config, model_slug
    from unito_amazon.card_creation.queries import default_query_text
    from unito_amazon.card_creation.retrieval import chunks_to_documents, load_chunks
    from unito_amazon.card_creation.vector_store import CHUNK_INSTRUCTION, FaissEvidenceRetriever

    QUERY_FORMATS["new_format"] = default_query_text
    INSTRUCTIONS["current_long"] = CHUNK_INSTRUCTION
    config = load_config(args.config)
    common = config.model_copy(update={
        "model_scoped_artifacts": True, "deduplication_enabled": False,
        "chunk_tokenizer_model": config.chunk_tokenizer,
    })
    create_chunks(common)  # cached when chunks.jsonl is current
    chunk_models = load_chunks(common)
    documents = chunks_to_documents(chunk_models, common.include_section_titles)
    chunks = {chunk.chunk_id: chunk.model_dump(mode="json") for chunk in chunk_models}
    labels = json.loads(common.labels_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or common.output_dir / "diagnostics" / "query_instructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_k = max(args.top_k)

    results, listings = [], []
    for model in args.models:
        current = common.model_copy(update={"embedding_model": model})
        retriever = FaissEvidenceRetriever(
            model_name=model, instruction=CHUNK_INSTRUCTION, batch_size=current.embedding_batch_size,
            max_tokens=current.embedding_max_tokens, device=current.device,
        )
        try:
            retriever.load_or_index(documents, current.chunk_index_path)
            # BGE-M3 is not instruction-tuned: only the query formats are compared.
            instructions = {"no_instruction": None} if model == "BAAI/bge-m3" else INSTRUCTIONS
            model_rankings = {}
            for (format_name, build), (instruction_name, instruction) in itertools.product(
                QUERY_FORMATS.items(), instructions.items(),
            ):
                texts = [format_query(instruction, build(label, definition)) for label, definition in labels.items()]
                vectors = retriever._encode(texts)
                scores, positions = retriever.faiss_index.search(vectors, max_k)
                rankings = {
                    label: [retriever.documents[int(position)].id for position in row]
                    for label, row in zip(labels, positions)
                }
                model_rankings[(format_name, instruction_name)] = rankings
                row = {"model": model, "query_format": format_name, "instruction": instruction_name,
                       **query_similarity(vectors)}
                for k in args.top_k:
                    row.update(hubness(rankings, k))
                    if k <= 10:
                        row.update(composition(rankings, chunks, k))
                results.append(row)
                for label, ranked, label_scores in zip(labels, rankings.values(), scores):
                    listings.append({
                        "model": model, "query_format": format_name, "instruction": instruction_name,
                        "label": label, "query_text": texts[list(labels).index(label)],
                        "top10": [
                            {"rank": rank, "chunk_id": chunk, "score": round(float(score), 4),
                             "document": chunks[chunk]["source"]["markdown_file"],
                             "section": " > ".join(chunks[chunk]["section_path"]),
                             "start": chunks[chunk]["body_text"][:160]}
                            for rank, (chunk, score) in enumerate(zip(ranked[:10], label_scores[:10]), start=1)
                        ],
                    })
            reference = model_rankings.get(REFERENCE_VARIANT) or model_rankings[("new_format", "no_instruction")]
            for row in results:
                if row["model"] == model:
                    rankings = model_rankings[(row["query_format"], row["instruction"])]
                    row["jaccard_vs_reference@10"] = round(overlap_with(rankings, reference, 10), 3)
        finally:
            retriever.close()

    (output_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "top10_by_variant.jsonl").open("w", encoding="utf-8") as handle:
        for item in listings:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "summary.md").write_text(render_markdown(results, args.top_k), encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


def render_markdown(results: list[dict], top_k: list[int]) -> str:
    columns = ["query_format", "instruction", "mean_query_cosine", "max_query_cosine"]
    for k in top_k:
        columns += [f"distinct_chunks@{k}", f"chunks_in_3plus_labels@{k}", f"max_labels_per_chunk@{k}"]
    columns += ["distinct_documents@10", "top_document_share@10", "statistical_share@10", "jaccard_vs_reference@10"]
    lines = [
        "# Query format and instruction diagnostic", "",
        "Dense retrieval only (before reranking); 11 EDOS queries; no human labels.",
        "Lower query cosine, more distinct chunks and fewer chunks shared by 3+ labels mean less hubness.",
        f"Reference for Jaccard: {REFERENCE_VARIANT[0]} + {REFERENCE_VARIANT[1]}.", "",
    ]
    for model in dict.fromkeys(row["model"] for row in results):
        lines += [f"## {model}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for row in (row for row in results if row["model"] == model):
            values = [f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in columns]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
