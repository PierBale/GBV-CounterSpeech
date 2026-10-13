import json
from collections import Counter

import numpy as np

from unito_amazon.retrieval.schema import Document

from .cache import StageCache
from .config import CardCreationConfig
from .deduplication import (
    DEDUPLICATION_METHOD, SEMDEDUP_REFERENCE, normalized_text_hash, select_nonredundant,
)
from .io import read_jsonl, write_jsonl
from .progress import report
from .schema import Chunk, DensePassage, RankedPassage
from .vector_store import CHUNK_INSTRUCTION, FaissEvidenceRetriever


def load_chunks(config: CardCreationConfig) -> list[Chunk]:
    report("retrieval", f"Caricamento e validazione chunk: {config.chunks_path}")
    chunks = [Chunk.model_validate(record) for record in read_jsonl(config.chunks_path)]
    if not chunks:
        raise ValueError("Chunk library is empty; run the chunk stage first.")
    report("retrieval", f"Caricati {len(chunks)} chunk")
    return chunks


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    return [
        Document(
            id=chunk.chunk_id, text=chunk.text,
            metadata={"source": chunk.source.model_dump(), "section_path": chunk.section_path},
        )
        for chunk in chunks
    ]


def chunk_retriever(config: CardCreationConfig) -> FaissEvidenceRetriever:
    return FaissEvidenceRetriever(
        model_name=config.embedding_model, instruction=CHUNK_INSTRUCTION,
        batch_size=config.embedding_batch_size, max_tokens=config.embedding_max_tokens,
        device=config.device, retriever_name="paragraphs_qwen3_faiss",
    )


def index_chunks(
    config: CardCreationConfig, force: bool = False, cache: StageCache | None = None,
) -> None:
    cache = cache or StageCache(config)
    ticket = cache.prepare("index", files=[config.chunks_path], force=force, preserve_existing=True)
    if ticket.mode == "cached":
        return
    report("index", f"Preparazione indice paragrafi: {config.chunk_index_path}")
    retriever = chunk_retriever(config)
    try:
        retriever.load_or_index(
            chunks_to_documents(load_chunks(config)), config.chunk_index_path,
            force_reindex=force,
        )
    finally:
        retriever.close()
    cache.checkpoint(ticket, complete=True)


def _group_passages(passages, labels: dict[str, str], config: CardCreationConfig) -> dict:
    groups = {}
    seen = set()
    for passage in passages:
        key = (passage.label, passage.chunk.chunk_id)
        if key in seen:
            raise ValueError(f"Duplicate label/chunk pair in checkpoint: {key}")
        seen.add(key)
        if labels.get(passage.label) != passage.definition or passage.embedding_model != config.embedding_model:
            raise ValueError("Checkpoint passages do not match current categories/embedding model.")
        if isinstance(passage, RankedPassage) and passage.reranker_model != config.reranker_model:
            raise ValueError("Checkpoint passages do not match the current reranker model.")
        groups.setdefault(passage.label, []).append(passage)
    return groups


def _dense_retrieval(config: CardCreationConfig, cache: StageCache, labels: dict[str, str]):
    ticket = cache.prepare(
        "dense",
        files=[config.chunks_path, config.chunk_index_path,
               config.chunk_index_path.with_suffix(".faiss.json"), config.labels_path],
        allow_partial=True,
    )
    if ticket.mode == "cached":
        return [DensePassage.model_validate(item) for item in read_jsonl(config.dense_path)]
    passages = (
        [DensePassage.model_validate(item) for item in read_jsonl(config.dense_path)]
        if ticket.mode == "resume" and config.dense_path.exists() else []
    )
    groups = _group_passages(passages, labels, config)
    chunks = load_chunks(config)
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    expected_count = min(config.dense_top_j, len(chunks))
    for group in groups.values():
        if (len(group) != expected_count
                or sorted(item.dense_rank for item in group) != list(range(1, expected_count + 1))
                or any(lookup.get(item.chunk.chunk_id) != item.chunk for item in group)):
            raise ValueError("Invalid partial dense-retrieval checkpoint.")
    remaining = [label for label in labels if label not in groups]
    report("retrieve", f"Checkpoint dense: {len(groups)}/{len(labels)} categorie completate")
    if remaining:
        retriever = chunk_retriever(config)
        try:
            retriever.load(config.chunk_index_path, chunks_to_documents(chunks))
            for label in remaining:
                position = list(labels).index(label) + 1
                report("retrieve", f"Retrieval categoria {position}/{len(labels)}: {label}")
                query = f"Sexism type: {label}\nDescription: {labels[label]}"
                results = retriever.retrieve(query, config.dense_top_j)
                if len(results) != expected_count:
                    raise ValueError(f"Unexpected number of dense candidates for {label}.")
                passages.extend(
                    DensePassage(
                        label=label, definition=labels[label], embedding_model=config.embedding_model,
                        chunk=lookup[item.document.id], dense_rank=item.rank, dense_score=item.score,
                    )
                    for item in results
                )
                write_jsonl(config.dense_path, (item.model_dump(mode="json") for item in passages))
                cache.checkpoint(ticket)
                report("retrieve", f"{label}: {len(results)} candidati salvati in {config.dense_path}")
        finally:
            retriever.close()
    cache.checkpoint(ticket, complete=True)
    return passages


def _rerank_passages(config: CardCreationConfig, cache: StageCache, labels: dict[str, str], dense):
    ticket = cache.prepare("rerank", files=[config.dense_path], allow_partial=True)
    if ticket.mode == "cached":
        return [RankedPassage.model_validate(item) for item in read_jsonl(config.ranked_path)]
    ranked = (
        [RankedPassage.model_validate(item) for item in read_jsonl(config.ranked_path)]
        if ticket.mode == "resume" and config.ranked_path.exists() else []
    )
    dense_groups = _group_passages(dense, labels, config)
    ranked_groups = _group_passages(ranked, labels, config)
    for label, group in ranked_groups.items():
        expected = {item.chunk.chunk_id: item for item in dense_groups[label]}
        if (len(group) != len(expected)
                or sorted(item.reranker_rank for item in group) != list(range(1, len(expected) + 1))):
            raise ValueError("Invalid partial reranking checkpoint.")
        for item in group:
            original = expected.get(item.chunk.chunk_id)
            if (original is None or original.chunk != item.chunk
                    or original.dense_rank != item.dense_rank or original.dense_score != item.dense_score):
                raise ValueError("Reranking checkpoint differs from dense retrieval.")
    remaining = [label for label in labels if label not in ranked_groups]
    report("retrieve", f"Checkpoint reranking: {len(ranked_groups)}/{len(labels)} categorie completate")
    if remaining:
        from .reranking import QwenReranker

        reranker = QwenReranker(
            config.reranker_model, config.reranker_batch_size,
            config.reranker_max_tokens, config.device,
        )
        try:
            for label in remaining:
                position = list(labels).index(label) + 1
                candidates = sorted(dense_groups[label], key=lambda item: item.dense_rank)
                query = f"Sexism type: {label}\nDescription: {labels[label]}"
                report("retrieve", f"Reranking categoria {position}/{len(labels)}: {label}")
                scores = reranker.score(query, [item.chunk.text for item in candidates])
                ordered = sorted(zip(candidates, scores, strict=True), key=lambda pair: (-pair[1], pair[0].dense_rank))
                ranked.extend(
                    RankedPassage(
                        **candidate.model_dump(mode="json"), reranker_model=config.reranker_model,
                        reranker_rank=rank, reranker_score=score,
                    )
                    for rank, (candidate, score) in enumerate(ordered, start=1)
                )
                write_jsonl(config.ranked_path, (item.model_dump(mode="json") for item in ranked))
                cache.checkpoint(ticket)
                report("retrieve", f"{label}: ranking completo salvato in {config.ranked_path}")
        finally:
            reranker.close()
    cache.checkpoint(ticket, complete=True)
    return ranked


def _deduplicate(config: CardCreationConfig, cache: StageCache, labels: dict[str, str], ranked) -> None:
    index_inputs = [config.chunk_index_path, config.chunk_index_path.with_suffix(".faiss.json")]
    ticket = cache.prepare(
        "dedup", files=[config.ranked_path, *(index_inputs if config.deduplication_enabled else [])],
    )
    if ticket.mode == "cached":
        return
    groups = _group_passages(ranked, labels, config)
    candidate_embeddings = {}
    if config.deduplication_enabled:
        retriever = chunk_retriever(config)
        try:
            retriever.load(config.chunk_index_path, chunks_to_documents(load_chunks(config)))
            candidate_ids = list(dict.fromkeys(item.chunk.chunk_id for item in ranked))
            vectors = retriever.document_embeddings(candidate_ids)
            candidate_embeddings = dict(zip(candidate_ids, vectors, strict=True))
            report("dedup", f"Riutilizzati {len(candidate_embeddings)} embedding; nessun nuovo modello")
        finally:
            retriever.close()
    output = []
    audit = []
    for position, label in enumerate(labels, start=1):
        passages = sorted(groups[label], key=lambda item: item.reranker_rank)
        embeddings = (
            np.stack([candidate_embeddings[item.chunk.chunk_id] for item in passages])
            if config.deduplication_enabled else None
        )
        result = select_nonredundant(
            [item.chunk.text for item in passages], embeddings, top_k=config.rerank_top_k,
            similarity_threshold=config.deduplication_similarity_threshold,
            enabled=config.deduplication_enabled,
        )
        output.extend(passages[index].model_dump(mode="json") for index in result.selected_indices)
        for decision in result.decisions:
            passage = passages[decision.index]
            reference = passages[decision.reference_index] if decision.reference_index is not None else None
            is_duplicate = decision.reason in {"exact_duplicate", "near_duplicate"}
            audit.append({
                "label": label, "chunk_id": passage.chunk.chunk_id,
                "source": passage.chunk.source.model_dump(mode="json"),
                "dense_rank": passage.dense_rank, "reranker_rank": passage.reranker_rank,
                "reranker_score": passage.reranker_score, "decision": decision.reason,
                "compared_to": reference.chunk.chunk_id if reference else None,
                "duplicate_of": reference.chunk.chunk_id if is_duplicate and reference else None,
                "kept_source": reference.chunk.source.model_dump(mode="json") if is_duplicate and reference else None,
                "cosine_similarity": decision.cosine_similarity,
                "similarity_threshold": config.deduplication_similarity_threshold if config.deduplication_enabled else None,
                "deduplication_enabled": config.deduplication_enabled,
                "method": DEDUPLICATION_METHOD if config.deduplication_enabled else "disabled",
                "embedding_model": config.embedding_model,
                "normalized_text_sha256": normalized_text_hash(passage.chunk.text) if config.deduplication_enabled else None,
                "scientific_reference": SEMDEDUP_REFERENCE,
            })
            if is_duplicate:
                report(
                    "dedup", f"{decision.reason}: {passage.chunk.chunk_id} -> "
                    f"{reference.chunk.chunk_id}; coseno={decision.cosine_similarity:.4f}",
                )
        counts = Counter(item.reason for item in result.decisions)
        report(
            "dedup", f"Categoria {position}/{len(labels)}: {len(result.selected_indices)} selezionati, "
            f"{counts['exact_duplicate']} duplicati esatti, {counts['near_duplicate']} quasi-duplicati, "
            f"{counts['top_k_limit']} oltre il limite K",
        )
        if len(result.selected_indices) < config.rerank_top_k:
            report("dedup", f"Solo {len(result.selected_indices)} passaggi non ridondanti per {label}")
    write_jsonl(config.deduplication_log_path, audit)
    write_jsonl(config.retrieved_path, output)
    cache.checkpoint(ticket, complete=True)
    report("dedup", f"Audit salvato: {config.deduplication_log_path}")
    report("retrieve", f"{len(output)} passaggi selezionati salvati: {config.retrieved_path}")


def retrieve_passages(config: CardCreationConfig, cache: StageCache | None = None) -> None:
    cache = cache or StageCache(config)
    report("retrieve", f"Caricamento tipi di sessismo e descrizioni: {config.labels_path}")
    labels = json.loads(config.labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or not labels or any(
        not isinstance(label, str) or not isinstance(definition, str)
        or not label.strip() or not definition.strip() for label, definition in labels.items()
    ):
        raise ValueError("Labels JSON must map each category to a non-empty definition.")
    report("retrieve", f"Categorie={len(labels)}; top J={config.dense_top_j}; top K={config.rerank_top_k}")
    report(
        "dedup", f"Filtro attivo={config.deduplication_enabled}; "
        f"soglia coseno={config.deduplication_similarity_threshold}",
    )
    dense = _dense_retrieval(config, cache, labels)
    ranked = _rerank_passages(config, cache, labels, dense)
    _deduplicate(config, cache, labels, ranked)
