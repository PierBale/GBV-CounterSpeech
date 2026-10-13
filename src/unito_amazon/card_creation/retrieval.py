import json
from collections import Counter

from unito_amazon.retrieval.schema import Document

from .config import CardCreationConfig
from .deduplication import (
    DEDUPLICATION_METHOD, SEMDEDUP_REFERENCE, normalized_text_hash, select_nonredundant,
)
from .io import read_jsonl, write_jsonl
from .progress import report
from .schema import Chunk, RankedPassage
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


def index_chunks(config: CardCreationConfig, force: bool = False) -> None:
    report("index", f"Preparazione indice paragrafi: {config.chunk_index_path}")
    retriever = chunk_retriever(config)
    try:
        retriever.load_or_index(
            chunks_to_documents(load_chunks(config)), config.chunk_index_path,
            force_reindex=force,
        )
    finally:
        retriever.close()


def retrieve_passages(config: CardCreationConfig) -> None:
    import numpy as np

    from .reranking import QwenReranker

    report("retrieve", f"Caricamento tipi di sessismo e descrizioni: {config.labels_path}")
    labels = json.loads(config.labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or not labels or any(
        not isinstance(label, str) or not isinstance(definition, str)
        or not label.strip() or not definition.strip()
        for label, definition in labels.items()
    ):
        raise ValueError("Labels JSON must map each EDOS label to a non-empty definition.")
    report(
        "retrieve", f"Categorie: {len(labels)}; top J={config.dense_top_j} nel retrieval, "
        f"top K={config.rerank_top_k} dopo reranking",
    )
    report(
        "dedup", f"Filtro attivo={config.deduplication_enabled}; "
        f"soglia coseno={config.deduplication_similarity_threshold}; confronto per categoria",
    )
    chunks = load_chunks(config)
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    retriever = chunk_retriever(config)
    dense_results = {}
    candidate_embeddings = {}
    try:
        retriever.load(config.chunk_index_path, chunks_to_documents(chunks))
        for position, (label, definition) in enumerate(labels.items(), start=1):
            report("retrieve", f"Retrieval categoria {position}/{len(labels)}: {label}")
            query = f"Sexism type: {label}\nDescription: {definition}"
            dense_results[label] = (query, retriever.retrieve(query, config.dense_top_j))
            report("retrieve", f"Categoria {position}/{len(labels)}: recuperati {len(dense_results[label][1])} candidati")
        if config.deduplication_enabled:
            candidate_ids = list(dict.fromkeys(
                item.document.id
                for _, candidates in dense_results.values()
                for item in candidates
            ))
            vectors = retriever.document_embeddings(candidate_ids)
            candidate_embeddings = dict(zip(candidate_ids, vectors, strict=True))
            report("dedup", f"Riutilizzati {len(candidate_embeddings)} embedding dei chunk; nessun nuovo modello")
    finally:
        # The embedding model is released before loading the reranker.
        retriever.close()

    report("retrieve", "Retrieval dense completato; avvio reranking dei candidati")
    reranker = QwenReranker(
        config.reranker_model, config.reranker_batch_size,
        config.reranker_max_tokens, config.device,
    )
    output = []
    all_ranked = []
    deduplication_log = []
    try:
        for position, (label, (query, candidates)) in enumerate(dense_results.items(), start=1):
            report("retrieve", f"Reranking categoria {position}/{len(labels)}: {label}; {len(candidates)} candidati")
            scores = reranker.score(query, [item.document.text for item in candidates])
            ordered = sorted(
                zip(candidates, scores, strict=True),
                key=lambda pair: (-pair[1], pair[0].rank),
            )
            # Preserve the original reranker rank, including gaps after filtering.
            ranked = [
                RankedPassage(
                    label=label, definition=labels[label],
                    embedding_model=config.embedding_model,
                    reranker_model=config.reranker_model,
                    chunk=lookup[candidate.document.id],
                    dense_rank=candidate.rank, dense_score=candidate.score,
                    reranker_rank=rank, reranker_score=score,
                )
                for rank, (candidate, score) in enumerate(ordered, start=1)
            ]
            all_ranked.extend(passage.model_dump(mode="json") for passage in ranked)
            report("dedup", f"Categoria {position}/{len(labels)}: filtro prima della selezione dei top K")
            embeddings = (
                np.stack([candidate_embeddings[passage.chunk.chunk_id] for passage in ranked])
                if config.deduplication_enabled else None
            )
            result = select_nonredundant(
                [passage.chunk.text for passage in ranked], embeddings,
                top_k=config.rerank_top_k,
                similarity_threshold=config.deduplication_similarity_threshold,
                enabled=config.deduplication_enabled,
            )
            output.extend(ranked[index].model_dump(mode="json") for index in result.selected_indices)
            for decision in result.decisions:
                passage = ranked[decision.index]
                reference = (
                    ranked[decision.reference_index]
                    if decision.reference_index is not None else None
                )
                is_duplicate = decision.reason in {"exact_duplicate", "near_duplicate"}
                deduplication_log.append({
                    "label": label, "chunk_id": passage.chunk.chunk_id,
                    "source": passage.chunk.source.model_dump(mode="json"),
                    "dense_rank": passage.dense_rank,
                    "reranker_rank": passage.reranker_rank,
                    "reranker_score": passage.reranker_score,
                    "decision": decision.reason,
                    "compared_to": reference.chunk.chunk_id if reference else None,
                    "duplicate_of": reference.chunk.chunk_id if is_duplicate and reference else None,
                    "kept_source": (
                        reference.chunk.source.model_dump(mode="json")
                        if is_duplicate and reference else None
                    ),
                    "cosine_similarity": decision.cosine_similarity,
                    "similarity_threshold": (
                        config.deduplication_similarity_threshold
                        if config.deduplication_enabled else None
                    ),
                    "deduplication_enabled": config.deduplication_enabled,
                    "method": DEDUPLICATION_METHOD if config.deduplication_enabled else "disabled",
                    "embedding_model": config.embedding_model,
                    "normalized_text_sha256": (
                        normalized_text_hash(passage.chunk.text)
                        if config.deduplication_enabled else None
                    ),
                    "scientific_reference": SEMDEDUP_REFERENCE,
                })
                if is_duplicate:
                    report(
                        "dedup", f"{decision.reason}: {passage.chunk.chunk_id} -> "
                        f"{reference.chunk.chunk_id}; coseno={decision.cosine_similarity:.4f}",
                    )
            counts = Counter(decision.reason for decision in result.decisions)
            report(
                "dedup", f"Categoria {position}/{len(labels)} completata: "
                f"{len(result.selected_indices)} selezionati, {counts['exact_duplicate']} duplicati esatti, "
                f"{counts['near_duplicate']} quasi-duplicati, {counts['top_k_limit']} oltre il limite K",
            )
            if len(result.selected_indices) < config.rerank_top_k:
                report("dedup", f"Solo {len(result.selected_indices)} passaggi non ridondanti disponibili per {label}")
    finally:
        reranker.close()
    report("retrieve", f"Salvataggio ranking completo ({len(all_ranked)} passaggi): {config.ranked_path}")
    write_jsonl(config.ranked_path, all_ranked)
    report("dedup", f"Salvataggio audit ({len(deduplication_log)} decisioni): {config.deduplication_log_path}")
    write_jsonl(config.deduplication_log_path, deduplication_log)
    report("retrieve", f"Salvataggio di {len(output)} coppie categoria/passaggio: {config.retrieved_path}")
    write_jsonl(config.retrieved_path, output)
    report("retrieve", "Passaggi rerankati salvati")
