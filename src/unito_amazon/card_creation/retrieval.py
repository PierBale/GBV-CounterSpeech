import json

from unito_amazon.retrieval.schema import Document

from .config import CardCreationConfig
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
    chunks = load_chunks(config)
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    retriever = chunk_retriever(config)
    dense_results = {}
    try:
        retriever.load(config.chunk_index_path, chunks_to_documents(chunks))
        for position, (label, definition) in enumerate(labels.items(), start=1):
            report("retrieve", f"Retrieval categoria {position}/{len(labels)}: {label}")
            query = f"Sexism type: {label}\nDescription: {definition}"
            dense_results[label] = (query, retriever.retrieve(query, config.dense_top_j))
            report("retrieve", f"Categoria {position}/{len(labels)}: recuperati {len(dense_results[label][1])} candidati")
    finally:
        # The embedding model is released before loading the reranker.
        retriever.close()

    report("retrieve", "Retrieval dense completato; avvio reranking dei candidati")
    reranker = QwenReranker(
        config.reranker_model, config.reranker_batch_size,
        config.reranker_max_tokens, config.device,
    )
    output = []
    try:
        for position, (label, (query, candidates)) in enumerate(dense_results.items(), start=1):
            report("retrieve", f"Reranking categoria {position}/{len(labels)}: {label}; {len(candidates)} candidati")
            scores = reranker.score(query, [item.document.text for item in candidates])
            ordered = sorted(
                zip(candidates, scores, strict=True),
                key=lambda pair: (-pair[1], pair[0].rank),
            )[:config.rerank_top_k]
            for rank, (candidate, score) in enumerate(ordered, start=1):
                output.append(RankedPassage(
                    label=label, definition=labels[label],
                    embedding_model=config.embedding_model,
                    reranker_model=config.reranker_model,
                    chunk=lookup[candidate.document.id],
                    dense_rank=candidate.rank, dense_score=candidate.score,
                    reranker_rank=rank, reranker_score=score,
                ).model_dump(mode="json"))
            report("retrieve", f"Categoria {position}/{len(labels)} completata: {len(candidates)} candidati -> {len(ordered)} selezionati")
    finally:
        reranker.close()
    report("retrieve", f"Salvataggio di {len(output)} coppie categoria/passaggio: {config.retrieved_path}")
    write_jsonl(config.retrieved_path, output)
    report("retrieve", "Passaggi rerankati salvati")
