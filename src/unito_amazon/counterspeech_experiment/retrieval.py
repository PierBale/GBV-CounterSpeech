"""Dense retrieval over the v4 chunks (RAG) and over the card evidence (card-RAG).

Both use Qwen3-Embedding-8B, exact cosine search (FAISS) and the hateful message as
query; no reranking. The label strategies keep only the label's pool and take the
top-k of the same ranking: card-RAG + label the cards of the label, RAG + label the
chunks those cards were made from (the same 15 chunks per label, so the two differ
only in the evidence format: raw chunk vs card).
"""

from unito_amazon.card_creation.card_schema import EvidenceCard
from unito_amazon.card_creation.io import file_sha256, read_jsonl
from unito_amazon.card_creation.passage_text import format_chunk_text
from unito_amazon.card_creation.prompts import violence_type
from unito_amazon.card_creation.schema import Chunk
from unito_amazon.card_creation.vector_store import CARD_INSTRUCTION, FaissEvidenceRetriever
from unito_amazon.retrieval.schema import Document, RetrievalResult

from .config import ExperimentConfig
from .data import label_code


def chunk_documents(config: ExperimentConfig) -> list[Document]:
    documents = []
    for row in read_jsonl(config.chunks_path):
        chunk = Chunk.model_validate(row)
        documents.append(Document(
            id=chunk.chunk_id, text=format_chunk_text(chunk, config.include_section_titles),
            metadata={"evidence_type": "chunk", "chunk_id": chunk.chunk_id,
                      "source_file": chunk.source.markdown_file},
        ))
    if len({document.id for document in documents}) != len(documents):
        raise ValueError(f"Duplicate chunk IDs in {config.chunks_path}")
    return documents


def card_documents(config: ExperimentConfig, chunks: list[Document]) -> list[Document]:
    """Only the evidence is embedded and shown to the generator."""
    chunk_text = {document.id: document.metadata for document in chunks}
    documents = []
    for row in read_jsonl(config.cards_path):
        card = EvidenceCard.model_validate(row)
        if card.chunk.chunk_id not in chunk_text:
            raise ValueError(f"Card {card.card_id} comes from a chunk missing in {config.chunks_path}")
        documents.append(Document(
            id=card.card_id, text=card.evidence,
            metadata={"evidence_type": "card", "card_id": card.card_id,
                      "chunk_id": card.chunk.chunk_id, "source_file": card.source.markdown_file,
                      "edos_label": card.primary_edos_label, "edos_label_code": label_code(card.primary_edos_label),
                      "edos_definition": card.edos_definition,
                      "reranker_rank": card.retrieval.get("reranker_rank")},
        ))
    if len({document.id for document in documents}) != len(documents):
        raise ValueError(f"Duplicate card IDs in {config.cards_path}")
    return documents


def label_definitions(cards: list[Document]) -> dict[str, tuple[str, str]]:
    """EDOS code -> (category name without number, definition), as in the card prompt."""
    definitions = {}
    for card in cards:
        code = card.metadata["edos_label_code"]
        entry = (violence_type(card.metadata["edos_label"]), card.metadata["edos_definition"])
        if definitions.setdefault(code, entry) != entry:
            raise ValueError(f"Inconsistent label name/definition for {code} in the cards.")
    return definitions


def label_chunk_pools(cards: list[Document]) -> dict[str, set[str]]:
    """EDOS code -> ids of the chunks the cards of that label were made from."""
    pools = {}
    for card in cards:
        pools.setdefault(card.metadata["edos_label_code"], set()).add(card.metadata["chunk_id"])
    return pools


class ExperimentRetrievers:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.chunks = chunk_documents(config)
        self.cards = card_documents(config, self.chunks)
        self.definitions = label_definitions(self.cards)
        self.label_chunks = label_chunk_pools(self.cards)
        options = dict(model_name=config.embedding_model, instruction=CARD_INSTRUCTION,
                       batch_size=config.embedding_batch_size, max_tokens=config.embedding_max_tokens,
                       device=config.device)
        self.chunk_retriever = FaissEvidenceRetriever(retriever_name="rag_chunks_faiss", **options)
        self.card_retriever = FaissEvidenceRetriever(retriever_name="card_rag_faiss", **options)

    def load(self) -> None:
        # One encoder for both indexes and the queries: the weights are loaded once.
        self.card_retriever.encoder = self.chunk_retriever._encoder()
        self.card_retriever.max_tokens = self.chunk_retriever.max_tokens
        self.chunk_retriever.load_or_index(self.chunks, self.config.index_dir / "chunks_qwen3-embedding-8b.faiss")
        self.card_retriever.load_or_index(self.cards, self.config.index_dir / "cards_evidence_qwen3-embedding-8b.faiss")

    def retrieve(self, message: str, predicted_label: str,
                 gold_label: str | None = None) -> dict[str, list[RetrievalResult]]:
        k = self.config.top_k
        ranked_cards = self.card_retriever.retrieve(message, top_k=len(self.cards))
        ranked_chunks = self.chunk_retriever.retrieve(message, top_k=len(self.chunks))

        def within(ranked: list[RetrievalResult], label: str, kind: str) -> list[RetrievalResult]:
            """Top-k of the same ranking within the label's pool (same scores, ranks renumbered)."""
            code = label_code(label)
            if kind == "cards":
                filtered = [result for result in ranked if result.document.metadata["edos_label_code"] == code]
            else:
                filtered = [result for result in ranked if result.document.id in self.label_chunks.get(code, set())]
            if len(filtered) < k:
                raise ValueError(f"Only {len(filtered)} {kind} for label {label!r}; top_k={k}.")
            return [result.model_copy(update={"rank": rank}) for rank, result in enumerate(filtered[:k], start=1)]

        retrieved = {
            "rag": ranked_chunks[:k],
            "rag_label": within(ranked_chunks, predicted_label, "chunks"),
            "card_rag": ranked_cards[:k],
            "card_rag_label": within(ranked_cards, predicted_label, "cards"),
        }
        if gold_label is not None:
            retrieved["rag_gold_label"] = within(ranked_chunks, gold_label, "chunks")
            retrieved["card_rag_gold_label"] = within(ranked_cards, gold_label, "cards")
        return retrieved

    def metadata(self) -> dict:
        return {
            "embedding_model": self.config.embedding_model, "query_instruction": CARD_INSTRUCTION,
            "similarity": "cosine (normalized embeddings, FAISS IndexFlatIP)", "reranking": False,
            "top_k": self.config.top_k,
            "chunks_path": str(self.config.chunks_path), "chunks_sha256": file_sha256(self.config.chunks_path),
            "chunk_count": len(self.chunks), "chunk_text": "section titles + body" if self.config.include_section_titles else "body",
            "cards_path": str(self.config.cards_path), "cards_sha256": file_sha256(self.config.cards_path),
            "card_count": len(self.cards), "card_text": "evidence only",
            "label_pools": "label strategies: top-k within the cards of the label (card-RAG) or within "
                           "the chunks those cards were made from (RAG)",
            "label_chunk_pool_sizes": {code: len(pool) for code, pool in sorted(self.label_chunks.items())},
        }

    def close(self) -> None:
        self.card_retriever.encoder = None
        self.chunk_retriever.close()
