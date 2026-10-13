from pathlib import Path

from unito_amazon.retrieval.schema import Document

from .config import CardCreationConfig
from .cache import StageCache
from .io import read_jsonl
from .progress import report
from .card_schema import EvidenceCard
from .vector_store import CARD_INSTRUCTION, FaissEvidenceRetriever


def load_card_documents(
    path: str | Path, validated_only: bool = False,
) -> list[Document]:
    report("cards", f"Caricamento e validazione libreria: {path}; validated_only={validated_only}")
    documents = []
    seen = set()
    for record in read_jsonl(path):
        card = EvidenceCard.model_validate(record)
        if card.card_id in seen:
            raise ValueError(f"Duplicate card ID: {card.card_id}")
        seen.add(card.card_id)
        if validated_only and card.status != "validated":
            continue
        if not card.source_quote.strip() or card.source_quote not in card.source_passage:
            raise ValueError(f"Invalid supporting quote in {card.card_id}")
        # Do not index the full source passage: this is retrieval over cards.
        text = (
            f"Evidence: {card.evidence}\n"
            f"Evidence connection: {card.reasoning}\n"
            f"EDOS category: {card.primary_edos_label}\n"
            f"Keywords: {', '.join(card.retrieval_keywords)}\n"
            f"Source: {card.source.title}"
        )
        documents.append(Document(
            id=card.card_id, text=text,
            metadata={
                "evidence_type": "card", "card_id": card.card_id,
                "source_file": card.source.markdown_file,
                "source": card.source.model_dump(mode="json"),
                "source_chunk_id": card.source_chunk_id,
                "section_path": card.section_path,
                "primary_edos_label": card.primary_edos_label,
                "source_quote": card.source_quote, "evidence": card.evidence,
                "validation": card.validation, "status": card.status,
            },
        ))
    if not documents:
        raise ValueError(f"No usable cards in {path}; create cards before enabling card-RAG.")
    report("cards", f"Libreria pronta: {len(documents)} carte utilizzabili su {len(seen)} lette")
    return documents


def index_cards(config: CardCreationConfig, force: bool = False, cache: StageCache | None = None) -> None:
    cache = cache or StageCache(config)
    ticket = cache.prepare("index-cards", files=[config.cards_path], force=force, preserve_existing=True)
    if ticket.mode == "cached":
        return
    report("index-cards", f"Preparazione indice delle carte: {config.card_index_path}")
    retriever = FaissEvidenceRetriever(
        model_name=config.embedding_model, instruction=CARD_INSTRUCTION,
        batch_size=config.embedding_batch_size, max_tokens=config.embedding_max_tokens,
        device=config.device,
    )
    try:
        retriever.load_or_index(
            load_card_documents(config.cards_path), config.card_index_path,
            force_reindex=force,
        )
    finally:
        retriever.close()
    cache.checkpoint(ticket, complete=True)
