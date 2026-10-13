"""Shared source corpus for controlled chunk/card RAG experiments."""

from pathlib import Path

from unito_amazon.card_creation.io import read_jsonl
from unito_amazon.card_creation.schema import Chunk

from .schema import Document


def load_chunk_documents(path: str | Path) -> list[Document]:
    documents = []
    seen = set()
    for row in read_jsonl(path):
        chunk = Chunk.model_validate(row)
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate source chunk ID: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        documents.append(Document(
            id=chunk.chunk_id, text=chunk.text,
            metadata={
                "evidence_type": "chunk",
                "source_file": chunk.source.markdown_file,
                "source": chunk.source.model_dump(mode="json"),
                "chunk": chunk.model_dump(mode="json", exclude={"source"}),
            },
        ))
    if len(documents) < 3:
        raise ValueError(f"Chunk RAG requires at least three chunks in {path}.")
    return documents


def validate_card_sources(cards: list[Document], chunks: list[Document]) -> None:
    """Reject cards created from a different or stale source corpus."""
    originals = {document.id: document for document in chunks}
    for card in cards:
        chunk = card.metadata["chunk"]
        original = originals.get(chunk["chunk_id"])
        if (original is None or chunk != original.metadata["chunk"]
                or card.metadata["source"] != original.metadata["source"]):
            raise ValueError(
                f"Card {card.id} does not match the source chunk corpus; "
                "check CHUNK_LIBRARY_PATH and CARD_LIBRARY_PATH."
            )
