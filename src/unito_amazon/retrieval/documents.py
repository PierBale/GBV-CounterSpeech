import re
from pathlib import Path

import pandas as pd

from .schema import Document


DEFAULT_TEXT_FIELD = "verbalizzazione"


def markdown_directory_to_documents(
    directory: str | Path,
    min_chars: int = 40,
    max_chars: int = 1600,
) -> list[Document]:
    """Load Markdown files as paragraph-level retrieval documents."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Markdown directory not found: {directory}")

    paths = sorted(directory.rglob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown files found in: {directory}")

    documents: list[Document] = []
    seen_texts: set[str] = set()

    for path in paths:
        headings: dict[int, str] = {}
        document_index = 0
        raw_text = path.read_text(encoding="utf-8")

        for block in re.split(r"\n\s*\n+", raw_text):
            body_lines = []

            for line in block.splitlines():
                heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line.strip())
                if heading_match:
                    level = len(heading_match.group(1))
                    headings[level] = heading_match.group(2).strip()
                    headings = {
                        key: value
                        for key, value in headings.items()
                        if key <= level
                    }
                else:
                    body_lines.append(line)

            body = _normalize_markdown_text("\n".join(body_lines))
            if len(body) < min_chars:
                continue

            section_path = [headings[level] for level in sorted(headings)]
            section = " > ".join(section_path)

            for text in _split_long_text(body, max_chars=max_chars):
                deduplication_key = text.casefold()
                if deduplication_key in seen_texts:
                    continue
                seen_texts.add(deduplication_key)

                document_index += 1
                retrieval_text = f"{section}\n{text}" if section else text
                documents.append(
                    Document(
                        id=f"{path.stem}-{document_index}",
                        text=retrieval_text,
                        metadata={
                            "source_file": path.name,
                            "source_path": str(path),
                            "section": section or None,
                        },
                    )
                )

    if len(documents) < 3:
        raise ValueError(
            f"RAG requires at least three documents; loaded {len(documents)}."
        )

    return documents


def _normalize_markdown_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        additional_length = len(word) + (1 if current_words else 0)
        if current_words and current_length + additional_length > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += additional_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def dataframe_to_documents(
    df: pd.DataFrame,
    text_field: str = DEFAULT_TEXT_FIELD,
    id_field: str | None = None,
) -> list[Document]:
    """
    Converte un DataFrame normalizzato in una lista di Document.

    Parameters
    ----------
    df:
        Dataset normalizzato.
    text_field:
        Colonna da usare come testo indicizzabile.
        Esempi: "verbalizzazione", "risposta", "domanda".
    id_field:
        Colonna da usare come id documento. Se None, usa l'indice del DataFrame.
    """

    if text_field not in df.columns:
        raise ValueError(f"Colonna text_field non trovata nel DataFrame: {text_field}")

    documents = []

    for idx, row in df.iterrows():
        doc_id = str(row[id_field]) if id_field and id_field in df.columns else str(idx)

        text = row[text_field]

        if pd.isna(text):
            continue

        metadata = row.to_dict()

        documents.append(
            Document(
                id=doc_id,
                text=str(text),
                metadata=metadata,
            )
        )

    return documents
