import json
import re
import warnings
from collections.abc import Callable
from itertools import groupby

from .config import CardCreationConfig
from .io import fingerprint, write_jsonl
from .progress import report
from .schema import Chunk, Source


def markdown_paragraphs(markdown: str) -> list[tuple[int, tuple[str, ...], str]]:
    """Split on blank lines and headings; headings never become evidence."""
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    headings: dict[int, str] = {}
    paragraphs = []
    pending = []

    def flush():
        text = "\n".join(pending).strip()
        pending.clear()
        if not text or text == "NO_CONTENT_HERE":
            return
        # Remove presentation markup while retaining links' visible text.
        text = re.sub(r"!\[[^]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[*`]+", "", text).strip()
        if text and not re.fullmatch(r"[-_=\s]+", text):
            section = tuple(headings[level] for level in sorted(headings))
            paragraphs.append((len(paragraphs) + 1, section, text))

    for line in markdown.replace("\r\n", "\n").splitlines():
        match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = match.group(2).strip()
        elif not line.strip():
            flush()
        else:
            pending.append(line)
    flush()
    return paragraphs


def aggregate_paragraphs(
    paragraphs: list[tuple[int, tuple[str, ...], str]],
    count_tokens: Callable[[str], int],
    minimum: int = 100,
) -> list[dict]:
    """Greedily aggregate adjacent short paragraphs, without overlap or splits."""
    chunks = []

    def make_chunk(items):
        text = "\n\n".join(item[2] for item in items)
        return {
            "text": text,
            "paragraph_ids": [item[0] for item in items],
            "section_path": list(items[0][1]),
            "token_count": count_tokens(text),
        }

    # Group consecutive occurrences, not all paragraphs sharing a heading name.
    for _, section_group in groupby(paragraphs, key=lambda item: item[1]):
        section_chunks = []
        pending = []
        for paragraph in section_group:
            if count_tokens(paragraph[2]) >= minimum:
                if pending:
                    section_chunks.append(make_chunk([*pending, paragraph]))
                    pending = []
                else:
                    section_chunks.append(make_chunk([paragraph]))
            else:
                pending.append(paragraph)
                if count_tokens("\n\n".join(item[2] for item in pending)) >= minimum:
                    section_chunks.append(make_chunk(pending))
                    pending = []
        if pending:
            residual = make_chunk(pending)
            if section_chunks:
                previous = section_chunks[-1]
                previous["text"] += "\n\n" + residual["text"]
                previous["paragraph_ids"] += residual["paragraph_ids"]
                previous["token_count"] = count_tokens(previous["text"])
            else:
                section_chunks.append(residual)
        chunks.extend(section_chunks)
    return chunks


def create_chunks(config: CardCreationConfig) -> None:
    from transformers import AutoTokenizer

    report("chunk", f"Ricerca Markdown in {config.markdown_dir}")
    paths = sorted(config.markdown_dir.rglob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown files found in {config.markdown_dir}")
    report("chunk", f"Trovati {len(paths)} file; minimo desiderato: {config.min_chunk_tokens} token")
    report("chunk", f"Caricamento del solo tokenizer: {config.embedding_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
    report("chunk", "Tokenizer pronto; aggregazione senza overlap o spezzatura dei paragrafi")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    metadata = {}
    if config.source_metadata_path:
        report("chunk", f"Caricamento metadati bibliografici: {config.source_metadata_path}")
        metadata = json.loads(config.source_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Source metadata must map relative Markdown paths to objects.")

    chunks = []
    total_paragraphs = 0
    for position, path in enumerate(paths, start=1):
        relative = path.relative_to(config.markdown_dir).as_posix()
        report("chunk", f"File {position}/{len(paths)}: {relative}")
        paragraphs = markdown_paragraphs(path.read_text(encoding="utf-8"))
        total_paragraphs += len(paragraphs)
        report("chunk", f"Rilevati {len(paragraphs)} paragrafi; aggregazione in corso")
        manifest_path = path.with_suffix(".parse.json")
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists() else {}
        )
        source = Source.model_validate({
            "source_id": fingerprint(relative)[:16],
            "markdown_file": relative,
            "pdf_file": manifest.get("pdf_file"),
            "title": path.stem.replace("_", " "),
            **metadata.get(relative, {}),
        })
        file_chunks = aggregate_paragraphs(paragraphs, count_tokens, config.min_chunk_tokens)
        for item in file_chunks:
            if item["token_count"] > config.long_chunk_warning_tokens:
                warnings.warn(f"Long intact chunk in {relative}: {item['token_count']} tokens")
            chunk = Chunk(
                chunk_id="chunk_" + fingerprint(
                    [source.source_id, item["paragraph_ids"], item["text"]]
                )[:24],
                source=source, **item,
            )
            chunks.append(chunk.model_dump(mode="json"))
        report("chunk", f"File {position}/{len(paths)} completato: {len(file_chunks)} chunk")
    if not chunks:
        raise ValueError("Markdown files contain no non-empty paragraphs.")
    report("chunk", f"Salvataggio di {len(chunks)} chunk da {total_paragraphs} paragrafi")
    write_jsonl(config.chunks_path, chunks)
    report("chunk", f"Chunk salvati: {config.chunks_path}")
