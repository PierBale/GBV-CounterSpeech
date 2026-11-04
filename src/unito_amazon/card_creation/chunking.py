import json
import re
import warnings
from collections.abc import Callable
from itertools import groupby
from html.parser import HTMLParser
from typing import Literal

from .config import CardCreationConfig
from .cache import StageCache
from .io import file_sha256, fingerprint, write_jsonl
from .editorial import EDITORIAL_RULES_VERSION, filter_editorial_paragraphs
from .layout import strip_layout_noise
from .passage_text import compose_passage_text
from .progress import report
from .schema import Chunk, Source


WORD_COUNT_METHOD = "Unicode words/numbers; internal apostrophes and hyphens retained; HTML tags excluded"
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’\-‐‑][^\W_]+)*", re.UNICODE)


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def count_words(text: str) -> int:
    """Count visible words and numbers, ignoring markup and punctuation-only spans."""
    parser = _VisibleText()
    parser.feed(text)
    parser.close()
    return len(WORD_PATTERN.findall(" ".join(parser.parts)))


def markdown_paragraphs(
    markdown: str, *, clean_layout: bool = False, audit: list[dict] | None = None,
) -> list[tuple[int, tuple[str, ...], str]]:
    """Split on blank lines and headings; headings never become evidence."""
    if clean_layout:
        markdown, decisions = strip_layout_noise(markdown)
        if audit is not None:
            audit.extend(decisions)
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
    *, size_unit: Literal["tokens", "words"] = "tokens",
) -> list[dict]:
    """Greedily aggregate adjacent short paragraphs, without overlap or splits."""
    chunks = []

    def make_chunk(items):
        text = "\n\n".join(item[2] for item in items)
        return {
            "text": text,
            "paragraph_ids": [item[0] for item in items],
            "section_path": list(items[0][1]),
            ("word_count" if size_unit == "words" else "token_count"): count_tokens(text),
        }

    # Removed layout/editorial material does not split a surviving section.
    # Real section paths still delimit aggregation.
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
                previous["word_count" if size_unit == "words" else "token_count"] = count_tokens(previous["text"])
            else:
                section_chunks.append(residual)
        chunks.extend(section_chunks)
    return chunks


def build_document_chunks(
    paragraphs: list[tuple[int, tuple[str, ...], str]], source: Source,
    count_tokens: Callable[[str], int] | None = None, minimum: int = 100,
    include_section_titles: bool = True, audit: list[dict] | None = None,
    *, size_unit: Literal["tokens", "words"] = "tokens",
) -> list[Chunk]:
    """Materialize complete model inputs, enforcing the configured minimum.

    The minimum applies to the body as well: headings cannot turn a tiny
    fragment into an eligible evidence passage. Unmergeable sections are
    excluded with their full body in the audit, never silently discarded.
    """
    counter = count_words if size_unit == "words" else count_tokens
    if counter is None:
        raise ValueError("Token-based chunking requires a token counter.")
    size_key = "word_count" if size_unit == "words" else "token_count"
    result = []
    for item in aggregate_paragraphs(paragraphs, counter, minimum, size_unit=size_unit):
        body = item["text"]
        text = compose_passage_text(body, item["section_path"], include_section_titles)
        size = counter(text)
        body_size = item[size_key]
        if body_size < minimum or size < minimum:
            if audit is not None:
                audit.append({
                    "action": "excluded", "rules_version": EDITORIAL_RULES_VERSION,
                    "rule_id": f"below_minimum_{size_unit}", f"minimum_{size_unit}": minimum,
                    "paragraph_ids": item["paragraph_ids"], "section_path": item["section_path"],
                    "paragraph_text": body, "text": text,
                    f"body_{size_key}": body_size, size_key: size,
                })
            continue
        result.append(Chunk(
            chunk_id="chunk_" + fingerprint([source.source_id, item["paragraph_ids"], body])[:24],
            source=source, text=text, body_text=body,
            token_count=size if size_unit == "tokens" else None,
            body_token_count=body_size if size_unit == "tokens" else None,
            word_count=count_words(text), body_word_count=count_words(body),
            paragraph_ids=item["paragraph_ids"],
            section_path=item["section_path"],
        ))
    return result


def create_chunks(config: CardCreationConfig, cache: StageCache | None = None) -> None:
    cache = cache or StageCache(config)
    ticket = cache.prepare(
        "chunk", files=[config.source_metadata_path] if config.source_metadata_path else [],
        directories=[(config.markdown_dir, ("*.md", "*.parse.json"))],
    )
    if ticket.mode == "cached":
        return
    report("chunk", f"Ricerca Markdown in {config.markdown_dir}")
    paths = sorted(config.markdown_dir.rglob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown files found in {config.markdown_dir}")
    report("chunk", f"Trovati {len(paths)} file; minimo effettivo: {config.min_chunk_size} {config.chunk_size_unit}")
    count_tokens = None
    if config.chunk_size_unit == "tokens":
        from transformers import AutoTokenizer
        report("chunk", f"Caricamento del solo tokenizer: {config.chunk_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(config.chunk_tokenizer)
        count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    report("chunk", "Aggregazione senza overlap o spezzatura dei paragrafi")

    metadata = {}
    if config.source_metadata_path:
        report("chunk", f"Caricamento metadati bibliografici: {config.source_metadata_path}")
        metadata = json.loads(config.source_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Source metadata must map relative Markdown paths to objects.")

    chunks = []
    audit = []
    total_paragraphs = 0
    for position, path in enumerate(paths, start=1):
        relative = path.relative_to(config.markdown_dir).as_posix()
        report("chunk", f"File {position}/{len(paths)}: {relative}")
        decisions = []
        paragraphs = markdown_paragraphs(
            path.read_text(encoding="utf-8"), clean_layout=config.editorial_filter_enabled,
            audit=decisions,
        )
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
        retained, excluded = filter_editorial_paragraphs(paragraphs, config.editorial_filter_enabled)
        decisions.extend(excluded)
        file_chunks = build_document_chunks(
            retained, source, count_tokens, config.min_chunk_size,
            config.include_section_titles, audit=decisions, size_unit=config.chunk_size_unit,
        )
        markdown_digest = file_sha256(path)
        audit.extend({
            **decision, "source_id": source.source_id, "markdown_file": relative,
            "markdown_sha256": markdown_digest,
        } for decision in decisions)
        report("chunk", f"{len(decisions)} rimozioni/esclusioni registrate nell'audit")
        for chunk in file_chunks:
            if chunk.token_count is not None and chunk.token_count > config.long_chunk_warning_tokens:
                warnings.warn(f"Long intact chunk in {relative}: {chunk.token_count} tokens")
            chunks.append(chunk.model_dump(mode="json"))
        report("chunk", f"File {position}/{len(paths)} completato: {len(file_chunks)} chunk")
    write_jsonl(config.editorial_audit_path, audit)
    if not chunks:
        raise ValueError(f"No paragraphs remain for chunking; inspect {config.editorial_audit_path}.")
    report("chunk", f"Salvataggio di {len(chunks)} chunk da {total_paragraphs} paragrafi")
    write_jsonl(config.chunks_path, chunks)
    cache.checkpoint(ticket, complete=True)
    report("chunk", f"Chunk salvati: {config.chunks_path}")
