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
from .editorial import EDITORIAL_RULES_VERSION, clean_body_markup, filter_editorial_paragraphs, plain_heading
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

    lines = markdown.replace("\r\n", "\n").splitlines()
    caption_open = False
    for position, line in enumerate(lines):
        match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match and is_caption_heading(match.group(2), lines[position + 1:]):
            # A table/figure caption marked up as a heading: it describes the
            # following object and must not become the section of later text.
            flush()
            pending.append(plain_heading(match.group(2)))
            caption_open = True
            if audit is not None:
                audit.append({
                    "action": "converted", "rules_version": EDITORIAL_RULES_VERSION,
                    "rule_id": "caption_heading_as_text", "rule_pattern": CAPTION_HEADING,
                    "line_number": position + 1, "line_text": line,
                })
        elif match:
            flush()
            caption_open = False
            level = len(match.group(1))
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = match.group(2).strip()
        elif not line.strip():
            if not caption_open:
                flush()
        elif re.fullmatch(r"\s*<hr\s*/?>\s*", line, re.I):
            flush()
            caption_open = False
        else:
            caption_open = False
            pending.append(line)
    flush()
    return paragraphs


# Numbered table/figure captions; other captions only when a table follows.
CAPTION_HEADING = r"(?:table|figure|fig\.?|chart|graph|map)\s*\d+[a-z]?\b.*"
WEAK_CAPTION_HEADING = r".*\(\s*%.*|.*%\s*\).*|percentage\s+of\b.*"


def is_caption_heading(heading: str, following: list[str]) -> bool:
    text = plain_heading(heading)
    if re.fullmatch(CAPTION_HEADING, text, re.I):
        return True
    if not re.fullmatch(WEAK_CAPTION_HEADING, text, re.I):
        return False
    next_line = next((line for line in following if line.strip()), "")
    return next_line.lstrip().lower().startswith("<table")


def aggregate_paragraphs(
    paragraphs: list[tuple[int, tuple[str, ...], str]],
    count_tokens: Callable[[str], int],
    minimum: int = 100,
    *, size_unit: Literal["tokens", "words"] = "tokens", merge_short_sections: bool = True,
) -> list[dict]:
    """Greedily aggregate adjacent short paragraphs, without overlap or splits.

    Paragraphs are first aggregated within their own section. A section that
    stays below the minimum as a whole ("orphan", e.g. a short KEY FINDINGS box)
    is then merged with the adjacent unit sharing the longest heading prefix
    (at least the top-level heading; the following unit wins ties, since such
    boxes usually introduce what follows). The merged chunk takes the common
    heading prefix as its section; the deeper headings of each merged part are
    kept as a line in the body, so no source heading is lost. The minimum is
    always checked on paragraph text only ("evidence" size), never on headings.
    """
    size_key = "word_count" if size_unit == "words" else "token_count"

    def evidence_size(items):
        return count_tokens("\n\n".join(item[2] for item in items))

    def common_prefix(first, second):
        prefix = []
        for left, right in zip(first, second):
            if left != right:
                break
            prefix.append(left)
        return tuple(prefix)

    def make_chunk(items, path):
        parts, previous = [], tuple(path)
        for item in items:
            if tuple(item[1]) != previous:
                headings = [plain_heading(heading) for heading in item[1][len(path):]]
                headings = [heading for heading in headings if heading]
                if headings:
                    parts.append(" > ".join(headings))
                previous = tuple(item[1])
            parts.append(item[2])
        text = "\n\n".join(parts)
        return {
            "text": text,
            "paragraph_ids": [item[0] for item in items],
            "section_path": list(path),
            "merged_section_paths": list(dict.fromkeys(tuple(item[1]) for item in items)),
            "evidence_size": evidence_size(items),
            size_key: count_tokens(text),
        }

    units = []  # [items, path, is_short], in document order
    # Removed layout/editorial material does not split a surviving section.
    for path, section_group in groupby(paragraphs, key=lambda item: item[1]):
        section_units = []
        pending = []
        for paragraph in section_group:
            if count_tokens(paragraph[2]) >= minimum:
                section_units.append([*pending, paragraph])
                pending = []
            else:
                pending.append(paragraph)
                if evidence_size(pending) >= minimum:
                    section_units.append(pending)
                    pending = []
        if pending:
            if section_units:
                section_units[-1] = section_units[-1] + pending
            else:
                section_units.append(pending)
        units.extend([items, tuple(path), evidence_size(items) < minimum] for items in section_units)

    changed = merge_short_sections
    while changed:
        changed = False
        for position, (items, path, short) in enumerate(units):
            if not short:
                continue
            candidates = []
            for neighbor, is_next in ((position + 1, True), (position - 1, False)):
                if 0 <= neighbor < len(units):
                    prefix = common_prefix(path, units[neighbor][1])
                    if prefix:
                        candidates.append((len(prefix), is_next, neighbor, prefix))
            if not candidates:
                continue
            _, _, neighbor, prefix = max(candidates)
            first, last = sorted((position, neighbor))
            merged = units[first][0] + units[last][0]
            units[first:last + 1] = [[merged, prefix, evidence_size(merged) < minimum]]
            changed = True
            break
    return [make_chunk(items, path) for items, path, _ in units]


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
        # Inline subsection headings of merged sections never count as evidence.
        if item["evidence_size"] < minimum or body_size < minimum or size < minimum:
            if audit is not None:
                audit.append({
                    "action": "excluded", "rules_version": EDITORIAL_RULES_VERSION,
                    "rule_id": f"below_minimum_{size_unit}", f"minimum_{size_unit}": minimum,
                    "paragraph_ids": item["paragraph_ids"], "section_path": item["section_path"],
                    "paragraph_text": body, "text": text,
                    f"body_{size_key}": body_size, size_key: size,
                })
            continue
        if len(item["merged_section_paths"]) > 1 and audit is not None:
            audit.append({
                "action": "merged", "rules_version": EDITORIAL_RULES_VERSION,
                "rule_id": "short_section_merged_with_adjacent",
                "rule_pattern": "orphan section below minimum; longest common heading prefix",
                "paragraph_ids": item["paragraph_ids"], "section_path": item["section_path"],
                "merged_section_paths": [list(path) for path in item["merged_section_paths"]],
                "text": text,
            })
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
        if config.editorial_filter_enabled:
            retained = [
                (paragraph_id, section, cleaned) for paragraph_id, section, text in retained
                if (cleaned := clean_body_markup(text))
            ]
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
