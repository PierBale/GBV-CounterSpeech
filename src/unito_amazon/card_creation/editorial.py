"""Versioned syntactic exclusions; no relevance classifier or inferred metadata."""

import re
from html.parser import HTMLParser


EDITORIAL_RULES_VERSION = "4.0"

# Whole heading matches only. Forewords, summaries and explanatory notes survive.
EDITORIAL_SECTION = r"(?:imprint|disclaimer\s*(?:and\s*)?copyright)"
ACKNOWLEDGEMENTS_SECTION = r"acknowledg(?:e)?ments?"
DIRECTORY_SECTION = r"(?:list\s+of\s+(?:[\w&.-]+\s+){0,5}members|directory\s+of\s+(?:organisations|organizations|members))(?:\s*\([^)]*\))?"
# Reference lists contain no explanatory notes: the whole section is excluded.
REFERENCE_LIST_SECTION = r"(?:\d+[.)]?\s*)?(?:references|bibliography|list\s+of\s+references|works\s+cited)"
# Note sections mix citations and explanatory notes: judged note by note.
NOTES_SECTION = r"(?:\d+[.)]?\s*)?(?:endnotes|footnotes|notes|sources used to prepare the factsheet)"
BIBLIOGRAPHIC_SECTION = rf"(?:{REFERENCE_LIST_SECTION}|{NOTES_SECTION})"
CONTENTS_SECTION = r"(?:contents|table\s+of\s+contents|list\s+of\s+(?:tables|figures|boxes|charts|graphs|maps|illustrations))"
CONTENTS_ROW = r".+\S(?:[ \t.]+|[ \t]*/[ \t]*)(?:p\.?[ \t]*)?(?:\d{1,4}|[ivxlcdm]{1,8})"
HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)
NOTE_MARKER = r"^\s*(?:\(<sup>\s*\d{1,4}\s*</sup>\)|<sup>\s*\(?\d{1,4}\)?\s*</sup>|\[\d{1,4}\]|\(?\d{1,4}\)?[.)]?\s+)"
# Markers that identify a note anywhere in the body (unlike '1.', which also opens list items).
STRONG_NOTE_MARKER = r"^\s*(?:\(<sup>\s*\d{1,4}\s*</sup>\)|<sup>\s*\(?\d{1,4}\)?\s*</sup>|\[\d{1,4}\])"
# Syntactic signs that a note is (or contains) a source reference.
CITATION_SIGNAL = re.compile(
    r"https?://|www\.|\bdoi\b|\bavailable\s+(?:at|from|online)\b|\bretrieved\s+from\b|\baccessed\b|"
    r"\((?:19|20)\d{2}[a-z]?\)|\(n\.d\.\)|\bop\.?\s*cit\b|\bibid\b|\bet\s+al\b|"
    r",\s*\d{1,4}(?:[–-]\d{1,4})?\.\s*$",
    re.IGNORECASE,
)
# A quoted title: capitalised and short, followed by punctuation (case-sensitive,
# so quoted legal wording such as “to promote awareness ...” is not a title).
QUOTED_TITLE = re.compile(r"[“\"‘][A-Z0-9][^”\"’]{2,120}?[.,]?[”\"’]\s*(?:[,.(]|$)")
EXPLANATORY_NOTE_MIN_WORDS = 10
LONG_NOTE_MIN_WORDS = 40
# Lowercase only: capitalised words in quoted titles ("When Women Are ...") do not count.
PROSE_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|may|must|should|affects?|increases?|reduces?|"
    r"found|finds|shows?|showed|reports?|reported|estimates?|indicates?|suggests?|"
    r"provides?|concludes?|demonstrates?|experienced|requires?)\b"
)

# Whole paragraphs, except explicitly identified publisher boilerplate prefixes.
PARAGRAPH_RULES = (
    ("icon_only", r"icon:\s*[^\n]+(?:\n\s*icon:\s*[^\n]+)*"),
    ("logo_only", r"logo:\s*[^\n]+(?:\n\s*logo:\s*[^\n]+)*"),
    ("isbn_only", r"ISBN(?:-1[03])?\s*:?\s*\d[\d Xx-]{8,}(?:\s*\([^()\n]*\))?"),
    ("copyright_notice", r"©[^\n]*\b(?:19|20)\d{2}\b[^\n]*"),
    ("cover_credit", r"Cover\s+(?:photo|design(?:\s+and\s+layout)?)\s*:[^\n]+"),
    ("publisher_boilerplate", (
        r"(?:All rights reserved\.|Some rights reserved\.|"
        r"Under the terms of this licen[cs]e,|"
        r"Cataloguing-in-Publication\s*\(CIP\)\s*data\.|"
        r"Sales, rights and licensing\.|Third-party materials\.|General disclaimers\.)[\s\S]*"
    )),
)


def plain_heading(heading: str) -> str:
    """Remove presentation markup, preserving the words actually in the source."""
    heading = re.sub(r"</?(?:u|b|strong|em|i|span)(?:\s[^<>]*)?>", "", heading, flags=re.IGNORECASE)
    heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
    return re.sub(r"[*`]+|~~", "", heading).strip()


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def visible_text(text: str) -> str:
    parser = _PlainText()
    parser.feed(text)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def is_contents_table(table: str) -> bool:
    """Require a title plus a page number in every nonempty row."""
    rows = []
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr\s*>", table, re.I | re.S):
        cells = [visible_text(cell) for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", row, re.I | re.S)]
        cells = [cell for cell in cells if cell]
        if not cells:
            continue
        if len(cells) != 2 or not re.fullmatch(r"(?:p\.?\s*)?(?:\d{1,4}|[ivxlcdm]+)", cells[-1], re.I):
            return False
        rows.append(cells)
    return len(rows) >= 3


def is_text_contents(text: str) -> bool:
    lines = [visible_text(line) for line in text.splitlines() if visible_text(line)]
    if lines and all(re.fullmatch(CONTENTS_ROW, line, re.I) for line in lines):
        return True
    # Wrapped entries and short labels are common in extracted PDF indexes.
    entries = sum(bool(re.fullmatch(CONTENTS_ROW, line, re.I)) for line in lines)
    if entries < 3:
        return False
    pending = []
    for line in lines:
        if re.fullmatch(CONTENTS_ROW, line, re.I):
            pending.clear()
            continue
        if len(line) > 180 or re.search(r"[.!?;]$", line):
            return False
        pending.append(line)
        if len(pending) > 2:
            return False
    return True


def is_bibliographic_entry(text: str, *, in_bibliography: bool) -> bool:
    """Recognize citation syntax, never relevance or the mere presence of a URL."""
    marked_note = bool(re.match(NOTE_MARKER, text, re.I))
    if not in_bibliography and not marked_note:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        # PDF extraction can attach a citation to the following substantive text.
        # Every line must be an identifiable citation; ambiguous mixed blocks stay.
        return all(is_bibliographic_entry(line, in_bibliography=in_bibliography or marked_note) for line in lines)
    plain = re.sub(r"^\s*(?:\[\d{1,4}\]|\(?\d{1,4}\)?[.)]?)\s+", "", visible_text(text)).strip()
    if re.fullmatch(r"(?:https?://|www\.)\S+(?:\s+(?:accessed|retrieved)(?:\s+on)?\s+[^\n]+)?", plain.strip("~"), re.I):
        return True
    if re.fullmatch(r"(?:ibid\.?|ibidem\.?)", plain, re.I):
        return True
    year = re.search(r"\((?:19|20)\d{2}[a-z]?\)|(?<=\.)\s+(?:19|20)\d{2}[a-z]?\.", plain[:180])
    if year:
        tail = plain[year.end():].lstrip(" ,.:;")
        first_words = " ".join(tail.split()[:12])
        if re.search(r"\b(?:is|are|was|were|can|may|must|should|has|have|had|affects?|increases?|reduces?)\b", first_words, re.I):
            return False
        # Keep findings, explanations and recommendations, even with a citation.
        if re.match(r"(?:found|finds?|shows?|reports?|estimates?|indicates?|suggests?|argues?|notes?|recommends?|defines?|requires?|states?|concludes?|observed|demonstrates?)\b", tail, re.I):
            return False
        author = plain[:year.start()].strip(" .,")
        named_author = bool(re.match(r"^[A-ZÀ-ÖØ-Þ][\w’'-]+,\s+[A-ZÀ-ÖØ-Þ]", author))
        author_words = re.findall(r"[^\W_]+", author)
        corporate_author = bool(author_words) and all(
            word[0].isupper() or word.casefold() in {"of", "the", "and", "for", "on", "de", "van", "von"}
            for word in author_words
        )
        if not named_author and not corporate_author:
            return False
        if re.match(r"\d|(?:approximately|about|nearly|over|more than)\b", tail, re.I):
            return False
        citation_details = bool(re.search(r"https?://|\bdoi\b|\bavailable at\b|\bpublications office\b|\bpp?\.\s*\d|\[online\]", tail, re.I))
        if named_author or citation_details or (in_bibliography and len(author_words) <= 12):
            return bool(tail)
    # Author. “Title.” Publisher/date/URL, a common endnote style.
    quoted = re.match(r'^([A-ZÀ-ÖØ-Þ][^!?\n]{1,180}?)\.\s*[“‘"]', plain)
    if quoted:
        author = quoted[1]
        words = re.findall(r"[^\W_]+", author)
        if "," in author or all(w[0].isupper() or w.casefold() in {"of", "the", "and", "for", "on"} for w in words):
            return bool(re.search(r"\b(?:19|20)\d{2}\b|https?://|\bdoi\b", plain, re.I))
    return False


def is_explanatory_note(note: str) -> bool:
    """Short notes survive only as prose without source-reference syntax; long
    notes (explanations that end with their source) survive unless the note as
    a whole is a citation."""
    plain = visible_text(re.sub(NOTE_MARKER, "", note, count=1, flags=re.I))
    words = len(re.findall(r"[^\W_]+", plain))
    if words < EXPLANATORY_NOTE_MIN_WORDS:
        return False
    if is_bibliographic_entry(note, in_bibliography=True):
        return False
    if words < LONG_NOTE_MIN_WORDS and (CITATION_SIGNAL.search(plain) or QUOTED_TITLE.search(plain)):
        # A short note with a source survives only if it states something
        # (a finding or a claim), as with citations attached to findings.
        return bool(PROSE_VERB.search(re.sub(r"https?://\S+", "", plain)))
    return True


def split_notes(text: str, marker: str, *, wrapped_lines: bool) -> list[tuple[bool, str]] | None:
    """Split a block whose first line is a note into (is_note, text) units.

    In note sections an unmarked line continues the previous note. In body
    text it does not: parsers insert page-bottom notes mid-sentence, so the
    following unmarked line is usually the body resuming and is always kept.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not re.match(marker, lines[0], re.I):
        return None
    units: list[tuple[bool, str]] = []
    for line in lines:
        if re.match(marker, line, re.I):
            units.append((True, line))
        elif wrapped_lines and units and units[-1][0]:
            units[-1] = (True, units[-1][1] + "\n" + line)
        elif units and not units[-1][0]:
            units[-1] = (False, units[-1][1] + "\n" + line)
        else:
            units.append((False, line))
    return units


def in_section(section_path: tuple[str, ...], pattern: str) -> bool:
    return any(re.fullmatch(pattern, visible_text(plain_heading(h)), re.I) for h in section_path)


def exclusion_rule(section_path: tuple[str, ...], text: str) -> tuple[str, str] | None:
    headings = [visible_text(plain_heading(heading)) for heading in section_path]
    if any(re.fullmatch(EDITORIAL_SECTION, heading, re.IGNORECASE) for heading in headings):
        return "editorial_section", EDITORIAL_SECTION
    if any(re.fullmatch(REFERENCE_LIST_SECTION, heading, re.I) for heading in headings):
        return "reference_list_section", REFERENCE_LIST_SECTION
    if any(re.fullmatch(ACKNOWLEDGEMENTS_SECTION, heading, re.I) for heading in headings):
        return "acknowledgements_section", ACKNOWLEDGEMENTS_SECTION
    if any(re.fullmatch(DIRECTORY_SECTION, heading, re.I) for heading in headings):
        return "organisation_directory_section", DIRECTORY_SECTION
    # A contents heading alone is insufficient; require recognizable page entries.
    if any(re.fullmatch(CONTENTS_SECTION, heading, re.IGNORECASE) for heading in headings):
        if is_text_contents(text):
            return "contents_entries", CONTENTS_ROW
    in_bibliography = any(re.fullmatch(BIBLIOGRAPHIC_SECTION, heading, re.I) for heading in headings)
    if is_bibliographic_entry(text, in_bibliography=in_bibliography):
        return "bibliographic_entry", "citation syntax in bibliography or numbered note"
    for name, pattern in PARAGRAPH_RULES:
        if re.fullmatch(pattern, text, re.IGNORECASE):
            return name, pattern
    return None


LEADING_NOTE_MARKER = re.compile(
    r"^([ \t]*)(?:\(<sup>\s*(\d{1,4})\s*</sup>\)|<sup>\s*\(?(\d{1,4})\)?\s*</sup>)[ \t]*", re.M | re.I,
)
INLINE_NOTE_REFERENCE = re.compile(
    r"[ \t]*\(<sup>\s*\d{1,4}(?:\s*[,–-]\s*\d{1,4})*\s*</sup>\)"
    r"|[ \t]*<sup>\s*\(?\d{1,4}(?:\s*[,–-]\s*\d{1,4})*\)?\s*</sup>",
    re.I,
)


def clean_body_markup(text: str) -> str:
    """Remove presentation residue; words of the source are never changed.

    Applied after the editorial rules, which rely on note markers. Line-initial
    note markers become plain "[n]"; inline note calls are dropped; parser
    strikethrough and underline/small tags carry no meaning in these sources.
    """
    text = LEADING_NOTE_MARKER.sub(lambda m: f"{m[1]}[{m[2] or m[3]}] ", text)
    text = INLINE_NOTE_REFERENCE.sub("", text)
    text = text.replace("~~", "")
    text = re.sub(r"</?(?:u|small)\s*>", "", text, flags=re.I)
    return text.strip()


def filter_editorial_paragraphs(paragraphs: list[tuple[int, tuple[str, ...], str]], enabled: bool):
    retained = []
    excluded = []
    for paragraph_id, section, text in paragraphs:
        if enabled and any(re.fullmatch(CONTENTS_SECTION, visible_text(plain_heading(h)), re.I) for h in section):
            removed = []

            def remove_contents(match):
                if is_contents_table(match.group()):
                    removed.append(match.group())
                    return ""
                return match.group()

            cleaned = HTML_TABLE.sub(remove_contents, text).strip()
            if removed:
                excluded.append({
                    "action": "trimmed" if cleaned else "excluded", "rules_version": EDITORIAL_RULES_VERSION,
                    "rule_id": "html_contents_table", "rule_pattern": "title and page number columns, at least three rows",
                    "paragraph_id": paragraph_id, "section_path": list(section),
                    "paragraph_text": text, "removed_text": "\n\n".join(removed), "retained_text": cleaned,
                })
                if not cleaned:
                    continue
                text = cleaned
        if enabled and not in_section(section, REFERENCE_LIST_SECTION):
            # Notes are judged one by one: explanatory notes stay, references go.
            notes_section = in_section(section, NOTES_SECTION)
            marker = NOTE_MARKER if notes_section else STRONG_NOTE_MARKER
            units = split_notes(text, marker, wrapped_lines=notes_section)
            if units:
                notes = [unit for is_note, unit in units if is_note]
                kept = [unit for is_note, unit in units if not is_note or is_explanatory_note(unit)]
                if len(kept) < len(units):
                    cleaned = "\n\n".join(kept)
                    removed_notes = [note for note in notes if note not in kept]
                    excluded.append({
                        "action": "trimmed" if kept else "excluded",
                        "rules_version": EDITORIAL_RULES_VERSION,
                        "rule_id": (
                            "bibliographic_entry"
                            if all(is_bibliographic_entry(note, in_bibliography=True) for note in removed_notes)
                            else "non_explanatory_notes"
                        ),
                        "rule_pattern": f"note marker; < {EXPLANATORY_NOTE_MIN_WORDS} words or citation syntax",
                        "paragraph_id": paragraph_id, "section_path": list(section),
                        "paragraph_text": text,
                        "removed_text": "\n\n".join(note for note in notes if note not in kept),
                        "retained_text": cleaned,
                    })
                    if not kept:
                        continue
                    text = cleaned
        rule = exclusion_rule(section, text) if enabled else None
        if rule is None:
            retained.append((paragraph_id, section, text))
        else:
            excluded.append({
                "action": "excluded", "rules_version": EDITORIAL_RULES_VERSION,
                "rule_id": rule[0], "rule_pattern": rule[1],
                "paragraph_id": paragraph_id, "section_path": list(section),
                "paragraph_text": text,
            })
    return retained, excluded
