"""Versioned syntactic exclusions; no relevance classifier or inferred metadata."""

import re
from html.parser import HTMLParser


EDITORIAL_RULES_VERSION = "3.0"

# Whole heading matches only. Forewords, summaries and explanatory notes survive.
EDITORIAL_SECTION = r"(?:imprint|disclaimer\s*(?:and\s*)?copyright)"
ACKNOWLEDGEMENTS_SECTION = r"acknowledg(?:e)?ments?"
DIRECTORY_SECTION = r"(?:list\s+of\s+(?:[\w&.-]+\s+){0,5}members|directory\s+of\s+(?:organisations|organizations|members))(?:\s*\([^)]*\))?"
BIBLIOGRAPHIC_SECTION = r"(?:\d+[.)]?\s*)?(?:references|bibliography|endnotes|footnotes|notes|sources used to prepare the factsheet)"
CONTENTS_SECTION = r"(?:contents|table\s+of\s+contents)"
CONTENTS_ROW = r".+\S(?:[ \t.]+|[ \t]*/[ \t]*)(?:p\.?[ \t]*)?(?:\d{1,4}|[ivxlcdm]{1,8})"
HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)
NOTE_MARKER = r"^\s*(?:<sup>\s*\(?\d{1,4}\)?\s*</sup>|\[\d{1,4}\]|\(?\d{1,4}\)?[.)]?\s+)"

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


def exclusion_rule(section_path: tuple[str, ...], text: str) -> tuple[str, str] | None:
    headings = [visible_text(plain_heading(heading)) for heading in section_path]
    if any(re.fullmatch(EDITORIAL_SECTION, heading, re.IGNORECASE) for heading in headings):
        return "editorial_section", EDITORIAL_SECTION
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
