"""Versioned syntactic exclusions; no relevance classifier or inferred metadata."""

import re


EDITORIAL_RULES_VERSION = "1.0"

# Whole heading matches only. References, forewords, acknowledgements and
# publication summaries are deliberately not treated as editorial sections.
EDITORIAL_SECTION = r"(?:imprint|disclaimer\s*(?:and\s*)?copyright)"
CONTENTS_SECTION = r"(?:contents|table\s+of\s+contents)"
CONTENTS_ROW = r".+\S(?:[ \t.]+|[ \t]*/[ \t]*)(?:p\.?[ \t]*)?(?:\d{1,4}|[ivxlcdm]{1,8})"

# Whole paragraphs, except explicitly identified publisher boilerplate prefixes.
PARAGRAPH_RULES = (
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
    return re.sub(r"[*`]+", "", heading).strip()


def exclusion_rule(section_path: tuple[str, ...], text: str) -> tuple[str, str] | None:
    headings = [plain_heading(heading) for heading in section_path]
    if any(re.fullmatch(EDITORIAL_SECTION, heading, re.IGNORECASE) for heading in headings):
        return "editorial_section", EDITORIAL_SECTION
    # A contents heading alone is insufficient: every nonempty line must have
    # the syntactic form of an entry followed by a page number.
    if any(re.fullmatch(CONTENTS_SECTION, heading, re.IGNORECASE) for heading in headings):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and all(re.fullmatch(CONTENTS_ROW, line, re.IGNORECASE) for line in lines):
            return "contents_entries", CONTENTS_ROW
    for name, pattern in PARAGRAPH_RULES:
        if re.fullmatch(pattern, text, re.IGNORECASE):
            return name, pattern
    return None


def filter_editorial_paragraphs(paragraphs: list[tuple[int, tuple[str, ...], str]], enabled: bool):
    retained = []
    excluded = []
    for paragraph_id, section, text in paragraphs:
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
