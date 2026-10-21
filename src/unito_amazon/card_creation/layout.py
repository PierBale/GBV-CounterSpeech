"""Remove explicit graphics/page markers before interpreting Markdown headings.

Repeated running titles require two distinct observed page numbers and a title
found in the document's headings (possibly wrapped) or written in all caps.
No document text is generated, paraphrased, or dehyphenated.
"""

import re
from collections import defaultdict

from .editorial import EDITORIAL_RULES_VERSION, plain_heading


GRAPHIC_LINE = r"(?:icon|logo):\s*\S.*"
EXPLICIT_PAGE = r"(?:Page\s+\d+(?:\s*[►|]\s*.*)?|(?:.*?►\s*)Page\s+\d+)"
PAGE_FIRST = r"(?P<page>\d{1,4})(?:\s*[|►]\s*|\s+)(?P<title>.+)"
PAGE_LAST = r"(?P<title>.+?)\s+(?P<page>\d{1,4})"
TAGGED_PAGE_BLOCK = (
    r"^[ \t]*<page_(?P<kind>header|footer)>"
    r"(?P<body>(?:(?!</?page_(?:header|footer)>)[\s\S])*)"
    r"</page_(?P=kind)>[ \t]*$"
)


def display_line(line: str) -> str:
    return plain_heading(re.sub(r"^\s*#{1,6}\s+", "", line.strip()))


def title_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def strip_layout_noise(markdown: str) -> tuple[str, list[dict]]:
    lines = markdown.replace("\r\n", "\n").splitlines()
    visible = [display_line(line) for line in lines]
    heading_titles = set()
    tagged_blocks = []
    tagged_titles = set()
    normalized_markdown = "\n".join(lines)
    for match in re.finditer(TAGGED_PAGE_BLOCK, normalized_markdown, re.IGNORECASE | re.MULTILINE):
        first = normalized_markdown.count("\n", 0, match.start())
        last = first + match.group().count("\n")
        tagged_blocks.append((first, last, match["kind"].lower()))
        title = display_line(" ".join(match["body"].split()))
        heading_titles.add(title_key(title))
        if match["kind"].lower() == "header" and title:
            tagged_titles.add(title_key(title))
    # A publisher explicitly identified by a logo is source metadata, and can
    # confirm a repeated publisher/page-number footer without a made-up title.
    for text in visible:
        metadata = re.fullmatch(r"(?:logo|publisher):\s*(.+)", text, re.IGNORECASE)
        if metadata:
            heading_titles.add(title_key(metadata[1]))
    for position, line in enumerate(lines):
        if not re.match(r"^\s*#{1,6}\s+", line):
            continue
        title_parts = []
        # Also recognize titles wrapped across up to three consecutive nonempty
        # lines, as in the source's Technology-Facilitated / Gender-Based Violence.
        for next_position in range(position, len(lines)):
            text = visible[next_position]
            if not text:
                continue
            if len(text) > 180 or re.search(r"[.!?]$", text) or text == "---":
                break
            if next_position != position and not re.match(r"^\s*#{1,6}\s+", lines[next_position]):
                words = re.findall(r"\w+", text)
                if not words or not all(
                    word[0].isupper() or word.casefold() in {"a", "an", "the", "of", "on", "in", "and", "for", "to", "with", "against"}
                    for word in words
                ):
                    break
            title_parts.append(text)
            heading_titles.add(title_key(" ".join(title_parts)))
            if len(title_parts) == 3:
                break

    def plausible_title(text):
        return (
            len(text) <= 180 and len(re.findall(r"\w+", text)) >= 3
            and (title_key(text) in heading_titles or text.isupper())
        )

    candidates = []
    observations = defaultdict(set)
    for position, text in enumerate(visible):
        for pattern in (PAGE_FIRST, PAGE_LAST):
            match = re.fullmatch(pattern, text)
            if match and plausible_title(match["title"]):
                key = title_key(match["title"])
                observations[key].add(int(match["page"]))
                candidates.append((position, None, key, pattern))
                break
        if re.fullmatch(r"\d{1,4}", text):
            for direction in (-1, 1):
                neighbor = position + direction
                while 0 <= neighbor < len(lines) and (
                    not visible[neighbor] or visible[neighbor] == "---"
                ):
                    neighbor += direction
                if 0 <= neighbor < len(lines) and plausible_title(visible[neighbor]):
                    key = title_key(visible[neighbor])
                    observations[key].add(int(text))
                    candidates.append((neighbor, position, key, "title_adjacent_to_page_number"))

    removed = {}
    page_positions = set()

    def discard(position, rule, pattern, **extra):
        removed.setdefault(position, {
            "action": "excluded", "rules_version": EDITORIAL_RULES_VERSION,
            "rule_id": rule, "rule_pattern": pattern,
            "line_number": position + 1, "line_text": lines[position], **extra,
        })

    for position, text in enumerate(visible):
        if re.fullmatch(GRAPHIC_LINE, text, re.IGNORECASE):
            discard(position, "graphic_description_line", GRAPHIC_LINE)
        elif re.fullmatch(EXPLICIT_PAGE, text, re.IGNORECASE):
            discard(position, "explicit_page_marker", EXPLICIT_PAGE)
            page_positions.add(position)
    for first, last, kind in tagged_blocks:
        for position in range(first, last + 1):
            discard(position, f"explicit_page_{kind}", TAGGED_PAGE_BLOCK)
            page_positions.add(position)
    # Untagged duplicates of a parser-marked running header are removed by
    # exact normalized equality. Actual Markdown section headings are retained.
    for key in tagged_titles:
        positions = [
            position for position, text in enumerate(visible)
            if title_key(text) == key and not re.match(r"^\s*#{1,6}\s+", lines[position])
        ]
        if len(positions) >= 2:
            for position in positions:
                discard(position, "repeated_tagged_running_title", "exact_source_page_header",
                        title_signature=key, observed_occurrences=len(positions))
                page_positions.add(position)
    for position, page_position, key, pattern in candidates:
        if len(observations[key]) < 2:
            continue
        discard(position, "repeated_running_title", pattern,
                title_signature=key, observed_pages=sorted(observations[key]))
        page_positions.add(position)
        if page_position is not None:
            discard(page_position, "page_number_with_running_title", pattern,
                    title_signature=key, observed_pages=sorted(observations[key]))
            page_positions.add(page_position)

    skipped = set(removed)
    # A recognized page break is a layout boundary, not a paragraph or section
    # boundary. Remove surrounding padding only when joining two body fragments;
    # a genuine Markdown heading on either side retains its structural boundary.
    for position in page_positions:
        left, right = position - 1, position + 1
        while left >= 0 and (left in removed or not visible[left] or visible[left] == "---"):
            left -= 1
        while right < len(lines) and (right in removed or not visible[right] or visible[right] == "---"):
            right += 1
        if left < 0 or right >= len(lines):
            continue
        if re.match(r"^\s*#{1,6}\s+", lines[left]) or re.match(r"^\s*#{1,6}\s+", lines[right]):
            continue
        skipped.update(range(left + 1, right))
        removed[position]["joined_body_fragments"] = True
    return "\n".join(line for position, line in enumerate(lines) if position not in skipped), [
        removed[position] for position in sorted(removed)
    ]
