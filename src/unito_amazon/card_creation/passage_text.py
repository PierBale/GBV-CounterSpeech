"""One deterministic model-input representation, separate from source text."""

from .editorial import plain_heading
from .schema import Chunk


def format_chunk_text(chunk: Chunk, include_section_titles: bool = True) -> str:
    if not include_section_titles:
        return chunk.text
    headings = [plain_heading(heading) for heading in chunk.section_path]
    headings = [heading for heading in headings if heading]
    if not headings:
        return chunk.text
    return "Section: " + " > ".join(headings) + "\n\n" + chunk.text
