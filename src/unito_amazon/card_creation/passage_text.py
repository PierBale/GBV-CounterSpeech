"""One deterministic model-input representation, separate from source text."""

from .editorial import plain_heading
from .schema import Chunk


def compose_passage_text(body: str, section_path: list[str], include_section_titles: bool = True) -> str:
    if not include_section_titles:
        return body
    headings = [plain_heading(heading) for heading in section_path]
    headings = [heading for heading in headings if heading]
    if not headings:
        return body
    return "Section: " + " > ".join(headings) + "\n\n" + body


def format_chunk_text(chunk: Chunk, include_section_titles: bool = True) -> str:
    if chunk.body_text is None:
        # Legacy libraries stored only the body in text.
        return compose_passage_text(chunk.text, chunk.section_path, include_section_titles)
    expected = compose_passage_text(chunk.body_text, chunk.section_path, include_section_titles)
    if chunk.text != expected:
        raise ValueError("Chunk text differs from the configured model input; rebuild the chunk stage.")
    return chunk.text
