"""Evidence-summary cards (v3), separate from the legacy v1 card schemas.

Keeping generation schemas separate preserves upstream chunk/index checkpoints
when only the card output contract changes.
"""

from typing import Literal

from pydantic import Field, field_validator

from .schema import Source, StrictModel


class GeneratedContent(StrictModel):
    reasoning: str = Field(min_length=1)
    evidence: str = Field(min_length=1)

    @field_validator("reasoning", "evidence")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Generated fields cannot be blank.")
        return value


class CardChunk(StrictModel):
    chunk_id: str
    text: str = Field(min_length=1)
    section_path: list[str]
    paragraph_ids: list[int]
    token_count: int = Field(ge=1)


class CardPrompt(StrictModel):
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)


class EvidenceCard(GeneratedContent):
    schema_version: Literal["1.0"] = "1.0"
    card_id: str
    primary_edos_label: str
    edos_definition: str
    source: Source
    chunk: CardChunk
    retrieval: dict[str, str | int | float]
    generator_model: str
    prompt: CardPrompt
