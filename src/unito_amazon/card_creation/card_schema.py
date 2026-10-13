"""Evidence-summary cards (v2), separate from the legacy v1 card schemas.

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


class EvidenceCard(GeneratedContent):
    schema_version: Literal["2.0"] = "2.0"
    card_id: str
    status: Literal["candidate", "validated"] = "candidate"
    primary_edos_label: str
    edos_definition: str
    source: Source
    source_chunk_id: str
    source_passage: str
    source_quote: str = Field(min_length=1)
    section_path: list[str]
    retrieval_keywords: list[str]
    retrieval: dict[str, str | int | float]
    generator_model: str
    generation_key: str
    validation: dict = Field(default_factory=dict)
