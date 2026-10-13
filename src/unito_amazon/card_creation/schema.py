from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(StrictModel):
    source_id: str
    markdown_file: str
    title: str
    pdf_file: str | None = None
    publisher: str | None = None
    year: int | None = None
    url: str | None = None


class Chunk(StrictModel):
    chunk_id: str
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    paragraph_ids: list[int]
    section_path: list[str]
    source: Source


class DensePassage(StrictModel):
    label: str
    definition: str
    embedding_model: str
    chunk: Chunk
    dense_rank: int
    dense_score: float


class RankedPassage(StrictModel):
    label: str
    definition: str
    embedding_model: str
    reranker_model: str
    chunk: Chunk
    dense_rank: int
    dense_score: float
    reranker_rank: int
    reranker_score: float


class GeneratedContent(StrictModel):
    source_quote: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    argument: str = Field(min_length=1)

    @field_validator("source_quote", "reasoning", "argument")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Generated fields cannot be blank.")
        return value


class GenerationEnvelope(StrictModel):
    card: GeneratedContent | None
    rejection_reason: str | None


class EvidenceCard(GeneratedContent):
    schema_version: Literal["1.0"] = "1.0"
    card_id: str
    status: Literal["candidate", "validated"] = "candidate"
    primary_edos_label: str
    edos_definition: str
    source: Source
    source_chunk_id: str
    source_passage: str
    section_path: list[str]
    retrieval_keywords: list[str]
    retrieval: dict[str, str | int | float]
    generator_model: str
    generation_key: str
    validation: dict = Field(default_factory=dict)
