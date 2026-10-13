"""Defaults are resolved from the repository root, independent of the cwd."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CardCreationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_dir: Path = PROJECT_ROOT / "documents"
    markdown_dir: Path = PROJECT_ROOT / "src" / "1_parsed_results"
    output_dir: Path = PROJECT_ROOT / "data" / "card_creation"
    labels_path: Path = Path(__file__).with_name("edos_labels.json")
    source_metadata_path: Path | None = None
    llama_api_key_env: str = "LLAMA_CLOUD_API_KEY"
    llama_base_url: str = "https://api.cloud.eu.llamaindex.ai"
    llama_tier: str = "agentic"
    llama_version: str = "2026-09-13"
    min_chunk_tokens: int = Field(default=100, ge=1)
    long_chunk_warning_tokens: int = Field(default=2000, ge=1)
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_batch_size: int = Field(default=4, ge=1)
    embedding_max_tokens: int = Field(default=32768, ge=1)
    dense_top_j: int = Field(default=100, ge=1)
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    reranker_batch_size: int = Field(default=4, ge=1)
    reranker_max_tokens: int = Field(default=32768, ge=1)
    rerank_top_k: int = Field(default=10, ge=1)
    generator_model: str = "google/gemma-4-12B-it"
    generation_max_new_tokens: int = Field(default=900, ge=1)
    generation_retries: int = Field(default=2, ge=0)
    device: str | None = None

    @model_validator(mode="after")
    def check_top_k(self):
        if self.rerank_top_k > self.dense_top_j:
            raise ValueError("rerank_top_k cannot exceed dense_top_j.")
        return self

    @property
    def chunks_path(self) -> Path:
        return self.output_dir / "chunks.jsonl"

    @property
    def chunk_index_path(self) -> Path:
        return self.output_dir / "indexes" / "chunks.faiss"

    @property
    def retrieved_path(self) -> Path:
        return self.output_dir / "edos_passages.jsonl"

    @property
    def cards_path(self) -> Path:
        return self.output_dir / "cards.jsonl"

    @property
    def attempts_path(self) -> Path:
        return self.output_dir / "generation_attempts.jsonl"

    @property
    def card_index_path(self) -> Path:
        return self.output_dir / "indexes" / "cards.faiss"


def load_config(path: str | Path | None = None) -> CardCreationConfig:
    if path is None:
        return CardCreationConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Card creation configuration must be a JSON object.")
    for name in (
        "pdf_dir", "markdown_dir", "output_dir", "labels_path",
        "source_metadata_path",
    ):
        if raw.get(name) is not None:
            candidate = Path(raw[name])
            raw[name] = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return CardCreationConfig.model_validate(raw)
