"""Defaults are resolved from the repository root, independent of the cwd."""

import json
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EMBEDDING_MODELS = ("Qwen/Qwen3-Embedding-8B", "BAAI/bge-m3")
RERANKING_MODELS = ("BAAI/bge-reranker-v2-m3", "Qwen/Qwen3-Reranker-8B")


def model_slug(model: str) -> str:
    import re

    return re.sub(r"[^a-z0-9.-]+", "-", model.rsplit("/", 1)[-1].lower())


class CardCreationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_dir: Path = PROJECT_ROOT / "documents"
    markdown_dir: Path = PROJECT_ROOT / "src" / "1_parsed_results"
    output_dir: Path = PROJECT_ROOT / "data" / "card_creation"
    labels_path: Path = Path(__file__).with_name("edos_labels.json")
    source_metadata_path: Path | None = None
    queries_path: Path | None = None
    chunk_tokenizer_model: str | None = None
    editorial_filter_enabled: bool = True
    include_section_titles: bool = True
    generate_evidence: bool = True
    all_combinations: bool = False
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    # Used by the experiment command; ordinary commands retain legacy paths.
    model_scoped_artifacts: bool = False
    llama_api_key_env: str = "LLAMA_CLOUD_API_KEY"
    llama_base_url: str = "https://api.cloud.eu.llamaindex.ai"
    llama_tier: str = "agentic"
    llama_version: str = "2026-09-13"
    min_chunk_tokens: int = Field(default=100, ge=1)
    long_chunk_warning_tokens: int = Field(default=2000, ge=1)
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_batch_size: int = Field(default=4, ge=1)
    embedding_max_tokens: int = Field(default=32768, ge=1)
    dense_top_j: int = Field(
        default=100, ge=1, validation_alias=AliasChoices("retrieval_top_k", "dense_top_j"),
    )
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    reranker_batch_size: int = Field(default=4, ge=1)
    reranker_max_tokens: int = Field(default=32768, ge=1)
    rerank_top_k: int = Field(default=10, ge=1)
    deduplication_enabled: bool = True
    deduplication_similarity_threshold: float = Field(
        default=0.97, gt=0, le=1, allow_inf_nan=False,
    )
    generator_model: str = "google/gemma-4-12B-it"
    generation_max_new_tokens: int = Field(default=900, ge=1)
    generation_temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    generation_retries: int = Field(default=5, ge=0)
    device: str | None = None

    @model_validator(mode="after")
    def check_top_k(self):
        if self.rerank_top_k > self.dense_top_j:
            raise ValueError("rerank_top_k cannot exceed dense_top_j.")
        return self

    @property
    def retrieval_top_k(self) -> int:
        return self.dense_top_j

    @property
    def chunk_tokenizer(self) -> str:
        return self.chunk_tokenizer_model or self.embedding_model

    @property
    def pipeline_id(self) -> str:
        return f"{model_slug(self.embedding_model)}__{model_slug(self.reranker_model)}"

    @property
    def retrieval_work_dir(self) -> Path:
        if self.model_scoped_artifacts:
            return self.output_dir / "work" / "retrieval" / model_slug(self.embedding_model)
        return self.output_dir

    @property
    def pipeline_work_dir(self) -> Path:
        if self.model_scoped_artifacts:
            return self.output_dir / "work" / "reranking" / self.pipeline_id
        return self.output_dir

    @property
    def chunks_path(self) -> Path:
        return self.output_dir / "chunks.jsonl"

    @property
    def editorial_audit_path(self) -> Path:
        return self.output_dir / "editorial_exclusions.jsonl"

    @property
    def chunk_index_path(self) -> Path:
        return self.retrieval_work_dir / "indexes" / "chunks.faiss"

    @property
    def retrieved_path(self) -> Path:
        return self.pipeline_work_dir / "edos_passages.jsonl"

    @property
    def dense_path(self) -> Path:
        return self.retrieval_work_dir / "edos_passages_dense.jsonl"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def ranked_path(self) -> Path:
        return self.pipeline_work_dir / "edos_passages_ranked.jsonl"

    @property
    def deduplication_log_path(self) -> Path:
        return self.pipeline_work_dir / "deduplication_decisions.jsonl"

    @property
    def cards_path(self) -> Path:
        return self.pipeline_work_dir / "cards.jsonl"

    @property
    def attempts_path(self) -> Path:
        return self.pipeline_work_dir / "generation_attempts.jsonl"

    @property
    def generation_manifest_path(self) -> Path:
        return self.pipeline_work_dir / "generation_manifest.json"

    @property
    def card_index_path(self) -> Path:
        return self.pipeline_work_dir / "indexes" / "cards.faiss"


def load_config(path: str | Path | None = None) -> CardCreationConfig:
    if path is None:
        return CardCreationConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Card creation configuration must be a JSON object.")
    for name in (
        "pdf_dir", "markdown_dir", "output_dir", "labels_path",
        "source_metadata_path", "queries_path",
    ):
        if raw.get(name) is not None:
            candidate = Path(raw[name])
            raw[name] = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return CardCreationConfig.model_validate(raw)
