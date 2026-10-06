from dataclasses import dataclass
from typing import Literal


ModelRole = Literal["decoder", "encoder"]


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    huggingface_id: str
    role: ModelRole
    paper_name: str


MODEL_SPECS = (
    ModelSpec(
        alias="qwen3-emb-0.6b",
        huggingface_id="Qwen/Qwen3-Embedding-0.6B",
        role="encoder",
        paper_name="Qwen3-Emb-0.6B",
    ),
    ModelSpec(
        alias="bge-m3-0.6b",
        huggingface_id="BAAI/bge-m3",
        role="encoder",
        paper_name="BGE-m3-0.6B",
    ),
    ModelSpec(
        alias="gemma-emb-0.3b",
        huggingface_id="google/embeddinggemma-300m",
        role="encoder",
        paper_name="Gemma-Emb-0.3B",
    ),
    ModelSpec(
        alias="llama-3.2-1b",
        huggingface_id="meta-llama/Llama-3.2-1B-Instruct",
        role="decoder",
        paper_name="Llama-3.2-1B",
    ),
    ModelSpec(
        alias="llama-3.1-8b",
        huggingface_id="meta-llama/Llama-3.1-8B-Instruct",
        role="decoder",
        paper_name="Llama-3.1-8B",
    ),
    ModelSpec(
        alias="ministral-3-8b",
        huggingface_id="mistralai/Ministral-3-8B-Instruct-2512",
        role="decoder",
        paper_name="Ministral-3-8B",
    ),
    ModelSpec(
        alias="qwen-3.5-9b",
        huggingface_id="Qwen/Qwen3.5-9B",
        role="decoder",
        paper_name="Qwen-3.5-9B",
    ),
)


DECODER_ALIASES = tuple(
    spec.alias for spec in MODEL_SPECS if spec.role == "decoder"
)
ENCODER_ALIASES = tuple(
    spec.alias for spec in MODEL_SPECS if spec.role == "encoder"
)


def resolve_model(model: str, role: ModelRole | None = None) -> ModelSpec:
    normalized = model.strip().casefold()
    for spec in MODEL_SPECS:
        if normalized in {
            spec.alias.casefold(),
            spec.huggingface_id.casefold(),
            spec.paper_name.casefold(),
        }:
            if role is not None and spec.role != role:
                raise ValueError(
                    f"Model {model!r} has role {spec.role!r}, not {role!r}."
                )
            return spec

    available = [
        spec.alias
        for spec in MODEL_SPECS
        if role is None or spec.role == role
    ]
    raise ValueError(
        f"Unknown {role or 'registered'} model: {model!r}. "
        f"Available aliases: {', '.join(available)}."
    )


def resolve_decoder_model(model: str) -> ModelSpec:
    return resolve_model(model, role="decoder")


def resolve_encoder_model(model: str) -> ModelSpec:
    return resolve_model(model, role="encoder")
