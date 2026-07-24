from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class ChunkEmbeddingIndex:
    embeddings: np.ndarray
    chunk_ids: list[str]
    model_name: str


class HuggingFaceEmbedder:
    """Sentence-Transformers wrapper with Qwen's asymmetric retrieval prompts."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Install the project dependencies "
                "with: pip install -r requirements.txt"
            ) from exc

        kwargs = {"device": device} if device else {}
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return _l2_normalize(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        # Qwen's model card recommends its built-in "query" prompt for retrieval
        # queries and no prompt for documents.
        embedding = self.model.encode(
            [query],
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _l2_normalize(embedding)[0]


def save_chunk_embedding_index(
    path: str | Path,
    embeddings: np.ndarray,
    chunk_ids: Sequence[str],
    model_name: str,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = _l2_normalize(embeddings)
    ids = [str(chunk_id) for chunk_id in chunk_ids]
    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"Embedding rows ({matrix.shape[0]}) do not match chunk IDs ({len(ids)})"
        )
    np.savez_compressed(
        output,
        embeddings=matrix,
        chunk_ids=np.asarray(ids),
        model_name=np.asarray(model_name),
    )


def load_chunk_embedding_index(path: str | Path) -> ChunkEmbeddingIndex:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {input_path}")
    with np.load(input_path, allow_pickle=False) as data:
        embeddings = _l2_normalize(data["embeddings"])
        chunk_ids = [str(value) for value in data["chunk_ids"].tolist()]
        model_name = str(data["model_name"].item())
    if embeddings.shape[0] != len(chunk_ids):
        raise ValueError(
            f"Invalid index: {embeddings.shape[0]} embeddings for {len(chunk_ids)} chunk IDs"
        )
    return ChunkEmbeddingIndex(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        model_name=model_name,
    )
