from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChunkRetrievalResult:
    chunk: dict[str, Any]
    relevance_score: float
    mmr_score: float
    candidate_rank: int


class ChunkMMRRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        indexed_chunk_ids: list[str],
    ) -> None:
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
        if chunk_ids != indexed_chunk_ids:
            raise ValueError(
                "The chunks do not match the embedding index. Re-run "
                "03_encode_chunks.py after changing or reordering the chunk file."
            )
        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        if self.embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Found {self.embeddings.shape[0]} embeddings for {len(chunks)} chunks."
            )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 10,
        candidate_pool_size: int = 100,
        lambda_mult: float = 0.5,
    ) -> list[ChunkRetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if candidate_pool_size < top_k:
            raise ValueError("candidate_pool_size must be greater than or equal to top_k.")
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1.")
        if not self.chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match index dimension "
                f"{self.embeddings.shape[1]}."
            )
        query /= max(float(np.linalg.norm(query)), 1e-12)
        relevance = self.embeddings @ query

        pool_size = min(candidate_pool_size, len(self.chunks))
        ranked_all = np.argsort(-relevance, kind="stable")
        candidate_indices = ranked_all[:pool_size].tolist()
        candidate_rank = {
            chunk_index: rank
            for rank, chunk_index in enumerate(candidate_indices, start=1)
        }

        selected: list[int] = []
        remaining = candidate_indices[:]
        selection_scores: dict[int, float] = {}
        limit = min(top_k, pool_size)
        while remaining and len(selected) < limit:
            best_index: int | None = None
            best_mmr = -np.inf
            for chunk_index in remaining:
                if selected:
                    redundancy = float(
                        np.max(self.embeddings[selected] @ self.embeddings[chunk_index])
                    )
                else:
                    redundancy = 0.0
                mmr_score = (
                    lambda_mult * float(relevance[chunk_index])
                    - (1.0 - lambda_mult) * redundancy
                )
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_index = chunk_index

            assert best_index is not None
            selected.append(best_index)
            remaining.remove(best_index)
            selection_scores[best_index] = float(best_mmr)

        return [
            ChunkRetrievalResult(
                chunk=self.chunks[index],
                relevance_score=float(relevance[index]),
                mmr_score=selection_scores[index],
                candidate_rank=candidate_rank[index],
            )
            for index in selected
        ]
