"""Rank-ordered cosine deduplication inspired by SemDeDup (Abbas et al., 2023).

Reference: https://arxiv.org/abs/2303.09540 (Section 3).
This is an adaptation, not a reproduction: we compare retrieval candidates
within each category and prioritize reranker rank, without k-means clustering.
"""

import hashlib
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


SEMDEDUP_REFERENCE = "https://arxiv.org/abs/2303.09540"
DEDUPLICATION_METHOD = "rank_ordered_cosine_v1"


@dataclass(frozen=True)
class DeduplicationDecision:
    index: int
    reason: Literal["selected", "exact_duplicate", "near_duplicate", "top_k_limit"]
    reference_index: int | None = None
    cosine_similarity: float | None = None


@dataclass(frozen=True)
class DeduplicationResult:
    selected_indices: list[int]
    decisions: list[DeduplicationDecision]


def normalized_text_hash(text: str) -> str:
    """Ignore Unicode composition and whitespace, preserving case and facts."""
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalized:
        raise ValueError("Cannot deduplicate an empty passage.")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_nonredundant(
    texts: Sequence[str], embeddings: np.ndarray | None, top_k: int,
    similarity_threshold: float = 0.97, enabled: bool = True,
) -> DeduplicationResult:
    """Select up to K passages from an already reranker-ordered candidate list.

    Only selected passages can suppress later candidates. In particular, a
    discarded passage cannot start a transitive chain of further removals.
    Lower-ranked candidates are considered to fill slots freed by duplicates.
    Once K slots are filled, remaining items are audited but never selected.
    This function does not load models, mutate embeddings or remove corpus data.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if not math.isfinite(similarity_threshold) or not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be finite and in (0, 1].")
    if not enabled:
        return DeduplicationResult(
            selected_indices=list(range(min(top_k, len(texts)))),
            decisions=[
                DeduplicationDecision(index, "selected" if index < top_k else "top_k_limit")
                for index in range(len(texts))
            ],
        )
    if embeddings is None:
        raise ValueError("Stored document embeddings are required for cosine deduplication.")
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise ValueError("Expected a non-zero-dimensional embedding for each candidate.")
    if not np.isfinite(vectors).all():
        raise ValueError("Deduplication embeddings must contain only finite values.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Deduplication embeddings must have finite, non-zero norms.")
    # Re-normalize cached float32 vectors for a stable cosine calculation.
    vectors = vectors / norms[:, None]
    hashes = [normalized_text_hash(text) for text in texts]
    selected = []
    selected_hashes = {}
    decisions = []
    for index, text_hash in enumerate(hashes):
        exact_match = selected_hashes.get(text_hash)
        if exact_match is not None:
            similarity = float(np.clip(vectors[index] @ vectors[exact_match], -1.0, 1.0))
            decisions.append(DeduplicationDecision(index, "exact_duplicate", exact_match, similarity))
            continue
        reference = None
        similarity = None
        if selected:
            similarities = np.clip(vectors[selected] @ vectors[index], -1.0, 1.0)
            closest = int(np.argmax(similarities))
            reference = selected[closest]
            similarity = float(similarities[closest])
            if similarity >= similarity_threshold:
                decisions.append(DeduplicationDecision(index, "near_duplicate", reference, similarity))
                continue
        if len(selected) < top_k:
            selected.append(index)
            selected_hashes[text_hash] = index
            reason = "selected"
        else:
            reason = "top_k_limit"
        decisions.append(DeduplicationDecision(index, reason, reference, similarity))
    return DeduplicationResult(selected, decisions)
