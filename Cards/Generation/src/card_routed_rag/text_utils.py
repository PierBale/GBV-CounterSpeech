from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def chunk_words(text: str, chunk_size: int = 350, overlap: int = 80) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if len(chunk.split()) >= 40:
            chunks.append(chunk)
    return chunks


def quote_in_passage(quote: str, passage: str) -> bool:
    q = normalize_space(quote).lower()
    p = normalize_space(passage).lower()
    if not q or not p:
        return False
    if q in p:
        return True
    # Soft fallback for punctuation/line-break differences.
    q_tokens = tokenize(q)
    p_tokens = tokenize(p)
    if len(q_tokens) < 5:
        return False
    q_join = " ".join(q_tokens)
    p_join = " ".join(p_tokens)
    return q_join in p_join


def keyword_overlap_score(text: str, terms: Iterable[str]) -> float:
    toks = set(tokenize(text))
    term_tokens = [t for term in terms for t in tokenize(term)]
    if not term_tokens:
        return 0.0
    counts = Counter(term_tokens)
    hits = sum(counts[t] for t in set(term_tokens) if t in toks)
    return hits / max(1, sum(counts.values()))


def card_text(card: dict) -> str:
    source = card.get("source", {}) or {}
    parts = [
        card.get("argument") or card.get("claim", ""),
        card.get("reasoning", ""),
        card.get("edos_alignment", ""),
        card.get("chunk") or card.get("source_quote", ""),
        card.get("primary_edos_label", ""),
        " ".join(card.get("secondary_edos_labels", []) or []),
        " ".join(card.get("retrieval_keywords", []) or []),
        source.get("section", "") or "",
        source.get("title", "") or "",
    ]
    return normalize_space(" ".join(str(p) for p in parts if p))
