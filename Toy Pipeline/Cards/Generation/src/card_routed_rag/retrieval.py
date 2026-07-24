from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    from rank_bm25 import BM25Okapi
except Exception:
    class BM25Okapi:
        def __init__(self, corpus):
            from collections import Counter
            import math
            self.corpus = corpus
            self.N = len(corpus)
            self.avgdl = sum(len(d) for d in corpus) / max(1, self.N)
            self.df = Counter()
            for doc in corpus:
                for t in set(doc):
                    self.df[t] += 1
            self.idf = {t: math.log((self.N - df + 0.5) / (df + 0.5) + 1) for t, df in self.df.items()}
            self.doc_tf = [Counter(doc) for doc in corpus]
        def get_scores(self, query_tokens):
            import numpy as np
            k1, b = 1.5, 0.75
            scores = []
            for doc, tf in zip(self.corpus, self.doc_tf):
                dl = len(doc)
                score = 0.0
                for q in query_tokens:
                    f = tf.get(q, 0)
                    if not f:
                        continue
                    idf = self.idf.get(q, 0.0)
                    denom = f + k1 * (1 - b + b * dl / max(1e-9, self.avgdl))
                    score += idf * f * (k1 + 1) / denom
                scores.append(score)
            return np.array(scores, dtype=float)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_utils import tokenize, card_text
from .card_validation import validation_quality


@dataclass
class RetrievalResult:
    card: dict[str, Any]
    score: float
    breakdown: dict[str, float]


class CardRetriever:
    def __init__(self, cards: list[dict[str, Any]], weights: dict[str, float] | None = None):
        self.cards = cards
        self.texts = [card_text(c) for c in cards]
        self.tokens = [tokenize(t) for t in self.texts]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None
        self.weights = weights or {
            "lexical_similarity": 0.30,
            "semantic_similarity": 0.20,
            "primary_edos_match": 0.25,
            "secondary_edos_match": 0.08,
            "keyword_overlap": 0.07,
            "validation_quality": 0.10,
        }

    def _tfidf_scores(self, query: str) -> np.ndarray:
        if not self.texts:
            return np.array([])
        q = self.vectorizer.transform([query])
        return cosine_similarity(q, self.matrix).ravel()

    def _bm25_scores(self, query: str) -> np.ndarray:
        if not self.texts:
            return np.array([])
        raw = np.array(self.bm25.get_scores(tokenize(query)), dtype=float)
        if raw.max() > raw.min():
            return (raw - raw.min()) / (raw.max() - raw.min())
        return np.zeros_like(raw)

    def _semantic_scores(self, query: str) -> np.ndarray:
        # Optional dense retrieval. Falls back to TF-IDF when sentence-transformers is unavailable.
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            emb = model.encode([query] + self.texts, normalize_embeddings=True)
            return np.dot(emb[0], emb[1:].T)
        except Exception:
            return self._tfidf_scores(query)

    def basic(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        tfidf = self._tfidf_scores(query)
        bm25 = self._bm25_scores(query)
        scores = 0.5 * tfidf + 0.5 * bm25
        return self._rank(scores, [{"tfidf": float(a), "bm25": float(b)} for a, b in zip(tfidf, bm25)], top_k)

    def dense_mmr(self, query: str, top_k: int = 5, lambda_mult: float = 0.7) -> list[RetrievalResult]:
        rel = self._semantic_scores(query)
        if len(rel) == 0:
            return []
        # Use TF-IDF item-item similarity for diversity fallback; dense item-item would be costlier.
        item_sim = cosine_similarity(self.matrix) if self.matrix is not None else np.eye(len(self.cards))
        selected: list[int] = []
        candidates = set(range(len(self.cards)))
        while candidates and len(selected) < top_k:
            best_i = None
            best_score = -1e9
            for i in candidates:
                div = max([item_sim[i, j] for j in selected], default=0.0)
                score = lambda_mult * rel[i] - (1 - lambda_mult) * div
                if score > best_score:
                    best_score = score
                    best_i = i
            selected.append(best_i)  # type: ignore[arg-type]
            candidates.remove(best_i)  # type: ignore[arg-type]
        return [RetrievalResult(self.cards[i], float(rel[i]), {"semantic_similarity": float(rel[i]), "mmr_selected_rank": float(r + 1)}) for r, i in enumerate(selected)]

    def card_aware(self, query: str, edos_label: str | None = None, top_k: int = 5, source_diversity_bonus: float = 0.08, redundancy_penalty: float = 0.15, max_cards_per_source: int = 2) -> list[RetrievalResult]:
        tfidf = self._tfidf_scores(query)
        bm25 = self._bm25_scores(query)
        sem = self._semantic_scores(query)
        query_tokens = set(tokenize(query))
        results: list[RetrievalResult] = []
        for i, card in enumerate(self.cards):
            primary = card.get("primary_edos_label")
            secondary = set(card.get("secondary_edos_labels", []) or [])
            primary_match = 1.0 if edos_label and primary == edos_label else 0.0
            secondary_match = 1.0 if edos_label and edos_label in secondary else 0.0
            kw = set(tokenize(" ".join(card.get("retrieval_keywords", []) or [])))
            kw_overlap = len(query_tokens & kw) / max(1, len(query_tokens | kw)) if kw else 0.0
            val_q = validation_quality(card)
            breakdown = {
                "lexical_similarity": float(0.5 * tfidf[i] + 0.5 * bm25[i]),
                "semantic_similarity": float(sem[i]),
                "primary_edos_match": primary_match,
                "secondary_edos_match": secondary_match,
                "keyword_overlap": kw_overlap,
                "validation_quality": val_q,
            }
            score = sum(self.weights.get(k, 0.0) * v for k, v in breakdown.items())
            results.append(RetrievalResult(card, float(score), breakdown))

        # Greedy portfolio selection: score + source diversity - redundancy.
        selected: list[RetrievalResult] = []
        remaining = results[:]
        source_counts: dict[str, int] = {}
        text_by_id = {id(r): card_text(r.card) for r in remaining}
        while remaining and len(selected) < top_k:
            best = None
            best_adj = -1e9
            for r in remaining:
                source_id = (r.card.get("source") or {}).get("source_id", "UNKNOWN")
                if source_counts.get(source_id, 0) >= max_cards_per_source:
                    continue
                diversity = source_diversity_bonus if source_counts.get(source_id, 0) == 0 else 0.0
                redundancy = 0.0
                if selected:
                    # Simple redundancy by token Jaccard with already selected cards.
                    rt = set(tokenize(text_by_id[id(r)]))
                    sims = []
                    for s in selected:
                        st = set(tokenize(card_text(s.card)))
                        sims.append(len(rt & st) / max(1, len(rt | st)))
                    redundancy = max(sims)
                adjusted = r.score + diversity - redundancy_penalty * redundancy
                if adjusted > best_adj:
                    best_adj = adjusted
                    best = r
            if best is None:
                break
            best.breakdown["portfolio_adjusted_score"] = float(best_adj)
            selected.append(best)
            source_id = (best.card.get("source") or {}).get("source_id", "UNKNOWN")
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            remaining.remove(best)
        return selected

    def _rank(self, scores: np.ndarray, breakdowns: list[dict[str, float]], top_k: int) -> list[RetrievalResult]:
        idx = np.argsort(scores)[::-1][:top_k]
        return [RetrievalResult(self.cards[i], float(scores[i]), breakdowns[i]) for i in idx]
