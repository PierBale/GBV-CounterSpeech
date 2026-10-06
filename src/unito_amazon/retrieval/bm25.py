import math
import re
from collections import Counter

from .base import BaseRetriever
from .schema import Document, RetrievalResult


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        lowercase: bool = True,
        retriever_name: str = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.lowercase = lowercase
        self.retriever_name = retriever_name
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        self.inverse_document_frequencies: dict[str, float] = {}
        self.average_document_length = 0.0

    def index(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("[BM25Retriever] The document list is empty.")

        self.documents = documents
        tokenized_corpus = [self._tokenize(document.text) for document in documents]
        self.term_frequencies = [Counter(tokens) for tokens in tokenized_corpus]
        self.document_lengths = [len(tokens) for tokens in tokenized_corpus]
        self.average_document_length = sum(self.document_lengths) / len(documents)

        document_frequencies: Counter[str] = Counter()
        for tokens in tokenized_corpus:
            document_frequencies.update(set(tokens))

        document_count = len(documents)
        self.inverse_document_frequencies = {
            term: math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if not self.documents:
            raise RuntimeError("[BM25Retriever] Call index() before retrieve().")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_terms = set(self._tokenize(query))
        scored_indices = [
            (index, self._score_document(query_terms, index))
            for index in range(len(self.documents))
        ]
        scored_indices.sort(key=lambda item: (-item[1], item[0]))

        return [
            RetrievalResult(
                document=self.documents[index],
                score=score,
                rank=rank,
                retriever_name=self.retriever_name,
            )
            for rank, (index, score) in enumerate(
                scored_indices[: min(top_k, len(scored_indices))],
                start=1,
            )
        ]

    def _score_document(self, query_terms: set[str], index: int) -> float:
        frequencies = self.term_frequencies[index]
        document_length = self.document_lengths[index]
        length_normalization = 1 - self.b + self.b * (
            document_length / self.average_document_length
        )

        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue

            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * length_normalization
            score += self.inverse_document_frequencies.get(term, 0.0) * (
                numerator / denominator
            )

        return score

    def _tokenize(self, text: str) -> list[str]:
        text = str(text)
        if self.lowercase:
            text = text.lower()
        return re.findall(r"\b[\w']+\b", text, flags=re.UNICODE)
