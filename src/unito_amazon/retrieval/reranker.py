from abc import ABC, abstractmethod
from typing import Any

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .schema import RetrievalResult


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

# va bene per Qwen/Qwen3-Reranker-0.6B
class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        reranker_name: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.reranker_name = reranker_name or f"cross_encoder:{model_name}"

        self.model = CrossEncoder(model_name, trust_remote_code=True)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        documents = [
            result.document.text
            for result in results
        ]

        pairs = [
            (query, document)
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
        )

        reranked_items = []

        for result, score in zip(results, scores):
            reranked_items.append(
                RetrievalResult(
                    document=result.document,
                    score=float(score),
                    rank=result.rank,
                    retriever_name=f"{result.retriever_name}+{self.reranker_name}",
                )
            )

        reranked_items.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        if top_k is not None:
            reranked_items = reranked_items[:top_k]

        for rank, item in enumerate(reranked_items, start=1):
            item.rank = rank

        return reranked_items
    
# va bene per BAAI/bge-reranker-v2-m3
class SequenceClassificationReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
        reranker_name: str | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker_name = reranker_name or f"sequence_classifier:{model_name}"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        scores = self._score_pairs(
            query=query,
            results=results,
        )

        reranked_items = []

        for result, score in zip(results, scores):
            reranked_items.append(
                RetrievalResult(
                    document=result.document,
                    score=float(score),
                    rank=result.rank,
                    retriever_name=f"{result.retriever_name}+{self.reranker_name}",
                )
            )

        reranked_items.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        if top_k is not None:
            reranked_items = reranked_items[:top_k]

        for rank, item in enumerate(reranked_items, start=1):
            item.rank = rank

        return reranked_items

    def _score_pairs(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[float]:
        all_scores = []

        pairs = [
            [query, result.document.text]
            for result in results
        ]

        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    return_dict=True,
                )

                batch_scores = outputs.logits.view(-1).float().cpu().tolist()
                all_scores.extend(batch_scores)

        return all_scores

    def destroy(self) -> None:
        del self.model
        del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# va bene per jinaai/jina-reranker-v3 (custom architecture con trust_remote_code)
class JinaReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        reranker_name: str | None = None,
    ):
        from transformers import AutoModel

        self.model_name = model_name
        self.reranker_name = reranker_name or f"jina:{model_name}"

        self.model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        documents = [result.document.text for result in results]

        # returns list of dicts: {document, relevance_score, index}
        jina_results = self.model.rerank(
            query,
            documents,
            top_n=top_k,
        )

        # map original index → original RetrievalResult
        index_to_result = {i: results[i] for i in range(len(results))}

        reranked_items = []
        for rank, item in enumerate(jina_results, start=1):
            original = index_to_result[item["index"]]
            reranked_items.append(
                RetrievalResult(
                    document=original.document,
                    score=float(item["relevance_score"]),
                    rank=rank,
                    retriever_name=f"{original.retriever_name}+{self.reranker_name}",
                )
            )

        return reranked_items