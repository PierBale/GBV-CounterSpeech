from typing import Any

from .base import BaseRetriever
from .dense import DenseRetriever
from .bm25 import BM25Retriever
from .reranker import (
    BaseReranker,
    CrossEncoderReranker,
    JinaReranker,
    SequenceClassificationReranker,
)


class RetrieverFactory:
    @staticmethod
    def create(config: dict[str, Any]) -> BaseRetriever:
        retriever_type = config.get("type")

        if retriever_type == "bm25":
            return BM25Retriever(
                lowercase=config.get("lowercase", True),
                retriever_name=config.get("name", "bm25"),
            )

        if retriever_type == "dense":
            return DenseRetriever(
                model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
                index_folder=config.get("index_folder", "data/indexes"),
                index_name=config.get("index_name"),
                index_filename=config.get("index_filename"),
                normalize_embeddings=config.get("normalize_embeddings", True),
                retriever_name=config.get("name", "dense"),
                query_prompt_name=config.get("query_prompt_name"),
                document_prompt_name=config.get("document_prompt_name"),
            )

        raise ValueError(f"Retriever non supportato: {retriever_type}")


class RerankerFactory:
    @staticmethod
    def create(config: dict[str, Any]) -> BaseReranker:
        reranker_type = config.get("type")

        if reranker_type == "cross_encoder":
            return CrossEncoderReranker(
                model_name=config["model_name"],
                batch_size=config.get("batch_size", 16),
                reranker_name=config.get("name"),
            )

        if reranker_type == "sequence_classification":
            return SequenceClassificationReranker(
                model_name=config["model_name"],
                max_length=config.get("max_length", 512),
                batch_size=config.get("batch_size", 16),
                device=config.get("device"),
                reranker_name=config.get("name"),
            )

        if reranker_type == "jina":
            return JinaReranker(
                model_name=config["model_name"],
                reranker_name=config.get("name"),
            )

        raise ValueError(f"Reranker non supportato: {reranker_type}")