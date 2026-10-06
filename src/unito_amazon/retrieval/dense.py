import pickle
import re
from pathlib import Path

import numpy as np

from .base import BaseRetriever
from .schema import Document, RetrievalResult
from .filters import MetadataFilters, get_candidate_indices

def _slugify_model_name(model_name: str) -> str:
    slug = model_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        index_folder: str | Path | None = None,
        index_name: str | None = None,
        index_filename: str | None = None,
        normalize_embeddings: bool = True,
        retriever_name: str = "dense",
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.index_folder = Path(index_folder) if index_folder else None
        self.index_name = index_name
        self.index_filename = index_filename or self._default_index_filename()
        self.normalize_embeddings = normalize_embeddings
        self.retriever_name = retriever_name
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.device = device
        self.batch_size = batch_size

        self.index_path = self._resolve_index_path()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers. Install the "
                "generation requirements before using Qwen3 embeddings."
            ) from error

        self.embedding_model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("[DenseRetriever] La lista di documenti è vuota.")

        self.documents = documents
        texts = [document.text for document in documents]

        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress_bar": True,
            "batch_size": self.batch_size,
        }

        if self.document_prompt_name is not None:
            encode_kwargs["prompt_name"] = self.document_prompt_name

        embeddings = self.embedding_model.encode(
            texts,
            **encode_kwargs,
        )

        self.embeddings = embeddings.astype("float32")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        if self.embeddings is None:
            raise RuntimeError("[DenseRetriever] Devi chiamare index() prima di retrieve().")

        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
        }

        if self.query_prompt_name is not None:
            encode_kwargs["prompt_name"] = self.query_prompt_name

        query_embedding = self.embedding_model.encode(
            query,
            **encode_kwargs,
        ).astype("float32")

        scores = self._compute_scores(query_embedding)

        candidate_indices = get_candidate_indices(
            documents=self.documents,
            metadata_filters=metadata_filters,
        )

        if not candidate_indices:
            return []

        candidate_scores = scores[candidate_indices]

        ranked_candidate_positions = np.argsort(candidate_scores)[::-1][:top_k]

        ranked_indices = [
            candidate_indices[int(position)]
            for position in ranked_candidate_positions
        ]

        results = []

        for rank, idx in enumerate(ranked_indices, start=1):
            idx = int(idx)

            results.append(
                RetrievalResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    retriever_name=self.retriever_name,
                )
            )

        return results

    def save_index(self, path: str | Path | None = None) -> None:
        save_path = Path(path) if path else self.index_path

        if save_path is None:
            raise ValueError("[DenseRetriever] Nessun index_folder o path specificato.")

        if self.embeddings is None:
            raise RuntimeError("[DenseRetriever] Nessun indice da salvare.")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "retriever_name": self.retriever_name,
            "model_name": self.model_name,
            "index_name": self.index_name,
            "index_filename": self.index_filename,
            "normalize_embeddings": self.normalize_embeddings,
            "query_prompt_name": self.query_prompt_name,
            "document_prompt_name": self.document_prompt_name,
            "documents": [document.model_dump() for document in self.documents],
            "embeddings": self.embeddings,
        }

        with open(save_path, "wb") as file:
            pickle.dump(payload, file)

    def load_index(self, path: str | Path | None = None) -> None:
        load_path = Path(path) if path else self.index_path

        if load_path is None:
            raise ValueError("[DenseRetriever] Nessun index_folder o path specificato.")

        with open(load_path, "rb") as file:
            payload = pickle.load(file)

        loaded_model_name = payload.get("model_name")

        if loaded_model_name != self.model_name:
            raise ValueError(
                "[DenseRetriever] L'indice caricato è stato creato con un modello diverso. "
                f"Indice: {loaded_model_name}, retriever attuale: {self.model_name}"
            )
        
        loaded_query_prompt_name = payload.get("query_prompt_name")
        loaded_document_prompt_name = payload.get("document_prompt_name")

        if loaded_query_prompt_name != self.query_prompt_name:
            raise ValueError(
                "[DenseRetriever] L'indice caricato usa query_prompt_name diverso. "
                f"Indice: {loaded_query_prompt_name}, retriever attuale: {self.query_prompt_name}"
            )

        if loaded_document_prompt_name != self.document_prompt_name:
            raise ValueError(
                "[DenseRetriever] L'indice caricato usa document_prompt_name diverso. "
                f"Indice: {loaded_document_prompt_name}, retriever attuale: {self.document_prompt_name}"
            )

        self.documents = [
            Document(**document)
            for document in payload["documents"]
        ]

        self.embeddings = payload["embeddings"]
        self.normalize_embeddings = payload["normalize_embeddings"]

    def index_exists(self) -> bool:
        return self.index_path is not None and self.index_path.exists()

    def load_or_index(
        self,
        documents: list[Document],
        force_reindex: bool = False,
        save: bool = True,
    ) -> None:
        if self.index_exists() and not force_reindex:
            self.load_index()
            return

        self.index(documents)

        if save:
            self.save_index()

    def _default_index_filename(self) -> str:
        model_slug = _slugify_model_name(self.model_name)

        if self.index_name:
            index_slug = _slugify_model_name(self.index_name)
            return f"dense_{model_slug}_{index_slug}.pkl"

        return f"dense_{model_slug}.pkl"

    def _resolve_index_path(self) -> Path | None:
        if self.index_folder is None:
            return None

        return self.index_folder / self.index_filename

    def _compute_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("[DenseRetriever] Embeddings non disponibili.")

        if self.normalize_embeddings:
            return self.embeddings @ query_embedding

        return self._cosine_similarity_matrix(
            matrix=self.embeddings,
            vector=query_embedding,
        )

    @staticmethod
    def _cosine_similarity_matrix(
        matrix: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        matrix_norm = np.linalg.norm(matrix, axis=1)
        vector_norm = np.linalg.norm(vector)

        denominator = matrix_norm * vector_norm
        denominator = np.where(denominator == 0, 1e-12, denominator)

        return (matrix @ vector) / denominator
