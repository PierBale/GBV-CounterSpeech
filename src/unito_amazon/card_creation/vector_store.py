"""Exact cosine retrieval with normalized Qwen embeddings and FAISS."""

import gc
import hashlib
import json
from pathlib import Path

from unito_amazon.retrieval.base import BaseRetriever
from unito_amazon.retrieval.schema import Document, RetrievalResult

from .io import fingerprint
from .progress import report


CHUNK_INSTRUCTION = (
    "Given a type of sexism and its description, retrieve passages containing "
    "specific verifiable evidence useful to counter this type of sexist hate speech."
)
CARD_INSTRUCTION = (
    "Given a hateful message, retrieve evidence useful for writing a factual "
    "counter-speech that directly challenges its harmful idea."
)


class FaissEvidenceRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        instruction: str = CARD_INSTRUCTION,
        batch_size: int = 4,
        max_tokens: int = 32768,
        device: str | None = None,
        retriever_name: str | None = None,
    ):
        if batch_size < 1 or max_tokens < 1:
            raise ValueError("batch_size and max_tokens must be positive.")
        self.model_name = model_name
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.device = device
        self.retriever_name = retriever_name or f"cards_faiss:{model_name}"
        self.documents: list[Document] = []
        self.faiss_index = None
        self.encoder = None

    def _encoder(self):
        if self.encoder is None:
            report("embedding", f"Caricamento modello {self.model_name}; device={self.device or 'auto'}")
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name, device=self.device)
            self.max_tokens = min(self.max_tokens, self.encoder.max_seq_length)
            report("embedding", f"Modello pronto; limite input: {self.max_tokens} token")
        return self.encoder

    def _encode(self, texts: list[str]):
        import numpy as np

        encoder = self._encoder()
        report("embedding", f"Verifica lunghezza di {len(texts)} input, senza troncamento")
        for position, text in enumerate(texts):
            length = len(encoder.tokenizer.encode(text, add_special_tokens=True))
            if length > self.max_tokens:
                raise ValueError(
                    f"Embedding input {position} has {length} tokens, exceeding "
                    f"{self.max_tokens}; the pipeline does not split or truncate it."
                )
        batches = (len(texts) + self.batch_size - 1) // self.batch_size
        report("embedding", f"Encoding di {len(texts)} input in {batches} batch (batch_size={self.batch_size})")
        embeddings = np.asarray(
            encoder.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=True, prompt="",
            ), dtype="float32",
        )
        report("embedding", f"Encoding completato: {len(embeddings)} vettori normalizzati")
        return embeddings

    def index(self, documents: list[Document]) -> None:
        import faiss

        if not documents:
            raise ValueError("Cannot build a FAISS index from an empty library.")
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Retrieval document IDs must be unique.")
        report("faiss", f"Creazione indice per {len(documents)} documenti")
        embeddings = self._encode([document.text for document in documents])
        report("faiss", f"Inserimento vettori nell'indice esatto; dimensione={embeddings.shape[1]}")
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        self.documents = documents
        report("faiss", f"Indice pronto: {self.faiss_index.ntotal} documenti")

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip() or top_k < 1:
            raise ValueError("A non-empty query and positive top_k are required.")
        if self.faiss_index is None:
            raise RuntimeError("Load or build the FAISS index before retrieval.")
        report("faiss", f"Retrieval: top {min(top_k, len(self.documents))} su {len(self.documents)} documenti")
        query_vector = self._encode([f"Instruct: {self.instruction}\nQuery: {query}"])
        scores, positions = self.faiss_index.search(
            query_vector, min(top_k, len(self.documents))
        )
        return [
            RetrievalResult(
                document=self.documents[int(position)], score=float(score),
                rank=rank, retriever_name=self.retriever_name,
            )
            for rank, (position, score) in enumerate(
                zip(positions[0], scores[0], strict=True), start=1
            )
            if position >= 0
        ]

    def document_embeddings(self, document_ids: list[str]):
        """Recover stored document vectors without loading the embedding model."""
        import numpy as np

        if self.faiss_index is None:
            raise RuntimeError("Load or build the FAISS index before reconstructing vectors.")
        positions = {document.id: position for position, document in enumerate(self.documents)}
        missing = [document_id for document_id in document_ids if document_id not in positions]
        if missing:
            raise ValueError(f"Document IDs not found in the FAISS index: {missing[:5]}")
        report("faiss", f"Recupero di {len(document_ids)} embedding gia salvati, senza inferenza")
        if not document_ids:
            return np.empty((0, self.faiss_index.d), dtype="float32")
        return np.asarray([
            self.faiss_index.reconstruct(positions[document_id])
            for document_id in document_ids
        ], dtype="float32")

    def _manifest(self, documents: list[Document]) -> dict:
        records = [document.model_dump(mode="json") for document in documents]
        return {
            "model_name": self.model_name,
            "instruction": self.instruction,
            "normalized": True,
            "documents_sha256": fingerprint(records),
            "documents": records,
        }

    @staticmethod
    def metadata_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".json")

    def save(self, path: Path) -> None:
        import faiss

        if self.faiss_index is None:
            raise RuntimeError("No FAISS index to save.")
        report("faiss", f"Salvataggio indice e manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Python handles Unicode paths on Windows, including this repository's path.
        serialized = faiss.serialize_index(self.faiss_index).tobytes()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(serialized)
        temporary.replace(path)
        self.metadata_path(path).write_text(
            json.dumps(
                {**self._manifest(self.documents),
                 "index_sha256": hashlib.sha256(serialized).hexdigest()},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        report("faiss", f"Indice e manifest salvati: {path}")

    def load(self, path: Path, documents: list[Document]) -> None:
        import faiss
        import numpy as np

        report("faiss", f"Caricamento indice e verifica fingerprint/checksum: {path}")
        metadata = json.loads(self.metadata_path(path).read_text(encoding="utf-8"))
        expected = self._manifest(documents)
        for key in ("model_name", "instruction", "normalized", "documents_sha256"):
            if metadata.get(key) != expected[key]:
                raise ValueError(f"Stale/incompatible FAISS index ({key}): {path}")
        serialized = path.read_bytes()
        if hashlib.sha256(serialized).hexdigest() != metadata.get("index_sha256"):
            raise ValueError(f"FAISS index checksum mismatch: {path}")
        index = faiss.deserialize_index(np.frombuffer(serialized, dtype="uint8").copy())
        if index.ntotal != len(documents):
            raise ValueError(f"FAISS document count mismatch: {path}")
        self.faiss_index = index
        self.documents = documents
        report("faiss", f"Indice verificato e caricato: {index.ntotal} documenti")

    def load_or_index(
        self, documents: list[Document], path: Path, force_reindex: bool = False,
    ) -> None:
        if path.exists() and self.metadata_path(path).exists() and not force_reindex:
            try:
                self.load(path, documents)
                report("faiss", "Indice esistente riutilizzato: embedding dei documenti non ricalcolati")
                return
            except ValueError as error:
                report("faiss", f"Indice non compatibile, ricostruzione necessaria: {error}")
        elif force_reindex:
            report("faiss", "Ricostruzione richiesta con --force-reindex")
        else:
            report("faiss", "Indice o manifest assente: creazione di un nuovo indice")
        self.index(documents)
        self.save(path)

    def close(self) -> None:
        had_encoder = self.encoder is not None
        if had_encoder:
            report("embedding", f"Rilascio modello {self.model_name}")
        self.encoder = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if had_encoder:
            report("embedding", "Modello rilasciato")
