"""CPU-only tests with synthetic vectors and mocked retrieval/reranking."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from unito_amazon.card_creation.config import CardCreationConfig
from unito_amazon.card_creation.cache import StageCache
from unito_amazon.card_creation.deduplication import normalized_text_hash, select_nonredundant
from unito_amazon.card_creation.io import read_jsonl
from unito_amazon.card_creation.retrieval import retrieve_passages
from unito_amazon.card_creation.schema import Chunk, RankedPassage, Source
from unito_amazon.card_creation.vector_store import FaissEvidenceRetriever
from unito_amazon.retrieval.schema import Document, RetrievalResult


class DeduplicationTests(unittest.TestCase):
    def test_exact_duplicates_ignore_whitespace_and_unicode_composition(self):
        result = select_nonredundant(
            ["café  20%\nreported", "cafe\u0301 20% reported", "Other evidence"],
            np.array([[1, 0], [0, 1], [0, 1]]), top_k=2,
        )
        self.assertEqual(result.selected_indices, [0, 2])
        self.assertEqual(result.decisions[1].reason, "exact_duplicate")
        self.assertEqual(result.decisions[1].reference_index, 0)
        # Exact text matching does not depend on the semantic threshold.
        self.assertEqual(result.decisions[1].cosine_similarity, 0.0)

    def test_exact_normalization_preserves_numbers_negation_and_case(self):
        original = normalized_text_hash("Women report 20%")
        for other in ("Women report 30%", "Women do not report 20%", "women report 20%"):
            with self.subTest(other=other):
                self.assertNotEqual(original, normalized_text_hash(other))

    def test_near_duplicate_is_replaced_by_lower_ranked_distinct_candidate(self):
        embeddings = np.array([[5.0, 0.0], [4.99, 0.1], [0.0, 2.0]])
        original = embeddings.copy()
        result = select_nonredundant(["A", "Paraphrase A", "B"], embeddings, top_k=2)
        self.assertEqual(result.selected_indices, [0, 2])
        self.assertEqual(result.decisions[1].reason, "near_duplicate")
        self.assertEqual(result.decisions[1].reference_index, 0)
        self.assertGreaterEqual(result.decisions[1].cosine_similarity, 0.97)
        np.testing.assert_array_equal(embeddings, original)

    def test_cosine_threshold_is_inclusive(self):
        result = select_nonredundant(
            ["A", "B"], np.array([[1, 0], [2, 0]]), top_k=2, similarity_threshold=1.0,
        )
        self.assertEqual(result.selected_indices, [0])
        self.assertEqual(result.decisions[1].reason, "near_duplicate")

    def test_only_selected_representatives_can_remove_candidates(self):
        angles = np.deg2rad([0, 12, 24])
        embeddings = np.column_stack([np.cos(angles), np.sin(angles)])
        result = select_nonredundant(["A", "B", "C"], embeddings, top_k=3)
        # A is similar to B, B to C, but A is not similar enough to C.
        self.assertEqual(result.selected_indices, [0, 2])

    def test_nearest_selected_representative_is_recorded(self):
        result = select_nonredundant(
            ["A", "B", "Paraphrase B"],
            np.array([[1, 0], [0, 1], [0.1, 1]]), top_k=3,
        )
        self.assertEqual(result.selected_indices, [0, 1])
        self.assertEqual(result.decisions[2].reference_index, 1)

    def test_remaining_candidates_are_audited_after_top_k_is_full(self):
        result = select_nonredundant(
            ["A", "B", "Paraphrase A"],
            np.array([[1, 0], [0, 1], [1, 0]]), top_k=1,
        )
        self.assertEqual(result.selected_indices, [0])
        self.assertEqual([item.reason for item in result.decisions], [
            "selected", "top_k_limit", "near_duplicate",
        ])

    def test_does_not_force_k_if_candidates_are_redundant(self):
        result = select_nonredundant(["A", "B"], np.ones((2, 2)), top_k=10)
        self.assertEqual(result.selected_indices, [0])

    def test_disabled_mode_keeps_original_top_k_without_embeddings(self):
        result = select_nonredundant(["A", "A", "B"], None, top_k=2, enabled=False)
        self.assertEqual(result.selected_indices, [0, 1])
        self.assertEqual(result.decisions[2].reason, "top_k_limit")

    def test_invalid_vectors_are_rejected(self):
        for vectors in (
            None, np.array([1, 0]), np.ones((2, 2)), np.empty((1, 0)),
            np.zeros((1, 2)), np.array([[math.nan, 1]]), np.array([[math.inf, 1]]),
        ):
            with self.subTest(vectors=vectors):
                with self.assertRaises(ValueError):
                    select_nonredundant(["A"], vectors, top_k=1)

    def test_invalid_settings_are_rejected(self):
        for threshold in (0, -0.1, 1.1, math.nan, math.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    select_nonredundant(["A"], np.ones((1, 2)), 1, threshold)
        with self.assertRaises(ValueError):
            select_nonredundant(["A"], np.ones((1, 2)), top_k=0)
        for threshold in (0, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                CardCreationConfig(deduplication_similarity_threshold=threshold)


class StoredEmbeddingTests(unittest.TestCase):
    def test_reconstruction_preserves_id_order_without_loading_a_model(self):
        retriever = FaissEvidenceRetriever()
        retriever.documents = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
        stored = np.array([[1, 0], [0, 1]], dtype=np.float32)
        retriever.faiss_index = SimpleNamespace(d=2, reconstruct=Mock(side_effect=lambda i: stored[i]))
        with patch.object(retriever, "_encoder", side_effect=AssertionError("Model must not load")):
            vectors = retriever.document_embeddings(["b", "a"])
        np.testing.assert_array_equal(vectors, stored[[1, 0]])
        self.assertIsNone(retriever.encoder)
        with self.assertRaises(ValueError):
            retriever.document_embeddings(["missing"])
        self.assertEqual(retriever.document_embeddings([]).shape, (0, 2))

    def test_reconstruction_requires_an_index(self):
        with self.assertRaises(RuntimeError):
            FaissEvidenceRetriever().document_embeddings(["a"])


class RetrievalIntegrationTests(unittest.TestCase):
    def test_filter_backfills_per_category_and_saves_original_ranks_and_audit(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                labels_path = root / "labels.json"
                labels_path.write_text(json.dumps({"type_a": "Description A", "type_b": "Description B"}))
                config = CardCreationConfig(
                    output_dir=root / "output", labels_path=labels_path,
                    dense_top_j=3, rerank_top_k=2, deduplication_enabled=enabled,
                )
                source = Source(source_id="source", markdown_file="source.md", title="Source")
                chunks = [
                    Chunk(
                        chunk_id=key, text=text, token_count=100,
                        paragraph_ids=[i], section_path=[], source=source,
                    )
                    for i, (key, text) in enumerate([
                        ("a", "First evidence"), ("b", "Rephrased first evidence"), ("c", "Distinct evidence"),
                    ], start=1)
                ]
                lookup = {chunk.chunk_id: chunk for chunk in chunks}
                candidates = [
                    RetrievalResult(
                        document=Document(id=key, text=lookup[key].text),
                        score=1.0 / rank, rank=rank, retriever_name="mock",
                    )
                    for rank, key in enumerate(["a", "c", "b"], start=1)
                ]
                stored = {"a": [1, 0], "b": [0.999, 0.01], "c": [0, 1]}
                retriever = Mock()
                retriever.retrieve.return_value = candidates
                retriever.document_embeddings.side_effect = lambda ids: np.array([stored[key] for key in ids])
                reranker = Mock()
                reranker.score.return_value = [0.99, 0.80, 0.95]
                with (
                    patch("unito_amazon.card_creation.retrieval.load_chunks", return_value=chunks),
                    patch("unito_amazon.card_creation.retrieval.chunk_retriever", return_value=retriever),
                    patch("unito_amazon.card_creation.reranking.QwenReranker", return_value=reranker),
                    patch("unito_amazon.card_creation.retrieval.report"),
                ):
                    retrieve_passages(config)
                selected = [RankedPassage.model_validate(item) for item in read_jsonl(config.retrieved_path)]
                expected_ids = ["a", "c"] if enabled else ["a", "b"]
                self.assertEqual([item.chunk.chunk_id for item in selected], expected_ids * 2)
                self.assertEqual([item.label for item in selected], ["type_a", "type_a", "type_b", "type_b"])
                self.assertEqual([item.reranker_rank for item in selected], ([1, 3] if enabled else [1, 2]) * 2)
                self.assertEqual(len(read_jsonl(config.ranked_path)), 6)
                audit = read_jsonl(config.deduplication_log_path)
                self.assertEqual(len(audit), 6)
                if enabled:
                    retriever.document_embeddings.assert_called_once()
                    duplicates = [item for item in audit if item["decision"] == "near_duplicate"]
                    self.assertEqual(len(duplicates), 2)
                    self.assertTrue(all(item["chunk_id"] == "b" and item["duplicate_of"] == "a" for item in duplicates))
                    self.assertTrue(all(item["cosine_similarity"] >= 0.97 for item in duplicates))
                else:
                    retriever.document_embeddings.assert_not_called()
                    self.assertTrue(all(item["method"] == "disabled" for item in audit))
                self.assertEqual(retriever.close.call_count, 2 if enabled else 1)
                reranker.close.assert_called_once()
                # A second run must not reload Qwen or query FAISS again.
                retriever.reset_mock()
                reranker.reset_mock()
                with (
                    patch("unito_amazon.card_creation.retrieval.chunk_retriever", side_effect=AssertionError("No model/index reload")),
                    patch("unito_amazon.card_creation.reranking.QwenReranker", side_effect=AssertionError("No reranker reload")),
                    patch("unito_amazon.card_creation.retrieval.report"),
                ):
                    retrieve_passages(config)
                if enabled:
                    # Changing only the threshold must reuse dense retrieval and reranking.
                    updated = config.model_copy(update={"deduplication_similarity_threshold": 0.99999})
                    retriever.retrieve.side_effect = AssertionError("No new dense retrieval")
                    with (
                        patch("unito_amazon.card_creation.retrieval.load_chunks", return_value=chunks),
                        patch("unito_amazon.card_creation.retrieval.chunk_retriever", return_value=retriever),
                        patch("unito_amazon.card_creation.reranking.QwenReranker", side_effect=AssertionError("No new reranking")),
                        patch("unito_amazon.card_creation.retrieval.report"),
                    ):
                        retrieve_passages(updated)
                    selected = read_jsonl(config.retrieved_path)
                    self.assertEqual([item["chunk"]["chunk_id"] for item in selected], ["a", "b"] * 2)
                    # An interrupted dense pass resumes at the remaining category.
                    StageCache(config).invalidate("dense")
                    retriever.retrieve.side_effect = [candidates, RuntimeError("Interrupted category")]
                    with (
                        patch("unito_amazon.card_creation.retrieval.load_chunks", return_value=chunks),
                        patch("unito_amazon.card_creation.retrieval.chunk_retriever", return_value=retriever),
                        patch("unito_amazon.card_creation.reranking.QwenReranker", return_value=reranker),
                        patch("unito_amazon.card_creation.retrieval.report"),
                    ):
                        with self.assertRaises(RuntimeError):
                            retrieve_passages(config)
                        self.assertEqual(len(read_jsonl(config.dense_path)), 3)
                        retriever.reset_mock()
                        retriever.retrieve.side_effect = None
                        retriever.retrieve.return_value = candidates
                        retrieve_passages(config)
                    retriever.retrieve.assert_called_once()
                    # A partially scored pool must retain completed categories too.
                    StageCache(config).invalidate("rerank")
                    reranker.score.side_effect = [[0.99, 0.80, 0.95], RuntimeError("Interrupted reranking")]
                    with (
                        patch("unito_amazon.card_creation.retrieval.load_chunks", return_value=chunks),
                        patch("unito_amazon.card_creation.retrieval.chunk_retriever", return_value=retriever),
                        patch("unito_amazon.card_creation.reranking.QwenReranker", return_value=reranker),
                        patch("unito_amazon.card_creation.retrieval.report"),
                    ):
                        with self.assertRaises(RuntimeError):
                            retrieve_passages(config)
                        self.assertEqual(len(read_jsonl(config.ranked_path)), 3)
                        reranker.reset_mock()
                        reranker.score.side_effect = None
                        retrieve_passages(config)
                    reranker.score.assert_called_once()


if __name__ == "__main__":
    unittest.main()
