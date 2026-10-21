"""Exercise the real stages/checkpoints with small deterministic model doubles."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from unito_amazon.card_creation.config import CardCreationConfig, EMBEDDING_MODELS, RERANKING_MODELS, model_slug
from unito_amazon.card_creation.experiment import run_experiment
from unito_amazon.card_creation.generation import generate_cards
from unito_amazon.card_creation.io import file_sha256, read_jsonl, write_jsonl
from unito_amazon.card_creation.queries import load_queries
from unito_amazon.card_creation.schema import Chunk, Source
from unito_amazon.card_creation.vector_store import FaissEvidenceRetriever
from unito_amazon.retrieval.schema import RetrievalResult


class RetrievalExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.labels = self.root / "labels.json"
        self.labels.write_text('{"label": "Definition"}', encoding="utf-8")
        self.queries = self.root / "queries.jsonl"
        write_jsonl(self.queries, [
            {"query_id": "q1", "query_text": "First query", "edos_label": "label", "metadata": {"category": "test"}},
            {"query_id": "q2", "query_text": "Second query", "edos_label": "label"},
        ])
        self.config = CardCreationConfig(
            output_dir=self.root / "output", markdown_dir=self.root / "markdown",
            labels_path=self.labels, queries_path=self.queries, retrieval_top_k=3,
            rerank_top_k=2, all_combinations=True, generate_evidence=False, run_id="test-run",
        )
        self.chunks = [Chunk(
            chunk_id=f"chunk-{i}", text="Repeated text" if i < 3 else "Other text",
            token_count=100, paragraph_ids=[i], section_path=["Section"],
            source=Source(source_id=f"doc-{i}", markdown_file=f"{i}.md", title=f"Document {i}"),
        ) for i in range(1, 4)]
        self.retrieval_calls = []
        self.index_calls = []
        self.rerank_calls = []
        self.chunk_calls = []
        self.failure = None
        self.retrieval_failure = None
        stack = ExitStack()
        self.addCleanup(stack.close)
        for module in ("cache", "retrieval", "generation", "experiment"):
            stack.enter_context(patch(f"unito_amazon.card_creation.{module}.report"))
        stack.enter_context(patch("unito_amazon.card_creation.experiment.create_chunks", side_effect=self.chunk_stage))
        stack.enter_context(patch("unito_amazon.card_creation.retrieval.chunk_retriever", side_effect=self.retriever))
        stack.enter_context(patch("unito_amazon.card_creation.reranking.create_reranker", side_effect=self.reranker))
        self.generator = stack.enter_context(patch("unito_amazon.card_creation.generation.GemmaCardGenerator"))
        self.generator.side_effect = AssertionError("Evidence must be completely disabled")

    def chunk_stage(self, config, cache):
        ticket = cache.prepare("chunk", directories=[(config.markdown_dir, ("*.md", "*.parse.json"))])
        if ticket.mode != "cached":
            self.chunk_calls.append(config.chunk_tokenizer)
            write_jsonl(config.chunks_path, (chunk.model_dump(mode="json") for chunk in self.chunks))
            cache.checkpoint(ticket, complete=True)

    def retriever(self, config):
        fake = Mock()

        def index(documents, path, force_reindex=False):
            self.index_calls.append(config.embedding_model)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(config.embedding_model.encode())
            path.with_suffix(".faiss.json").write_text("{}", encoding="utf-8")

        def retrieve(query, top_k):
            self.retrieval_calls.append((config.embedding_model, query, top_k))
            if self.retrieval_failure and self.retrieval_failure(config, query):
                raise RuntimeError("Synthetic retrieval failure")
            chunks = self.chunks if config.embedding_model == EMBEDDING_MODELS[0] else self.chunks[::-1]
            return [RetrievalResult(
                document={"id": chunk.chunk_id, "text": chunk.text}, score=0.9 - rank / 10,
                rank=rank, retriever_name="fake",
            ) for rank, chunk in enumerate(chunks[:top_k], 1)]

        fake.load_or_index.side_effect = index
        fake.retrieve.side_effect = retrieve
        return fake

    def reranker(self, config):
        fake = Mock()

        def score(query, texts):
            self.rerank_calls.append((config.embedding_model, config.reranker_model, query, texts))
            if self.failure and self.failure(config, query):
                raise RuntimeError("Synthetic scoring failure")
            # An exact tie must retain retrieval order; negative raw scores survive.
            return [-5.0, 2.0, 2.0]

        fake.score.side_effect = score
        return fake

    def test_matrix_preserves_candidates_provenance_duplicates_and_skips_cached_models(self):
        root = run_experiment(self.config)
        self.assertEqual(len(self.chunk_calls), 1)
        self.assertEqual(self.index_calls, list(EMBEDDING_MODELS))
        self.assertEqual(len(self.retrieval_calls), 4)  # two models, two queries
        self.assertEqual(len(self.rerank_calls), 8)  # four combinations, two queries
        for embedding in EMBEDDING_MODELS:
            retrieval = read_jsonl(root / "retrieval" / f"{model_slug(embedding)}.jsonl")
            self.assertEqual(len(retrieval), 6)
            originals = {(row["query_id"], row["chunk_id"]): row for row in retrieval}
            for reranker in RERANKING_MODELS:
                pipeline = f"{model_slug(embedding)}__{model_slug(reranker)}"
                selected = read_jsonl(root / "reranking" / f"{pipeline}.jsonl")
                full = read_jsonl(root / "reranking_full" / f"{pipeline}.jsonl")
                self.assertEqual(len(selected), 4)
                self.assertEqual(len(full), 6)
                for row in full:
                    for key, value in originals[(row["query_id"], row["chunk_id"])].items():
                        self.assertEqual(row[key], value)
                    self.assertEqual(row["document_id"], row["chunk"]["source"]["source_id"])
                    self.assertEqual(row["metadata"]["section_path"], ["Section"])
                    self.assertEqual(row["run_id"], self.config.run_id)
                    self.assertEqual(row["pipeline_id"], pipeline)
                    self.assertEqual(row["edos_label"], "label")
                for query in ("q1", "q2"):
                    rows = [row for row in full if row["query_id"] == query]
                    self.assertEqual([row["retrieval_rank"] for row in rows], [2, 3, 1])
                    self.assertEqual([row["rerank_score"] for row in rows], [2.0, 2.0, -5.0])
            calls = [call for call in self.rerank_calls if call[0] == embedding]
            self.assertEqual(calls[0][2:], calls[2][2:])
            self.assertEqual(calls[1][2:], calls[3][2:])
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse((root / "evidence").exists())
        self.generator.assert_not_called()
        run_experiment(self.config)
        self.assertEqual(len(self.index_calls), 2)
        self.assertEqual(len(self.retrieval_calls), 4)
        self.assertEqual(len(self.rerank_calls), 8)
        # A new archived run also reuses compatible work-stage results.
        other = run_experiment(self.config.model_copy(update={"run_id": "new-run"}))
        self.assertEqual(len(self.retrieval_calls), 4)
        self.assertEqual(read_jsonl(other / "retrieval" / "bge-m3.jsonl")[0]["run_id"], "new-run")

    def test_scoring_failure_is_explicit_and_resumes_only_missing_query(self):
        self.failure = lambda config, query: config.reranker_model == RERANKING_MODELS[1] and query == "Second query"
        with self.assertRaisesRegex(RuntimeError, "Synthetic"):
            run_experiment(self.config)
        root = self.config.output_dir / "runs" / self.config.run_id
        self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "failed")
        self.assertTrue((root / "retrieval" / "qwen3-embedding-8b.jsonl").exists())
        failed_output = root / "reranking" / "qwen3-embedding-8b__qwen3-reranker-8b.jsonl"
        self.assertFalse(failed_output.exists())
        error = read_jsonl(root / "errors.jsonl")[-1]
        self.assertEqual((error["stage"], error["query_id"], error["edos_label"]), ("reranking", "q2", "label"))
        self.failure = None
        run_experiment(self.config)
        self.assertEqual(len(self.retrieval_calls), 4)
        self.assertEqual(len(self.rerank_calls), 9)  # eight successes plus one failed query
        self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "complete")

    def test_evidence_uses_selected_chunks_and_has_separate_outputs(self):
        self.generator.side_effect = None
        self.generator.return_value.generate.return_value = '{"reasoning":"Reason","evidence":"Evidence"}'
        root = run_experiment(self.config.model_copy(update={"generate_evidence": True}))
        files = sorted((root / "evidence").glob("*.jsonl"))
        self.assertEqual(len(files), 8)  # cards and attempts per combination
        self.assertEqual(self.generator.call_count, 4)
        self.assertEqual(self.generator.return_value.generate.call_count, 16)
        for file in files:
            for record in read_jsonl(file):
                self.assertEqual(record["run_id"], self.config.run_id)
                if "retrieval" in record:
                    self.assertIn(record["retrieval"]["query_id"], {"q1", "q2"})
                    self.assertLessEqual(record["retrieval"]["reranker_rank"], 2)
        before = {path: file_sha256(path) for path in root.glob("retrieval/*.jsonl")}
        run_experiment(self.config.model_copy(update={"generate_evidence": True}))
        self.assertEqual(before, {path: file_sha256(path) for path in before})
        self.assertEqual(self.generator.call_count, 4)

    def test_runtime_evidence_error_keeps_retrieval_archives(self):
        self.generator.side_effect = None
        self.generator.return_value.generate.side_effect = RuntimeError("LLM failed")
        with self.assertRaisesRegex(RuntimeError, "LLM failed"):
            run_experiment(self.config.model_copy(update={"generate_evidence": True}))
        root = self.config.output_dir / "runs" / self.config.run_id
        self.assertEqual(len(list(root.glob("retrieval/*.jsonl"))), 2)
        self.assertEqual(len(list(root.glob("reranking/*.jsonl"))), 4)
        self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "failed")

    def test_retrieval_failure_records_query_and_resumes_partial_checkpoint(self):
        self.retrieval_failure = lambda config, query: query == "Second query"
        with self.assertRaisesRegex(RuntimeError, "Synthetic retrieval"):
            run_experiment(self.config)
        root = self.config.output_dir / "runs" / self.config.run_id
        error = read_jsonl(root / "errors.jsonl")[-1]
        self.assertEqual((error["stage"], error["query_id"]), ("retrieval", "q2"))
        self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "failed")
        self.retrieval_failure = None
        run_experiment(self.config)
        self.assertEqual(len(self.retrieval_calls), 5)
        self.assertEqual(len(self.rerank_calls), 8)

    def test_changing_top_k_reuses_full_ranking_without_new_inference(self):
        run_experiment(self.config)
        root = run_experiment(self.config.model_copy(update={"run_id": "top-one", "rerank_top_k": 1}))
        self.assertEqual(len(self.retrieval_calls), 4)
        self.assertEqual(len(self.rerank_calls), 8)
        for path in root.glob("reranking/*.jsonl"):
            self.assertEqual(len(read_jsonl(path)), 2)
        for path in root.glob("reranking_full/*.jsonl"):
            self.assertEqual(len(read_jsonl(path)), 6)

    def test_disabled_generation_does_not_even_construct_cache(self):
        with patch("unito_amazon.card_creation.generation.StageCache", side_effect=AssertionError("No cache or LLM")):
            generate_cards(self.config)

    def test_changed_query_cannot_overwrite_existing_run(self):
        root = run_experiment(self.config)
        checksum = file_sha256(root / "retrieval" / "bge-m3.jsonl")
        queries = read_jsonl(self.queries)
        queries[0]["query_text"] = "Changed query"
        write_jsonl(self.queries, queries)
        with self.assertRaisesRegex(ValueError, "different inputs/settings"):
            run_experiment(self.config)
        self.assertEqual(file_sha256(root / "retrieval" / "bge-m3.jsonl"), checksum)


class QueryAndEncodingTests(unittest.TestCase):
    def test_legacy_queries_have_stable_ids_and_allow_multiple_queries_per_label(self):
        config = CardCreationConfig()
        first = load_queries(config)
        self.assertEqual(first, load_queries(config))
        self.assertEqual(len({query.query_id for query in first}), len(first))
        self.assertTrue(all(query.query_id != query.edos_label for query in first))

    def test_top_k_alias_and_validation(self):
        self.assertEqual(CardCreationConfig(retrieval_top_k=20).dense_top_j, 20)
        self.assertEqual(CardCreationConfig(dense_top_j=20).retrieval_top_k, 20)
        with self.assertRaises(ValidationError):
            CardCreationConfig(retrieval_top_k=5, rerank_top_k=10)

    def test_only_qwen_queries_receive_instruction(self):
        qwen = FaissEvidenceRetriever(model_name=EMBEDDING_MODELS[0], instruction="Task")
        bge = FaissEvidenceRetriever(model_name=EMBEDDING_MODELS[1], instruction="Task")
        self.assertEqual(qwen.format_query("Query"), "Instruct: Task\nQuery: Query")
        self.assertEqual(bge.format_query("Query"), "Query")


if __name__ == "__main__":
    unittest.main()
