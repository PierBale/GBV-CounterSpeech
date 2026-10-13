"""Checkpoint tests with temporary files and mocked models; no GPU required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from unito_amazon.card_creation.cache import DEPENDENCIES, StageCache
from unito_amazon.card_creation.config import CardCreationConfig
from unito_amazon.card_creation.generation import generate_cards
from unito_amazon.card_creation.io import read_jsonl, write_json, write_jsonl
from unito_amazon.card_creation.card_schema import CardChunk, EvidenceCard
from unito_amazon.card_creation.schema import Chunk, RankedPassage, Source


class CacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.markdown = self.root / "markdown"
        self.markdown.mkdir()
        self.source_path = self.markdown / "source.md"
        self.source_path.write_text("Original evidence", encoding="utf-8")
        self.labels_path = self.root / "labels.json"
        self.labels_path.write_text('{"type": "description"}', encoding="utf-8")
        self.config = CardCreationConfig(
            output_dir=self.root / "output", markdown_dir=self.markdown, labels_path=self.labels_path,
        )
        reporter = patch("unito_amazon.card_creation.cache.report")
        reporter.start()
        self.addCleanup(reporter.stop)

    def populate(self):
        cache = StageCache(self.config)
        for stage, parents in DEPENDENCIES.items():
            files = [cache.outputs[parent][0] for parent in parents]
            directories = [(self.markdown, ("*.md",))] if stage == "chunk" else []
            ticket = cache.prepare(stage, files=files, directories=directories)
            for path in cache.outputs[stage]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(stage, encoding="utf-8")
            cache.checkpoint(ticket, complete=True)
        return cache

    def test_completed_chunk_is_reused(self):
        self.populate()
        cache = StageCache(self.config)
        ticket = cache.prepare("chunk", directories=[(self.markdown, ("*.md",))])
        self.assertEqual(ticket.mode, "cached")

    def test_source_changes_invalidate_descendants_but_preserve_markdown(self):
        self.populate()
        self.source_path.write_text("Updated evidence", encoding="utf-8")
        cache = StageCache(self.config)
        ticket = cache.prepare("chunk", directories=[(self.markdown, ("*.md",))])
        self.assertEqual(ticket.mode, "run")
        self.assertFalse(self.config.cards_path.exists())
        self.assertFalse(self.config.chunk_index_path.exists())
        self.assertEqual(self.source_path.read_text(encoding="utf-8"), "Updated evidence")

    def test_new_markdown_file_invalidates_chunk_cache(self):
        self.populate()
        (self.markdown / "new.md").write_text("New evidence", encoding="utf-8")
        ticket = StageCache(self.config).prepare("chunk", directories=[(self.markdown, ("*.md",))])
        self.assertEqual(ticket.mode, "run")

    def test_generator_change_keeps_all_retrieval_artifacts(self):
        cache = self.populate()
        updated = self.config.model_copy(update={"generator_model": "test/other-generator"})
        StageCache(updated)
        for stage in ("chunk", "index", "dense", "rerank", "dedup"):
            self.assertTrue(all(path.exists() for path in cache.outputs[stage]))
        self.assertFalse(self.config.cards_path.exists())
        self.assertFalse(self.config.card_index_path.exists())

    def test_threshold_change_keeps_dense_and_reranked_passages(self):
        self.populate()
        updated = self.config.model_copy(update={"deduplication_similarity_threshold": 0.99})
        StageCache(updated)
        self.assertTrue(self.config.dense_path.exists())
        self.assertTrue(self.config.ranked_path.exists())
        self.assertTrue(self.config.chunk_index_path.exists())
        self.assertFalse(self.config.retrieved_path.exists())
        self.assertFalse(self.config.cards_path.exists())

    def test_temperature_change_keeps_retrieval_but_invalidates_cards(self):
        cache = self.populate()
        StageCache(self.config.model_copy(update={"generation_temperature": 0.5}))
        for stage in ("chunk", "index", "dense", "rerank", "dedup"):
            self.assertTrue(all(path.exists() for path in cache.outputs[stage]))
        self.assertFalse(self.config.cards_path.exists())
        self.assertFalse(self.config.card_index_path.exists())

    def test_top_j_change_keeps_chunking_and_index(self):
        self.populate()
        StageCache(self.config.model_copy(update={"dense_top_j": 50}))
        self.assertTrue(self.config.chunks_path.exists())
        self.assertTrue(self.config.chunk_index_path.exists())
        self.assertFalse(self.config.dense_path.exists())
        self.assertFalse(self.config.ranked_path.exists())

    def test_performance_settings_do_not_erase_completed_work(self):
        self.populate()
        StageCache(self.config.model_copy(update={
            "embedding_batch_size": 1, "reranker_batch_size": 1,
            "generation_retries": 5, "device": "cpu", "long_chunk_warning_tokens": 5000,
        }))
        self.assertTrue(self.config.cards_path.exists())
        self.assertTrue(self.config.ranked_path.exists())

    def test_partial_checkpoint_can_resume_after_new_atomic_writes(self):
        cache = StageCache(self.config)
        ticket = cache.prepare("dense", files=[self.labels_path], allow_partial=True)
        write_jsonl(self.config.dense_path, [{"label": "first"}])
        cache.checkpoint(ticket)
        write_jsonl(self.config.dense_path, [{"label": "first"}, {"label": "second"}])
        resumed = StageCache(self.config).prepare("dense", files=[self.labels_path], allow_partial=True)
        self.assertEqual(resumed.mode, "resume")
        self.assertEqual(len(read_jsonl(self.config.dense_path)), 2)

    def test_single_stage_refuses_stale_upstream_inputs(self):
        self.populate()
        self.source_path.write_text("Changed input", encoding="utf-8")
        with self.assertRaises(ValueError):
            StageCache(self.config).prepare("generate", files=[self.config.retrieved_path])
        self.assertTrue(self.source_path.exists())

    def test_missing_completed_output_is_rebuilt_with_descendants(self):
        self.populate()
        self.config.dense_path.unlink()
        ticket = StageCache(self.config).prepare("dense", files=[self.config.chunk_index_path])
        self.assertEqual(ticket.mode, "run")
        self.assertFalse(self.config.ranked_path.exists())
        self.assertTrue(self.config.chunk_index_path.exists())
        self.assertTrue(self.source_path.exists())

    def test_invalidation_refuses_paths_outside_output_directory(self):
        cache = self.populate()
        cache.outputs["chunk"] = [self.source_path]
        with self.assertRaises(ValueError):
            cache.invalidate("chunk")
        self.assertTrue(self.source_path.exists())
        self.assertTrue(self.config.chunk_index_path.exists())

    def test_manually_edited_cards_are_preserved_when_indexing(self):
        cache = self.populate()
        source = Source(source_id="s", markdown_file="source.md", title="Source")
        card = EvidenceCard(
            card_id="card", primary_edos_label="type", edos_definition="description", source=source,
            chunk=CardChunk(chunk_id="chunk", text="Evidence", section_path=[], paragraph_ids=[1], token_count=100),
            reasoning="Reason", evidence="Qualitative evidence", retrieval={},
            generator_model=self.config.generator_model,
            prompt={"system": "System instructions", "user": "Full source prompt"},
        )
        write_jsonl(self.config.cards_path, [card.model_dump(mode="json")])
        write_jsonl(self.config.attempts_path, [{"status": "accepted", "generation_key": "key", "card_id": "card"}])
        write_json(self.config.output_dir / "generation_manifest.json", {"signature": "legacy"})
        ticket = cache.prepare("generate", files=[self.config.retrieved_path], preserve_existing=True)
        cache.checkpoint(ticket, complete=True)
        reviewed = card.model_copy(update={"evidence": "Manually corrected summary"})
        write_jsonl(self.config.cards_path, [reviewed.model_dump(mode="json")])
        ticket = StageCache(self.config).prepare("index-cards", files=[self.config.cards_path])
        self.assertEqual(ticket.mode, "run")
        self.assertEqual(read_jsonl(self.config.cards_path)[0]["evidence"], "Manually corrected summary")


class GenerationResumeTests(unittest.TestCase):
    def test_model_failure_resumes_missing_cards_and_completed_run_skips_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CardCreationConfig(output_dir=Path(directory) / "output")
            source = Source(source_id="s", markdown_file="source.md", title="Source")
            records = []
            for number in (1, 2):
                chunk = Chunk(
                    chunk_id=f"chunk_{number}", text=f"Evidence {number}", token_count=100,
                    paragraph_ids=[number], section_path=[], source=source,
                )
                records.append(RankedPassage(
                    label="type", definition="description", embedding_model=config.embedding_model,
                    reranker_model=config.reranker_model, chunk=chunk, dense_rank=number,
                    dense_score=0.9, reranker_rank=number, reranker_score=0.9,
                ).model_dump(mode="json"))
            write_jsonl(config.retrieved_path, records)

            def response(number):
                return json.dumps({
                    "reasoning": "Reason", "evidence": f"Qualitative evidence {number}",
                })

            failed = Mock()
            failed.generate.side_effect = [response(1), RuntimeError("Simulated interruption")]
            with patch("unito_amazon.card_creation.generation.GemmaCardGenerator", return_value=failed):
                with self.assertRaises(RuntimeError):
                    generate_cards(config)
            self.assertEqual(len(read_jsonl(config.cards_path)), 1)
            resumed = Mock()
            resumed.generate.return_value = response(2)
            with patch("unito_amazon.card_creation.generation.GemmaCardGenerator", return_value=resumed):
                generate_cards(config)
            self.assertEqual(resumed.generate.call_count, 1)
            self.assertEqual(len(read_jsonl(config.cards_path)), 2)
            cards = read_jsonl(config.cards_path)
            self.assertEqual(cards[0]["chunk"]["text"], records[0]["chunk"]["text"])
            self.assertEqual(cards[0]["evidence"], "Qualitative evidence 1")
            self.assertEqual(cards[0]["schema_version"], EvidenceCard.model_fields["schema_version"].default)
            self.assertEqual(cards[0]["prompt"]["user"], failed.generate.call_args_list[0].args[0])
            self.assertNotIn("generation_key", cards[0])
            self.assertNotIn("argument", cards[0])
            with patch("unito_amazon.card_creation.generation.GemmaCardGenerator", side_effect=AssertionError("No model load")):
                generate_cards(config)


if __name__ == "__main__":
    unittest.main()
