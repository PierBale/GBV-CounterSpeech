"""Word-based segmentation, shared-corpus caching and size validation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from unito_amazon.card_creation.cache import StageCache
from unito_amazon.card_creation.chunking import build_document_chunks, count_words, create_chunks
from unito_amazon.card_creation.config import CardCreationConfig
from unito_amazon.card_creation.io import read_jsonl, write_jsonl
from unito_amazon.card_creation.retrieval import load_chunks
from unito_amazon.card_creation.schema import Source
from unito_amazon.card_creation.tests import test_retrieval_experiment as experiments


class WordChunkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.markdown = self.root / "markdown"
        self.markdown.mkdir()
        self.config = CardCreationConfig(
            output_dir=self.root / "output", markdown_dir=self.markdown,
            chunk_size_unit="words", min_chunk_words=100, model_scoped_artifacts=True,
        )
        self.source = Source(source_id="doc", markdown_file="doc.md", title="Document")

    def test_visible_words_ignore_html_markup_and_keep_compounds(self):
        self.assertEqual(count_words("women’s gender-based well‑being aren't 2026"), 5)
        self.assertEqual(count_words('<table class="ignored"><tr><td>Women</td><td>42</td></tr></table>'), 2)
        self.assertEqual(count_words("women&nbsp;and girls &amp; women"), 4)
        self.assertEqual(count_words(" | --- + = & "), 0)

    def test_short_paragraphs_merge_within_section_and_absorb_tail(self):
        paragraphs = [(1, ("Results",), "word " * 60), (3, ("Results",), "word " * 40),
                      (5, ("Results",), "tail " * 10), (7, ("Other",), "separate " * 99)]
        audit = []
        chunks = build_document_chunks(paragraphs, self.source, minimum=100, size_unit="words", audit=audit)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].paragraph_ids, [1, 3, 5])
        self.assertEqual(chunks[0].body_word_count, 110)
        self.assertIsNone(chunks[0].token_count)
        self.assertIsNone(chunks[0].body_token_count)
        self.assertEqual(audit[0]["rule_id"], "below_minimum_words")
        self.assertEqual(audit[0]["body_word_count"], 99)

    def test_headings_cannot_pad_body_and_long_paragraphs_remain_intact(self):
        paragraphs = [(1, ("long heading " * 100,), "body " * 99),
                      (2, ("Results",), "evidence " * 1500)]
        chunks = build_document_chunks(paragraphs, self.source, minimum=100, size_unit="words")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].body_word_count, 1500)
        self.assertEqual(chunks[0].body_text, paragraphs[1][2])

    def test_chunk_stage_never_imports_transformers_and_retrieval_accepts_word_counts(self):
        (self.markdown / "doc.md").write_text("# Results\n\n" + "evidence " * 100, encoding="utf-8")
        with patch.dict("sys.modules", {"transformers": None}):
            create_chunks(self.config)
        chunks = load_chunks(self.config)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].body_word_count, 100)
        self.assertIsNone(self.config.chunk_tokenizer)
        # Both metadata counts and actual body content must meet the word minimum.
        records = read_jsonl(self.config.chunks_path)
        records[0]["body_word_count"] = 1000
        write_jsonl(self.config.chunks_path, records)
        with self.assertRaisesRegex(ValueError, "undersized"):
            load_chunks(self.config)

    def test_word_boundaries_and_cache_do_not_depend_on_embedder_or_tokenizer(self):
        path = self.markdown / "doc.md"
        path.write_text("# Results\n\n" + "evidence " * 100, encoding="utf-8")
        create_chunks(self.config)
        original = self.config.chunks_path.read_bytes()
        other = self.config.model_copy(update={"embedding_model": "BAAI/bge-m3", "chunk_tokenizer_model": "unused/tokenizer"})
        self.assertEqual(StageCache(other).settings["chunk"], StageCache(self.config).settings["chunk"])
        with patch.dict("sys.modules", {"transformers": None}):
            create_chunks(other)
        self.assertEqual(other.chunks_path.read_bytes(), original)

    def test_word_threshold_change_invalidates_chunk_cache(self):
        (self.markdown / "doc.md").write_text("# Results\n\n" + "evidence " * 150, encoding="utf-8")
        create_chunks(self.config)
        other = self.config.model_copy(update={"min_chunk_words": 120})
        ticket = StageCache(other).prepare("chunk", directories=[(self.markdown, ("*.md", "*.parse.json"))])
        self.assertEqual(ticket.mode, "run")
        self.assertTrue((self.markdown / "doc.md").is_file())


class WordExperimentTests(experiments.RetrievalExperimentTests):
    """Reuse the complete four-model/resume suite with word-mode chunk metadata."""

    def setUp(self):
        super().setUp()
        self.config = self.config.model_copy(update={"chunk_size_unit": "words", "min_chunk_words": 100})
        for chunk in self.chunks:
            chunk.body_word_count = count_words(chunk.body_text)
            chunk.word_count = count_words(chunk.text)
            chunk.body_token_count = None
            chunk.token_count = None

    def test_manifest_records_words_and_no_chunk_tokenizer(self):
        from unito_amazon.card_creation.experiment import run_experiment
        root = run_experiment(self.config)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunking"]["size_unit"], "words")
        self.assertEqual(manifest["chunking"]["minimum_body_words"], 100)
        self.assertIsNone(manifest["encoding"]["chunk_tokenizer"])
        self.assertEqual(len(manifest["reranking"]), 4)


if __name__ == "__main__":
    unittest.main()
