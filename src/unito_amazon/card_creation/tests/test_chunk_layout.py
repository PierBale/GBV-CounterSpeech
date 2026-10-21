"""Deterministic page cleanup, hard minimum and saved model input contracts."""

import unittest
import tempfile
from pathlib import Path

from unito_amazon.card_creation.chunking import build_document_chunks, markdown_paragraphs
from unito_amazon.card_creation.layout import strip_layout_noise
from unito_amazon.card_creation.passage_text import format_chunk_text
from unito_amazon.card_creation.schema import Source
from unito_amazon.card_creation.config import CardCreationConfig
from unito_amazon.card_creation.io import write_jsonl
from unito_amazon.card_creation.retrieval import load_chunks


class ChunkLayoutTests(unittest.TestCase):
    def setUp(self):
        self.source = Source(source_id="doc", markdown_file="doc.md", title="Document")
        self.count = lambda text: len(text.split())

    def test_repeated_footer_joins_a_paragraph_before_applying_size(self):
        markdown = (
            "# Technology-Facilitated\n# Gender-Based Violence\n\nAn Overview\n\n"
            "# Threats\n\nWomen receive threats on a\n\n"
            "Technology-Facilitated Gender-Based Violence: An Overview 15\n\n"
            "daily basis across multiple years.\n\n"
            "Another paragraph before a second\n\n"
            "Technology-Facilitated Gender-Based Violence: An Overview 17\n\npage ends here."
        )
        audit = []
        paragraphs = markdown_paragraphs(markdown, clean_layout=True, audit=audit)
        threat_paragraphs = [item for item in paragraphs if item[1] == ("Threats",)]
        self.assertEqual(len(threat_paragraphs), 2)
        self.assertIn("on a\ndaily basis", threat_paragraphs[0][2])
        self.assertNotIn("An Overview 15", threat_paragraphs[0][2])
        self.assertEqual(len([row for row in audit if row["rule_id"] == "repeated_running_title"]), 2)
        self.assertTrue(all(row["joined_body_fragments"] for row in audit))

    def test_page_first_and_explicit_grevio_markers_are_removed(self):
        markdown = (
            "# Technology-Facilitated Gender-Based Violence\n\n"
            "# Findings\n\nFirst part\n\n12 | Technology-Facilitated Gender-Based Violence\n\ncontinues.\n\n"
            "Another part\n\n14 | Technology-Facilitated Gender-Based Violence\n\ncontinues.\n\n"
            "Last part\n\nPage 18 ► GREVIO General Recommendation\n\ncontinues.\n\n► Page 19"
        )
        cleaned, audit = strip_layout_noise(markdown)
        self.assertNotIn("12 |", cleaned)
        self.assertNotIn("14 |", cleaned)
        self.assertNotIn("Page 18", cleaned)
        self.assertNotIn("Page 19", cleaned)
        self.assertIn("First part\ncontinues.", cleaned)
        self.assertTrue(all(row["line_number"] >= 1 and row["line_text"] for row in audit))

    def test_running_markdown_titles_do_not_replace_the_real_section(self):
        markdown = (
            "# YOUTH GUIDE TO END ONLINE GENDER-BASED VIOLENCE\n\nCover\n\n"
            "# Threats\n\nFirst part\n\n13\n\n# YOUTH GUIDE TO END ONLINE GENDER-BASED VIOLENCE\n\n"
            "continues.\n\nAnother part\n\n15\n\nYOUTH GUIDE TO END ONLINE GENDER-BASED VIOLENCE\n\ncontinues.\n\n"
            "# Recommendations\n\nActual new section."
        )
        paragraphs = markdown_paragraphs(markdown, clean_layout=True)
        self.assertIn(("Threats",), [item[1] for item in paragraphs])
        self.assertIn(("Recommendations",), [item[1] for item in paragraphs])
        self.assertEqual(sum(item[1] == ("Threats",) for item in paragraphs), 2)

    def test_icons_are_removed_even_between_substantive_lines(self):
        markdown = "# Findings\n\nFirst line\nicon: microphone\nsecond line\n\nicon: cross mark"
        cleaned, audit = strip_layout_noise(markdown)
        self.assertNotIn("icon:", cleaned)
        self.assertIn("First line\nsecond line", cleaned)
        self.assertEqual(len(audit), 2)

    def test_unconfirmed_numbered_text_and_real_headings_survive(self):
        markdown = "# Findings\n\nWomen affected 20\n\nWomen affected 30\n\n# Recommendations\n\nReal text."
        cleaned, audit = strip_layout_noise(markdown)
        self.assertEqual(cleaned, markdown)
        self.assertEqual(audit, [])

    def test_unmergeable_short_section_is_audited_not_indexed(self):
        paragraphs = [(1, (), "Gender-based violence"), (2, ("Findings",), "word " * 205)]
        audit = []
        chunks = build_document_chunks(paragraphs, self.source, self.count, minimum=200, audit=audit)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(audit[0]["rule_id"], "below_minimum_tokens")
        self.assertEqual(audit[0]["paragraph_text"], "Gender-based violence")
        self.assertEqual(audit[0]["paragraph_ids"], [1])
        self.assertGreaterEqual(chunks[0].body_token_count, 200)
        self.assertGreaterEqual(chunks[0].token_count, 200)

    def test_titles_cannot_pad_a_tiny_body_to_the_minimum(self):
        audit = []
        chunks = build_document_chunks(
            [(1, ("long heading " * 150,), "three body words")], self.source,
            self.count, minimum=200, audit=audit,
        )
        self.assertEqual(chunks, [])
        self.assertEqual(audit[0]["body_token_count"], 3)

    def test_same_section_merges_across_removed_ids_and_absorbs_short_tail(self):
        paragraphs = [(1, ("Findings",), "word " * 120), (3, ("Findings",), "word " * 100),
                      (5, ("Findings",), "tail " * 20)]
        chunks = build_document_chunks(paragraphs, self.source, self.count, minimum=200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].paragraph_ids, [1, 3, 5])
        self.assertEqual(chunks[0].body_token_count, 240)
        self.assertEqual(chunks[0].text, "Section: Findings\n\n" + chunks[0].body_text)
        self.assertEqual(format_chunk_text(chunks[0]), chunks[0].text)
        self.assertEqual(chunks[0].token_count, self.count(chunks[0].text))

    def test_mismatched_title_configuration_requires_rechunking(self):
        chunk = build_document_chunks([(1, ("Results",), "word " * 200)], self.source, self.count)[0]
        with self.assertRaisesRegex(ValueError, "rebuild the chunk stage"):
            format_chunk_text(chunk, False)

    def test_legacy_or_short_library_cannot_bypass_the_new_chunk_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CardCreationConfig(output_dir=Path(directory))
            chunk = build_document_chunks([(1, ("Results",), "word " * 200)], self.source, self.count)[0]
            write_jsonl(config.chunks_path, [chunk.model_dump(mode="json")])
            self.assertEqual(load_chunks(config)[0].text, chunk.text)
            legacy = chunk.model_dump(mode="json", exclude={"body_text", "body_token_count"})
            legacy["text"], legacy["token_count"] = "Gender-based violence", 3
            write_jsonl(config.chunks_path, [legacy])
            with self.assertRaisesRegex(ValueError, "Legacy or undersized"):
                load_chunks(config)


if __name__ == "__main__":
    unittest.main()
