"""Syntactic exclusions and exact model inputs. No model dependencies required."""

import unittest

from unito_amazon.card_creation.chunking import aggregate_paragraphs, markdown_paragraphs
from unito_amazon.card_creation.editorial import EDITORIAL_RULES_VERSION, filter_editorial_paragraphs
from unito_amazon.card_creation.passage_text import format_chunk_text
from unito_amazon.card_creation.prompts import build_card_prompt
from unito_amazon.card_creation.retrieval import chunks_to_documents
from unito_amazon.card_creation.schema import Chunk, RankedPassage, Source


class EditorialFilterTests(unittest.TestCase):
    def test_explicit_rules_record_excluded_text_and_original_paragraph_ids(self):
        paragraphs = markdown_paragraphs(
            "# Findings\n\nEvidence about violence.\n\nlogo: WHO\nlogo: UN Women\n\n"
            "Further evidence.\n\nISBN 978-92-4-011702-0 (electronic version)\n\n"
            "© World Health Organization 2025\n\n"
            "Some rights reserved. This work is available under a licence."
        )
        retained, audit = filter_editorial_paragraphs(paragraphs, enabled=True)
        self.assertEqual([item[0] for item in retained], [1, 3])
        self.assertEqual([row["paragraph_id"] for row in audit], [2, 4, 5, 6])
        self.assertEqual([row["rule_id"] for row in audit], [
            "logo_only", "isbn_only", "copyright_notice", "publisher_boilerplate",
        ])
        self.assertEqual(audit[0]["paragraph_text"], "logo: WHO\nlogo: UN Women")
        self.assertTrue(all(row["rules_version"] == EDITORIAL_RULES_VERSION for row in audit))
        # Removed editorial material no longer splits the same logical section.
        chunks = aggregate_paragraphs(retained, lambda text: len(text.split()), minimum=100)
        self.assertEqual([chunk["paragraph_ids"] for chunk in chunks], [[1, 3]])

    def test_contents_need_an_exact_heading_and_entry_syntax(self):
        paragraphs = [
            (1, ("<u>TABLE OF CONTENTS</u>",), "Introduction 5\nMethods .... 12"),
            (2, ("Contents",), "Acknowledgements / p.v\nMethods / p.vi\nIntroduction / p.1"),
            (3, ("Contents",), "This substantive paragraph has no page entries."),
            (4, ("Survey results",), "Women 5\nMen 12"),
            (5, ("Contents of abusive messages",), "Example 12"),
        ]
        retained, audit = filter_editorial_paragraphs(paragraphs, enabled=True)
        self.assertEqual([row["paragraph_id"] for row in audit], [1, 2])
        self.assertEqual([item[0] for item in retained], [3, 4, 5])

    def test_only_explicit_editorial_sections_are_removed(self):
        paragraphs = markdown_paragraphs(
            "# IMPRINT\n\nPublisher: Example.\n\n## Credits\n\nDesign: Example.\n\n"
            "# Findings\n\nEvidence.\n\n# DISCLAIMERANDCOPYRIGHT\n\nPublication notice.\n\n"
            "# Copyright and intimate image abuse\n\nSubstantive legal discussion."
        )
        retained, audit = filter_editorial_paragraphs(paragraphs, enabled=True)
        self.assertEqual([item[2] for item in retained], ["Evidence.", "Substantive legal discussion."])
        self.assertEqual(len(audit), 3)

    def test_mixed_content_references_tables_and_ambiguous_cover_text_survive(self):
        paragraphs = [
            (1, ("Results",), "Copyright law may affect responses to image-based abuse."),
            (2, ("Results",), "logo: WHO\nSubstantive findings about violence."),
            (3, ("Results",), "| Group | Finding |\n| Women | Exposure to abuse |"),
            (4, ("References",), "A paper about gender-based violence. 2025."),
            (5, ("Foreword",), "Context about gender-based violence."),
            (6, (), "Preventing violence against women, second edition"),
            (7, ("Copyright",), "A substantive discussion without a publishing notice."),
        ]
        retained, audit = filter_editorial_paragraphs(paragraphs, enabled=True)
        self.assertEqual(retained, paragraphs)
        self.assertEqual(audit, [])

    def test_disabled_filter_preserves_all_paragraphs(self):
        paragraphs = [(1, ("IMPRINT",), "logo: WHO")]
        self.assertEqual(filter_editorial_paragraphs(paragraphs, enabled=False), (paragraphs, []))


class PassageContextTests(unittest.TestCase):
    def setUp(self):
        self.chunk = Chunk(
            chunk_id="chunk", text="Exact body\n\nwith a second paragraph.", token_count=10,
            paragraph_ids=[2, 3], section_path=["**Results**", "<u>Digital violence</u>"],
            source=Source(source_id="doc", markdown_file="doc.md", title="Document"),
        )

    def test_model_text_includes_only_source_headings_and_preserves_body(self):
        before = self.chunk.model_dump()
        expected = "Section: Results > Digital violence\n\n" + self.chunk.text
        self.assertEqual(format_chunk_text(self.chunk), expected)
        self.assertEqual(chunks_to_documents([self.chunk])[0].text, expected)
        self.assertEqual(self.chunk.model_dump(), before)
        passage = RankedPassage(
            label="Label", definition="Definition", embedding_model="embedding", reranker_model="reranker",
            chunk=self.chunk, dense_rank=1, dense_score=0.8, reranker_rank=1, reranker_score=-2.0,
        )
        self.assertTrue(build_card_prompt(passage).endswith("Source passage:\n" + expected))
        self.assertTrue(build_card_prompt(passage, False).endswith("Source passage:\n" + self.chunk.text))

    def test_no_headings_or_disabled_option_leave_body_unchanged(self):
        self.assertEqual(format_chunk_text(self.chunk, False), self.chunk.text)
        self.assertEqual(chunks_to_documents([self.chunk], False)[0].text, self.chunk.text)
        self.assertEqual(format_chunk_text(self.chunk.model_copy(update={"section_path": []})), self.chunk.text)


if __name__ == "__main__":
    unittest.main()
