"""Section-aware aggregation tests using a lightweight token counter."""

import unittest

from unito_amazon.card_creation.chunking import aggregate_paragraphs, markdown_paragraphs


class ChunkSectionTests(unittest.TestCase):
    def test_short_paragraphs_merge_only_within_same_hierarchy(self):
        paragraphs = markdown_paragraphs(
            "# Section\n\n## First subsection\n\nFirst paragraph.\n\nSecond paragraph.\n\n"
            "### Detail\n\nDetail paragraph.\n\n## Second subsection\n\nOther paragraph."
        )
        chunks = aggregate_paragraphs(paragraphs, lambda text: len(text.split()), minimum=100)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["section_path"], ["Section", "First subsection"])
        self.assertEqual(chunks[0]["paragraph_ids"], [1, 2])
        self.assertEqual(chunks[0]["text"], "First paragraph.\n\nSecond paragraph.")
        self.assertEqual(chunks[1]["section_path"], ["Section", "First subsection", "Detail"])
        self.assertEqual(chunks[2]["section_path"], ["Section", "Second subsection"])
        self.assertEqual(chunks[2]["paragraph_ids"], [4])


if __name__ == "__main__":
    unittest.main()
