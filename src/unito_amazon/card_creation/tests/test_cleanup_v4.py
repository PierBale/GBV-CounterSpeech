"""Rules introduced with editorial version 4.0, on patterns seen in the corpus."""

import unittest

from unito_amazon.card_creation.chunking import aggregate_paragraphs, markdown_paragraphs
from unito_amazon.card_creation.editorial import clean_body_markup, filter_editorial_paragraphs
from unito_amazon.card_creation.layout import strip_layout_noise


WORDS = lambda text: len(text.split())


class LayoutV4Tests(unittest.TestCase):
    def test_triangle_page_markers_are_removed_and_body_rejoined(self):
        markdown = (
            "# Context\n\nSexist behaviour is on a continuum and\n\n"
            "Page **10 ▶ Preventing and combating sexism**\n\n---\n\n"
            "includes hate speech.\n\n**Recommendation CM/Rec(2019)1** ▶ Page **11**\n"
        )
        cleaned, audit = strip_layout_noise(markdown)
        self.assertNotIn("▶", cleaned)
        self.assertIn("continuum and\nincludes hate speech.", cleaned)
        self.assertEqual({row["rule_id"] for row in audit}, {"explicit_page_marker"})

    def test_generated_visual_lines_removed_but_source_captions_kept(self):
        markdown = ("Text before.\n\nphoto: woman speaking into a megaphone\n\n"
                    "Image: Infographic included in the training.\n")
        cleaned, audit = strip_layout_noise(markdown)
        self.assertNotIn("megaphone", cleaned)
        self.assertIn("Image: Infographic", cleaned)
        self.assertEqual(audit[0]["rule_id"], "generated_visual_description")

    def test_page_footnotes_move_after_the_rejoined_paragraph(self):
        markdown = (
            "## Workplace\n\nWomen may be excluded from\n\n<hr/>\n\n"
            "9. Council (2016), “Kit”, available at https://example.org.\n"
            "10. Article 26.2 requires parties to promote awareness of workplace harassment.\n\n"
            "**Recommendation CM/Rec(2019)1** ▶ Page **21**\n\n---\n\n"
            "important opportunities.\n\nNext paragraph.\n"
        )
        cleaned, audit = strip_layout_noise(markdown)
        self.assertIn("excluded from\nimportant opportunities.", cleaned)
        self.assertLess(cleaned.index("important opportunities."), cleaned.index("10. Article"))
        self.assertNotIn("<hr/>", cleaned)
        paragraphs = [text for _, _, text in markdown_paragraphs(cleaned)]
        self.assertIn("10. Article 26.2 requires parties to promote awareness of workplace harassment.", paragraphs)
        self.assertEqual(sum(row["rule_id"] == "page_footnote_relocated" for row in audit), 2)


class ParagraphV4Tests(unittest.TestCase):
    def test_caption_heading_does_not_become_the_section_of_following_text(self):
        markdown = (
            "## Causes\n\nIntro.\n\n### **Prevalence of harassment** (%, EU, 2014)\n\n"
            "<table><tr><td>21</td></tr></table>\n\nForms of sexist hate speech.\n\n"
            "# TABLE 3: SHELTERS (2024)\n\nCountry notes.\n"
        )
        paragraphs = markdown_paragraphs(markdown)
        self.assertEqual(paragraphs[1][1], ("Causes",))
        self.assertTrue(paragraphs[1][2].startswith("Prevalence of harassment (%, EU, 2014)\n<table>"))
        self.assertEqual(paragraphs[2], (3, ("Causes",), "Forms of sexist hate speech."))
        self.assertEqual(paragraphs[3][1], ("Causes",))

    def test_reference_list_sections_are_excluded_whole(self):
        paragraphs = [(1, ("REFERENCES",), "Supporting analysis"),
                      (2, ("<u>REFERENCES</u>",), "Association for Progressive Communications. Undated.")]
        retained, audit = filter_editorial_paragraphs(paragraphs, True)
        self.assertEqual(retained, [])
        self.assertEqual({row["rule_id"] for row in audit}, {"reference_list_section"})

    def test_list_of_tables_is_a_contents_section(self):
        text = "Table 2.2.1: Product of literature search 22\nTable 2.3.1: Excluded studies 24\nTable 2.4.1: Studies 24"
        retained, audit = filter_editorial_paragraphs([(1, ("List of Tables",), text)], True)
        self.assertEqual(retained, [])
        self.assertEqual(audit[0]["rule_id"], "contents_entries")

    def test_endnote_block_is_split_and_only_explanatory_notes_survive(self):
        block = ("25 Economist Intelligence Unit. “Measuring the Prevalence of Online Violence.”\n"
                 "27 Throughout the report, we provide data on TFGBV against LGBTQ+ individuals when available.\n"
                 "36 Henry and Powell. “Embodied Harms,” 770.")
        retained, audit = filter_editorial_paragraphs([(1, ("Endnotes",), block)], True)
        self.assertEqual(retained[0][2], "27 Throughout the report, we provide data on TFGBV against LGBTQ+ individuals when available.")
        self.assertEqual(audit[0]["action"], "trimmed")

    def test_body_line_after_an_inserted_note_is_kept(self):
        block = ("<sup>124</sup> See EPRS (2024) Cyberviolence against women in the EU, p. 3.\n"
                 "One of the distinct challenges in addressing TFVAWG is its connection with offline violence.")
        retained, audit = filter_editorial_paragraphs([(1, ("Context",), block)], True)
        self.assertEqual(retained[0][2], "One of the distinct challenges in addressing TFVAWG is its connection with offline violence.")
        self.assertEqual(audit[0]["action"], "trimmed")

    def test_markup_residue_is_removed_without_changing_words(self):
        text = ("~~85% of EU Member States~~ have a helpline (<sup>2</sup>), as women<sup>75</sup> "
                "report <u>here</u>.\n(<sup>3</sup>) Explanatory note.\nkm<sup>a</sup>")
        self.assertEqual(
            clean_body_markup(text),
            "85% of EU Member States have a helpline, as women report here.\n[3] Explanatory note.\nkm<sup>a</sup>",
        )


class ShortSectionMergeTests(unittest.TestCase):
    def test_short_key_findings_merge_with_following_child_section(self):
        paragraphs = [
            (1, ("Previous chapter",), "word " * 120),
            (2, ("KEY FINDINGS: HARASSMENT AT WORK",), "One in three women " * 10),
            (3, ("KEY FINDINGS: HARASSMENT AT WORK", "EXTENT"), "word " * 120),
        ]
        chunks = aggregate_paragraphs(paragraphs, WORDS, minimum=100)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1]["section_path"], ["KEY FINDINGS: HARASSMENT AT WORK"])
        self.assertEqual(chunks[1]["paragraph_ids"], [2, 3])
        self.assertIn("\n\nEXTENT\n\n", chunks[1]["text"])
        self.assertEqual(chunks[1]["evidence_size"], 160)

    def test_sections_without_a_shared_heading_are_not_merged(self):
        paragraphs = [(1, ("A",), "word " * 120), (2, ("B",), "short text")]
        chunks = aggregate_paragraphs(paragraphs, WORDS, minimum=100)
        self.assertEqual([chunk["paragraph_ids"] for chunk in chunks], [[1], [2]])


if __name__ == "__main__":
    unittest.main()
