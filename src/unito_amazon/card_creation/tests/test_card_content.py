"""Card output contract and retrieval checks; no model loading."""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from unito_amazon.card_creation.card_retrieval import load_card_documents
from unito_amazon.card_creation.card_schema import GeneratedContent
from unito_amazon.card_creation.config import CardCreationConfig
from unito_amazon.card_creation.generation import normalize_card
from unito_amazon.card_creation.io import write_jsonl
from unito_amazon.card_creation.prompts import build_card_prompt
from unito_amazon.card_creation.schema import Chunk, RankedPassage, Source


class CardContentTests(unittest.TestCase):
    def setUp(self):
        self.config = CardCreationConfig()
        self.passage = RankedPassage(
            label="Target label", definition="Target definition",
            embedding_model=self.config.embedding_model, reranker_model=self.config.reranker_model,
            chunk=Chunk(
                chunk_id="chunk", text="Full original passage with details not in the summary.",
                token_count=100, paragraph_ids=[1], section_path=[],
                source=Source(source_id="source", markdown_file="source.md", title="Source"),
            ),
            dense_rank=1, dense_score=0.9, reranker_rank=1, reranker_score=0.9,
        )

    def test_only_reasoning_and_evidence_are_accepted(self):
        content = GeneratedContent.model_validate_json('{"reasoning":"Why relevant","evidence":"Summary"}')
        self.assertEqual(list(content.model_dump()), ["reasoning", "evidence"])
        for record in (
            {"reasoning": "Why", "evidence": " "},
            {"reasoning": "Why", "argument": "Old format"},
            {"reasoning": "Why", "evidence": "Summary", "source_quote": "Not requested"},
            {"card": None, "rejection_reason": "Old envelope"},
        ):
            with self.assertRaises(ValidationError):
                GeneratedContent.model_validate(record)

    def test_source_is_saved_by_code_but_not_indexed_as_card_content(self):
        card = normalize_card(
            GeneratedContent(reasoning="Why relevant", evidence="Qualitative summary"),
            self.passage, self.config, "key",
        )
        self.assertEqual(card.source_quote, self.passage.chunk.text)
        self.assertEqual(card.source_passage, self.passage.chunk.text)
        self.assertNotIn("quote_exact_match", card.validation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cards.jsonl"
            write_jsonl(path, [card.model_dump(mode="json")])
            document = load_card_documents(path)[0]
        self.assertIn("Qualitative summary", document.text)
        self.assertNotIn(self.passage.chunk.text, document.text)
        self.assertEqual(document.metadata["source_quote"], self.passage.chunk.text)

    def test_prompt_includes_full_source_label_and_definition(self):
        prompt = build_card_prompt(self.passage)
        self.assertTrue(prompt.endswith(self.passage.chunk.text))
        self.assertIn(self.passage.label, prompt)
        self.assertIn(self.passage.definition, prompt)
        self.assertIn('"evidence": "self-contained qualitative evidence summary"', prompt)
        self.assertNotIn('"source_quote":', prompt)
        self.assertNotIn("rejection_reason", prompt)


if __name__ == "__main__":
    unittest.main()
