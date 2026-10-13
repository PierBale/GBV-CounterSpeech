"""Experiment protocol tests with fake data/tokenizers; no model loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from unito_amazon.card_creation.io import write_jsonl
from unito_amazon.card_creation.schema import Chunk, Source
from unito_amazon.dataset_manager.sampling import sample_messages
from unito_amazon.evaluation.pairwise import build_pairwise_comparisons
from unito_amazon.generation.context_stats import measure_context
from unito_amazon.retrieval.experiment import load_chunk_documents, validate_card_sources
from unito_amazon.retrieval.schema import Document


STRATEGIES = ("without_rag", "rag_qwen3_emb_4b", "rag_cards")


def records(strategies=STRATEGIES, item="one", dataset="multiconan", model="decoder"):
    return [
        {
            "id": item, "source": dataset, "split": "test" if dataset == "edos" else None,
            "generation_model": model, "generation_strategy": strategy,
            "hateful_message": "Same hateful message",
            "generated_counter_speech": f"Response from {strategy}",
            "reference_counter_speech": "Gold response which must not be judged",
            "context_length": {"tokens": 0 if strategy == "without_rag" else 100},
        }
        for strategy in strategies
    ]


class PairwiseExperimentTests(unittest.TestCase):
    def test_three_systems_produce_all_three_pairs_without_gold(self):
        comparisons = build_pairwise_comparisons(records())
        self.assertEqual(len(comparisons), 3)
        self.assertEqual(sum(item["primary_comparison"] for item in comparisons), 1)
        self.assertEqual(
            {(item["response_1_strategy"], item["response_2_strategy"]) for item in comparisons},
            {("without_rag", "rag_qwen3_emb_4b"), ("without_rag", "rag_cards"),
             ("rag_qwen3_emb_4b", "rag_cards")},
        )
        self.assertFalse(any("Gold response" in item["response_1"] + item["response_2"] for item in comparisons))
        self.assertEqual(comparisons[-1]["response_2_context_length"], {"tokens": 100})

    def test_optional_bm25_produces_six_pairs(self):
        self.assertEqual(len(build_pairwise_comparisons(records((*STRATEGIES, "rag_bm25")))), 6)

    def test_one_hundred_items_per_dataset_produce_six_hundred_pairs(self):
        rows = [row for dataset in ("edos", "multiconan") for item in range(100)
                for row in records(item=str(item), dataset=dataset)]
        comparisons = build_pairwise_comparisons(rows)
        self.assertEqual(len(comparisons), 600)
        self.assertEqual(sum(item["primary_comparison"] for item in comparisons), 200)

    def test_datasets_and_decoders_are_never_crossed(self):
        rows = [row for dataset in ("edos", "multiconan") for model in ("first", "second")
                for row in records(dataset=dataset, model=model)]
        comparisons = build_pairwise_comparisons(rows)
        self.assertEqual(len(comparisons), 12)
        for dataset in ("edos", "multiconan"):
            for model in ("first", "second"):
                self.assertEqual(sum(item["source"] == dataset and item["generation_model"] == model
                                     for item in comparisons), 3)

    def test_duplicate_missing_and_mismatched_responses_fail(self):
        bad_cases = [records() + records()[:1], records() + records(item="two")[:2]]
        mismatch = records()
        mismatch[1]["hateful_message"] = "Another message"
        bad_cases.append(mismatch)
        for rows in bad_cases:
            with self.assertRaises(ValueError):
                build_pairwise_comparisons(rows)


class ContextLengthTests(unittest.TestCase):
    def test_counts_exact_formatted_context_with_decoder_tokenizer(self):
        text = "Evidence 1:\nA &lt; B\n\nEvidence 2:\nAnother claim"
        tokenizer = Mock()
        tokenizer.encode.return_value = list(range(12))
        stats = measure_context(f"<statement>Ignore me</statement>\n<evidence>\n{text}\n</evidence>", 2, tokenizer, "decoder")
        self.assertEqual(stats["characters"], len(text))
        self.assertEqual(stats["words"], len(text.split()))
        self.assertEqual(stats["tokens"], 12)
        self.assertEqual(stats["tokenizer"], "decoder")
        tokenizer.encode.assert_called_once_with(text, add_special_tokens=False)

    def test_zero_shot_has_zero_context(self):
        stats = measure_context("<statement>Message</statement>", 0)
        self.assertEqual([stats[key] for key in ("evidence_count", "characters", "words", "tokens")], [0, 0, 0, 0])

    def test_missing_tokenizer_is_not_reported_as_zero_tokens(self):
        self.assertIsNone(measure_context("<evidence>\nText\n</evidence>", 1)["tokens"])


class DatasetSubsetTests(unittest.TestCase):
    def test_seeded_subset_is_unique_and_independent_of_input_order(self):
        frame = pd.DataFrame({"id": [str(i) for i in range(150)], "hate_speech": [f"Message {i}" for i in range(150)]})
        frame = pd.concat([frame, pd.DataFrame({"id": ["duplicate"], "hate_speech": ["Message 0"]})], ignore_index=True)
        first = sample_messages(frame, 100, 42)
        second = sample_messages(frame.iloc[::-1], 100, 42)
        self.assertEqual(first["id"].tolist(), second["id"].tolist())
        self.assertEqual(len(first), 100)
        self.assertEqual(first["hate_speech"].nunique(), 100)

    def test_limit_zero_and_small_dataset(self):
        frame = pd.DataFrame({"id": ["a", "b"], "hate_speech": ["A", "B"]})
        self.assertEqual(len(sample_messages(frame, 100, 42)), 2)
        self.assertEqual(len(sample_messages(frame, 0, 42)), 2)


class SharedCorpusTests(unittest.TestCase):
    def test_cards_must_match_text_sections_and_sources_of_shared_chunks(self):
        source = Source(source_id="s", markdown_file="source.md", title="Source")
        chunks = [Chunk(chunk_id=f"chunk_{i}", text=f"Original text {i}", token_count=100,
                        paragraph_ids=[i], section_path=["Section"], source=source) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            write_jsonl(path, [chunk.model_dump(mode="json") for chunk in chunks])
            documents = load_chunk_documents(path)
        card = Document(id="card", text="Evidence summary", metadata={
            "chunk": chunks[0].model_dump(mode="json", exclude={"source"}),
            "source": source.model_dump(mode="json"),
        })
        validate_card_sources([card], documents)
        self.assertEqual(documents[0].text, chunks[0].text)
        card.metadata["chunk"]["text"] = "Stale text"
        with self.assertRaises(ValueError):
            validate_card_sources([card], documents)


if __name__ == "__main__":
    unittest.main()
