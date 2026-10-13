"""Integration checks with mocked encoders, datasets and judge outputs."""

import unittest
from unittest.mock import Mock, patch

import pandas as pd

import main as generation_main
from judge import build_summary
from unito_amazon.card_creation.vector_store import CARD_INSTRUCTION
from unito_amazon.evaluation.pairwise import build_pairwise_comparisons
from unito_amazon.retrieval.schema import Document


class FakeFaissRetriever:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.encoder = None

    def load_or_index(self, documents, path, **kwargs):
        self.documents = documents
        self.index_path = path

    def _encoder(self):
        if self.encoder is None:
            self.encoder = object()
        return self.encoder


class ExperimentIntegrationTests(unittest.TestCase):
    def test_chunk_and_card_use_shared_encoder_instruction_and_three_results(self):
        chunks = [Document(id=f"chunk_{i}", text=f"Text {i}") for i in range(3)]
        cards = [Document(id=f"card_{i}", text=f"Evidence {i}") for i in range(3)]
        with (
            patch.object(generation_main, "RAG_RETRIEVERS", ("qwen3-emb-4b", "cards")),
            patch.object(generation_main, "RUN_WITHOUT_RAG", True),
            patch.object(generation_main, "load_chunk_documents", return_value=chunks),
            patch.object(generation_main, "file_sha256", return_value="shared-corpus"),
            patch.object(generation_main, "validate_card_sources") as validate,
            patch("unito_amazon.card_creation.card_retrieval.load_card_documents", return_value=cards),
            patch("unito_amazon.card_creation.vector_store.FaissEvidenceRetriever", FakeFaissRetriever),
        ):
            generators = generation_main.build_generators(Mock())
        self.assertEqual(set(generators), {"without_rag", "rag_qwen3_emb_4b", "rag_cards"})
        chunk_rag, card_rag = generators["rag_qwen3_emb_4b"], generators["rag_cards"]
        self.assertIs(chunk_rag.retriever.encoder, card_rag.retriever.encoder)
        self.assertEqual(chunk_rag.retriever.model_name, card_rag.retriever.model_name)
        self.assertEqual(chunk_rag.retriever.instruction, CARD_INSTRUCTION)
        self.assertEqual(card_rag.retriever.instruction, CARD_INSTRUCTION)
        self.assertNotEqual(chunk_rag.retriever.index_path, card_rag.retriever.index_path)
        validate.assert_called_once_with(cards, chunks)
        for generator in (chunk_rag, card_rag):
            self.assertEqual(generator.top_k, 3)
            self.assertFalse(generator.experiment_metadata["reranking"])
            self.assertEqual(generator.experiment_metadata["source_corpus_sha256"], "shared-corpus")

    def test_limit_is_applied_separately_after_edos_sexist_filter(self):
        edos = pd.DataFrame({
            "id": [f"edos-{i}" for i in range(180)], "source": "edos",
            "hate_speech": [f"EDOS message {i}" for i in range(180)],
            "label_sexist": ["sexist"] * 140 + ["not sexist"] * 40,
        })
        conan = pd.DataFrame({
            "id": [f"conan-{i}" for i in range(180)], "source": "multiconan",
            "hate_speech": [f"MultiCONAN message {i}" for i in range(180)],
        })
        with (
            patch.object(generation_main, "DATASET", "both"),
            patch.object(generation_main, "LIMIT", 100),
            patch.object(generation_main, "INCLUDE_NON_SEXIST", False),
            patch.object(generation_main.DatasetLoader, "load_dataset", side_effect=[edos, conan]),
        ):
            subset = generation_main.load_generation_data()
        self.assertEqual(subset.groupby("source").size().to_dict(), {"edos": 100, "multiconan": 100})
        self.assertTrue(subset.loc[subset["source"] == "edos", "label_sexist"].eq("sexist").all())

    def test_summary_separates_datasets_and_marks_controlled_pair(self):
        rows = []
        for dataset in ("edos", "multiconan"):
            for strategy in ("without_rag", "rag_qwen3_emb_4b", "rag_cards"):
                rows.append({"id": "message", "source": dataset, "generation_model": "decoder",
                             "generation_strategy": strategy, "hateful_message": "Message",
                             "generated_counter_speech": strategy})
        comparisons = build_pairwise_comparisons(rows)
        for item in comparisons:
            item["pairwise_judge"] = {"winner": "response_2"}
        summary = build_summary(comparisons)
        self.assertEqual(summary["valid_comparisons"], 6)
        by_dataset = summary["by_dataset_generation_model_and_pair"]
        self.assertEqual(len(by_dataset), 6)
        self.assertEqual(sum(item["primary_comparison"] for item in by_dataset.values()), 2)


if __name__ == "__main__":
    unittest.main()
