"""Experiment logic with fake retrievers, decoder and judge (no model weights)."""

import dataclasses
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from unito_amazon.counterspeech_experiment import generation, judging
from unito_amazon.counterspeech_experiment.config import STRATEGIES, ExperimentConfig
from unito_amazon.counterspeech_experiment.retrieval import ExperimentRetrievers
from unito_amazon.evaluation.schema import PairwiseJudgeResult
from unito_amazon.llm.schema import BatchLLMResponse
from unito_amazon.retrieval.schema import Document, RetrievalResult


def fake_factory(llm):
    """Stand-in for unito_amazon.llm.factory, which needs transformers."""
    module = types.ModuleType("unito_amazon.llm.factory")
    module.LLMFactory = types.SimpleNamespace(create=lambda config: llm)
    return mock.patch.dict(sys.modules, {"unito_amazon.llm.factory": module})


def card(index: int, code: str) -> Document:
    return Document(id=f"card{index}", text=f"evidence {index}", metadata={
        "edos_label_code": code, "edos_label": f"{code} label {code}", "edos_definition": f"definition {code}"})


def result(document: Document, rank: int) -> RetrievalResult:
    return RetrievalResult(document=document, score=1 - rank / 100, rank=rank, retriever_name="fake")


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k):
        return [result(document, rank) for rank, document in enumerate(self.documents[:top_k], start=1)]


class RetrievalTests(unittest.TestCase):
    def test_label_filter_keeps_the_card_ranking_within_the_label(self):
        retrievers = ExperimentRetrievers.__new__(ExperimentRetrievers)
        retrievers.config = ExperimentConfig()
        retrievers.cards = [card(i, "2.1" if i % 2 else "3.2") for i in range(20)]
        retrievers.card_retriever = FakeRetriever(retrievers.cards)
        retrievers.chunk_retriever = FakeRetriever([Document(id=f"c{i}", text="chunk") for i in range(9)])
        retrieved = retrievers.retrieve("message", "2.1 descriptive attacks")
        self.assertEqual([r.document.id for r in retrieved["card_rag"]], [f"card{i}" for i in range(5)])
        self.assertEqual([r.document.id for r in retrieved["card_rag_label"]], ["card1", "card3", "card5", "card7", "card9"])
        self.assertEqual([r.rank for r in retrieved["card_rag_label"]], [1, 2, 3, 4, 5])
        self.assertEqual(len(retrieved["rag"]), 5)
        with self.assertRaises(ValueError):
            retrievers.retrieve("message", "1.1 threats of harm")


class FakeLLM:
    tokenizer = None

    def __init__(self):
        self.calls = 0

    def generate_batch(self, prompts, config):
        self.calls += len(prompts)
        return BatchLLMResponse(texts=[json.dumps({"counter_narrative": f"reply {i}"}) for i in range(len(prompts))],
                                model_name="fake", provider="fake")

    def destroy(self):
        pass


class GenerationAndJudgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = dataclasses.replace(ExperimentConfig(), output_dir=Path(self.directory.name),
                                          generation_batch_size=3)
        items = [{"id": f"m{i}", "source": "edos" if i < 2 else "multiconan", "split": None, "target": "WOMEN",
                  "hateful_message": f"message {i}", "gold_edos_label": None, "reference_counter_speech": None,
                  "predicted_edos_label": "2.1 descriptive attacks", "classifier": {}} for i in range(4)]
        cards = [card(i, "2.1") for i in range(6)]
        retrieved = {"rag": [result(Document(id=f"c{i}", text=f"chunk {i}"), i) for i in range(1, 6)],
                     "card_rag": [result(c, i) for i, c in enumerate(cards[:5], start=1)],
                     "card_rag_label": [result(c, i) for i, c in enumerate(cards[1:6], start=1)]}
        fake = mock.MagicMock()
        fake.retrieve.return_value = retrieved
        fake.definitions = {"2.1": ("descriptive attacks", "definition 2.1")}
        fake.metadata.return_value = {}
        self.patches = [
            mock.patch.object(generation, "attach_predicted_labels", return_value=items),
            mock.patch.object(generation, "load_or_create_subset", return_value=items),
            mock.patch.object(generation, "ExperimentRetrievers", return_value=fake),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.directory.cleanup()

    def test_generation_resumes_and_judge_covers_all_pairs_in_both_orders(self):
        llm = FakeLLM()
        with fake_factory(llm):
            generation.run_generation(self.config)
            generation.run_generation(self.config)  # nothing left to generate
        self.assertEqual(llm.calls, 16)
        records = [json.loads(line) for line in self.config.generations_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 16)
        by_strategy = {r["generation_strategy"]: r for r in records if r["id"] == "m0"}
        self.assertNotIn("<evidence>", by_strategy["zero_shot"]["prompt"])
        self.assertIn("Evidence 5:", by_strategy["rag"]["prompt"])
        self.assertNotIn("<category>", by_strategy["card_rag"]["prompt"])
        self.assertIn("Category of gender-based violence: descriptive attacks", by_strategy["card_rag_label"]["prompt"])
        self.assertTrue(all(r["response_format_valid"] for r in records))

        comparisons = judging.build_comparisons(records, STRATEGIES)
        self.assertEqual(len(comparisons), 4 * 6 * 2)

        def fake_judgments(batch):
            # Always prefers the first-shown response, except card_rag_label which always wins.
            out = []
            for record in batch:
                if "card_rag_label" in (record["response_1_strategy"], record["response_2_strategy"]):
                    first = record["response_1_strategy"] == "card_rag_label"
                    out.append(PairwiseJudgeResult(winner="response_1" if first else "response_2",
                                                   response_1_score=8 if first else 5,
                                                   response_2_score=5 if first else 8, raw_response="x"))
                else:
                    out.append(PairwiseJudgeResult(winner="response_1", response_1_score=7,
                                                   response_2_score=6, raw_response="x"))
            return out

        judge = mock.MagicMock()
        judge.judge_batch.side_effect = fake_judgments
        with fake_factory(FakeLLM()), \
                mock.patch.object(judging, "LLMJudge", return_value=judge):
            judging.run_judge(self.config)
        summary = json.loads((self.config.output_dir / "judge_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["valid_judgments"], 48)
        position_bias = summary["pairs"]["all|zero_shot|rag"]
        self.assertEqual((position_bias["a_wins"], position_bias["ties"], position_bias["b_wins"]), (0, 4, 0))
        self.assertEqual(position_bias["order_agreement"], 0)
        label = summary["pairs"]["all|card_rag|card_rag_label"]
        self.assertEqual((label["a_wins"], label["b_wins"], label["a_score"]), (0, 4, 0))
        self.assertEqual(label["mean_judgelm_score_difference_a_minus_b"], -3)
        self.assertEqual(summary["pairs"]["edos|card_rag|card_rag_label"]["messages"], 2)


if __name__ == "__main__":
    unittest.main()
