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
from unito_amazon.counterspeech_experiment.retrieval import ExperimentRetrievers, label_chunk_pools
from unito_amazon.evaluation.prompts import CounterSpeechPairwisePrompt
from unito_amazon.evaluation.schema import PairwiseJudgeResult
from unito_amazon.generation.schema import parse_counter_narrative
from unito_amazon.llm.schema import BatchLLMResponse
from unito_amazon.retrieval.schema import Document, RetrievalResult


def fake_factory(llm):
    """Stand-in for unito_amazon.llm.factory, which needs transformers."""
    module = types.ModuleType("unito_amazon.llm.factory")
    module.LLMFactory = types.SimpleNamespace(create=lambda config: llm)
    return mock.patch.dict(sys.modules, {"unito_amazon.llm.factory": module})


def card(index: int, code: str) -> Document:
    return Document(id=f"card{index}", text=f"evidence {index}", metadata={
        "edos_label_code": code, "edos_label": f"{code} label {code}", "edos_definition": f"definition {code}",
        "chunk_id": f"c{index}"})


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
        retrievers.config = ExperimentConfig()  # top_k 4, hybrid 2 cards + 2 chunks
        retrievers.cards = [card(i, "2.1" if i % 2 else "3.2") for i in range(20)]
        retrievers.card_retriever = FakeRetriever(retrievers.cards)
        retrievers.chunks = [Document(id=f"c{i}", text="chunk") for i in range(25)]
        retrievers.chunk_retriever = FakeRetriever(retrievers.chunks)
        retrievers.label_chunks = label_chunk_pools(retrievers.cards)
        retrieved = retrievers.retrieve("message", "2.1 descriptive attacks", "3.2 stereotypes")
        ids = {name: [r.document.id for r in results] for name, results in retrieved.items()}
        self.assertEqual(ids["card_rag"], [f"card{i}" for i in range(4)])
        self.assertEqual(ids["card_rag_label"], ["card1", "card3", "card5", "card7"])
        self.assertEqual([r.rank for r in retrieved["card_rag_label"]], [1, 2, 3, 4])
        self.assertEqual(ids["card_rag_gold_label"], ["card0", "card2", "card4", "card6"])
        self.assertEqual(ids["rag"], [f"c{i}" for i in range(4)])
        # RAG + label: the chunks the cards of the label come from, in the chunk ranking.
        self.assertEqual(ids["rag_label"], ["c1", "c3", "c5", "c7"])
        self.assertEqual([r.rank for r in retrieved["rag_label"]], [1, 2, 3, 4])
        self.assertEqual(ids["rag_gold_label"], ["c0", "c2", "c4", "c6"])
        # Hybrid: top 2 cards and top 2 chunks, alternated card first, within the label's pools.
        self.assertEqual(ids["hybrid_rag"], ["card0", "c0", "card1", "c1"])
        self.assertEqual([r.rank for r in retrieved["hybrid_rag"]], [1, 2, 3, 4])
        self.assertEqual(ids["hybrid_rag_label"], ["card1", "c1", "card3", "c3"])
        self.assertEqual(ids["hybrid_rag_gold_label"], ["card0", "c0", "card2", "c2"])
        without_gold = retrievers.retrieve("message", "2.1 descriptive attacks")
        for name in ("card_rag_gold_label", "rag_gold_label", "hybrid_rag_gold_label"):
            self.assertNotIn(name, without_gold)
        with self.assertRaises(ValueError):
            retrievers.retrieve("message", "1.1 threats of harm")


class ParsingAndPromptTests(unittest.TestCase):
    def test_counter_speech_key_and_legacy_key(self):
        self.assertEqual(parse_counter_narrative('{"counter_speech": " Hi. "}'), ("Hi.", True))
        self.assertEqual(parse_counter_narrative('```json\n{"counter_narrative": "Old."}\n```'), ("Old.", True))
        self.assertFalse(parse_counter_narrative('{"answer": "x"}')[1])

    def test_judge_criteria(self):
        social = CounterSpeechPairwisePrompt("social_media").build("hs", "r1", "r2")
        self.assertIn("Friendly and colloquial language is preferred.", social)
        self.assertIn("based on the criteria above", social)
        self.assertNotIn("Number of facts", social)
        self.assertIn("Number of facts", CounterSpeechPairwisePrompt().build("hs", "r1", "r2"))
        with self.assertRaises(ValueError):
            CounterSpeechPairwisePrompt("other")


class FakeLLM:
    tokenizer = None

    def __init__(self):
        self.calls = 0

    def generate_batch(self, prompts, config):
        self.calls += len(prompts)
        return BatchLLMResponse(texts=[json.dumps({"counter_speech": f"reply {i}"}) for i in range(len(prompts))],
                                model_name="fake", provider="fake")

    def destroy(self):
        pass


class GenerationAndJudgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        # counterspeech_v3 settings: short prompts, hybrid systems included.
        self.config = dataclasses.replace(ExperimentConfig(), output_dir=Path(self.directory.name),
                                          generation_batch_size=3, prompt_style="short", strategies=STRATEGIES,
                                          judge_all_pairs=False)
        items = [{"id": f"m{i}", "source": "edos" if i < 2 else "multiconan", "split": None, "target": "WOMEN",
                  "hateful_message": f"message {i}", "reference_counter_speech": None,
                  "gold_edos_label": "3.2 immutable gender differences" if i < 2 else None,
                  "predicted_edos_label": "2.1 descriptive attacks", "classifier": {}} for i in range(4)]
        cards = [card(i, "2.1") for i in range(6)] + [card(i, "3.2") for i in range(6, 11)]

        def chunks(first):
            return [result(Document(id=f"c{i}", text=f"chunk {i}"), i - first + 1) for i in range(first, first + 4)]

        retrieved = {"rag": chunks(1), "rag_label": chunks(6), "rag_gold_label": chunks(11),
                     "card_rag": [result(c, i) for i, c in enumerate(cards[:4], start=1)],
                     "card_rag_label": [result(c, i) for i, c in enumerate(cards[1:5], start=1)],
                     "card_rag_gold_label": [result(c, i) for i, c in enumerate(cards[6:10], start=1)]}
        for name, label_cards in (("hybrid_rag", cards[:2]), ("hybrid_rag_label", cards[1:3]),
                                  ("hybrid_rag_gold_label", cards[6:8])):
            mixed = [doc for pair in zip(label_cards, [Document(id=f"h{name}{i}", text=f"hchunk {i}") for i in range(2)])
                     for doc in pair]
            retrieved[name] = [result(doc, i) for i, doc in enumerate(mixed, start=1)]
        fake = mock.MagicMock()
        fake.retrieve.return_value = retrieved
        fake.definitions = {"2.1": ("descriptive attacks", "definition 2.1"),
                            "3.2": ("immutable gender differences", "definition 3.2")}
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

    def test_v4_guidelines_prompts_without_hybrid(self):
        config = dataclasses.replace(ExperimentConfig(), output_dir=Path(self.directory.name) / "v4",
                                     generation_batch_size=3)  # defaults: counterspeech_v4
        with fake_factory(FakeLLM()):
            generation.run_generation(config)
        records = [json.loads(line) for line in config.generations_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 2 * 5 + 2 * 5)
        self.assertFalse(any(r["generation_strategy"].startswith("hybrid_") or "gold" in r["generation_strategy"]
                             for r in records))
        by_strategy = {r["generation_strategy"]: r for r in records if r["id"] == "m0"}

        self.assertEqual(len({r["system_prompt"] for r in by_strategy.values()}), 1)
        system = by_strategy["rag"]["system_prompt"]
        self.assertTrue(system.startswith("Given the hateful claim provided in HS, generate a two-sentence "
                                          "counter-speech in English following the guidelines"))
        self.assertIn("divisive labels (e.g., sexist, misogynist)", system)
        self.assertIn('"counter_speech"', system)
        self.assertEqual(by_strategy["zero_shot"]["prompt"],
                         "HS: message 0\nBased on the provided HS, generate a two-sentence counter-speech in "
                         "English.\n### Counter-speech:")
        rag = by_strategy["rag"]["prompt"]
        self.assertTrue(rag.startswith("EVIDENCE:\n[1] chunk 1\n\n[2] chunk 2"))
        self.assertIn("[4] chunk 4\nHS: message 0\nBased on the provided HS and EVIDENCE", rag)
        self.assertNotIn("[5]", rag)
        self.assertIn("The EVIDENCE consists of paragraphs that address gender-based violence. You must use", rag)
        self.assertIn("The EVIDENCE consists of summaries that address gender-based violence.",
                      by_strategy["card_rag"]["prompt"])
        card_label = by_strategy["card_rag_label"]["prompt"]
        self.assertTrue(card_label.startswith("EVIDENCE:\nCategory of gender-based violence: descriptive attacks. "
                                              "Definition: definition 2.1\n\n[1] evidence 1"))
        self.assertEqual(card_label.count("Category of gender-based violence:"), 1)
        self.assertIn("a category of gender-based violence and summaries that address that specific category",
                      card_label)
        self.assertIn("a category of gender-based violence and paragraphs that address that specific category",
                      by_strategy["rag_label"]["prompt"])
        self.assertTrue(rag.endswith("Ground your response exclusively in the EVIDENCE.\n### Counter-speech:"))
        for record in by_strategy.values():
            self.assertTrue(record["prompt"].endswith("\n### Counter-speech:"))
            self.assertEqual(record["generated_counter_speech"][:5], "reply")
        # Every pair of the 5 systems (10), both orders, 4 messages.
        self.assertEqual(len(judging.build_comparisons(records, config.strategies, config.judge_all_pairs)), 4 * 10 * 2)
        judge = mock.MagicMock()
        judge.judge_batch.side_effect = lambda batch: [
            PairwiseJudgeResult(winner="response_1", response_1_score=7, response_2_score=6, raw_response="x")
            for _ in batch]
        with fake_factory(FakeLLM()), mock.patch.object(judging, "LLMJudge", return_value=judge):
            judging.run_judge(config)
        summary = json.loads((config.output_dir / "judge_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["valid_judgments"], 80)
        self.assertTrue(summary["judge_all_pairs"])
        for source in ("edos", "multiconan"):
            self.assertEqual(sum(key.startswith(source + "|") for key in summary["pairs"]), 10)
        for pair in ("zero_shot|rag_label", "rag|rag_label", "rag_label|card_rag"):
            self.assertEqual(summary["pairs"][f"all|{pair}"]["messages"], 4)

    def test_generation_resumes_and_judge_covers_all_pairs_in_both_orders(self):
        llm = FakeLLM()
        with fake_factory(llm):
            generation.run_generation(self.config)
            generation.run_generation(self.config)  # nothing left to generate
        self.assertEqual(llm.calls, 2 * 10 + 2 * 7)  # gold-label strategies only for EDOS
        records = [json.loads(line) for line in self.config.generations_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 34)
        self.assertEqual({r["generation_strategy"] for r in records if r["source"] == "multiconan"},
                         set(STRATEGIES) - {"rag_gold_label", "card_rag_gold_label", "hybrid_rag_gold_label"})
        self.assertTrue(all(r["response_format_valid"] for r in records))
        self.assertEqual(records[0]["generated_counter_speech"], "reply 0")
        by_strategy = {r["generation_strategy"]: r for r in records if r["id"] == "m0"}

        # Short prompts (Damo et al. 2025): the same JSON-format system prompt for every system.
        self.assertEqual(len({r["system_prompt"] for r in by_strategy.values()}), 1)
        self.assertIn('"counter_speech"', by_strategy["rag"]["system_prompt"])
        zero_shot = by_strategy["zero_shot"]["prompt"]
        self.assertTrue(zero_shot.startswith("Hateful message: message 0\nGenerate a counter-speech"))
        self.assertTrue(zero_shot.endswith("Counter-speech:"))
        self.assertNotIn("Evidence", zero_shot)
        for name in ("rag", "card_rag", "hybrid_rag"):
            prompt = by_strategy[name]["prompt"]
            self.assertTrue(prompt.startswith("Here are four evidence summaries you MUST use"))
            self.assertIn("Evidence 4:", prompt)
            self.assertNotIn("Evidence 5:", prompt)
            self.assertNotIn("category of gender-based violence", prompt)
            self.assertTrue(prompt.endswith("Hateful message: message 0\nCounter-speech:"))
        rag_label = by_strategy["rag_label"]
        self.assertIn("category of gender-based violence: descriptive attacks (definition: definition 2.1)",
                      rag_label["prompt"])
        self.assertEqual([e["id"] for e in rag_label["evidence"]], [f"c{i}" for i in range(6, 10)])
        rag_gold = by_strategy["rag_gold_label"]
        self.assertIn("category of gender-based violence: immutable gender differences", rag_gold["prompt"])
        self.assertEqual(rag_gold["edos_label_used_kind"], "gold")
        self.assertEqual([e["id"] for e in rag_gold["evidence"]], [f"c{i}" for i in range(11, 15)])
        gold = by_strategy["card_rag_gold_label"]
        self.assertIn("category of gender-based violence: immutable gender differences", gold["prompt"])
        self.assertEqual((gold["edos_label_used_kind"], by_strategy["card_rag_label"]["edos_label_used_kind"]),
                         ("gold", "predicted"))
        self.assertEqual([e["id"] for e in gold["evidence"]], [f"card{i}" for i in range(6, 10)])
        hybrid_gold = by_strategy["hybrid_rag_gold_label"]
        self.assertIn("category of gender-based violence: immutable gender differences", hybrid_gold["prompt"])
        self.assertEqual([e["id"] for e in by_strategy["hybrid_rag_label"]["evidence"]][::2], ["card1", "card2"])

        comparisons = judging.build_comparisons(records, STRATEGIES)
        self.assertEqual(len(comparisons), 2 * 19 * 2 + 2 * 12 * 2)  # judged pairs only

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
                mock.patch.object(judging, "LLMJudge", return_value=judge) as judge_class:
            judging.run_judge(self.config)
        self.assertEqual(judge_class.call_args.kwargs["prompt"].criteria, "social_media")
        judgments = [json.loads(line) for line in self.config.judgments_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({j["judge_criteria"] for j in judgments}, {"social_media"})
        summary = json.loads((self.config.output_dir / "judge_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["judge_criteria"], "social_media")
        self.assertEqual(summary["valid_judgments"], 124)
        self.assertNotIn("multiconan|card_rag|card_rag_gold_label", summary["pairs"])
        self.assertEqual(summary["pairs"]["edos|card_rag_label|card_rag_gold_label"]["messages"], 2)
        position_bias = summary["pairs"]["all|zero_shot|rag"]
        self.assertEqual((position_bias["a_wins"], position_bias["ties"], position_bias["b_wins"]), (0, 4, 0))
        self.assertEqual(position_bias["order_agreement"], 0)
        label = summary["pairs"]["all|card_rag|card_rag_label"]
        self.assertEqual((label["a_wins"], label["b_wins"], label["a_score"]), (0, 4, 0))
        self.assertEqual(label["mean_judgelm_score_difference_a_minus_b"], -3)
        self.assertEqual(summary["pairs"]["edos|card_rag|card_rag_label"]["messages"], 2)
        # The label strategies are compared only with card-RAG / hybrid using the same label.
        self.assertEqual(summary["pairs"]["all|rag_label|card_rag_label"]["messages"], 4)
        self.assertEqual(summary["pairs"]["edos|rag_gold_label|card_rag_gold_label"]["messages"], 2)
        self.assertNotIn("multiconan|rag_gold_label|card_rag_gold_label", summary["pairs"])
        self.assertFalse(any("|rag|rag_label" in key or "zero_shot|rag_label" in key for key in summary["pairs"]))
        self.assertEqual(summary["pairs"]["multiconan|card_rag_label|hybrid_rag_label"]["messages"], 2)
        self.assertNotIn("multiconan|card_rag_gold_label|hybrid_rag_gold_label", summary["pairs"])

        # Judgments made with other criteria are never mixed in the same folder.
        with self.assertRaises(ValueError), fake_factory(FakeLLM()):
            judging.run_judge(dataclasses.replace(self.config, judge_criteria="facts"))


class SubsetExtensionTests(unittest.TestCase):
    def test_a_larger_sample_with_the_same_seed_extends_the_saved_one(self):
        from unito_amazon.counterspeech_experiment.data import extends

        def subset(size, ids):
            return {"sample_size_per_dataset": size, "seed": 42, "edos_split": "test",
                    "items": [{"id": i, "source": i[0]} for i in ids]}

        small = subset(1, ["e1", "m1"])
        self.assertTrue(extends(small, subset(2, ["e1", "e2", "m1", "m2"])))
        self.assertFalse(extends(small, subset(2, ["e2", "e1", "m1", "m2"])))
        self.assertFalse(extends(small, {**subset(2, ["e1", "e2", "m1", "m2"]), "seed": 7}))


class MetricsTests(unittest.TestCase):
    def test_multiconan_responses_are_scored_against_all_references_of_the_message(self):
        from unito_amazon.counterspeech_experiment import metrics

        with tempfile.TemporaryDirectory() as directory:
            config = dataclasses.replace(ExperimentConfig(), output_dir=Path(directory), strategies=STRATEGIES)
            records = [{"id": f"m{i}", "source": source, "generation_strategy": strategy, "generation_model": "g",
                        "hateful_message": f" message {i} ", "generated_counter_speech": f"reply {i} {strategy}"}
                       for i, source in ((0, "multiconan"), (1, "multiconan"), (2, "edos"))
                       for strategy in STRATEGIES if source == "edos" or "gold" not in strategy]
            config.generations_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            references = {"message 0": ["ref a", "ref b"], "message 1": ["ref c"]}
            calls = []

            class FakeMetric:
                def __init__(self, name):
                    self.name = name

                def compute(self, predictions, references, **kwargs):
                    calls.append((self.name, predictions, references, kwargs))
                    return {"rouge": {"rougeL": [0.1, 0.3]}, "bleu": {"bleu": 0.05},
                            "bertscore": {"f1": [0.5, 0.7]}}[self.name]

            fake_metrics = {name: FakeMetric(name) for name in ("rouge", "bleu", "bertscore")}
            with mock.patch.object(metrics, "multiconan_references", return_value=references), \
                    mock.patch.object(metrics, "load_metrics", return_value=fake_metrics):
                metrics.run_metrics(config)
            self.assertEqual(len(calls), 3 * 7)  # one call per metric and system, reference-based on MultiCONAN only
            name, predictions, refs, kwargs = calls[0]
            self.assertEqual((name, predictions), ("rouge", ["reply 0 zero_shot", "reply 1 zero_shot"]))
            self.assertIn(refs[0], ["ref a", "ref b"])  # one reference, drawn at random
            self.assertEqual(refs[1], "ref c")
            self.assertTrue(all(call[2] == refs for call in calls))  # same references for every system
            self.assertTrue(kwargs["use_stemmer"])
            self.assertEqual(calls[1][3]["model_type"], "bert-base-multilingual-cased")
            summary = json.loads((config.output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
            multiconan, edos = summary["datasets"]["multiconan"], summary["datasets"]["edos"]
            self.assertEqual(list(multiconan["strategies"]), [s for s in STRATEGIES if "gold" not in s])
            self.assertEqual(multiconan["strategies"]["zero_shot"],
                             {"rougeL": 0.2, "bleu": 0.05, "bertscore": 0.6, "gen_len": 3.0, "n": 2})
            self.assertEqual((multiconan["messages_with_several_references"],
                              multiconan["mean_available_reference_count"]), (1, 1.5))
            self.assertEqual(list(edos["strategies"]), list(STRATEGIES))
            self.assertEqual(edos["strategies"]["card_rag_gold_label"],
                             {"rougeL": None, "bleu": None, "bertscore": None, "gen_len": 3.0, "n": 1})

            import counterspeech_experiment_tables as tables
            table = tables.metrics_table(summary)
            self.assertIn(r"Card-RAG + EDOS label (gold) & -- & -- & -- & 3.000 \\", table)
            self.assertIn(r"\textbf{0.200} & \textbf{0.050} & \textbf{0.600} & 3.000 \\", table)


if __name__ == "__main__":
    unittest.main()
