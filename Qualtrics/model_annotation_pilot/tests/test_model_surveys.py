"""Validate sampling, draft preservation, source coverage, and QSF exports."""

import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_surveys", HERE / "build_model_surveys.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def visible_text(markup):
    parser = TextParser()
    parser.feed(markup)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


class ModelSurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surveys, cls.maps, cls.manifest = builder.build_package()
        cls.template = json.loads(builder.TEMPLATE.read_text(encoding="utf-8"))
        cls.source = {}
        for path in builder.INPUT_FILES:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["retrieval"]["reranker_rank"] <= 6:
                    cls.source[row["card_id"], row["generator_model"]] = row

    def test_exact_top_six_coverage_keeps_both_models(self):
        actual = {}
        for rows in self.maps.values():
            for item in rows:
                key = item["card_id"], item["generator_model"]
                self.assertNotIn(key, actual)
                actual[key] = item
                source = self.source[key]
                self.assertEqual(item["reranker_rank"], source["retrieval"]["reranker_rank"])
                self.assertEqual(item["chunk_id"], source["chunk"]["chunk_id"])
                self.assertIn(item["card_id"], item["question_id"])
        self.assertEqual(set(actual), set(self.source))
        self.assertEqual(len(actual), 132)
        self.assertEqual(len({item["question_id"] for item in actual.values()}), 132)
        by_model_label = Counter((item["generator_model"], item["primary_edos_label"]) for item in actual.values())
        self.assertEqual(len(by_model_label), 22)
        self.assertEqual(set(by_model_label.values()), {6})

    def test_largest_remainders_and_private_balance_across_seeds(self):
        items, _ = builder.load_pool()
        labels = {item["row"]["primary_edos_label"] for item in items}
        for seed in (builder.SEED, 0, 1, -12, 42, 123456789):
            pools = builder.split_pool(items, seed)
            self.assertEqual({name: len(pool) for name, pool in pools.items()},
                             {"comune": 45, "marem": 29, "arianna": 29, "marta": 29})
            counts = {name: Counter(item["row"]["primary_edos_label"] for item in pool)
                      for name, pool in pools.items()}
            # All label populations are 12: quota 45/11 gives ten 4s and one 5.
            self.assertEqual(sorted(counts["comune"].values()), [4] * 10 + [5])
            common_models = Counter(item["row"]["generator_model"] for item in pools["comune"])
            self.assertEqual(sorted(common_models.values()), [22, 23])
            ids = [item["question_id"] for pool in pools.values() for item in pool]
            self.assertEqual(len(ids), len(set(ids)))
            for label in labels:
                by_model = Counter(item["row"]["generator_model"] for item in pools["comune"]
                                   if item["row"]["primary_edos_label"] == label)
                expected = [2, 3] if counts["comune"][label] == 5 else [2, 2]
                self.assertEqual(sorted(by_model.values()), expected)
                private = [counts[name][label] for name in builder.ANNOTATORS]
                self.assertLessEqual(max(private) - min(private), 1)
                self.assertEqual(sum(private) + counts["comune"][label], 12)
        self.assertEqual(self.manifest["items_per_annotator"], 74)
        self.assertEqual(self.manifest["rating_answers_per_annotator"], 148)

    def test_extra_common_card_can_come_from_either_model(self):
        items, _ = builder.load_pool()
        extra_models = set()
        for seed in range(32):
            common = builder.split_pool(items, seed)["comune"]
            counts = Counter((item["row"]["primary_edos_label"], item["row"]["generator_model"])
                             for item in common)
            extra = [model for (_, model), count in counts.items() if count == 3]
            self.assertEqual(len(extra), 1)
            extra_models.update(extra)
        self.assertEqual(extra_models, {item["row"]["generator_model"] for item in items})
        for label, counts in self.manifest["counts_by_label_and_model"].items():
            expected = Counter(row["generator_model"] for row in self.maps["comune"]
                               if row["primary_edos_label"] == label)
            self.assertEqual(counts["comune"], dict(expected))

    def test_draft_questions_blocks_and_linear_flow(self):
        originals = {e["PrimaryAttribute"]: e["Payload"] for e in self.template["SurveyElements"] if e["Element"] == "SQ"}
        mutable = {"QuestionID", "DataExportTag", "QuestionDescription", "QuestionText"}
        for name, survey in self.surveys.items():
            elements = survey["SurveyElements"]
            questions = {e["PrimaryAttribute"]: e["Payload"] for e in elements if e["Element"] == "SQ"}
            blocks = next(e["Payload"] for e in elements if e["Element"] == "BL")
            flow_payload = next(e["Payload"] for e in elements if e["Element"] == "FL")
            flow = flow_payload["Flow"]
            self.assertEqual(len(blocks), len(self.maps[name]) + 1)
            self.assertEqual([node["ID"] for node in flow], [block["ID"] for block in blocks.values()])
            self.assertEqual(len({node["FlowID"] for node in flow}), len(flow))
            self.assertTrue(all(node["Type"] == "Standard" and "Flow" not in node for node in flow))
            self.assertEqual(flow_payload["Properties"]["Count"], len(flow) + 1)
            self.assertTrue(all(block["Options"]["RandomizeQuestions"] == "false" for block in blocks.values()))
            refs = [entry["QuestionID"] for block in blocks.values() for entry in block["BlockElements"]]
            self.assertEqual(len(refs), len(set(refs)))
            self.assertEqual(set(refs), set(questions))
            self.assertEqual(len({q["DataExportTag"] for q in questions.values()}), len(questions))
            self.assertEqual(questions[refs[0]]["QuestionText"], originals["QID7"]["QuestionText"])
            selectors = [q for q in questions.values() if q["DataExportTag"] == "Annotator"]
            self.assertEqual(len(selectors), int(name == "comune"))
            if selectors:
                self.assertEqual([v["Display"] for v in selectors[0]["Choices"].values()], list(builder.ANNOTATORS))
            for item, block in zip(self.maps[name], list(blocks.values())[1:]):
                expected_refs = [item[key] for key in ("instruction_question_id", "faithfulness_question_id",
                                                       "answer_relevance_question_id", "notes_question_id")]
                self.assertEqual([entry["QuestionID"] for entry in block["BlockElements"]], expected_refs)
                for qid, original_id in zip(expected_refs, ("QID21", "QID29", "QID31", "QID4")):
                    question = questions[qid]
                    self.assertEqual(question["QuestionID"], qid)
                    self.assertEqual({k: v for k, v in question.items() if k not in mutable},
                                     {k: v for k, v in originals[original_id].items() if k not in mutable})
                    if original_id in ("QID21", "QID4"):
                        self.assertEqual(question["QuestionText"], originals[original_id]["QuestionText"])
                for dimension in ("faithfulness", "answer_relevance"):
                    self.assertEqual(questions[item[f"{dimension}_question_id"]]["DataExportTag"],
                                     item[f"{dimension}_export_tag"])
                    self.assertIn(item["question_id"], item[f"{dimension}_export_tag"])
                self.assertEqual(questions[item["notes_question_id"]]["DataExportTag"], item["notes_export_tag"])
            count = next(e["SecondaryAttribute"] for e in elements if e["Element"] == "QC")
            self.assertEqual(int(count), len(questions))

    def test_source_text_and_exact_rating_prompts(self):
        for name, survey in self.surveys.items():
            questions = {e["PrimaryAttribute"]: e["Payload"] for e in survey["SurveyElements"] if e["Element"] == "SQ"}
            for item in self.maps[name]:
                source = self.source[item["card_id"], item["generator_model"]]
                faithful = questions[item["faithfulness_question_id"]]["QuestionText"]
                relevant = questions[item["answer_relevance_question_id"]]["QuestionText"]
                expected_paragraph = re.sub(r"\ASection:[ \t]*", "", source["chunk"]["text"], count=1)
                self.assertIn(visible_text(expected_paragraph), visible_text(faithful))
                self.assertNotIn("Section:", faithful)
                for text in (faithful, relevant):
                    self.assertIn(re.sub(r"\s+", " ", source["evidence"]), visible_text(text))
                    self.assertNotIn(source["generator_model"], text)
                self.assertIn(source["primary_edos_label"], visible_text(relevant))
                self.assertIn(source["edos_definition"], visible_text(relevant))
                self.assertTrue(faithful.endswith("<strong>How relevant is the argument to the paragraph?</strong>"))
                self.assertTrue(relevant.endswith("<strong>How relevant is the argument to the description of the EDOS label?</strong>"))
                for table in re.findall(r"<table\b.*?</table>", expected_paragraph, flags=re.S):
                    self.assertIn(table, faithful)
        synthetic = {"chunk": {"text": "Section: Title\n\nA <5% change & <script>literal</script>."}}
        rendered = builder.render_paragraph(synthetic)
        self.assertIn("&lt;5%", rendered)
        self.assertNotIn("<script>", rendered)

    def test_seeded_order_stability_without_global_random_state(self):
        before = random.getstate()
        self.assertEqual(builder.build_package(), (self.surveys, self.maps, self.manifest))
        self.assertEqual(before, random.getstate())
        items, _ = builder.load_pool()
        pools = builder.split_pool(items)
        self.assertEqual(pools, builder.split_pool(list(reversed(items))))
        self.assertNotEqual(pools, builder.split_pool(items, builder.SEED + 1))
        for name, pool in pools.items():
            keys = [builder.label_key(item["row"]["primary_edos_label"]) for item in pool]
            self.assertEqual(keys, sorted(keys))
            # Independently reproduce the precise presentation stream required.
            rng = random.Random(f"{builder.SEED}:presentation:{name}")
            expected = []
            for label in sorted({item["row"]["primary_edos_label"] for item in items}, key=builder.label_key):
                ids = sorted(item["question_id"] for item in pool if item["row"]["primary_edos_label"] == label)
                rng.shuffle(ids)
                expected.extend(ids)
            self.assertEqual([item["question_id"] for item in pool], expected)

    def test_artifact_bytes_and_sha256_match(self):
        outputs = builder.serialize_package(self.surveys, self.maps, self.manifest)
        for name, content in outputs.items():
            self.assertEqual((HERE / name).read_bytes(), content.encode("utf-8"))
        manifest = json.loads(outputs["annotation_assignment.json"])
        for name, digest in manifest["output_files_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest)

    def test_cli_reproducibility_across_process_hash_seeds(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, hash_seed in ((first, "1"), (second, "321")):
                subprocess.run([sys.executable, str(HERE / "build_model_surveys.py"), "--seed", "20261114",
                                "--output-dir", directory], check=True, capture_output=True,
                               env={**os.environ, "PYTHONHASHSEED": hash_seed})
            for path in Path(first).iterdir():
                self.assertEqual(path.read_bytes(), (Path(second) / path.name).read_bytes())
                self.assertEqual(path.read_bytes(), (HERE / path.name).read_bytes())

    def test_reject_missing_duplicate_and_noninteger_ranks(self):
        rows = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                for path in builder.INPUT_FILES]
        for invalid_rank in (2, 0, "1", True, 16):
            changed = copy.deepcopy(rows)
            changed[0][0]["retrieval"]["reranker_rank"] = invalid_rank
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / f"model_{i}.jsonl" for i in range(2)]
                for path, contents in zip(paths, changed):
                    path.write_text("\n".join(json.dumps(row) for row in contents), encoding="utf-8")
                with self.assertRaises(ValueError):
                    builder.load_pool(paths)


if __name__ == "__main__":
    unittest.main()
