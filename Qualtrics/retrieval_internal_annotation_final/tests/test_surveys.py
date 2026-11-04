"""Check real source coverage, questionnaire routing, notes, and offline ordering."""

import csv
import importlib.util
import io
import json
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_surveys", HERE / "build_surveys.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FinalSurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surveys, cls.maps, cls.manifest = builder.build_package()
        cls.template = json.loads((HERE / "Draft-retrieval-setting.qsf").read_text(encoding="utf-8"))

    def test_exact_top_four_coverage_and_no_duplicates(self):
        expected = {}
        for path in builder.INPUT_DIR.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["reranker_rank"] <= 4:
                    expected[path.stem, row["query_id"], row["chunk_id"]] = row
        actual = {}
        pairs, ids = [], []
        for rows in self.maps.values():
            for row in rows:
                pairs.append((row["query_id"], row["chunk_id"]))
                ids.append(row["question_id"])
                for source in json.loads(row["source_matches"]):
                    key = source["pipeline_id"], row["query_id"], row["chunk_id"]
                    self.assertNotIn(key, actual)
                    actual[key] = source["reranker_rank"]
                    self.assertEqual(source["reranker_score"], expected[key]["reranker_score"])
        self.assertEqual(actual, {key: row["reranker_rank"] for key, row in expected.items()})
        self.assertEqual(len(expected), 176)
        self.assertEqual(len(pairs), 111)
        self.assertEqual(len(set(pairs)), 111)
        self.assertEqual(len(set(ids)), 111)
        self.assertEqual(set(pairs), {(query, chunk) for _, query, chunk in expected})

    def test_assignment_and_label_balance(self):
        self.assertEqual({name: len(rows) for name, rows in self.maps.items()},
                         {"comune": 27, "marem": 28, "arianna": 28, "marta": 28})
        per_annotator = Counter()
        for name, rows in self.maps.items():
            for annotator in builder.ANNOTATORS if name == "comune" else [name]:
                per_annotator[annotator] += len(rows)
        self.assertEqual(dict(per_annotator), {name: 55 for name in builder.ANNOTATORS})
        self.assertEqual(sum(per_annotator.values()), 165)
        for counts in self.manifest["counts_by_label"].values():
            self.assertGreaterEqual(counts["comune"], 1)
            private = [counts[name] for name in builder.ANNOTATORS]
            self.assertLessEqual(max(private) - min(private), 1)

    def test_questionnaire_flow_and_notes_match_draft(self):
        originals = {e["PrimaryAttribute"]: e["Payload"] for e in self.template["SurveyElements"] if e["Element"] == "SQ"}
        for name, survey in self.surveys.items():
            elements = survey["SurveyElements"]
            questions = {e["PrimaryAttribute"]: e["Payload"] for e in elements if e["Element"] == "SQ"}
            blocks = next(e["Payload"] for e in elements if e["Element"] == "BL")
            flow = next(e["Payload"] for e in elements if e["Element"] == "FL")["Flow"]
            self.assertEqual(len(flow), len(self.maps[name]) + 1)
            self.assertTrue(all(node["Type"] == "Standard" and "Flow" not in node for node in flow))
            self.assertEqual([node["ID"] for node in flow], [block["ID"] for block in blocks.values()])
            references = [entry["QuestionID"] for block in blocks.values() for entry in block["BlockElements"]]
            self.assertEqual(len(references), len(set(references)))
            self.assertEqual(set(references), set(questions))
            tags = [question["DataExportTag"] for question in questions.values()]
            self.assertEqual(len(tags), len(set(tags)))
            self.assertEqual(questions[references[0]]["QuestionText"], originals["QID7"]["QuestionText"])
            for row, block in zip(self.maps[name], list(blocks.values())[1:]):
                self.assertEqual(block["Options"]["RandomizeQuestions"], "false")
                self.assertEqual([q["QuestionID"] for q in block["BlockElements"]],
                                 [row["qualtrics_question_id"], row["notes_question_id"]])
                rating = questions[row["qualtrics_question_id"]]
                notes = questions[row["notes_question_id"]]
                self.assertEqual(rating["DataExportTag"], row["question_id"])
                self.assertEqual(rating["Choices"], originals["QID29"]["Choices"])
                self.assertEqual(rating["Validation"], originals["QID29"]["Validation"])
                self.assertEqual(notes["DataExportTag"], row["notes_export_tag"])
                for field in ("QuestionText", "QuestionType", "Selector", "Validation"):
                    self.assertEqual(notes[field], originals["QID4"][field])
                self.assertEqual(notes["Validation"]["Settings"]["ForceResponse"], "OFF")
            count = next(e["SecondaryAttribute"] for e in elements if e["Element"] == "QC")
            self.assertEqual(int(count), len(questions))
            selectors = [q for q in questions.values() if q["DataExportTag"] == "Annotator"]
            self.assertEqual(len(selectors), int(name == "comune"))
            if selectors:
                self.assertEqual([choice["Display"] for choice in selectors[0]["Choices"].values()], list(builder.ANNOTATORS))

    def test_label_order_and_offline_shuffle_are_reproducible(self):
        repeated = builder.build_package()
        self.assertEqual((self.surveys, self.maps, self.manifest), repeated)
        _, changed, _ = builder.build_package(seed=builder.SEED + 1)
        self.assertNotEqual(self.maps, changed)
        for rows in self.maps.values():
            labels = [builder.label_key(row["edos_label"]) for row in rows]
            self.assertEqual(labels, sorted(labels))
            self.assertEqual([r["position"] for r in rows], list(range(1, len(rows) + 1)))

    def test_exported_files_match_builder_and_keep_all_source_text(self):
        source = {}
        for path in builder.INPUT_DIR.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                source[row["query_id"], row["chunk_id"]] = row
        for name, survey in self.surveys.items():
            exported = json.loads((HERE / f"{name}.qsf").read_text(encoding="utf-8"))
            self.assertEqual(exported, survey)
            with (HERE / f"{name}_question_map.csv").open(encoding="utf-8", newline="") as stream:
                exported_map = list(csv.DictReader(stream))
            self.assertEqual(exported_map, [{k: str(v) for k, v in row.items()} for row in self.maps[name]])
            questions = {e["PrimaryAttribute"]: e["Payload"] for e in exported["SurveyElements"] if e["Element"] == "SQ"}
            for row in self.maps[name]:
                original = source[row["query_id"], row["chunk_id"]]
                self.assertEqual(row["definition"], original["definition"])
                self.assertEqual(questions[row["qualtrics_question_id"]]["QuestionText"], builder.render_context(original))


if __name__ == "__main__":
    unittest.main()
