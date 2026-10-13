"""LaTeX export checks with synthetic summaries; no models or metrics computed."""

import tempfile
import unittest
from pathlib import Path

from stats import build_report, write_report
from unito_amazon.evaluation.latex import automatic_latex_tables, latex_escape, pairwise_latex_tables


def summary(dataset="edos", baseline="without_rag", system="rag_cards", wins=5, losses=2, ties=1, total=10):
    return {
        "source": dataset, "generation_model": "Llama-3.1-8B", "judge_model": "JudgeLM-7B",
        "baseline_strategy": baseline, "challenger_strategy": system,
        "total_comparisons": total, "valid_comparisons": wins + losses + ties,
        "challenger_wins": wins, "baseline_wins": losses, "ties": ties,
    }


class LatexTableTests(unittest.TestCase):
    def test_percentages_use_valid_comparisons_and_all_outcomes_are_present(self):
        row = summary()
        text = pairwise_latex_tables([row])
        self.assertIn(r"\begin{table*}[t]", text)
        self.assertIn(r"\begin{tabular}{llrrrr}", text)
        self.assertIn(r"8 / 10 & 5 (62.50\%) & 2 (25.00\%) & 1 (12.50\%)", text)
        self.assertIn("Card-RAG (Qwen3-Emb-4B)", text)
        self.assertNotIn("MultiCONAN Reference", text)
        self.assertNotIn("BM25", text)
        self.assertNotIn("All datasets", text)

    def test_bold_is_best_percentage_within_same_baseline_and_dataset(self):
        rows = [summary(wins=5), summary(system="rag_qwen3_emb_4b", wins=6),
                summary(baseline="rag_qwen3_emb_4b", wins=7)]
        text = pairwise_latex_tables(rows)
        self.assertIn(r"\textbf{6 (66.67\%)}", text)
        self.assertNotIn(r"\textbf{5 (62.50\%)}", text)
        self.assertNotIn(r"\textbf{7 (70.00\%)}", text)

    def test_no_valid_judgments_do_not_produce_invented_percentages(self):
        row = summary(wins=0, losses=0, ties=0, total=4)
        text = pairwise_latex_tables([row])
        self.assertIn("0 / 4 & 0 (n/a) & 0 (n/a) & 0 (n/a)", text)

    def test_single_table_with_dataset_sections_without_pooled_results(self):
        rows = [summary(), summary(dataset="multiconan")]
        text = pairwise_latex_tables(rows)
        self.assertIn(r"\textit{EDOS}", text)
        self.assertIn(r"\textit{MultiCONAN}", text)
        self.assertNotIn("All datasets", text)
        self.assertNotIn("16 / 20", text)
        self.assertEqual(text.count(r"\begin{table*}"), 1)
        labels = [line for line in text.splitlines() if line.startswith(r"\label")]
        self.assertEqual(len(labels), 1)
        self.assertLess(text.index(r"\textit{EDOS}"), text.index(r"\textit{MultiCONAN}"))

    def test_bold_is_selected_independently_in_each_dataset_section(self):
        rows = [summary(wins=5), summary(system="rag_qwen3_emb_4b", wins=6),
                summary(dataset="multiconan", wins=7),
                summary(dataset="multiconan", system="rag_qwen3_emb_4b", wins=4)]
        text = pairwise_latex_tables(rows)
        edos, multiconan = text.split(r"\textit{MultiCONAN}", 1)
        self.assertIn(r"\textbf{6 (66.67\%)}", edos)
        self.assertNotIn(r"\textbf{5 (62.50\%)}", edos)
        self.assertIn(r"\textbf{7 (70.00\%)}", multiconan)
        self.assertNotIn(r"\textbf{4 (57.14\%)}", multiconan)

    def test_judges_are_separated_and_special_characters_escaped(self):
        first, second = summary(), summary()
        second["judge_model"] = "Judge & evaluator_2"
        second["generation_model"] = "Model_2%"
        text = pairwise_latex_tables([first, second])
        self.assertEqual(text.count(r"\begin{table*}"), 2)
        self.assertIn(r"Judge \& evaluator\_2", text)
        self.assertIn(r"Model\_2\%", text)
        self.assertEqual(latex_escape("{A} #1"), r"\{A\} \#1")

    def test_stats_export_writes_tex_automatically_without_metric_inputs(self):
        records = [
            {"id": str(i), "source": "edos", "generation_model": "Llama-3.1-8B",
             "judge_model": "JudgeLM-7B", "baseline_strategy": "rag_qwen3_emb_4b",
             "challenger_strategy": "rag_cards", "pairwise_judge": {"winner": winner}}
            for i, winner in enumerate(("response_2", "response_1", "tie"))
        ]
        report = build_report(records, [], [], [])
        with tempfile.TemporaryDirectory() as directory:
            write_report(report, Path(directory))
            text = (Path(directory) / "judgelm_pairwise.tex").read_text(encoding="utf-8")
        self.assertIn("3 / 3", text)
        self.assertIn(r"1 (33.33\%)", text)
        self.assertIn("Chunk-RAG (Qwen3-Emb-4B)", text)
        self.assertIn("Card-RAG (Qwen3-Emb-4B)", text)

    def test_empty_judgments_do_not_create_fake_rows(self):
        text = pairwise_latex_tables([])
        self.assertNotIn(r"\begin{table*}", text)

    def test_stats_never_pool_datasets_and_ignore_edos_metrics(self):
        judges = [
            {**summary(dataset=dataset), "pairwise_judge": {"winner": winner}}
            for dataset, winner in (("edos", "response_1"), ("multiconan", "response_2"))
        ]
        metrics = [
            {"id": str(i), "source": dataset, "generation_model": "Model_1",
             "generation_strategy": "rag_cards", "generated_counter_speech": "Response",
             "reference_counter_speech": reference, "automatic_metrics": {"bleu_4": value}}
            for i, (dataset, reference, value) in enumerate([
                ("edos", "Not a gold reference", 1.0),
                ("multiconan", "Gold", 0.2), ("MultiCONAN", "Gold", 0.4),
                ("multiconan", "", 0.9),
            ])
        ]
        report = build_report(judges, metrics, [], [])
        self.assertNotIn("overall", report)
        self.assertEqual(report["excluded_metric_records"], 2)
        edos, conan = (report["by_dataset"][key] for key in ("edos", "multiconan"))
        self.assertEqual(edos["overall"]["pairwise_judge"]["baseline_wins"], 1)
        self.assertEqual(conan["overall"]["pairwise_judge"]["challenger_wins"], 1)
        self.assertEqual(edos["automatic_metrics"]["by_system"], [])
        systems = conan["automatic_metrics"]["by_system"]
        self.assertEqual(systems[0]["metric_statistics"]["bleu_4"]["mean"], 0.3)
        text = automatic_latex_tables(systems)
        self.assertIn(r"\begin{tabular}{lrrrrr}", text)
        self.assertNotIn("Distinct", text)
        self.assertNotIn("Repetition", text)
        self.assertIn(r"0.3000 \pm 0.1414", text)
        self.assertIn("(2)", text)
        self.assertIn(r"Model\_1", text)
        self.assertNotIn("EDOS", text)
        self.assertIn("n/a (0)", text)
        with tempfile.TemporaryDirectory() as directory:
            write_report(report, Path(directory))
            self.assertEqual((Path(directory) / "automatic_metrics.tex").read_text(encoding="utf-8"), text)
            csv_text = (Path(directory) / "judgelm_pairwise.csv").read_text(encoding="utf-8-sig")
            self.assertTrue(csv_text.startswith("source,"))
            self.assertIn("edos,", csv_text)
            self.assertIn("multiconan,", csv_text)

    def test_automatic_table_empty_and_single_sample(self):
        self.assertNotIn(r"\begin{table*}", automatic_latex_tables([]))
        metrics = [{"source": "multiconan", "id": "one", "generation_model": "model",
                    "generation_strategy": "without_rag", "generated_counter_speech": "Response",
                    "reference_counter_speech": "Gold", "automatic_metrics": {"bleu_4": 0.0}}]
        report = build_report([], metrics, [], [])
        rows = report["by_dataset"]["multiconan"]["automatic_metrics"]["by_system"]
        self.assertIn(r"0.0000 \pm \mathrm{n/a}$ (1)", automatic_latex_tables(rows))
        self.assertNotIn(r"\begin{table*}", automatic_latex_tables([{**rows[0], "source": "edos"}]))


if __name__ == "__main__":
    unittest.main()
