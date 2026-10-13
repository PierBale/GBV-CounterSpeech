"""LaTeX export checks with synthetic summaries; no models or metrics computed."""

import tempfile
import unittest
from pathlib import Path

from stats import build_report, write_report
from unito_amazon.evaluation.latex import latex_escape, pairwise_latex_tables


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
        text = pairwise_latex_tables([row], [row])
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
        text = pairwise_latex_tables(rows, rows)
        self.assertIn(r"\textbf{6 (66.67\%)}", text)
        self.assertNotIn(r"\textbf{5 (62.50\%)}", text)
        self.assertNotIn(r"\textbf{7 (70.00\%)}", text)

    def test_no_valid_judgments_do_not_produce_invented_percentages(self):
        row = summary(wins=0, losses=0, ties=0, total=4)
        text = pairwise_latex_tables([row], [row])
        self.assertIn("0 / 4 & 0 (n/a) & 0 (n/a) & 0 (n/a)", text)

    def test_dataset_sections_and_overall_are_not_mixed(self):
        rows = [summary(), summary(dataset="multiconan")]
        overall = summary(wins=10, losses=4, ties=2, total=20)
        text = pairwise_latex_tables(rows, [overall])
        self.assertIn(r"\textit{EDOS}", text)
        self.assertIn(r"\textit{MultiCONAN}", text)
        self.assertIn(r"\textit{All datasets}", text)
        self.assertIn("16 / 20", text)
        self.assertEqual(text.count(r"\begin{table*}"), 1)

    def test_judges_are_separated_and_special_characters_escaped(self):
        first, second = summary(), summary()
        second["judge_model"] = "Judge & evaluator_2"
        second["generation_model"] = "Model_2%"
        text = pairwise_latex_tables([first, second], [first, second])
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
        text = pairwise_latex_tables([], [])
        self.assertNotIn(r"\begin{table*}", text)


if __name__ == "__main__":
    unittest.main()
