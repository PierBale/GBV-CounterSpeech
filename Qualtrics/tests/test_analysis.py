"""Regression tests for statistical conventions and survey attribution."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_retrieval import (HERE, ROOT, aggregate_votes, dcg, expand_selections, krippendorff_alpha,
                              load_survey, ndcg, rank_correlations)


class MetricsTests(unittest.TestCase):
    def test_alpha_matches_published_missing_data_example(self):
        # Krippendorff, Computing Alpha-Reliability, section C (pp. 4-5)
        # and ordinal result in section D (p. 8).
        missing = np.nan
        ratings = np.array([
            [1, 2, 3, 3, 2, 1, 4, 1, 2, missing, missing, missing],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, missing, 3],
            [missing, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, missing],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, missing],
        ]).T
        self.assertAlmostEqual(krippendorff_alpha(ratings, "nominal"), .743, delta=.0005)
        self.assertAlmostEqual(krippendorff_alpha(ratings, "ordinal"), .815, delta=.0005)
        # The last item has only one rating and must not affect the marginals.
        self.assertAlmostEqual(krippendorff_alpha(ratings), krippendorff_alpha(ratings[:-1]))

    def test_alpha_perfect_negative_and_constant(self):
        self.assertEqual(krippendorff_alpha([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), 1)
        self.assertAlmostEqual(krippendorff_alpha([[0, 1], [1, 0]]), -.5)
        self.assertTrue(np.isnan(krippendorff_alpha([[0, 0], [0, 0]])))

    def test_ordinal_alpha_for_two_annotators(self):
        # Marginals (2, 1, 3), ordinal squared distances (2.25, 12.25, 4).
        # Observed weighted coincidence sum = 8; expected sum = 180; n = 6.
        self.assertAlmostEqual(krippendorff_alpha([[0, 0], [1, 2], [2, 2]]), 7 / 9)

    def test_soft_labels_preserve_distribution_but_linear_scores_equal_mean(self):
        consensus = aggregate_votes(pd.DataFrame([[1, 1, 1], [0, 1, 2], [0, 2, 2], [0, np.nan, 2]]))
        np.testing.assert_allclose(consensus[['p_relevance_0', 'p_relevance_1', 'p_relevance_2']].sum(axis=1), 1)
        np.testing.assert_array_equal(consensus.soft_expected_relevance, consensus.mean_relevance)
        self.assertAlmostEqual(consensus.loc[2, 'mean_relevance'], 4 / 3)
        self.assertEqual(consensus.loc[2, 'median_relevance'], 2)
        self.assertAlmostEqual(consensus.loc[0, 'soft_gain_exponential_0_2'], 2 / 3)
        self.assertAlmostEqual(consensus.loc[1, 'soft_gain_exponential_0_2'], 8 / 9)
        self.assertAlmostEqual(consensus.loc[2, 'soft_gain_exponential_0_2'], 4 / 3)
        self.assertEqual(consensus.loc[3, 'n_ratings'], 2)
        self.assertTrue(consensus.drop(columns='n_ratings').ge(0).all().all())
        self.assertTrue(consensus.drop(columns='n_ratings').le(2).all().all())

    def test_soft_gain_changes_ndcg_without_changing_mean(self):
        consensus = aggregate_votes(pd.DataFrame([[1, 1, 1], [0, 1, 2], [2, 2, 2]]))
        mean_gain = consensus.mean_gain_exponential_0_2.tolist()
        soft_gain = consensus.soft_gain_exponential_0_2.tolist()
        self.assertNotAlmostEqual(ndcg(mean_gain, mean_gain), ndcg(soft_gain, soft_gain))
        np.testing.assert_allclose(consensus.soft_expected_relevance, consensus.mean_relevance)

    def test_soft_labels_reject_items_without_votes(self):
        with self.assertRaises(ValueError):
            aggregate_votes(pd.DataFrame([[np.nan, np.nan]]))

    def test_pooled_ndcg_does_not_reward_bad_selection(self):
        self.assertAlmostEqual(ndcg([1, 1, 1], [1, 1, 1]), 1)
        self.assertAlmostEqual(ndcg([1, 1, 1], [2, 2, 2, 1, 1, 1]), .5)
        self.assertTrue(np.isnan(ndcg([0, 0, 0], [0, 0, 0])))
        self.assertLess(dcg([0, 1, 2]), dcg([2, 1, 0]))
        self.assertAlmostEqual(ndcg([2, 1, 0], [2, 1, 0], exponential=True), 1)

    def test_correlations_use_score_direction_and_preserve_ties(self):
        self.assertAlmostEqual(rank_correlations([.9, .6, .2], [2, 1, 0])[0], 1)
        self.assertAlmostEqual(rank_correlations([.9, .6, .2], [0, 1, 2])[0], -1)
        self.assertAlmostEqual(rank_correlations([.9, .6, .2], [2, 2, 0])[0], np.sqrt(2 / 3))
        self.assertTrue(np.isnan(rank_correlations([.9, .6, .2], [1, 1, 1])[0]))
        self.assertTrue(np.isnan(rank_correlations([.9], [2])[0]))


class SurveyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping, cls.annotations, cls.matrix, cls.audit = load_survey(
            HERE / "Best-retrieval-results.csv", HERE / "Best_retrieval_config_question_map.csv",
            HERE / "Best_retrieval_config.qsf")
        cls.selections = expand_selections(cls.mapping, ROOT / "data/card_creation_comparison/runs/confronto-02")

    def test_survey_export_is_decoded_and_metadata_excluded(self):
        self.assertEqual(self.matrix.shape, (101, 3))
        self.assertEqual(len(self.annotations), 303)
        self.assertEqual(set(self.annotations.relevance), {0, 1, 2})
        self.assertEqual(int((self.annotations.relevance == 0).sum()), 112)
        self.assertEqual(int((self.annotations.relevance == 2).sum()), 122)
        self.assertFalse(self.matrix.isna().any().any())
        self.assertTrue(self.audit.included.all())

    def test_shared_items_restore_all_original_pipeline_ranks(self):
        self.assertEqual(len(self.selections), 132)
        self.assertEqual(self.selections.pipeline_id.nunique(), 4)
        self.assertTrue((self.selections.groupby("pipeline_id").size() == 33).all())
        for item in self.mapping.itertuples():
            import json
            actual = self.selections[self.selections.question_id == item.question_id]
            expected = {m["pipeline_id"]: m["reranker_rank"] for m in json.loads(item.source_matches)}
            self.assertEqual(dict(zip(actual.pipeline_id, actual["rank"])), expected)

    def test_question_map_matches_raw_top3_sample_independently(self):
        # Reconstruct membership directly from the extraction files, independently
        # of source_matches, so an omitted shared membership cannot pass unnoticed.
        import json
        from build_retrieval_survey import render_question

        raw = {}
        run_dir = ROOT / 'data/card_creation_comparison/runs/confronto-02/reranking'
        for path in run_dir.glob('*.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row['reranker_rank'] <= 3:
                    raw[path.stem, row['query_id'], row['chunk_id']] = row
        actual = {(r.pipeline_id, r.query_id, r.chunk_id): r.rank
                  for r in self.selections.itertuples()}
        self.assertEqual(actual, {key: row['reranker_rank'] for key, row in raw.items()})
        self.assertEqual(set(zip(self.mapping.query_id, self.mapping.chunk_id)),
                         {(query, chunk) for _, query, chunk in raw})
        survey = json.loads((HERE / 'Best_retrieval_config.qsf').read_text(encoding='utf-8'))
        questions = {e['PrimaryAttribute']: e['Payload'] for e in survey['SurveyElements']
                     if e['Element'] == 'SQ'}
        for item in self.mapping.itertuples():
            sources = [row for (_, query, chunk), row in raw.items()
                       if (query, chunk) == (item.query_id, item.chunk_id)]
            for source in sources:
                self.assertEqual(questions[item.qualtrics_question_id]['QuestionText'],
                                 render_question(source))


if __name__ == "__main__":
    unittest.main()
