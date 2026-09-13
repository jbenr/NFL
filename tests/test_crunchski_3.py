import unittest
import numpy as np
import pandas as pd
import polars as pl
import data_crunchski_3 as dc
import shared_scoring as ss
from tests.test_shared_scoring import games


class MatchupPreparationTests(unittest.TestCase):
    def test_percentile_reuses_production_columns(self):
        data = games()
        data['away_off_pass_ypp'] = .3
        data['away_def_pass_ypp'] = -.2
        a, b = dc.matchup_representation(data.iloc[:8], data.iloc[8:], 'percentile', ['pass_ypp'])
        np.testing.assert_allclose(a.diff_pass_ypp.iloc[:8], .3)
        np.testing.assert_allclose(a.diff_pass_ypp.iloc[8:], .2)
        np.testing.assert_allclose(b.diff_pass_ypp, [.3, .2])

    def test_zscore_reuses_production_z_columns_like_percentile_reuses_rank_columns(self):
        data = games()
        data['away_off_pass_ypp_z'] = .3
        data['away_def_pass_ypp_z'] = -.2
        a, b = dc.matchup_representation(data.iloc[:8], data.iloc[8:], 'zscore', ['pass_ypp'])
        np.testing.assert_allclose(a.diff_pass_ypp.iloc[:8], .3)
        np.testing.assert_allclose(a.diff_pass_ypp.iloc[8:], .2)
        np.testing.assert_allclose(b.diff_pass_ypp, [.3, .2])

    def test_diff_importance_is_used_by_metric_selection(self):
        from optimus_prime import rank_metrics
        ranked = rank_metrics(pd.Series({'diff_pass_ypp': 3., 'diff_run_ypp': 1.}), ['run_ypp', 'pass_ypp'])
        self.assertEqual(ranked.index[0], 'pass_ypp')

    def test_model_reuses_extracted_preparation(self):
        self.assertIs(ss.prepare, dc.prepare)
        self.assertIs(ss.referee_tendencies, dc.referee_tendencies)

    def test_pandas_and_polars_match_with_missing_values_and_shuffled_index(self):
        data = games().iloc[::-1].copy()
        data.iloc[0, data.columns.get_loc('away_raw_off_pass_ypp')] = np.nan
        for mode in ['separate', 'differential']:
            actual = dc.scoring_rows(data, mode)
            pd.testing.assert_frame_equal(actual, dc.scoring_rows(pl.from_pandas(data), mode))
            if mode == 'differential':
                expected = data.away_raw_off_pass_ypp - data.home_raw_def_pass_ypp
                np.testing.assert_array_equal(actual.diff_pass_ypp.iloc[:len(data)], expected)
