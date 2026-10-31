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


class ForecastLockTests(unittest.TestCase):
    """A rerun must see the same weather as the first build, or a new
    forecast capture moves picks with nothing else changed."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.dir = Path(tempfile.mkdtemp())
        self.hist = self.dir / 'hist.parquet'
        self.fc = self.dir / 'fc.parquet'
        self.lock = self.dir / 'lock.parquet'
        played = {c: [1.0] for c in dc.HISTORICAL_WEATHER}
        pd.DataFrame({'game_id': ['2026_07_A_B'], **played}).to_parquet(self.hist)
        self.panel = pd.DataFrame({'game_id': ['2026_07_A_B', '2026_08_C_D'], 'roof': ['outdoors', 'outdoors']})

    def capture(self, wind, issued):
        row = {c: 0.0 for c in dc.HISTORICAL_WEATHER}
        row.update(game_id='2026_08_C_D', wind_mph=wind, issued_at=pd.Timestamp(issued, tz='UTC'))
        old = pd.read_parquet(self.fc) if self.fc.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_parquet(self.fc)

    def wind(self, lock=True):
        out = dc.attach_historical_weather(self.panel, self.hist, self.fc, lock_path=self.lock if lock else None)
        return out.set_index('game_id').wind_mph

    def test_a_newer_capture_does_not_move_a_locked_game(self):
        self.capture(10., '2026-10-27')
        self.assertEqual(self.wind()['2026_08_C_D'], 10.)
        self.capture(18., '2026-10-28')
        self.assertEqual(self.wind()['2026_08_C_D'], 10.)
        self.assertEqual(self.wind(lock=False)['2026_08_C_D'], 18., 'without a lock the newest capture wins')

    def test_unlocking_a_week_takes_the_newest_capture_and_locks_it(self):
        self.capture(10., '2026-10-27')
        self.wind()
        self.capture(18., '2026-10-28')
        self.assertEqual(dc.unlock_forecasts(2026, 8, self.lock), 1)
        self.assertEqual(self.wind()['2026_08_C_D'], 18.)
        self.capture(25., '2026-10-29')
        self.assertEqual(self.wind()['2026_08_C_D'], 18.)

    def test_played_games_never_read_the_lock(self):
        self.capture(10., '2026-10-27')
        self.assertEqual(self.wind()['2026_07_A_B'], 1.)
        self.assertNotIn('2026_07_A_B', set(pd.read_parquet(self.lock).game_id))

    def test_unlock_only_frees_the_named_week(self):
        self.capture(10., '2026-10-27')
        self.wind()
        self.assertEqual(dc.unlock_forecasts(2026, 9, self.lock), 0)
        self.assertEqual(len(pd.read_parquet(self.lock)), 1)
