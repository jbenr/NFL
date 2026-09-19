import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from backtester import configure_workers, evaluate, history_weeks, two_sided_season


class BacktesterTests(unittest.TestCase):
    def test_preparation_workers_override_environment_not_training(self):
        args = SimpleNamespace(prep_jobs=4, jobs=8)
        with patch.dict(os.environ, {'NFL_WORKERS': '16'}):
            configure_workers(args)
            self.assertEqual(os.environ['NFL_WORKERS'], '4')
            self.assertEqual(args.jobs, 8)
            args.prep_jobs = 1
            configure_workers(args)
            self.assertEqual(os.environ['NFL_WORKERS'], '1')

    def test_invalid_workers_leave_environment_unchanged(self):
        for prep, training in [(0, 8), (-1, 8), (1, 0), (1, -1)]:
            with self.subTest(prep=prep, training=training), patch.dict(os.environ, {'NFL_WORKERS': '2'}):
                with self.assertRaises(ValueError):
                    configure_workers(SimpleNamespace(prep_jobs=prep, jobs=training))
                self.assertEqual(os.environ['NFL_WORKERS'], '2')

    def test_history_is_derived_from_start_and_end(self):
        schedule = pd.DataFrame([dict(season=s, week=w, game_type='REG')
                                 for s in range(2008, 2026) for w in range(1, 18)])
        args = SimpleNamespace(start_season=2010, season=2025, week=22)
        with patch('backtester.pd.read_parquet', return_value=schedule):
            self.assertEqual(history_weeks(args), 16 * 17 + 20)
            args.week = 10
            self.assertEqual(history_weeks(args), 15 * 17 + 10 + 19)
            args.start_season = 2008
            with self.assertRaises(ValueError):
                history_weeks(args)

    def _schedule(self, seasons=range(2008, 2026), weeks=range(1, 18)):
        return pd.DataFrame([dict(season=s, week=w, game_type='REG', game_id=f'{s}_{w}',
                                  away_score=10., home_score=20.) for s in seasons for w in weeks])

    def test_two_sided_plan_spans_start_season_through_season(self):
        # --plan short-circuits before build_panel/attach_historical_weather --
        # cheap enough to run for real, no mocked model fit needed.
        args = SimpleNamespace(season=2025, week=17, start_season=2010, lookback=20, train_window=100,
                               iterations=100, jobs=8, epochs=100, seed=1337, prep_jobs=1,
                               weather_file=None, plan=True, output=None)
        with patch('backtester.pd.read_parquet', return_value=self._schedule()), \
             patch('backtester.Path.exists', return_value=True):
            config = two_sided_season(args)
        self.assertEqual(config['start_season'], 2010)
        self.assertEqual(config['season'], 2025)
        self.assertEqual(config['end_week'], 17)

    def test_two_sided_default_start_season_is_single_season(self):
        # Matches the pre-multi-season behavior when --start-season isn't given.
        args = SimpleNamespace(season=2025, week=17, start_season=2025, lookback=20, train_window=100,
                               iterations=100, jobs=8, epochs=100, seed=1337, prep_jobs=1,
                               weather_file=None, plan=True, output=None)
        with patch('backtester.pd.read_parquet', return_value=self._schedule()), \
             patch('backtester.Path.exists', return_value=True):
            config = two_sided_season(args)
        self.assertEqual(config['start_season'], 2025)

    def test_two_sided_model_20_plan_uses_weighted_settings(self):
        args = SimpleNamespace(season=2025, week=17, start_season=2024, lookback=20, train_window=100,
                               iterations=100, jobs=8, epochs=100, seed=1337, prep_jobs=1,
                               weather_file=None, plan=True, output=None, model_version='model_2.0')
        with patch('backtester.pd.read_parquet', return_value=self._schedule()), \
             patch('backtester.Path.exists', return_value=True):
            config = two_sided_season(args)
        self.assertEqual(config['model'], 'model-2.0')
        self.assertEqual(config['calculation'], 'weighted')
        self.assertEqual(config['lookback'], 20)
        self.assertEqual(config['train_window'], 100)

    def test_two_sided_requires_a_complete_window(self):
        # 2026 wk1 is scheduled but unplayed (NaN scores) -- asking for it
        # should fail clearly, not silently evaluate a partial season.
        schedule = pd.concat([self._schedule(), pd.DataFrame([dict(
            season=2026, week=1, game_type='REG', game_id='2026_1',
            away_score=float('nan'), home_score=float('nan'))])], ignore_index=True)
        args = SimpleNamespace(season=2026, week=1, start_season=2026, lookback=20, train_window=100,
                               iterations=100, jobs=8, epochs=100, seed=1337, prep_jobs=1,
                               weather_file=None, plan=True, output=None)
        with patch('backtester.pd.read_parquet', return_value=schedule), \
             patch('backtester.Path.exists', return_value=True):
            with self.assertRaises(ValueError):
                two_sided_season(args)

    def test_cutoffs_do_not_change_with_validation_outcomes(self):
        rows = []
        for season in [2024, 2025]:
            for week in range(1, 9):
                rows.append(dict(season=season, week=week, week_id=(season-2024)*8+week, prediction=3., actual=5.,
                                 market_base=0., edge=3., variance=1., residual=5.,
                                 positive_odds=-110., negative_odds=-110.))
        data = pd.DataFrame(rows)
        a, _ = evaluate(data, 2025, 4)
        data.loc[data.season.eq(2025), 'actual'] = -5
        data.loc[data.season.eq(2025), 'residual'] = -5
        b, _ = evaluate(data, 2025, 4)
        self.assertEqual(a['diff_cutoff'], b['diff_cutoff'])
        self.assertEqual(a['sd_cutoff'], b['sd_cutoff'])
        self.assertGreater(a['validation']['pnl_units'], b['validation']['pnl_units'])


if __name__ == '__main__':
    unittest.main()
