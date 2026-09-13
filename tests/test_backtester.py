import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from backtester import configure_workers, evaluate, history_weeks


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
