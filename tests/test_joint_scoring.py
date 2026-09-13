import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import joint_scoring as js
from tests.test_shared_scoring import games


class JointTests(unittest.TestCase):
    def test_importance_excludes_same_week_and_future_results(self):
        rows = []
        for week in [1, 2, 3]:
            for away, home in [('ARI', 'LA'), ('SF', 'SEA')]:
                rows.append(dict(season=2025, week=week, game_type='REG',
                                 away_team=away, home_team=home, away_score=20., home_score=10.))
        schedule = pd.DataFrame(rows)
        first = js.game_importance(schedule)
        schedule.loc[schedule.week >= 2, ['away_score', 'home_score']] = [0., 99.]
        second = js.game_importance(schedule)
        columns = [s + '_' + n for s in ['away', 'home'] for n in js.IMPORTANCE]
        pd.testing.assert_frame_equal(first.loc[first.week == 2, columns],
                                      second.loc[second.week == 2, columns])
        self.assertTrue(first.away_importance.between(0, 1).all())
        self.assertIn('away_playoff_if_win', first)

    def test_targets_and_prior_only_transforms(self):
        data = games()
        data['away_importance'], data['home_importance'] = .3, .7
        for side in ['away', 'home']:
            for name in js.IMPORTANCE[1:]:
                data[side + '_' + name] = 0
        target, x, y, xp, names = js.prepare(data, 2026, 1, ['importance'])
        self.assertEqual(x.shape, (8, 2 * (17 + len(js.IMPORTANCE))))
        np.testing.assert_array_equal(y['spread'], -3.)
        np.testing.assert_array_equal(y['total'], data.iloc[:8].away_score + data.iloc[:8].home_score)
        changed = data.copy()
        changed.loc[8, ['away_score', 'home_score']] = 999.
        changed = pd.concat([changed, changed.iloc[[-1]].assign(week=2)])
        _, x2, y2, xp2, names2 = js.prepare(changed, 2026, 1, ['importance'])
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(xp, xp2)
        np.testing.assert_array_equal(y['total'], y2['total'])
        self.assertEqual(names, names2)

    def test_joint_report_keeps_both_matchups_and_implied_scores(self):
        import weekly_packet as wp
        data = games().iloc[[-1]].copy()
        data = data.assign(market='total', market_base=44., actual=np.nan, prediction=46.,
                           variance=4., baseline=40., edge=2., qualifies=False, pnl=0.,
                           odds=-110., assumed_odds=True, away_points=23., home_points=23.,
                           scores_implied=True, model_family='joint',
                           attr_away_off_run_ypp=2., attr_away_def_run_ypp=4.,
                           # headline_table() re-settles the saved {market}_details.csv
                           # to apply its own high-confidence cutoffs -- needs these three
                           # even though this fixture's own qualifies/pnl/odds above are
                           # already pre-settled for the per-game card, not the headline.
                           positive_odds=-110., negative_odds=-110., residual=np.nan)
        config = dict(model='joint test', calculation='joint-matchup-v1', market='total',
                      lookback=20, input_mode='differential', status='PASS', reason='test',
                      context_note='Test context availability note')
        importance = pd.DataFrame(dict(feature=['away_off_run_ypp'], importance=[1.], std=[.1]))
        with tempfile.TemporaryDirectory() as directory, patch.object(wp, 'display_stats', return_value=pd.DataFrame()):
            wp.write_packets(data, data, importance, config, Path(directory))
            html = (Path(directory) / '2026_01' / 'total.html').read_text()
            self.assertIn('ARI offense', html)
            self.assertIn('ARI defense', html)
            self.assertIn('Implied score', html)
            self.assertIn('Test context availability note', html)
            self.assertTrue((Path(directory) / '2026_01' / 'importance.html').exists())

    def test_swap_symmetry_and_cross_matchup_awareness(self):
        from modelo_workers import initialize_worker
        initialize_worker()
        import tensorflow as tf
        x = np.random.default_rng(4).normal(size=(3, 34)).astype('float32')
        reverse = np.c_[x[:, 17:], x[:, :17]]
        for market, sign in [('spread', -1), ('total', 1)]:
            tf.keras.utils.set_random_seed(9)
            model = js.build_model(17, market)
            p = np.asarray(model(x, training=False))
            q = np.asarray(model(reverse, training=False))
            np.testing.assert_allclose(p, sign * q, atol=1e-5)
            changed = x.copy()
            changed[:, 20] += 2
            self.assertFalse(np.allclose(p, model(changed, training=False)))

    def test_fit_cache_and_attribution_accounting(self):
        data = games()
        with tempfile.TemporaryDirectory() as directory:
            def cache(namespace, identity, sources):
                return Path(directory) / (identity[0] + '.parquet')
            with patch.object(js.utils, 'cache_path', side_effect=cache):
                _, details, importance = js.fit_panel(data, 2026, 1, 2, 1, 13, 1)
                with patch.object(js, 'fit_member', side_effect=AssertionError('cache missed')):
                    _, cached, _ = js.fit_panel(data, 2026, 1, 2, 1, 13, 1)
        for market in ['spread', 'total']:
            pd.testing.assert_frame_equal(details[market], cached[market])
            self.assertTrue(np.isfinite(details[market].prediction).all())
            self.assertLess(abs(details[market].integration_residual.iloc[0]), .1)
            self.assertIn('away_off_run_ypp', importance[market].feature.values)
            self.assertIn('away_def_run_ypp', importance[market].feature.values)
        row = details['spread'].iloc[0]
        self.assertAlmostEqual(row.away_points - row.home_points, row.prediction, places=5)
        self.assertAlmostEqual(row.away_points + row.home_points,
                               details['total'].prediction.iloc[0], places=5)

    def weather_games(self):
        return pd.DataFrame(dict(game_id=['a', 'b', 'c'], gameday=['2025-09-07'] * 3,
            gametime=['13:00'] * 3, roof=['outdoors', 'closed', None],
            temp=[95., np.nan, np.nan], wind=[20., np.nan, np.nan]))

    def test_weather_does_not_silently_use_recorded_conditions(self):
        d = js.weather_features(self.weather_games())
        self.assertTrue(pd.isna(d.weather_temperature_f.iloc[0]))
        self.assertEqual(d.weather_temperature_f_missing.iloc[0], 1)
        self.assertEqual(d.weather_temperature_f.iloc[1], 72)
        self.assertEqual(d.weather_wind_mph.iloc[1], 0)
        self.assertEqual(d.weather_precip_probability.iloc[1], 0)
        self.assertEqual(d.weather_roof_missing.iloc[2], 1)
        recorded = js.weather_features(self.weather_games(), 'recorded')
        self.assertEqual(recorded.weather_temperature_f.iloc[0], 95)

    def test_forecast_cutoff_and_valid_time(self):
        forecasts = pd.DataFrame(dict(game_id=['a'] * 3,
            issued_at=['2025-09-06T16:00:00Z', '2025-09-07T16:00:00Z', '2025-09-06T16:30:00Z'],
            valid_at=['2025-09-07T17:00:00Z', '2025-09-07T17:00:00Z', '2025-09-08T17:00:00Z'],
            temperature_f=[70, 99, 88], wind_mph=[5, 90, 80], precip_probability=[.2, .9, .8]))
        d = js.weather_features(self.weather_games(), forecasts=forecasts)
        self.assertEqual(d.weather_temperature_f.iloc[0], 70)
        forecasts.loc[0, 'issued_at'] = '2025-09-06 16:00:00'
        with self.assertRaises(ValueError):
            js.weather_features(self.weather_games(), forecasts=forecasts)


if __name__ == '__main__':
    unittest.main()
