"""Leakage, settlement, uncertainty, cache and packet regression checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import data_crunchski_2 as dc
import optimize_picks as op
import weekly_packet as wp


def sample():
    rng = np.random.default_rng(42)
    n = 60
    data = pd.DataFrame(dict(season=2025, week=np.repeat(np.arange(1, 21), 3),
                             away_team=['BUF', 'SF', 'LA'] * 20,
                             home_team=['DEN', 'SEA', 'CHI'] * 20,
                             week_id=np.repeat(np.arange(20), 3),
                             x=rng.normal(size=n), home_field_adv=1,
                             positive_odds=-110, negative_odds=-105, market='spread', market_base=-3))
    data['residual'] = 2 * data.x + rng.normal(size=n)
    data['actual'] = data.residual + data.market_base
    return data


class ResearchTests(unittest.TestCase):
    def test_fixed_model_rule_choice_uses_calibration_not_validation(self):
        import hashlib
        from types import SimpleNamespace
        import model_shredski as ms
        records = []
        for wid in range(30):
            season = 2023 if wid < 22 else 2024 if wid < 26 else 2025
            week = wid + 1 if wid < 22 else wid - 21 if wid < 26 else wid - 25
            records.append(dict(season=season, week=week, week_id=wid, game_type='REG',
                                away_team='BUF', home_team='DEN', away_score=21, home_score=17,
                                margin=4, points=38, spread_line=1, total_line=40, x=float(wid),
                                away_spread_odds=-110, home_spread_odds=-110, over_odds=-110, under_odds=-110))
        panel = pd.DataFrame(records)
        def forecast(data, season, week, tag, **kwargs):
            self.assertEqual(kwargs['iterations'], 100)
            self.assertEqual(kwargs['features'], ['x'])
            prior = data[(data.season < season) | ((data.season == season) & (data.week < week))]
            self.assertEqual(len(prior), 20)
            self.assertEqual(len(data), 21)
            return pd.DataFrame(dict(away_team=['BUF'], home_team=['DEN'], prediction=[.5], variance=[1.0]))
        with tempfile.TemporaryDirectory() as folder:
            def cache(kind, config, sources):
                return Path(folder) / (kind + hashlib.sha256(str(config).encode()).hexdigest() + '.parquet')
            args = SimpleNamespace(output=folder, season=2025, week=4, history_weeks=50,
                                   markets=['spread'], start_season=2024, validation_season=2025, seed=1337, min_bets=1)
            with patch.object(op, 'build_panel', return_value=panel), patch.object(op, 'feature_names', return_value=['x']), \
                 patch.object(ms, 'modelo', side_effect=forecast), patch.object(op.utils, 'cache_path', side_effect=cache):
                first = op.production_rule_scan(args)['spread']
                panel.loc[panel.season == 2025, ['away_score', 'home_score', 'margin']] = [0, 30, -30]
                second = op.production_rule_scan(args)['spread']
            self.assertEqual(first['diff_cutoff'], second['diff_cutoff'])
            self.assertEqual(first['sd_cutoff'], second['sd_cutoff'])
            self.assertGreater(first['validation']['pnl_units'], second['validation']['pnl_units'])

    def test_sd_conversion_preserves_bets_and_pnl(self):
        data = sample().assign(edge=lambda d: d.x, variance=np.linspace(0, 4, 60))
        for variance in [None, .5, 1, 2.25, 4]:
            sd = op.policy_sd_cutoff({'var_cutoff': variance})
            old = op.settle(data, .2, var_cutoff=variance)
            new = op.settle(data, .2, sd_cutoff=sd)
            pd.testing.assert_series_equal(old.qualifies, new.qualifies)
            pd.testing.assert_series_equal(old.pnl, new.pnl)
        with self.assertRaises(ValueError):
            op.settle(data, sd_cutoff=-1)

    def test_highlights_require_matching_past_validation_and_unrounded_limits(self):
        from main import validated_highlights
        from copy import deepcopy
        spec = op.neural_spec(['home_field_adv'], 20, 'legacy')
        policy = dict(model_spec=spec, status='PAPER QUALIFIED', evaluation='weekly_walk_forward',
                      validated_through=[2025, 22], calibration=dict(n=100, pnl_units=10),
                      validation=dict(n=100, pnl_units=10), validation_roi_95=[.01, .2],
                      diff_cutoff=4, sd_cutoff=1.04)
        data = pd.DataFrame(dict(diff=[4.1, 4.1, 3.999, 0], sd=[1.03, 1.049, .5, .5]))
        result = validated_highlights(data, policy, spec, (2026, 1))
        self.assertEqual(result.tolist(), [True, False, False, False])
        for key, value in [('status', 'PASS'), ('validated_through', [2026, 1]),
                           ('validation_roi_95', [-.01, .2]), ('evaluation', 'in_sample')]:
            bad = deepcopy(policy)
            bad[key] = value
            self.assertFalse(validated_highlights(data, bad, spec, (2026, 1)).any())
        mismatch = dict(spec, iterations=20)
        self.assertFalse(validated_highlights(data, policy, mismatch, (2026, 1)).any())
        self.assertFalse(validated_highlights(data, {}, spec, (2026, 1)).any())

    def test_weighted_rate_uses_weighted_denominator(self):
        data = pd.DataFrame(dict(posteam=['BUF'] * 3, play_type='run',
                                 game_date=['2024-01-01', '2024-06-01', '2024-09-01'], success=1))
        with patch.object(dc, 'RATE_MODE', 'weighted'):
            numerator = dc.slicer(data, 'run', 'posteam', 'success', 'sum')
            denominator = dc.slicer(data, 'run', 'posteam', 'success', 'count')
        np.testing.assert_allclose(numerator / denominator, 1)

    def test_game_importance_cannot_see_current_or_future_results(self):
        teams = [t for t in dc.TEAM_META if t != 'LAR']
        rows = [dict(season=2025, week=week, game_type='REG', away_team=teams[i],
                     home_team=teams[i+1], away_score=7, home_score=14)
                for week in [1, 2] for i in range(0, len(teams), 2)]
        schedule = pd.DataFrame(rows)
        before = dc.matchup_importance(schedule)
        schedule.loc[schedule.week == 2, ['away_score', 'home_score']] = [50, 0]
        after = dc.matchup_importance(schedule)
        columns = ['home_importance', 'away_importance']
        pd.testing.assert_frame_equal(before[columns], after[columns])

    def test_price_push_zero_edge_and_no_variance_filter(self):
        data = pd.DataFrame(dict(edge=[1, -1, 1, 0, 1], residual=[1, 1, 0, -1, np.nan],
                                 variance=1, positive_odds=150, negative_odds=-110, week_id=1))
        result = op.score(data, var_cutoff=np.nan)
        self.assertEqual((result['n'], result['wins'], result['losses'], result['pushes']), (3, 1, 1, 1))
        self.assertAlmostEqual(result['pnl_units'], .5)
        self.assertEqual(op.score(data, var_cutoff=.5)['n'], 0)

    def test_future_outcomes_do_not_change_earlier_predictions_and_cache_reuse(self):
        data = sample()
        with tempfile.TemporaryDirectory() as folder:
            def cache(kind, config, sources):
                import hashlib
                return Path(folder) / (hashlib.sha256(str(config).encode()).hexdigest() + '.parquet')
            with patch.object(op.utils, 'cache_path', side_effect=cache):
                first = op.weekly_predict(data, ['x', 'home_field_adv'], 5, bags=3)
                with patch.object(op, 'make_pipeline', side_effect=AssertionError('cache miss')):
                    pd.testing.assert_frame_equal(first, op.weekly_predict(data, ['x', 'home_field_adv'], 5, bags=3))
                data.loc[data.week_id >= 15, ['actual', 'residual']] += 1000
                second = op.weekly_predict(data, ['x', 'home_field_adv'], 5, bags=3)
            np.testing.assert_allclose(first.loc[first.week_id <= 15, 'prediction'], second.loc[second.week_id <= 15, 'prediction'])
            np.testing.assert_allclose(first.prediction, first.baseline + first.filter(like='attr_').sum(axis=1), atol=1e-10)
            self.assertTrue((first.variance >= 0).all())

    def test_records_reset_at_team_and_season_boundaries(self):
        data = pd.DataFrame(dict(season=[2024, 2024, 2025], week=[1, 2, 1],
                                 home_team=['BUF'] * 3, away_team=['DEN'] * 3,
                                 home_score=[20, 30, 30], away_score=[10, 10, 10]))
        records = dc._records_pregame(data)
        self.assertTrue(records.loc[records.week == 1, ['W_pg', 'L_pg', 'T_pg']].eq(0).all().all())

    def test_minimum_sample_does_not_fall_back_to_small_samples(self):
        data = sample().assign(edge=1, variance=1)
        self.assertFalse(op.cutoff_grid(data, min_bets=100).eligible.any())

    def test_attribution_shows_baseline_and_residual(self):
        row = pd.Series(dict(prediction=5, baseline=2, attr_x=3))
        html = wp.attribution(row)
        self.assertIn('prediction +5.00', html)
        self.assertIn('residual +0.0000', html)


if __name__ == '__main__':
    unittest.main()
