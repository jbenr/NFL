import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import data_crunchski_2 as dc2


class ZScoreNormalizationTests(unittest.TestCase):
    def test_z_it_is_mean_zero_unit_std_and_monotonic(self):
        x = pd.Series([1., 2., 3., 4., 5.])
        z = dc2.z_it(x)
        self.assertAlmostEqual(z.mean(), 0., places=9)
        self.assertAlmostEqual(z.std(ddof=0), 1., places=9)
        self.assertTrue((np.diff(z.to_numpy()) > 0).all())

    def test_rev_z_it_is_negation(self):
        x = pd.Series([1., 2., 3., 4., 5.])
        np.testing.assert_allclose(dc2.rev_z_it(x), -dc2.z_it(x))

    def test_z_it_handles_zero_variance_without_dividing_by_zero(self):
        np.testing.assert_allclose(dc2.z_it(pd.Series([5., 5., 5.])), [0., 0., 0.])

    def test_normalize_stats_orients_off_def_and_exceptions_like_production(self):
        # A: best raw pass offense, worst raw pass defense (allows the most),
        # fewest turnovers committed, fewest turnovers forced. C is A's mirror.
        stats = pd.DataFrame({
            'off_pass_ypp': [9., 5., 1.],
            'def_pass_ypp': [9., 5., 1.],      # allowed -- higher is worse defense
            'off_turnovers': [0., 2., 4.],     # committed -- higher is worse offense
            'def_turnovers': [0., 2., 4.],     # forced -- higher is better defense
            'off_pass_%': [.6, .5, .4],        # usage rate -- must stay untouched
        }, index=['A', 'B', 'C'])
        for rank_fn, rev_fn in [(dc2.rank_it, dc2.rev_rank_it), (dc2.z_it, dc2.rev_z_it)]:
            normalized = dc2._normalize_stats(stats, rank_fn, rev_fn)
            pd.testing.assert_series_equal(normalized['off_pass_%'], stats['off_pass_%'])
            self.assertGreater(normalized.loc['A', 'off_pass_ypp'], normalized.loc['C', 'off_pass_ypp'])
            self.assertLess(normalized.loc['A', 'def_pass_ypp'], normalized.loc['C', 'def_pass_ypp'])
            self.assertGreater(normalized.loc['A', 'off_turnovers'], normalized.loc['C', 'off_turnovers'])
            self.assertLess(normalized.loc['A', 'def_turnovers'], normalized.loc['C', 'def_turnovers'])


class CompStatsZScoreTests(unittest.TestCase):
    def _stats(self):
        # Ranking population is every team in the snapshot (A-D), not just
        # the two teams playing (A@B) -- mirrors comp_stats seeing bye teams.
        return pd.DataFrame({
            'off_pass_ypp': [9., 5., 1., 3.],
            'def_pass_ypp': [9., 5., 1., 3.],
            'off_pass_%': [.9, .5, .1, .5],
            'off_run_%': [.1, .5, .9, .5],
        }, index=['A', 'B', 'C', 'D'])

    def _sched(self):
        return pd.DataFrame({'away_team': ['A'], 'home_team': ['B'], 'week': [1]})

    def test_percentile_and_zscore_columns_agree_in_sign(self):
        comp = dc2.comp_stats(self._stats(), self._sched(), use_scaling=False)
        self.assertIn('away_off_pass_ypp_z', comp)
        self.assertIn('away_def_pass_ypp_z', comp)
        self.assertEqual(np.sign(comp.away_off_pass_ypp.iloc[0]), np.sign(comp.away_off_pass_ypp_z.iloc[0]))
        self.assertEqual(np.sign(comp.away_def_pass_ypp.iloc[0]), np.sign(comp.away_def_pass_ypp_z.iloc[0]))

    def test_scaling_applies_post_normalization_to_zscore_too(self):
        stats = self._stats()
        sched = self._sched()
        scaled = dc2.comp_stats(stats, sched, use_scaling=True)
        unscaled = dc2.comp_stats(stats, sched, use_scaling=False)
        multiplier = stats.loc['A', 'off_pass_%'] + 0.5
        self.assertAlmostEqual(scaled.away_off_pass_ypp.iloc[0], unscaled.away_off_pass_ypp.iloc[0] * multiplier)
        self.assertAlmostEqual(scaled.away_off_pass_ypp_z.iloc[0], unscaled.away_off_pass_ypp_z.iloc[0] * multiplier)


def _plays(rows):
    """rows: list of dicts, one per play, with only the fields a test cares
    about -- filled out to calc_stats' full column set with harmless
    defaults (0/None) so every feature computes without KeyErrors."""
    defaults = dict(complete_pass=0, series=None, series_success=0, first_down=0,
                    third_down_converted=0, third_down_failed=0, fourth_down_converted=0,
                    fourth_down_failed=0, interception=0, fumble_lost=0, penalty=0,
                    drive=1, drive_time_of_possession='0:30', sack=0, qb_hit=0,
                    epa=0.0, success=0,      # Model 2.1's EPA inputs
                    defteam='OPP')
    return pd.DataFrame([{**defaults, **row} for row in rows])


class CalcStatsPerGameTests(unittest.TestCase):
    def _three_games(self):
        # game_a: most recent (days_from_max=0), light volume. game_b: 7
        # days older, light volume. game_c: 70 days older, MUCH higher
        # volume (10 run plays/10 pass attempts vs. 2/2) -- unequal per-game
        # sample sizes, so pooling (sum of plays over sum of plays) gives a
        # genuinely different answer than averaging each game's own rate
        # first. That distinction is the entire point of this fixture.
        rows = []
        for game, date, run_yards, completions, attempts in [
            ('a', '2024-12-29', [0, 10], 2, 2),
            ('b', '2024-12-22', [2, 4], 1, 2),
            ('c', '2024-10-20', [20] * 10, 2, 10),
        ]:
            for y in run_yards:
                rows.append(dict(game_id=game, game_date=date, posteam='BUF', play_type='run', yards_gained=y))
            for i in range(attempts):
                rows.append(dict(game_id=game, game_date=date, posteam='BUF', play_type='pass',
                                 yards_gained=5, complete_pass=int(i < completions)))
        return _plays(rows)

    def test_mean_pools_plays_rather_than_averaging_per_game_rates(self):
        stats_ = None
        with patch.object(dc2, 'RATE_MODE', 'mean'):
            stats_ = dc2.calc_stats(self._three_games())
        # off_run_ypp: sum(0+10+2+4+20*10) / count(2+2+10) -- NOT
        # mean(5, 3, 20)=9.33, which is what per-game averaging would give.
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_ypp'], (0 + 10 + 2 + 4 + 200) / 14)
        # off_pass_completion_%: sum(2+1+2) / sum(2+2+10) -- NOT
        # mean(1.0, 0.5, 0.2)=0.567, which is what per-game averaging would give.
        self.assertAlmostEqual(stats_.loc['BUF', 'off_pass_completion_%'], (2 + 1 + 2) / (2 + 2 + 10))

    def test_median_still_uses_per_game_rates_for_ratio_stats(self):
        with patch.object(dc2, 'RATE_MODE', 'median'):
            stats_ = dc2.calc_stats(self._three_games())
        # completion_% per game: 1.0, 0.5, 0.2 -- median of those three is 0.5,
        # unaffected by game_c's much larger attempt volume (unlike pooling).
        self.assertAlmostEqual(stats_.loc['BUF', 'off_pass_completion_%'], 0.5)

    def test_median_uses_a_true_per_play_median_for_ypp_not_per_game_rates(self):
        with patch.object(dc2, 'RATE_MODE', 'median'):
            stats_ = dc2.calc_stats(self._three_games())
        # Per-game means would be [5, 3, 20] -> median 5. The true per-play
        # median, pooling all 14 individual run yardages
        # ([0,2,4,10] + ten 20s), is 20 -- a different, deliberately chosen
        # design (median mode's exception for continuous per-play stats).
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_ypp'], 20.0)

    def test_weighted_matches_hand_computed_pooled_weighted_ratio(self):
        df = self._three_games()
        dates = pd.to_datetime(df.groupby('game_id')['game_date'].first())
        days_from_max = (dates.max() - dates).dt.days.to_numpy()
        w = dict(zip(dates.index, dc2.gradual_acceleration_with_floor(days_from_max, **dc2.DECAY_PRESETS['weighted'])))
        run_yards, run_plays = {'a': 10, 'b': 6, 'c': 200}, {'a': 2, 'b': 2, 'c': 10}
        expected = sum(w[g] * run_yards[g] for g in w) / sum(w[g] * run_plays[g] for g in w)
        with patch.object(dc2, 'RATE_MODE', 'weighted'):
            stats_ = dc2.calc_stats(df)
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_ypp'], expected)

    def test_steep_matches_hand_computed_pooled_weighted_ratio_with_its_own_preset(self):
        df = self._three_games()
        dates = pd.to_datetime(df.groupby('game_id')['game_date'].first())
        days_from_max = (dates.max() - dates).dt.days.to_numpy()
        w = dict(zip(dates.index, dc2.gradual_acceleration_with_floor(days_from_max, **dc2.DECAY_PRESETS['steep'])))
        run_yards, run_plays = {'a': 10, 'b': 6, 'c': 200}, {'a': 2, 'b': 2, 'c': 10}
        expected = sum(w[g] * run_yards[g] for g in w) / sum(w[g] * run_plays[g] for g in w)
        with patch.object(dc2, 'RATE_MODE', 'steep'):
            stats_ = dc2.calc_stats(df)
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_ypp'], expected)

    def _two_seasons(self):
        """One game in each season, same team: 'carryover' should halve the older one."""
        rows = []
        for game, date, run_yards in [('2025_01_BUF_NYJ', '2025-09-07', [10, 10]),
                                      ('2024_20_BUF_KC', '2025-01-26', [0, 0])]:
            for y in run_yards:
                rows.append(dict(game_id=game, game_date=date, posteam='BUF', play_type='run', yards_gained=y))
                rows.append(dict(game_id=game, game_date=date, posteam='BUF', play_type='pass',
                                 yards_gained=5, complete_pass=1))
        return _plays(rows)

    def test_importance_preset_scales_each_game_by_its_playoff_leverage(self):
        """'importance' = weighted decay x prior-season halving x leverage."""
        df = self._two_seasons()     # 2025 game (10 ypp), 2024 game (0 ypp)
        leverage = {('2025_01_BUF_NYJ', 'BUF'): 1.0, ('2024_20_BUF_KC', 'BUF'): 0.5}
        dates = pd.to_datetime(df.groupby('game_id')['game_date'].first())
        days = (dates.max() - dates).dt.days.to_numpy()
        w = dict(zip(dates.index, dc2.gradual_acceleration_with_floor(days, **dc2.DECAY_PRESETS['importance'])))
        yards = {'2025_01_BUF_NYJ': 20, '2024_20_BUF_KC': 0}
        plays = {'2025_01_BUF_NYJ': 2, '2024_20_BUF_KC': 2}
        # prior season halves as well, so the old game carries 0.5 (season) x 0.5 (leverage)
        extra = {'2025_01_BUF_NYJ': 1.0, '2024_20_BUF_KC': 0.5 * 0.5}
        expected = (sum(w[g] * extra[g] * yards[g] for g in w) / sum(w[g] * extra[g] * plays[g] for g in w))
        with patch.object(dc2, 'RATE_MODE', 'importance'), patch.object(dc2, 'GAME_IMPORTANCE', leverage):
            stats_ = dc2.calc_stats(df)
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_ypp'], expected)

    def test_importance_floors_a_meaningless_game_instead_of_dropping_it(self):
        df = self._two_seasons()
        floor = dc2.IMPORTANCE_WEIGHT['importance']['floor']
        zero = {('2025_01_BUF_NYJ', 'BUF'): 1.0, ('2024_20_BUF_KC', 'BUF'): 0.0}
        floored = {('2025_01_BUF_NYJ', 'BUF'): 1.0, ('2024_20_BUF_KC', 'BUF'): floor}
        with patch.object(dc2, 'RATE_MODE', 'importance'), patch.object(dc2, 'GAME_IMPORTANCE', zero):
            at_zero = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        with patch.object(dc2, 'RATE_MODE', 'importance'), patch.object(dc2, 'GAME_IMPORTANCE', floored):
            at_floor = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        self.assertAlmostEqual(at_zero, at_floor)     # zero leverage is clipped to the floor
        self.assertLess(at_zero, 10.0)                # and the old game still pulls the average down

    def test_unknown_leverage_leaves_a_game_at_full_weight(self):
        df = self._two_seasons()
        with patch.object(dc2, 'RATE_MODE', 'carryover'):
            carryover_ypp = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        with patch.object(dc2, 'RATE_MODE', 'importance'), patch.object(dc2, 'GAME_IMPORTANCE', {('x', 'BUF'): 0.5}):
            unknown_ypp = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        self.assertAlmostEqual(unknown_ypp, carryover_ypp)

    def test_epa_metrics_come_from_the_play_level_epa_and_success_columns(self):
        """Model 2.1's inputs: EPA per play and success rate, split pass/run."""
        rows = [dict(game_id='2025_01_BUF_NYJ', game_date='2025-09-07', posteam='BUF', play_type='run',
                     yards_gained=4, epa=0.5, success=1),
                dict(game_id='2025_01_BUF_NYJ', game_date='2025-09-07', posteam='BUF', play_type='run',
                     yards_gained=0, epa=-0.9, success=0),
                dict(game_id='2025_01_BUF_NYJ', game_date='2025-09-07', posteam='BUF', play_type='pass',
                     yards_gained=12, epa=1.5, success=1, complete_pass=1),
                dict(game_id='2025_01_BUF_NYJ', game_date='2025-09-07', posteam='BUF', play_type='pass',
                     yards_gained=0, epa=-0.5, success=0, complete_pass=0)]
        with patch.object(dc2, 'RATE_MODE', 'mean'):
            stats_ = dc2.calc_stats(_plays(rows))
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_epa_pp'], (0.5 - 0.9) / 2)
        self.assertAlmostEqual(stats_.loc['BUF', 'off_pass_epa_pp'], (1.5 - 0.5) / 2)
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_success_%'], 0.5)
        self.assertAlmostEqual(stats_.loc['BUF', 'off_pass_success_%'], 0.5)

    def test_epa_metrics_join_the_model_input_set_only_for_2_1(self):
        import data_crunchski_3 as dc3
        base = dc3.use_epa(False)
        self.assertNotIn('pass_epa_pp', base)
        with_epa = dc3.use_epa(True)
        self.assertEqual(with_epa[-4:], dc3.EPA_METRICS)
        # FEATURES tracks METRICS, and both lists are shared by every importer.
        self.assertIn('off_pass_epa_pp', dc3.FEATURES)
        self.assertIn('def_run_success_%', dc3.FEATURES)
        dc3.use_epa(False)
        self.assertNotIn('off_pass_epa_pp', dc3.FEATURES)

    def test_carryover_halves_games_from_the_previous_season(self):
        df = self._two_seasons()
        dates = pd.to_datetime(df.groupby('game_id')['game_date'].first())
        days = (dates.max() - dates).dt.days.to_numpy()
        w = dict(zip(dates.index, dc2.gradual_acceleration_with_floor(days, **dc2.DECAY_PRESETS['carryover'])))
        yards = {'2025_01_BUF_NYJ': 20, '2024_20_BUF_KC': 0}
        plays = {'2025_01_BUF_NYJ': 2, '2024_20_BUF_KC': 2}
        half = {g: (0.5 if g.startswith('2024') else 1.0) for g in w}
        expected = (sum(w[g] * half[g] * yards[g] for g in w) / sum(w[g] * half[g] * plays[g] for g in w))
        with patch.object(dc2, 'RATE_MODE', 'carryover'):
            stats_ = dc2.calc_stats(df)
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_ypp'], expected)

    def test_carryover_leans_more_on_this_season_than_weighted_does(self):
        df = self._two_seasons()   # this season 10 ypp, last season 0 ypp
        with patch.object(dc2, 'RATE_MODE', 'weighted'):
            weighted_ypp = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        with patch.object(dc2, 'RATE_MODE', 'carryover'):
            carryover_ypp = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        self.assertGreater(carryover_ypp, weighted_ypp)

    def test_carryover_matches_weighted_inside_one_season(self):
        df = self._three_games()   # all 2024-25 dates, synthetic ids -> one season
        with patch.object(dc2, 'RATE_MODE', 'weighted'):
            weighted_ypp = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        with patch.object(dc2, 'RATE_MODE', 'carryover'):
            carryover_ypp = dc2.calc_stats(df).loc['BUF', 'off_run_ypp']
        self.assertAlmostEqual(carryover_ypp, weighted_ypp)

    def test_steep_discounts_the_high_volume_old_game_harder_than_weighted(self):
        df = self._three_games()
        with patch.object(dc2, 'RATE_MODE', 'weighted'):
            weighted_stats = dc2.calc_stats(df)
        with patch.object(dc2, 'RATE_MODE', 'steep'):
            steep_stats = dc2.calc_stats(df)
        # game_c (oldest, all 20-yard runs) dominates the pool by sheer
        # volume -- steep's faster decay/lower floor should discount its
        # weight harder, pulling off_run_ypp further down than 'weighted' does.
        self.assertLess(steep_stats.loc['BUF', 'off_run_ypp'], weighted_stats.loc['BUF', 'off_run_ypp'])

    def test_run_and_pass_percent_are_offense_only(self):
        with patch.object(dc2, 'RATE_MODE', 'mean'):
            stats_ = dc2.calc_stats(self._three_games())
        self.assertIn('off_run_%', stats_)
        self.assertNotIn('def_run_%', stats_)
        self.assertNotIn('def_pass_%', stats_)

    def test_undefined_game_excluded_from_median_without_affecting_other_features(self):
        # game_a has 3rd-down opportunities; game_b has none (0 converted,
        # 0 failed -> 0/0 -> NaN that game) but still has normal run plays.
        # This only matters for median (the one mode with a per-game step
        # for ratio stats) -- mean/weighted/steep pool numerator and
        # denominator directly, so an all-zero game just contributes zeros.
        rows = [
            dict(game_id='a', game_date='2024-12-29', posteam='BUF', play_type='run', yards_gained=4),
            dict(game_id='a', game_date='2024-12-29', posteam='BUF', play_type='pass', yards_gained=5,
                third_down_converted=1),
            dict(game_id='b', game_date='2024-12-22', posteam='BUF', play_type='run', yards_gained=6),
            dict(game_id='b', game_date='2024-12-22', posteam='BUF', play_type='pass', yards_gained=5),
        ]
        with patch.object(dc2, 'RATE_MODE', 'median'):
            stats_ = dc2.calc_stats(_plays(rows))
        # third_down_% only had one defined game (a): 1 converted / 1 faced = 1.0,
        # not diluted by game b's undefined 0/0.
        self.assertAlmostEqual(stats_.loc['BUF', 'off_third_down_%'], 1.0)
        # run_ypp (true per-play median) unaffected by third-down being
        # undefined in game b: plays [4, 6] -> median 5.
        self.assertAlmostEqual(stats_.loc['BUF', 'off_run_ypp'], 5.0)

    def test_calculation_choices_reject_legacy(self):
        with patch.object(dc2, 'RATE_MODE', 'legacy'):
            with self.assertRaises(ValueError):
                dc2.calc_stats(self._three_games())


if __name__ == '__main__':
    unittest.main()
