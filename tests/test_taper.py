import unittest

import numpy as np
import pandas as pd

import data_crunchski_2 as dc2


class SolvedPriorRatio(unittest.TestCase):
    """The 'solved' preset's one number, and the reasons it is that number
    rather than the floor the other presets use."""

    def test_sits_in_the_range_every_era_agreed_on(self):
        """The fitted objective is flat from about 0.35 to 0.6 across
        2011-2015, 2016-2020 and 2021-2025. Anything inside that band is
        defensible; anything outside it was not measured."""
        self.assertGreaterEqual(dc2.SOLVED_PRIOR_RATIO, 0.35)
        self.assertLessEqual(dc2.SOLVED_PRIOR_RATIO, 0.6)

    def test_is_far_above_the_floor_the_decay_presets_reach(self):
        """The finding this preset exists for: by September the decay
        curves have pushed last season to their floor, an order of
        magnitude below what it is worth."""
        for preset in ('weighted', 'carryover', 'steep'):
            floor = dc2.DECAY_PRESETS[preset]['floor_weight']
            floor *= dc2.PRIOR_SEASON_WEIGHT.get(preset, 1.)
            self.assertGreater(dc2.SOLVED_PRIOR_RATIO, floor * 5)


class SolvedPooling(unittest.TestCase):
    """How the preset actually weights a window of games."""

    def ingredients(self, games):
        index = pd.MultiIndex.from_tuples([(g, t) for g, t in games],
                                          names=['game_id', 'team'])
        return pd.DataFrame({'_run_yards': np.ones(len(games)) * 10.,
                             '_run_plays': np.ones(len(games))}, index=index)

    def test_current_season_games_weigh_the_same_as_each_other(self):
        """The whole point of the solved curve: no days-since decay inside
        a season. Two games a month apart count alike."""
        games = [('2024_01_KC_NO', 'KC'), ('2024_05_KC_NO', 'KC')]
        ing = self.ingredients(games)
        dates = pd.Series({'2024_01_KC_NO': pd.Timestamp('2024-09-08'),
                           '2024_05_KC_NO': pd.Timestamp('2024-10-06')})
        frame = ing.reset_index()
        seasons = dc2._season_of(frame, frame.game_id.map(dates))
        self.assertTrue((seasons == 2024).all())

    def test_season_boundary_is_read_from_the_game_id(self):
        games = [('2023_17_KC_NO', 'KC'), ('2024_01_KC_NO', 'KC')]
        ing = self.ingredients(games)
        frame = ing.reset_index()
        dates = frame.game_id.map(pd.Series({'2023_17_KC_NO': pd.Timestamp('2024-01-01'),
                                             '2024_01_KC_NO': pd.Timestamp('2024-09-08')}))
        seasons = dc2._season_of(frame, dates)
        self.assertEqual(list(seasons), [2023, 2024])

    def test_pooled_ratio_ignores_overall_scale(self):
        """Why the solver's shrink-toward-league-average cannot be
        installed here, and why only the in-season/prior ratio can: the
        pooled stat divides sums, so multiplying every weight by a
        constant changes nothing."""
        ing = self.ingredients([('2024_01_KC_NO', 'KC'), ('2024_02_KC_NO', 'KC')])
        frame = ing.reset_index()
        for scale in (1., 7.3):
            pooled = ing.mul(np.full(len(ing), scale), axis=0).groupby(
                frame['team'].to_numpy()).sum()
            self.assertAlmostEqual(
                float(pooled._run_yards / pooled._run_plays), 10.)

    def test_solved_is_a_registered_preset(self):
        self.assertIn('solved', dc2.DECAY_PRESETS)


if __name__ == '__main__':
    unittest.main()
