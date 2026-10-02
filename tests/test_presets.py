import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import data_crunchski_2 as dc2


class PresetShapes(unittest.TestCase):
    """What each recency preset pays a game, as a function of its age."""

    DAYS = np.array([7., 35., 70., 105., 140., 300.])

    def curve(self, preset):
        """Weights relative to the newest game in the window."""
        weights = dc2.gradual_acceleration_with_floor(self.DAYS, **dc2.DECAY_PRESETS[preset])
        return weights / weights[0]

    def test_gradual_sits_between_flat_and_the_production_curve(self):
        """The point of it: a real taper, but nothing like as harsh as
        'weighted', which has a September game at its 0.05 floor."""
        for age, mild, harsh in zip(self.DAYS, self.curve('gradual'), self.curve('weighted')):
            self.assertGreaterEqual(mild, harsh, f'gradual is steeper than weighted at {age:.0f} days')
            self.assertLessEqual(mild, 1.0)

    def test_gradual_actually_tapers(self):
        """Distinct from 'mean' -- a preset that decays by nothing is one
        we already have."""
        gradual = self.curve('gradual')
        self.assertLess(gradual[-1], 0.6, 'old games are barely discounted')
        self.assertGreater(gradual[-1], 0.2, 'old games are discounted to nothing')
        self.assertTrue(all(a >= b for a, b in zip(gradual, gradual[1:])), 'not monotonic')

    def test_pure_importance_ignores_age(self):
        """Its whole premise: leverage is the only thing weighting a game.
        A flat curve underneath means the leverage multiplier IS the
        weight."""
        self.assertTrue(np.allclose(self.curve('pure_importance'), 1.0),
                        'pure_importance still decays with age')
        self.assertIn('pure_importance', dc2.IMPORTANCE_WEIGHT,
                      'without this the backtester never computes leverage for it')
        self.assertNotIn('pure_importance', dc2.PRIOR_SEASON_WEIGHT,
                         'a season boundary discount is what this preset is testing without')

    def test_it_is_gentler_than_the_combined_importance_preset(self):
        """'importance' floors a dead rubber at 5%; on its own, leverage
        should not erase a game that quietly."""
        self.assertGreater(dc2.IMPORTANCE_WEIGHT['pure_importance']['floor'],
                           dc2.IMPORTANCE_WEIGHT['importance']['floor'])

    def test_mean_is_flat_and_needed_no_new_preset(self):
        """The simple baseline already existed -- 'mean' weights every game
        in the window alike, handled before the decay curves."""
        self.assertNotIn('mean', dc2.DECAY_PRESETS)
        ingredients = pd.DataFrame(
            {'_run_yards': [100., 300.], '_run_plays': [20., 20.]},
            index=pd.MultiIndex.from_tuples([('2024_01_KC_NO', 'KC'), ('2024_02_KC_NO', 'KC')],
                                            names=['game_id', 'team']))
        pooled = dc2._pool_ingredients(ingredients, None, 'mean')
        rate = (pooled._run_yards / pooled._run_plays).iloc[0]
        self.assertAlmostEqual(float(rate), 10.0,
                               msg='a 5.0 and a 15.0 game should average 10.0 when weighted alike')


class BacktesterAcceptsThem(unittest.TestCase):
    def test_mean_and_median_are_selectable(self):
        """They are not decay curves, so they are not in DECAY_PRESETS --
        but calc_stats has always taken them and the flat one is the
        baseline the curves are judged against."""
        import backtester
        source = Path('backtester.py').read_text(encoding='utf-8')
        self.assertIn("'mean', 'median', *dc.DECAY_PRESETS", source,
                      'the flat and median modes are not offered on the command line')
        self.assertIn('calculation', backtester.SWEEPABLE)


if __name__ == '__main__':
    unittest.main()
