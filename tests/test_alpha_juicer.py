import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import alpha_juicer as aj


def frame(n=900, hit_rate=.5, seed=7, pushes=0):
    """A synthetic graded backtest: seasons 2010-2025, weeks 1-22."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(dict(
        season=rng.integers(2010, 2026, n),
        week=rng.integers(1, 23, n),
        abs_edge=rng.uniform(0, 12, n),
        sd=rng.uniform(3, 6, n),
        leverage=rng.uniform(0, 1, n),
        away_importance=rng.uniform(0, 1, n),
        home_importance=rng.uniform(0, 1, n),
        market_base=rng.uniform(38, 54, n),
    ))
    data['edge'] = data.abs_edge * rng.choice([-1, 1], n)
    data['leverage_gap'] = (data.away_importance - data.home_importance).abs()
    data['picked_leverage'] = np.where(data.edge > 0, data.away_importance, data.home_importance)
    data['opponent_leverage'] = np.where(data.edge > 0, data.home_importance, data.away_importance)
    data['line_magnitude'] = data.market_base
    data['win'] = rng.random(n) < hit_rate
    data['push'] = np.r_[np.ones(pushes, bool), np.zeros(n - pushes, bool)]
    data.loc[data.push, 'win'] = False
    return data


ARGS = SimpleNamespace(min_n=100, cross_min_n=60, floor=.54, era_floor=.50, era_min_n=25,
                       half_floor=.52, rank='worst', draws=50)


class RecordTests(unittest.TestCase):
    def test_pushes_are_not_graded_and_cost_nothing(self):
        data = frame(n=200, hit_rate=1., pushes=20)
        n, rate, units = aj.record(np.ones(len(data), bool), data)
        self.assertEqual(n, 180)
        self.assertEqual(rate, 1.)
        self.assertAlmostEqual(units, 180 * (100 / 110))

    def test_a_losing_rule_is_priced_at_one_unit_a_bet(self):
        data = frame(n=200, hit_rate=0., seed=3)
        _, rate, units = aj.record(np.ones(len(data), bool), data)
        self.assertEqual((rate, units), (0., -200))


class BandTests(unittest.TestCase):
    def test_bands_match_the_packet(self):
        self.assertEqual((aj.band(.64), aj.band(.58), aj.band(.55)), ('S', 'A', 'B'))
        self.assertIsNone(aj.band(.53))


class SpaceTests(unittest.TestCase):
    def test_every_dimension_can_be_left_out(self):
        """The 'any' option is what makes subsets searchable -- differential
        and leverage with no SD condition has to be a candidate."""
        for market in ['spread', 'total']:
            for name, options in aj.dimensions(market).items():
                self.assertTrue(any(label.startswith('any') for label in options),
                                f'{market}/{name} has no opt-out')

    def test_the_search_is_the_full_product(self):
        space = aj.dimensions('total')
        expected = np.prod([len(options) for options in space.values()])
        self.assertEqual(len(list(aj.combos('total'))), expected)

    def test_a_week_restriction_shrinks_the_search(self):
        self.assertLess(len(list(aj.combos('spread', (1, 12)))), len(list(aj.combos('spread'))))

    def test_a_label_names_only_the_conditions_that_bite(self):
        combo = (('week', 'any week'), ('edge', 'edge>=5'), ('sd', 'any SD'), ('leverage', 'leverage>=0.7'))
        self.assertEqual(aj.label_of(combo), 'edge>=5 · leverage>=0.7')
        self.assertEqual(aj.label_of((('week', 'any week'), ('edge', 'any edge'),
                                      ('sd', 'any SD'), ('leverage', 'any leverage'))), 'everything')


class BuildTests(unittest.TestCase):
    def test_a_combo_is_the_intersection_of_its_conditions(self):
        data = frame()
        cache = aj.precompute({'run': data}, 'total')['run']
        combo = (('week', 'wk9-12'), ('edge', 'edge>=5'), ('sd', 'SD<=p25'), ('leverage', 'any leverage'))
        expected = (data.week.between(9, 12) & (data.abs_edge >= 5)
                    & (data.sd <= data.sd.quantile(.25))).to_numpy()
        np.testing.assert_array_equal(aj.build(combo, cache), expected)

    def test_bands_come_from_the_sample_being_searched(self):
        """A ladder written for one model is blind on another: the two-sided
        runs sit near SD 4.2, the shared run near 1.65."""
        wide, tight = frame(seed=4), frame(seed=5)
        tight['sd'] = tight.sd / 3          # a model on a different scale
        tight['abs_edge'] = tight.abs_edge * 2
        cuts = aj.cut_points(tight.abs_edge, aj.EDGE_QUANTILES, .5)
        self.assertGreater(max(cuts), max(aj.cut_points(wide.abs_edge, aj.EDGE_QUANTILES, .5)))
        cache = aj.precompute({'wide': wide, 'tight': tight}, 'spread', sample=wide)
        combo = (('week', 'any week'), ('edge', 'any edge'),
                 ('sd', [k for k in aj.dimensions('spread', sample=wide)['sd'] if k.startswith('SD<=p25')][0]),
                 ('leverage', 'any leverage'))
        # Same label, each run resolving it against its own distribution.
        for name, data in [('tight', tight), ('wide', wide)]:
            selected = data[aj.build(combo, cache[name])].sd
            self.assertLessEqual(selected.max(), data.sd.quantile(.25))
            self.assertAlmostEqual(selected.max(), data.sd.quantile(.25), delta=.05)
            self.assertAlmostEqual(len(selected) / len(data), .25, delta=.02)
        self.assertLess(tight[aj.build(combo, cache['tight'])].sd.max(),
                        wide[aj.build(combo, cache['wide'])].sd.max())

    def test_the_same_combo_rebuilds_on_another_run(self):
        """How cross-model scoring works: one combo, every frame."""
        frames = {'a': frame(seed=1), 'b': frame(seed=2)}
        cache = aj.precompute(frames, 'spread')
        combo = (('week', 'wk13+'), ('edge', 'edge>=3'), ('sd', 'any SD'), ('leverage', 'leverage>=0.7'))
        self.assertIn('edge>=3', aj.dimensions('spread')['edge'])
        for run, data in frames.items():
            expected = ((data.week >= 13) & (data.abs_edge >= 3) & (data.leverage >= .7)).to_numpy()
            np.testing.assert_array_equal(aj.build(combo, cache[run]), expected)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.combo = (('week', 'any week'), ('edge', 'any edge'), ('sd', 'any SD'), ('leverage', 'any leverage'))
        self.good = frame(n=1200, hit_rate=.60, seed=11)

    def evaluate(self, frames, primary='primary', args=ARGS):
        return aj.evaluate(self.combo, frames, aj.precompute(frames, 'spread'), primary, 'spread', args)

    def test_a_profitable_consistent_rule_passes(self):
        result = self.evaluate({'primary': self.good})
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['rate'], .54)
        self.assertIn(result['tier'], ['S', 'A', 'B'])

    def test_a_losing_rule_is_refused(self):
        self.assertIsNone(self.evaluate({'primary': frame(n=1200, hit_rate=.50, seed=5)}))

    def test_a_model_that_disagrees_kills_it(self):
        """The consistency requirement: ranked by the worst model, not the best."""
        frames = {'primary': self.good, 'other': frame(n=1200, hit_rate=.44, seed=12)}
        self.assertIsNone(self.evaluate(frames))

    def test_a_model_with_too_few_picks_gets_no_vote(self):
        thin = frame(n=40, hit_rate=.30, seed=13)
        self.assertIsNotNone(self.evaluate({'primary': self.good, 'thin': thin}))

    def test_ranking_by_the_primary_run_ignores_the_others(self):
        frames = {'primary': self.good, 'other': frame(n=1200, hit_rate=.44, seed=12)}
        loose = SimpleNamespace(**{**vars(ARGS), 'rank': 'primary'})
        self.assertIsNotNone(self.evaluate(frames, args=loose))


class NullTests(unittest.TestCase):
    def test_coin_flips_still_clear_the_gates(self):
        """The reason the report leads with this: on random outcomes a search
        this size still hands back rules that pass everything."""
        data = frame(n=1500, hit_rate=.5, seed=21)
        masks = [aj.build(c, aj.precompute({'r': data}, 'spread')['r'])
                 for c in list(aj.combos('spread'))[:400]]
        best, survivors = aj.noise_floor(data, masks, ARGS, draws=40, seed=3)
        self.assertGreater(np.median(best), .54)
        self.assertGreaterEqual(np.median(survivors), 0)

    def test_thin_cells_never_count_as_survivors(self):
        data = frame(n=300, hit_rate=.5, seed=23)
        masks = [np.r_[np.ones(20, bool), np.zeros(len(data) - 20, bool)]]
        _, survivors = aj.noise_floor(data, masks, ARGS, draws=20, seed=4)
        self.assertEqual(survivors.max(), 0)


class FoldTests(unittest.TestCase):
    def test_near_duplicates_collapse_to_one_family(self):
        base = np.zeros(500, bool)
        base[:200] = True
        nearly = base.copy()
        nearly[195:200] = False
        separate = np.zeros(500, bool)
        separate[300:450] = True
        results = [dict(mask=m, rate=.6) for m in [base, nearly, separate]]
        self.assertEqual(len(aj.fold(results)), 2)


if __name__ == '__main__':
    unittest.main()
