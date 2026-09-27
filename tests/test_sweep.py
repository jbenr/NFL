import argparse
import unittest

import backtester


def namespace(**overrides):
    base = dict(model='two-sided', model_version='model_2.0', calculation=None,
                lookback=20, train_window=100, inputs='differential', normalize='raw',
                start_season=2013, season=2025, week=22, output=None)
    base.update(overrides)
    return argparse.Namespace(**base)


class Combinations(unittest.TestCase):
    """--calculation a b c should run three backtests, not reject the list."""

    def test_a_plain_command_is_still_one_run(self):
        runs = backtester.combinations(namespace())
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].calculation, None)

    def test_one_list_gives_one_run_per_value(self):
        runs = backtester.combinations(namespace(calculation=['mean', 'gradual', 'pure_importance']))
        self.assertEqual([r.calculation for r in runs], ['mean', 'gradual', 'pure_importance'])

    def test_two_lists_give_every_pairing(self):
        runs = backtester.combinations(namespace(calculation=['mean', 'gradual'],
                                                 model_version=['model_2.0', 'model_2.1']))
        self.assertEqual(len(runs), 4)
        self.assertEqual({(r.model_version, r.calculation) for r in runs},
                         {('model_2.0', 'mean'), ('model_2.0', 'gradual'),
                          ('model_2.1', 'mean'), ('model_2.1', 'gradual')})

    def test_runs_do_not_share_state(self):
        """Each run is validated and mutated in place (start_season, output,
        groups), so they must not be the same object."""
        runs = backtester.combinations(namespace(calculation=['mean', 'gradual']))
        runs[0].output = 'somewhere'
        self.assertIsNone(runs[1].output)

    def test_every_sweepable_option_exists_on_the_parser(self):
        """A name in SWEEPABLE that argparse does not produce would silently
        never vary."""
        for name in backtester.SWEEPABLE:
            self.assertTrue(hasattr(namespace(), name), f'{name} is not a real argument')


class Labels(unittest.TestCase):
    def test_only_the_options_that_differ_are_named(self):
        runs = backtester.combinations(namespace(calculation=['mean', 'gradual']))
        label = backtester.describe(runs[0], runs)
        self.assertIn('calculation mean', label)
        self.assertNotIn('lookback', label, 'lookback is the same in every run')

    def test_a_lone_run_says_so(self):
        runs = backtester.combinations(namespace())
        self.assertEqual(backtester.describe(runs[0], runs), 'single run')

    def test_a_two_axis_sweep_names_both(self):
        runs = backtester.combinations(namespace(calculation=['mean', 'gradual'], lookback=[12, 20]))
        label = backtester.describe(runs[0], runs)
        self.assertIn('calculation', label)
        self.assertIn('lookback', label)


if __name__ == '__main__':
    unittest.main()
