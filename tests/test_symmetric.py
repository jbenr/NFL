import unittest
import numpy as np
import pandas as pd
import optimize_picks as op


class SymmetricTests(unittest.TestCase):
    def test_removes_only_away_usage_and_does_not_mutate(self):
        data = pd.DataFrame({'away_off_pass_ypp': [.44], 'away_raw_off_pass_%': [.6],
                             'away_def_pass_ypp': [-.2], 'away_off_run_ypp': [.27],
                             'away_raw_off_run_%': [.4], 'home_field_adv': [1]})
        original = data.copy()
        actual = op.symmetric_features(data)
        self.assertAlmostEqual(actual.away_off_pass_ypp.iloc[0], .4)
        self.assertAlmostEqual(actual.away_off_run_ypp.iloc[0], .3)
        self.assertEqual(actual.away_def_pass_ypp.iloc[0], -.2)
        pd.testing.assert_frame_equal(data, original)

    def test_swapping_teams_swaps_and_negates_matchups(self):
        # Same metric: away offense .8 - home defense .4; away defense .3 - home offense .7.
        original = pd.DataFrame({'away_off_pass_ypp': [.4 * 1.1],
                                 'away_def_pass_ypp': [-.4], 'away_raw_off_pass_%': [.6]})
        swapped = pd.DataFrame({'away_off_pass_ypp': [.4 * 1.2],
                                'away_def_pass_ypp': [-.4], 'away_raw_off_pass_%': [.7]})
        a, b = op.symmetric_features(original), op.symmetric_features(swapped)
        self.assertAlmostEqual(a.away_off_pass_ypp.iloc[0], -b.away_def_pass_ypp.iloc[0])
        self.assertAlmostEqual(a.away_def_pass_ypp.iloc[0], -b.away_off_pass_ypp.iloc[0])

    def test_invalid_usage_fails(self):
        data = pd.DataFrame({'away_off_pass_ypp': [1.], 'away_raw_off_pass_%': [np.nan]})
        with self.assertRaises(ValueError):
            op.symmetric_features(data)


if __name__ == '__main__':
    unittest.main()
