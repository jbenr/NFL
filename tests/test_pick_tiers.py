import unittest

import numpy as np
import pandas as pd

import weekly_packet as wp


def settled(**overrides):
    """One settled row, as apply_tiers receives it."""
    row = dict(season=2025, week=3, market_base=-3.5, prediction=-8., edge=-4.5, variance=16.,
               total_game_importance=.5, qualifies=True)
    row.update(overrides)
    return pd.DataFrame([row])


class TierTests(unittest.TestCase):
    def test_no_tier_outside_the_validated_windows(self):
        self.assertIsNone(wp.pick_tier('spread', week=3, sd=4.))
        self.assertIsNone(wp.pick_tier('spread', week=12, sd=4.))
        self.assertIsNone(wp.pick_tier('total', week=2, line=48.))
        self.assertIsNone(wp.pick_tier('total', week=10, line=44.))   # the 42-46 dead band

    def test_the_windows_that_do_earn_a_tier(self):
        self.assertEqual(wp.pick_tier('spread', week=13, sd=4.), 'S')
        self.assertEqual(wp.pick_tier('spread', week=20, sd=4.), 'S')
        self.assertIsNone(wp.pick_tier('spread', week=13, sd=6.))     # ensemble disagrees
        self.assertEqual(wp.pick_tier('total', week=13, line=50.), 'S')   # 64.6%
        self.assertEqual(wp.pick_tier('total', week=8, line=50.), 'S')    # 60.0%
        self.assertEqual(wp.pick_tier('total', week=10, line=50.), 'A')   # 59.5%
        self.assertEqual(wp.pick_tier('total', week=16, line=50.), 'B')   # 56.3%
        self.assertEqual(wp.pick_tier('spread', week=16, sd=3.5), 'B')    # 54.9%
        self.assertIsNone(wp.pick_tier('spread', week=16, sd=4.2))        # 44.9%, no bet

    def test_a_tierless_game_is_not_a_pick_however_big_the_edge(self):
        """The catch-all tier used to make every 3+ point disagreement a
        graded pick; those measured 49.1%, so they are no bet at all now."""
        early = wp.apply_tiers(settled(week=3, edge=-9.), 'spread')
        self.assertIsNone(early.tier.iloc[0])
        self.assertFalse(bool(early.qualifies.iloc[0]))
        late = wp.apply_tiers(settled(week=13, edge=-9.), 'spread')
        self.assertEqual(late.tier.iloc[0], 'S')
        self.assertTrue(bool(late.qualifies.iloc[0]))

    def test_a_tier_cannot_rescue_a_pick_that_failed_the_edge_test(self):
        losing = wp.apply_tiers(settled(week=13, qualifies=False), 'spread')
        self.assertFalse(bool(losing.qualifies.iloc[0]))

    def test_every_bucket_is_measured_documented_and_bettable(self):
        for market, spec in wp.PICK_BUCKETS.items():
            for bucket in spec['buckets']:
                tier = wp.tier_band(bucket['rate'])
                self.assertIsNotNone(tier, f'{market}: {bucket["rule"]} is below the bottom band')
                self.assertIn(tier, wp.TIER_COLORS)
                self.assertGreaterEqual(bucket['rate'], .524)   # never ship a losing bucket
                self.assertGreaterEqual(bucket['n'], 100)       # nor one measured on a handful of games
                for field in ['rule', 'volume', 'eras', 'note']:
                    self.assertTrue(bucket[field])
            self.assertIn(market, wp.NO_PICK)          # the dash row is documented too

    def test_the_letter_follows_the_measured_rate(self):
        self.assertEqual(wp.tier_band(.64), 'S')
        self.assertEqual(wp.tier_band(.60), 'S')
        self.assertEqual(wp.tier_band(.58), 'A')
        self.assertEqual(wp.tier_band(.549), 'B')
        self.assertIsNone(wp.tier_band(.539))
        self.assertIsNone(wp.tier_band(.50))


if __name__ == '__main__':
    unittest.main()
