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
        self.assertIsNone(wp.pick_tier('spread', week=3, importance=.9))    # leverage means nothing early
        self.assertIsNone(wp.pick_tier('spread', week=12, importance=.9))
        self.assertIsNone(wp.pick_tier('spread', week=14, importance=.4))   # late, but nothing riding on it
        self.assertIsNone(wp.pick_tier('spread', week=14, importance=None))
        self.assertIsNone(wp.pick_tier('total', week=2, line=48.))
        self.assertIsNone(wp.pick_tier('total', week=10, line=44.))   # the 42-46 dead band

    def test_the_windows_that_do_earn_a_tier(self):
        self.assertEqual(wp.pick_tier('spread', week=13, importance=.75), 'S')
        self.assertEqual(wp.pick_tier('spread', week=17, importance=.9), 'S')   # covers weeks 15-18
        self.assertEqual(wp.pick_tier('spread', week=20, importance=1.), 'S')
        self.assertEqual(wp.pick_tier('total', week=13, line=50.), 'S')   # 64.6%
        self.assertEqual(wp.pick_tier('total', week=8, line=50.), 'S')    # 60.0%
        self.assertEqual(wp.pick_tier('total', week=10, sd=3.5, line=50.), 'S')   # 68.3%, ensemble agrees
        self.assertEqual(wp.pick_tier('total', week=10, sd=5., line=50.), 'B')    # 55.2%, it does not
        self.assertEqual(wp.pick_tier('total', week=16, sd=5., line=50.), 'B')

    def test_a_tierless_game_is_not_a_pick_however_big_the_edge(self):
        """A catch-all tier used to make every qualifying disagreement a
        graded pick; those measured 49.1%, so they are no bet at all now."""
        early = wp.apply_tiers(settled(week=3, edge=-9., total_game_importance=.95), 'spread')
        self.assertIsNone(early.tier.iloc[0])
        self.assertFalse(bool(early.qualifies.iloc[0]))
        late = wp.apply_tiers(settled(week=13, edge=-9., total_game_importance=.95), 'spread')
        self.assertEqual(late.tier.iloc[0], 'S')
        self.assertTrue(bool(late.qualifies.iloc[0]))

    def test_a_late_game_nobody_needs_to_win_is_not_a_pick(self):
        dead = wp.apply_tiers(settled(week=17, edge=-9., total_game_importance=.2), 'spread')
        self.assertIsNone(dead.tier.iloc[0])
        self.assertFalse(bool(dead.qualifies.iloc[0]))

    def test_a_tier_cannot_rescue_a_pick_that_failed_the_edge_test(self):
        losing = wp.apply_tiers(settled(week=13, qualifies=False, total_game_importance=.95), 'spread')
        self.assertFalse(bool(losing.qualifies.iloc[0]))

    def test_every_bucket_is_measured_documented_and_bettable(self):
        for market, spec in wp.PICK_BUCKETS.items():
            for bucket in spec['buckets']:
                tier = wp.tier_band(bucket['rate'])
                self.assertIsNotNone(tier, f'{market}: {bucket["rule"]} is below the bottom band')
                self.assertIn(tier, wp.TIER_COLORS)
                self.assertGreaterEqual(bucket['rate'], .524)   # never ship a losing bucket
                self.assertGreaterEqual(bucket['n'], 100)       # nor one measured on a handful of games
                for field in ['rule', 'model', 'worth', 'eras', 'siblings', 'note']:
                    self.assertTrue(bucket[field])
            self.assertIn(market, wp.NO_PICK)          # the dash row is documented too

    def test_the_guide_warns_when_the_packet_ran_a_different_model(self):
        """Rates are model-specific -- the spread S bucket alone runs 60.0% on
        the source model and 49.4% on 2.1-importance -- so a packet built with
        something else must not present them as its own."""
        self.assertNotIn('tier-warn', wp.tier_guide(wp.TIER_MODEL))
        self.assertNotIn('tier-warn', wp.tier_guide(None))      # unknown: no claim either way
        warned = wp.tier_guide('Model 2.1 · importance')
        self.assertIn('tier-warn', warned)
        self.assertIn('Model 2.1 · importance', warned)

    def test_every_bucket_names_the_run_it_was_measured_on(self):
        for market, spec in wp.PICK_BUCKETS.items():
            for bucket in spec['buckets']:
                self.assertIn(bucket['model'], wp.tier_guide())
            # One list, one model: the no-pick row has to describe the same
            # run its buckets were measured on, or the guide contradicts itself.
            owners = {bucket['model'] for bucket in spec['buckets']}
            self.assertEqual(len(owners), 1, f'{market} mixes models: {owners}')
            self.assertEqual(wp.NO_PICK[market]['model'], owners.pop())

    def test_a_bucket_never_fires_on_another_model(self):
        """The weeks 1-12 band is 54.9% on the shared model and 47-49% on
        every two-sided run, so applying it to the wrong one is a leak."""
        shared_pick = dict(market='spread', week=6, sd=1.5, line=-3., edge=4.)
        self.assertEqual(wp.pick_tier(**shared_pick, running=wp.SHARED_MODEL), 'B')
        self.assertIsNone(wp.pick_tier(**shared_pick, running=wp.TIER_MODEL))
        self.assertIsNone(wp.pick_tier(**shared_pick))          # default is the packet's model
        # ...and the two-sided buckets stay put.
        self.assertEqual(wp.pick_tier('spread', 14, importance=.8, sd=4., line=-3., edge=4.), 'S')
        self.assertIsNone(wp.pick_tier('spread', 14, importance=.8, sd=4., line=-3., edge=4.,
                                       running=wp.SHARED_MODEL))

    def test_the_letter_follows_the_measured_rate(self):
        self.assertEqual(wp.tier_band(.64), 'S')
        self.assertEqual(wp.tier_band(.60), 'S')
        self.assertEqual(wp.tier_band(.58), 'A')
        self.assertEqual(wp.tier_band(.549), 'B')
        self.assertIsNone(wp.tier_band(.539))
        self.assertIsNone(wp.tier_band(.50))


class ProfileTests(unittest.TestCase):
    """Lookback, training window and stat preset belong to the model, not to
    the command line -- a bucket's hit rate was measured with specific ones."""

    def test_every_bucket_names_a_model_the_packet_knows_how_to_run(self):
        import model_spec
        for market, spec in wp.PICK_BUCKETS.items():
            for bucket in spec['buckets']:
                self.assertIn(bucket['model'], model_spec.PROFILES,
                              f'{market}: no profile for {bucket["model"]}, so the packet cannot reproduce it')

    def test_a_profile_carries_the_whole_recipe(self):
        import model_spec
        for name in model_spec.PROFILES:
            settings = model_spec.profile(name)
            for field in ['version', 'architecture', 'lookback', 'train_window', 'calculation']:
                self.assertIsNotNone(settings[field], f'{name} is missing {field}')

    def test_the_season_is_split_between_profiles_not_shared(self):
        import model_spec
        self.assertEqual(model_spec.owner('spread', 3), model_spec.SHARED_PROFILE)
        self.assertEqual(model_spec.owner('spread', 12), model_spec.SHARED_PROFILE)
        self.assertEqual(model_spec.owner('spread', 13), model_spec.DEFAULT_PROFILE)
        self.assertEqual(model_spec.owner('total', 3), model_spec.DEFAULT_PROFILE)
        # and the bucket that fires has to belong to the profile that owns the week
        for week in [3, 13]:
            owner = model_spec.owner('spread', week)
            bucket = wp.pick_bucket('spread', week, importance=.8, sd=2., line=-3., edge=4., running=owner)
            if bucket:
                self.assertEqual(bucket['model'], owner)

    def test_late_weeks_need_only_one_model(self):
        import model_spec
        _, order = model_spec.lineup(14)
        self.assertEqual(order, [model_spec.DEFAULT_PROFILE])
        _, early = model_spec.lineup(3)
        self.assertEqual(len(early), 2)
        self.assertEqual(early[0], model_spec.DEFAULT_PROFILE)   # owns the panel the other reuses


class PlumbingTests(unittest.TestCase):
    """The packet runs two models now, so the tier table has to follow the
    predictions, not the folder."""

    def test_write_packets_tiers_against_the_model_that_made_the_predictions(self):
        """Regression: the cards used to read {market}_config.json for the
        model, but that file is written AFTER the pages render -- so a
        shared-model packet silently graded its picks against the two-sided
        table and threw every pick away."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        seen = []
        real = wp.apply_tiers

        def spy(frame, market, running=None):
            seen.append(running)
            return real(frame, market, running)

        data = pd.DataFrame(dict(
            season=[2026], week=[3], away_team=['ATL'], home_team=['GB'], market='spread',
            market_base=[-3.], prediction=[-7.5], edge=[-4.5], variance=[4.], actual=[np.nan],
            residual=[np.nan], away_points=[20.], home_points=[27.5], baseline=[0.],
            integration_residual=[0.], qualifies=[True], win=[False], push=[False], pnl=[np.nan],
            odds=[-110.], assumed_odds=[False], positive_odds=[-110.], negative_odds=[-110.],
            attr_away_off_run_ypp=[1.], total_game_importance=[.5], sd=[2.]))
        config = dict(model=f'{wp.SHARED_MODEL} · 100 members', tier_model=wp.SHARED_MODEL,
                      calculation='model-2.1', feature_calculation='weighted', lookback=20,
                      market='spread', status='PASS', reason='test')
        importance = pd.DataFrame(dict(feature=['own_off_run_ypp'], importance=[1.], std=[.1]))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(wp, 'display_stats', return_value=pd.DataFrame()), \
                patch.object(wp, 'apply_tiers', side_effect=spy):
            wp.write_packets(data, data, importance, config, Path(folder))
        self.assertIn(wp.SHARED_MODEL, seen)
        self.assertNotIn(wp.TIER_MODEL, seen)


if __name__ == '__main__':
    unittest.main()
