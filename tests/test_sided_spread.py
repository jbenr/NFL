import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import data_crunchski_3 as dc
import shared_scoring as ss
from tests.test_two_sided import fixture


class ObjectiveTests(unittest.TestCase):
    """sided-spread: each row is asked for the margin from its own side, and
    the two estimates are averaged rather than subtracted."""

    def test_the_two_rows_get_equal_and_opposite_margins(self):
        data = fixture()
        played = data[data.away_score.notna() & data.home_score.notna()]
        _, _, y, _, offset = dc.prepare_two_sided(data, 2026, 1, objective='margin')
        n = len(y) // 2
        np.testing.assert_allclose(y[:n], -y[n:], atol=1e-5)
        margin = (played.away_score - played.home_score).to_numpy()[:n]
        np.testing.assert_allclose(y[:n] + offset, margin, atol=1e-4)

    def test_the_points_objective_is_untouched(self):
        data = fixture()
        _, _, y, _, offset = dc.prepare_two_sided(data, 2026, 1)
        played = data[data.away_score.notna() & data.home_score.notna()]
        n = len(y) // 2
        np.testing.assert_allclose(y[:n] + offset, played.away_score.to_numpy()[:n], atol=1e-4)

    def test_an_unknown_objective_is_refused(self):
        with self.assertRaises(ValueError):
            dc.prepare_two_sided(fixture(), 2026, 1, objective='spread')

    def test_a_margin_centred_target_has_no_offset(self):
        """[margin, -margin] averages to zero, so nothing is recentred."""
        _, _, _, _, offset = dc.prepare_two_sided(fixture(), 2026, 1, objective='margin')
        self.assertAlmostEqual(offset, 0., places=4)


class CombinationTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        import tempfile
        from pathlib import Path
        import utils
        self.folder = tempfile.TemporaryDirectory()

        def keyed(name, identity, sources=()):
            # A real cache keyed on identity, not one fixed path -- otherwise
            # two different fits would read each other's file and the test
            # below could never fail.
            digest = hashlib.sha256(repr((name, identity)).encode()).hexdigest()[:16]
            return Path(self.folder.name) / f'{digest}.parquet'

        self.cache = patch.object(utils, 'cache_path', side_effect=keyed)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.addCleanup(self.folder.cleanup)

    def test_averaging_not_subtracting_and_no_total(self):
        details, _ = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1,
                                      objective='margin')
        self.assertTrue(details.total_prediction.isna().all(), 'a margin model cannot forecast a total')
        self.assertTrue(details.away_points.isna().all(), 'neither row is a score')
        self.assertTrue(np.isfinite(details.prediction).all())
        # completeness still holds: prediction - baseline == sum of attributions
        self.assertTrue((details.integration_residual.abs() < .05).all())

    def test_the_points_objective_still_forecasts_both_markets(self):
        details, _ = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1)
        for column in ['prediction', 'total_prediction', 'away_points', 'home_points']:
            self.assertTrue(np.isfinite(details[column]).all(), column)
        self.assertTrue((details.total_integration_residual.abs() < .05).all())

    def test_the_two_objectives_do_not_share_a_cache_entry(self):
        """Same games, same seed, different question -- so a cache that keyed
        only on the data would hand back the wrong model's answer."""
        points, _ = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1)
        margin, _ = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1, objective='margin')
        self.assertFalse(np.allclose(points.prediction, margin.prediction))
        self.assertTrue(np.isfinite(points.total_prediction).all())
        self.assertTrue(margin.total_prediction.isna().all())


    def test_each_side_keeps_its_own_estimate(self):
        """The average is the bet, but both views are kept: they estimate the
        same number, and how far apart they land is its own signal."""
        details, _ = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1,
                                      objective='margin')
        for column in ['spread_from_away', 'spread_from_home', 'side_gap']:
            self.assertIn(column, details)
            self.assertTrue(np.isfinite(details[column]).all(), column)
        # both are in away-minus-home terms, so they average to the prediction
        np.testing.assert_allclose((details.spread_from_away + details.spread_from_home) / 2,
                                   details.prediction, atol=1e-6)
        np.testing.assert_allclose(details.side_gap,
                                   details.spread_from_away - details.spread_from_home, atol=1e-6)

    def test_the_points_objective_has_no_side_views_to_keep(self):
        details, _ = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1)
        self.assertNotIn('spread_from_away', details)
        for column in ['away_points', 'home_points']:
            self.assertTrue(np.isfinite(details[column]).all())


if __name__ == '__main__':
    unittest.main()
