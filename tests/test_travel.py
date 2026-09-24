import unittest

import numpy as np
import pandas as pd

import data_crunchski_3 as dc
import travel


def schedule():
    """Two seasons: a normal home slate, a London neutral-site game, and a
    season where one team is displaced to a second venue."""
    rows = [
        # season, week, away, home, stadium, location
        (2023, 1, 'SEA', 'MIA', 'MIA00', 'Home'),
        (2023, 2, 'MIA', 'SEA', 'SEA00', 'Home'),
        (2023, 3, 'SEA', 'BUF', 'LON00', 'Neutral'),
        (2023, 4, 'BUF', 'MIA', 'MIA00', 'Home'),
        (2023, 5, 'MIA', 'BUF', 'BUF00', 'Home'),
        (2023, 6, 'SEA', 'BUF', 'BUF00', 'Home'),
        # BUF hosts twice at home and once in Toronto -- displaced game, not a
        # change of base.
        (2023, 7, 'MIA', 'BUF', 'BUF01', 'Home'),
        (2024, 1, 'BUF', 'SEA', 'SEA00', 'Home'),
    ]
    frame = pd.DataFrame(rows, columns=['season', 'week', 'away_team', 'home_team', 'stadium_id', 'location'])
    frame['game_id'] = [f'{r.season}_{r.week:02d}_{r.away_team}_{r.home_team}' for r in frame.itertuples()]
    return frame


def venues():
    return pd.DataFrame([('MIA00', 25.957962, -80.238842), ('SEA00', 47.595151, -122.331626),
                         ('BUF00', 42.773753, -78.786954), ('BUF01', 43.641659, -79.389089),
                         ('LON00', 51.555984, -0.279590)],
                        columns=['stadium_id', 'latitude', 'longitude'])


class DistanceTests(unittest.TestCase):
    def test_haversine_matches_known_distances(self):
        # Seattle to Miami is about 2,730 miles; the same point is zero.
        self.assertAlmostEqual(travel.haversine_miles(47.595151, -122.331626, 25.957962, -80.238842), 2733, delta=15)
        self.assertEqual(travel.haversine_miles(42.773753, -78.786954, 42.773753, -78.786954), 0.)

    def test_longitude_sign_matters(self):
        """The bug this module would otherwise inherit: San Diego at +117
        longitude is in China, eight thousand miles from the real thing."""
        real = travel.haversine_miles(25.957962, -80.238842, 32.783219, -117.119592)
        flipped = travel.haversine_miles(25.957962, -80.238842, 32.783219, 117.119592)
        self.assertAlmostEqual(real, 2265, delta=25)
        self.assertGreater(flipped, 8000)


class HomeVenueTests(unittest.TestCase):
    def test_modal_home_venue_survives_a_displaced_game(self):
        homes = travel.home_venues(schedule())
        self.assertEqual(homes[(2023, 'BUF')], 'BUF00')   # not the Toronto one-off
        self.assertEqual(homes[(2023, 'MIA')], 'MIA00')

    def test_neutral_sites_never_become_a_home_venue(self):
        homes = travel.home_venues(schedule())
        self.assertEqual(homes[(2023, 'SEA')], 'SEA00')   # despite "hosting" nothing but Seattle

    def test_a_season_with_no_home_games_inherits_the_last_known_venue(self):
        homes = travel.home_venues(schedule())
        self.assertEqual(homes[(2024, 'BUF')], 'BUF00')   # BUF hosts nothing in 2024


class GameTravelTests(unittest.TestCase):
    def setUp(self):
        self.travel = travel.game_travel(schedule(), venues()).set_index('game_id')

    def test_home_team_travels_nothing(self):
        row = self.travel.loc['2023_01_SEA_MIA']
        self.assertEqual(row.home_travel_miles, 0.)
        self.assertAlmostEqual(row.away_travel_miles, 2733, delta=15)
        self.assertAlmostEqual(row.away_travel_adv, row.away_travel_miles)

    def test_both_teams_travel_to_a_neutral_site(self):
        row = self.travel.loc['2023_03_SEA_BUF']
        self.assertGreater(row.home_travel_miles, 3000)   # Buffalo to London
        self.assertGreater(row.away_travel_miles, 4000)   # Seattle to London
        self.assertAlmostEqual(row.away_travel_adv, row.away_travel_miles - row.home_travel_miles)

    def test_a_displaced_home_game_gives_the_home_team_travel_too(self):
        row = self.travel.loc['2023_07_MIA_BUF']       # Buffalo "hosting" in Toronto
        self.assertAlmostEqual(row.home_travel_miles, 67, delta=3)
        self.assertLess(row.away_travel_adv, row.away_travel_miles)

    def test_attach_travel_raises_rather_than_filling_a_gap(self):
        panel = schedule()[['game_id']].copy()
        attached = travel.attach_travel(panel, schedule(), venues())
        self.assertEqual(list(attached.columns), ['game_id'] + travel.COLUMNS)
        with self.assertRaises(ValueError) as caught:
            travel.attach_travel(pd.concat([panel, pd.DataFrame({'game_id': ['2023_09_SEA_LAC']})]),
                                 schedule(), venues())
        self.assertIn('2023_09_SEA_LAC', str(caught.exception))


class ModelInputTests(unittest.TestCase):
    """travel_advantage as the model sees it -- see tests/test_two_sided.py
    for the rest of the two-sided row construction."""

    def tearDown(self):
        dc.use_travel(False)

    def test_the_input_appears_only_when_the_version_asks_for_it(self):
        from tests.test_two_sided import fixture
        data = fixture()
        data['away_travel_miles'], data['home_travel_miles'] = 1500., 0.
        self.assertNotIn(dc.TRAVEL_CONTEXT, dc.two_sided_rows(data).columns)
        dc.use_travel(True)
        self.assertIn(dc.TRAVEL_CONTEXT, dc.two_sided_rows(data).columns)

    def test_the_two_rows_carry_equal_and_opposite_thousands_of_miles(self):
        from tests.test_two_sided import fixture
        dc.use_travel(True)
        data = fixture()
        data['away_travel_miles'], data['home_travel_miles'] = 1500., 200.
        rows, n = dc.two_sided_rows(data), len(data)
        np.testing.assert_allclose(rows[dc.TRAVEL_CONTEXT].iloc[:n], 1.3)
        np.testing.assert_allclose(rows[dc.TRAVEL_CONTEXT].iloc[n:], -1.3)

    def test_use_travel_toggles_the_shared_context_list_in_place(self):
        shared = dc.BASE_CONTEXT
        self.assertEqual(dc.use_travel(True), ['home_field', 'rest_advantage', 'travel_advantage'])
        self.assertIs(shared, dc.BASE_CONTEXT)           # importers see the change
        self.assertEqual(dc.use_travel(True), dc.use_travel(True))   # idempotent
        self.assertEqual(dc.use_travel(False), ['home_field', 'rest_advantage'])


class VersionTests(unittest.TestCase):
    def tearDown(self):
        import model_spec
        model_spec.select('2.0')

    def test_selecting_versions_swaps_the_input_set_both_ways(self):
        import model_spec
        model_spec.select('2.2')
        self.assertIn(dc.TRAVEL_CONTEXT, dc.BASE_CONTEXT)
        self.assertNotIn('pass_epa_pp', dc.METRICS)      # 2.2 is built on 2.0, not 2.1
        self.assertEqual(model_spec.RESULTS.name, 'model_2.2')
        model_spec.select('2.1')
        self.assertNotIn(dc.TRAVEL_CONTEXT, dc.BASE_CONTEXT)
        self.assertIn('pass_epa_pp', dc.METRICS)


if __name__ == '__main__':
    unittest.main()
