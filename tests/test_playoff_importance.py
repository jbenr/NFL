import unittest

import numpy as np
import pandas as pd

import data_crunchski_2 as dc
import playoff_importance as pi


class PlayoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = pd.read_parquet('data/sched.parquet').replace(
            {'away_team': dc.RELOCATED_TEAMS, 'home_team': dc.RELOCATED_TEAMS})

    def test_field_sizes_and_division_champions(self):
        teams = sorted(set(dc.TEAM_META) - {'LAR'})
        n = len(teams)
        conf = np.array([dc.TEAM_META[t][0] for t in teams])
        div = np.array([dc.TEAM_META[t][1] for t in teams])
        played = np.ones((n, n)) - np.eye(n)
        points = (np.arange(n)[:, None] > np.arange(n)).astype(float)[None]
        for season, count in [(2019, 6), (2020, 7)]:
            result = pi.seeds(points, played, conf, div, np.zeros((1, n)), season)[0]
            for conference in ['AFC', 'NFC']:
                self.assertEqual(int(((result > 0) & (conf == conference)).sum()), count)
                champions = np.where((result > 0) & (result <= 4) & (conf == conference))[0]
                self.assertEqual(len(set(div[champions])), 4)

    def test_known_final_week_cases(self):
        d = pi.week_importance(self.schedule, 2023, 18, 128)
        hou = d[d.away_team.eq('HOU')].iloc[0]
        self.assertEqual(hou.away_playoff_if_win, 1)
        self.assertEqual(hou.away_playoff_if_loss, 0)
        gb = d[d.home_team.eq('GB')].iloc[0]
        self.assertEqual(gb.home_playoff_if_win, 1)
        self.assertEqual(gb.away_playoff_if_win, 0)  # Chicago already eliminated.
        bal = d[d.home_team.eq('BAL')].iloc[0]
        self.assertEqual(bal.home_importance, 0)
        self.assertEqual(bal.home_top_seed_if_loss, 1)
        car = d[d.home_team.eq('CAR')].iloc[0]
        self.assertEqual(car.home_importance, 0)

    def test_clinched_playoffs_can_still_have_bye_leverage(self):
        d = pi.week_importance(self.schedule, 2024, 18, 64)
        row = d[d.home_team.eq('DET')].iloc[0]
        for side in ['away', 'home']:
            self.assertEqual(row[side + '_playoff_if_win'], 1)
            self.assertEqual(row[side + '_playoff_if_loss'], 1)
            self.assertEqual(row[side + '_bye_swing'], 1)
            self.assertEqual(row[side + '_importance'], 1)

    def test_future_results_lines_and_row_order_do_not_change_features(self):
        original = self.schedule[self.schedule.season.eq(2023)].copy()
        first = pi.week_importance(original, 2023, 18, 32)
        changed = original.copy()
        changed.loc[changed.week.ge(18), ['away_score', 'home_score']] = [999, 0]
        changed['spread_line'] = 999
        changed = changed.sample(frac=1, random_state=1)
        second = pi.week_importance(changed, 2023, 18, 32)
        columns = ['away_team', 'home_team'] + [s + '_' + f for s in ['away', 'home'] for f in pi.FEATURES]
        pd.testing.assert_frame_equal(first[columns].sort_values('away_team').reset_index(drop=True),
                                      second[columns].sort_values('away_team').reset_index(drop=True))

    def test_postseason_is_advancement_not_qualification(self):
        d = pi.week_importance(self.schedule, 2023, 19, 16)
        self.assertTrue(d.away_advancement_swing.eq(1).all())
        self.assertTrue(d.away_playoff_swing.eq(0).all())
        self.assertTrue(d.away_importance.eq(1).all())


if __name__ == '__main__':
    unittest.main()
