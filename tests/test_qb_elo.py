import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import data_crunchski_2 as dc


class QBEloTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for week, date in [(1, '2025-09-01'), (2, '2025-09-08')]:
            base = dict(season=2025, week=week, game_date=date, home_team='HOME',
                        passer=None, rusher=None, defteam='HOME', qb_scramble=0,
                        rushing_yards=0., incomplete_pass=0, complete_pass=0,
                        passing_yards=0., pass_touchdown=0, interception=0,
                        sack=0, rush_attempt=0, rush_touchdown=0)
            rows.extend([
                dict(base, passer='A.Starter', complete_pass=1, passing_yards=10.),
                dict(base, passer='B.Backup', complete_pass=1, passing_yards=20.),
                dict(base, rusher='B.Backup', rush_attempt=1, rushing_yards=5.),
                dict(base, passer='B.Backup', qb_scramble=1, rush_attempt=1,
                     rushing_yards=3., rush_touchdown=1),
                dict(base, rusher='R.Runner', rush_attempt=1, rushing_yards=99.),
                dict(base, passer='H.Starter', defteam='AWAY', sack=1),
            ])
        sched = pd.DataFrame([dict(season=2025, week=w, away_team='AWAY',
                                   home_team='HOME', away_qb_name='A Starter',
                                   home_qb_name='H Starter') for w in [1, 2]])
        return pd.DataFrame(rows), sched

    def test_all_qb_production_absolute_and_existing_decay(self):
        plays, sched = self.fixture()
        original = dc.gradual_acceleration_with_floor
        with patch.object(dc, 'gradual_acceleration_with_floor', wraps=original) as decay:
            offense, defense = dc.calc_qb_elo(plays, sched)
        allowed = defense.set_index('team').def_qb_elo
        # Starter 3.5 + backup passing 5.5 + designed run 1.9 + scramble TD 16.6.
        self.assertAlmostEqual(allowed['HOME'], 27.5)
        self.assertAlmostEqual(allowed['AWAY'], -8.)
        self.assertAlmostEqual(offense.set_index('name').weighted_qb_elo['A.Starter'], 3.5)
        self.assertEqual(decay.call_args_list[0].kwargs['steepness'], 3)
        self.assertEqual(decay.call_args_list[0].kwargs['floor_weight'], .05)
        self.assertEqual(decay.call_args_list[1].kwargs['steepness'], 4)
        self.assertEqual(decay.call_args_list[1].kwargs['floor_weight'], .4)

    def test_game_totals_receive_one_recency_weight(self):
        plays, sched = self.fixture()
        plays.loc[(plays.week == 2) & plays.passer.eq('B.Backup'), 'passing_yards'] += 10.
        _, defense = dc.calc_qb_elo(plays, sched)
        weights = dc.gradual_acceleration_with_floor(np.array([7, 0]),
                    total_season_days=160, steepness=3, floor_weight=.05)
        expected = np.average([27.5, 31.5], weights=weights)
        self.assertAlmostEqual(defense.set_index('team').def_qb_elo['HOME'], expected)


if __name__ == '__main__':
    unittest.main()
