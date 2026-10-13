import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data_pullson


class RefreshStartersTests(unittest.TestCase):
    def refresh(self, rooms, ids, listed):
        games = [('SEA', 'Drew Lock', '00-LOCK'), ('CHI', 'Tyson Bagent', '00-BAGENT'),
                 ('BAL', 'Lamar Jackson', '00-LAMAR')]
        sched = pd.DataFrame([dict(game_id=f'2026_06_{t}_X', season=2026, week=6, away_team=t,
                                   home_team='X', away_qb_name=n, away_qb_id=i, away_score=None)
                              for t, n, i in games]
                             + [dict(game_id='2026_07_SEA_X', season=2026, week=7, away_team='SEA',
                                     home_team='X', away_qb_name=None, away_qb_id=None, away_score=None)])
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'sched.parquet')
            sched.to_parquet(path)
            with patch.object(data_pullson, 'quarterback_status', return_value=(rooms, ids, listed)):
                return data_pullson.refresh_starters([2026], path).set_index('game_id')

    def test_week_without_a_report_only_promotes_a_qb1_absent_from_the_last_one(self):
        # The newest report is week 5; the games are week 6.
        rooms = {('SEA', 5): [('Sam Darnold', None), ('Drew Lock', None)],
                 ('CHI', 5): [('Caleb Williams', None), ('Tyson Bagent', None)],
                 ('BAL', 5): [('Lamar Jackson', 'out'), ('Tyler Huntley', None)]}
        listed = {('CHI', 5): {'Caleb Williams'}, ('BAL', 5): {'Lamar Jackson'}}
        out = self.refresh(rooms, {('SEA', 'Sam Darnold'): '00-DARNOLD'}, listed)
        self.assertEqual(out.at['2026_06_SEA_X', 'away_qb_name'], 'Sam Darnold')
        self.assertEqual(out.at['2026_06_SEA_X', 'away_qb_id'], '00-DARNOLD')
        self.assertEqual(out.at['2026_06_CHI_X', 'away_qb_name'], 'Tyson Bagent')   # still listed
        self.assertEqual(out.at['2026_06_BAL_X', 'away_qb_name'], 'Lamar Jackson')  # stale "out"
        self.assertTrue(pd.isna(out.at['2026_07_SEA_X', 'away_qb_name']))           # unnamed week


if __name__ == '__main__':
    unittest.main()
