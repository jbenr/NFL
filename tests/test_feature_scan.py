import unittest
import numpy as np
import pandas as pd
from feature_scan_data import pregame_context, rolling_snapshots, METRICS, candidate_catalog
from feature_scan import chronological_folds, build_variants, paired_improvement


def schedule():
    return pd.DataFrame([
        dict(season=2024, week=1, game_type='REG', away_team='BUF', home_team='NE',
             away_score=21., home_score=10., location='Home', away_rest=7, home_rest=7),
        dict(season=2024, week=2, game_type='REG', away_team='MIA', home_team='NE',
             away_score=14., home_score=14., location='Neutral', away_rest=7, home_rest=7),
        dict(season=2024, week=3, game_type='REG', away_team='BUF', home_team='NE',
             away_score=17., home_score=20., location='Home', away_rest=14, home_rest=7),
    ])


class FeatureScanTests(unittest.TestCase):
    def test_context_uses_only_earlier_weeks_and_keeps_bye_records(self):
        s = schedule()
        actual = pregame_context(s)
        changed = s.copy()
        changed.loc[changed.week >= 2, ['away_score', 'home_score']] = [99., 0.]
        revised = pregame_context(changed)
        pd.testing.assert_frame_equal(actual[actual.week <= 2], revised[revised.week <= 2])
        self.assertEqual(actual.loc[2, 'away_wins'], 1)
        self.assertEqual(actual.loc[2, 'away_played'], 1)
        self.assertEqual(actual.loc[2, 'home_ties'], 1)
        self.assertEqual(actual.loc[1, 'home_field_adv'], 0)
        self.assertEqual(actual.loc[2, 'rest_days_diff'], 7)

    def test_new_season_resets_records(self):
        s = schedule()
        result = pregame_context(pd.concat([s, s.iloc[[0]].assign(season=2025)]))
        self.assertEqual(result.iloc[-1].away_wins, 0)
        self.assertEqual(result.iloc[-1].home_played, 0)

    def test_rolling_rates_pool_opportunities_and_exclude_current_game(self):
        rows = []
        for week, n, d in [(1, 1., 2.), (2, 9., 10.), (3, 999., 1000.)]:
            row = dict(season=2024, week=week, game_id=str(week), team='BUF', side='off')
            for metric in METRICS:
                row[f'{metric}_num'], row[f'{metric}_den'] = n, d
            rows.append(row)
        snap = rolling_snapshots(pd.DataFrame(rows), [(2024, 4)], lookback_games=2)
        self.assertAlmostEqual(snap[snap.week == 3].iloc[0].pass_epa, 10 / 12)
        self.assertEqual(snap[snap.week == 3].iloc[0].history_games, 2)

    def test_all_feature_trials_protect_home_control(self):
        variants, tests = build_variants(candidate_catalog())
        self.assertGreater(len(tests), 50)
        for name, features in variants.items():
            if name != 'without_home_control':
                self.assertIn('home_field_adv', features)
        self.assertEqual(len(variants['full']), len(set(variants['full'])))

    def test_folds_never_train_on_future_or_reserved_season(self):
        data = pd.DataFrame({'season': np.repeat([2021, 2022, 2023, 2024, 2025], 220),
                             'week': np.tile(np.arange(220) // 16 + 1, 5)})
        for year, train, val in chronological_folds(data, 2025):
            self.assertLess(data.iloc[train].season.max(), data.iloc[val].season.min())
            self.assertLess(year, 2025)

    def test_paired_gain_sign_and_alignment(self):
        ref = pd.DataFrame({'season': [2022]*4, 'week': [1, 1, 2, 2],
                            'away_team': ['BUF', 'MIA', 'BUF', 'MIA'], 'home_team': ['NE']*4,
                            'abs_error': [3.]*4})
        trial = ref.copy()
        trial['abs_error'] = 2.
        gain = paired_improvement(ref, trial)
        self.assertEqual(gain['gain'], 1.)
        self.assertEqual(gain['low'], 1.)
        with self.assertRaises(ValueError):
            paired_improvement(ref, trial.iloc[::-1])


if __name__ == '__main__':
    unittest.main()
