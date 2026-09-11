import tempfile
import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import joint_feature_selection as fs


class SelectionTests(unittest.TestCase):
    def test_holdout_cannot_reverse_development_selection(self):
        data = pd.DataFrame([dict(season=s, week=1, game_type='REG', away_team='A',
                                 home_team='B', away_score=20., home_score=17.) for s in [2024, 2025]])
        phases = []
        def forecast(evaluator, selected, market, start, stop, members, stride=1):
            holdout = start == 2025
            phases.append(holdout)
            row = data[data.season.eq(2025 if holdout else 2024)][fs.op.KEY].copy()
            row['actual'] = 3.
            # Fewer features win development but lose the final evaluation.
            row['prediction'] = 3. + (100 - len(selected) if holdout else len(selected))
            row['sd'] = 1.
            return row
        with tempfile.TemporaryDirectory() as folder:
            args = fs.parser().parse_args(['--groups', '--markets', 'spread', '--output', folder])
            with patch.object(fs.utils, 'cache_path', return_value=Path(folder) / 'panel.parquet'), \
                 patch.object(fs.research, 'history_weeks', return_value=20), \
                 patch.object(fs.op, 'build_panel', return_value=data), \
                 patch.object(fs.js, 'context_panel', return_value=data), \
                 patch.object(fs.Evaluator, 'forecasts', forecast), contextlib.redirect_stdout(io.StringIO()):
                fs.run(args)
            result = json.loads(next(Path(folder).glob('*/spread_selection.json')).read_text())
            self.assertTrue(result['removed_features'])
            self.assertGreater(result['holdout_selected']['mae'], result['holdout_baseline']['mae'])
            self.assertEqual(phases, sorted(phases))

    def test_symmetric_mask_and_categorical_groups(self):
        names = ['diff_run_ypp', 'stadium:A', 'stadium:B', 'home_field']
        x = np.ones((2, 8), dtype='float32')
        masked = fs.mask_features(x, names, {'stadium'})
        np.testing.assert_array_equal(masked[0], [0, 1, 1, 0, 0, 1, 1, 0])
        np.testing.assert_array_equal(x, np.ones_like(x))

    def test_weekly_folds_never_train_on_test_or_future(self):
        data = pd.DataFrame([dict(season=s, week=w, game_type='REG', away_team='A', home_team='B')
                             for s in [2023, 2024, 2025] for w in range(1, 18)])
        folds = list(fs.weekly_folds(data, 2025, 2026))
        self.assertEqual(len(folds), 17)
        for year, week, block in folds:
            train = block[(block.season < year) | ((block.season == year) & (block.week < week))]
            self.assertEqual(len(train), 20)
            self.assertEqual(len(block), 21)
            self.assertFalse(((block.season == year) & (block.week > week)).any())

    def test_playoffs_are_included_without_consuming_regular_window(self):
        data = pd.DataFrame([dict(season=2024, week=w, game_type='REG', away_team='A', home_team='B')
                             for w in range(1, 21)] +
                            [dict(season=2024, week=21, game_type='POST', away_team='A', home_team='B'),
                             dict(season=2025, week=1, game_type='REG', away_team='A', home_team='B')])
        _, _, block = next(fs.weekly_folds(data, 2025, 2026))
        self.assertEqual(len(block), 22)
        self.assertTrue(block.game_type.eq('POST').any())

    def test_search_finds_harmful_features_and_respects_budget(self):
        calls = []
        def score(features):
            calls.append(features)
            return float('bad' in features) + 2 * float('good' not in features)
        best, scores = fs.search_subsets(['good', 'bad', 'neutral'], score, 10, 2)
        self.assertEqual(best, ('good',))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(scores), 10)
        _, scores = fs.search_subsets(['a', 'b', 'c'], lambda x: len(x), 4, 2)
        self.assertEqual(len(scores), 4)

    def test_prediction_cache_and_only_requested_market(self):
        from tests.test_shared_scoring import games
        data = games()
        args = SimpleNamespace(groups=[], jobs=1, epochs=1, seed=42)
        target = data[data.season.eq(2026)]
        names = ['diff_run_ypp', 'home_field']
        prepared = (target, np.ones((8, 4), dtype='float32'),
                    {'spread': np.zeros(8, dtype='float32'), 'total': np.ones(8, dtype='float32') * 40},
                    np.ones((1, 4), dtype='float32'), names)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.parquet'
            evaluator = fs.Evaluator(data, args, Path(folder))
            with patch.object(fs, 'weekly_folds', return_value=[(2026, 1, data)]), \
                 patch.object(fs.js, 'prepare', return_value=prepared), \
                 patch.object(fs.utils, 'cache_path', return_value=path), \
                 patch.object(fs, 'predict_member', return_value=np.array([2.])) as predict:
                first = evaluator.forecasts(names, 'spread', 2026, 2027, 2)
                second = evaluator.forecasts(names, 'spread', 2026, 2027, 2)
            self.assertEqual(predict.call_count, 2)
            self.assertEqual(predict.call_args.args[4], 'spread')
            pd.testing.assert_frame_equal(first, second)


if __name__ == '__main__':
    unittest.main()
