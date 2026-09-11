import unittest
from unittest.mock import patch
import pandas as pd
import data_crunchski_2 as dc


class FeatureLoaderTests(unittest.TestCase):
    def test_lazy_window_reuses_and_evicts_weeks(self):
        old = dc._WORK
        self.addCleanup(setattr, dc, '_WORK', old)
        with patch.object(pd, 'read_parquet') as read:
            dc._WORK = dc._build_work([2024], pd.DataFrame(), pd.DataFrame(), 20)
            read.assert_not_called()
            read.return_value = pd.DataFrame({'week': [1, 2], 'value': [10, 20]})
            self.assertEqual(dc._load_window([(2024, 2), (2024, 1)]).value.tolist(), [20, 10])
            dc._load_window([(2024, 2)])
            self.assertEqual(read.call_count, 1)
            self.assertEqual(set(dc._WORK['index']), {(2024, 2)})
            read.return_value = pd.DataFrame({'week': [3], 'value': [30]})
            dc._load_window([(2024, 3), (2024, 2)])
            self.assertEqual(read.call_args.kwargs['filters'], [('week', 'in', [3])])
            self.assertEqual(set(dc._WORK['index']), {(2024, 2), (2024, 3)})


if __name__ == '__main__':
    unittest.main()
