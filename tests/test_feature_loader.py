import unittest
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import data_crunchski_2 as dc


class FeatureLoaderTests(unittest.TestCase):
    def test_only_explicit_inputs_with_different_season_schemas(self):
        schemas = [SimpleNamespace(names=list(dc._PBP_COLUMNS | {'defenders_in_box'})),
                   SimpleNamespace(names=list(dc._PBP_COLUMNS))]
        with patch.object(dc.pq, 'read_schema', side_effect=schemas) as schema, patch.object(pd, 'read_parquet') as read:
            columns = dc._pbp_needed_columns([2024, 2025])
            self.assertEqual(columns, dc._PBP_COLUMNS)
            self.assertNotIn('defenders_in_box', columns)
            self.assertNotIn('number_of_pass_rushers', columns)
            self.assertEqual(schema.call_count, 2)
            read.assert_not_called()

    def test_missing_required_field_names_season_and_column(self):
        with patch.object(dc.pq, 'read_schema', return_value=SimpleNamespace(names=list(dc._PBP_COLUMNS - {'yards_gained'}))):
            with self.assertRaisesRegex(ValueError, 'pbp_2025.parquet: missing required feature inputs: yards_gained'):
                dc._pbp_needed_columns([2025])

    def test_lazy_window_reuses_and_evicts_weeks(self):
        old = dc._WORK
        self.addCleanup(setattr, dc, '_WORK', old)
        with patch.object(pd, 'read_parquet') as read:
            dc._WORK = dc._build_work([2024], pd.DataFrame(), pd.DataFrame(), 20)
            read.assert_not_called()
            read.return_value = pd.DataFrame({'week': [1, 2], 'value': [10, 20]})
            self.assertEqual(dc._load_window([(2024, 2), (2024, 1)]).value.tolist(), [20, 10])
            self.assertEqual(set(read.call_args.kwargs['columns']), dc._PBP_COLUMNS)
            dc._load_window([(2024, 2)])
            self.assertEqual(read.call_count, 1)
            self.assertEqual(set(dc._WORK['index']), {(2024, 2)})
            read.return_value = pd.DataFrame({'week': [3], 'value': [30]})
            dc._load_window([(2024, 3), (2024, 2)])
            self.assertEqual(read.call_args.kwargs['filters'], [('week', 'in', [3])])
            self.assertEqual(set(dc._WORK['index']), {(2024, 2), (2024, 3)})


if __name__ == '__main__':
    unittest.main()
