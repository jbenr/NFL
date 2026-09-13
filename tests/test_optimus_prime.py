import unittest
from unittest.mock import patch

import pandas as pd
import optimus_prime as optimus


class CalendarBoundsTests(unittest.TestCase):
    def test_shared_warmup_and_member_defaults(self):
        with patch.object(optimus.pd, 'read_parquet', return_value=self.weeks), patch.object(optimus.op, 'build_panel') as build:
            optimus.load_shared_panel(2011, 21, 2010, 10, 20, 'mean', True)
            short = build.call_args.args[2]
            optimus.load_shared_panel(2011, 21, 2010, 10, 40, 'mean', True)
            self.assertEqual(build.call_args.args[2] - short, 20)
            self.assertEqual(optimus._estimate_weeks_per_config(2011, 21, 2010, 20),
                             optimus._estimate_weeks_per_config(2011, 21, 2010, 100))
        self.assertEqual(optimus.SHARED_SCREEN_ITERATIONS, 50)
        self.assertEqual(optimus.SHARED_CONFIRM_ITERATIONS, 100)

    def test_member_counts_have_distinct_result_paths(self):
        args = ('spread', 20, 40, 'mean', True, 'separate', None)
        self.assertNotEqual(optimus._shared_run_dir(*args, iterations=50),
                            optimus._shared_run_dir(*args, iterations=100))

    def setUp(self):
        # 2003 start (not 2006): needs >= max(LOOKBACK_WINDOWS) REG weeks
        # before 2010 for the first test below to succeed, while still
        # leaving fewer than that before 2006 for the insufficient-warmup
        # case in the second test to still raise.
        self.weeks = pd.DataFrame([
            (season, week, 'REG' if week <= 17 else 'POST')
            for season in range(2003, 2012) for week in range(1, 22)
        ], columns=['season', 'week', 'game_type'])

    def test_counter_reaches_week_one_for_regular_and_postseason_ends(self):
        for end in [(2010, 1), (2011, 10), (2011, 21)]:
            with patch.object(optimus.pd, 'read_parquet', return_value=self.weeks):
                remaining = optimus.history_for_start(2010, *end)
            index = self.weeks.index[(self.weeks.season == end[0]) & (self.weeks.week == end[1])][0]
            # Mirror the loader: include current week, then decrement on REG.
            included = []
            while remaining >= 0:
                first = self.weeks.iloc[index]
                if first.season >= 2010:
                    included.append((first.season, first.week))
                index -= 1
                remaining -= int(self.weeks.iloc[index].game_type == 'REG')
            self.assertEqual(min(included), (2010, 1))
            self.assertEqual(max(included), end)

    def test_invalid_bounds_and_insufficient_warmup(self):
        with patch.object(optimus.pd, 'read_parquet', return_value=self.weeks):
            for bounds in [(2012, 2011, 21), (2010, 2011, 99), (2006, 2011, 21)]:
                with self.assertRaises(ValueError):
                    optimus.history_for_start(*bounds)
