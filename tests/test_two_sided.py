import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import data_crunchski_3 as dc
import shared_scoring as ss
import utils
from tests.test_shared_scoring import games


def fixture():
    data = games()
    values = {}
    for i, metric in enumerate(dc.METRICS):
        values[f'away_off_{metric}_z'] = np.arange(len(data)) * .1 + i
        values[f'away_def_{metric}_z'] = np.arange(len(data)) * .2 - i
    data = pd.concat([data, pd.DataFrame(values)], axis=1).copy()
    for side, usage in [('away', .7), ('home', .4)]:
        data[f'{side}_raw_off_pass_%'] = usage
        data[f'{side}_raw_off_run_%'] = 1 - usage
    for column in dc.HISTORICAL_WEATHER:
        data[column] = 1.
    data['weather_indoor'] = 0.
    data['game_id'] = [f'game{i}' for i in range(len(data))]
    data['roof'] = 'outdoors'
    return data


class TwoSidedTests(unittest.TestCase):
    def test_both_matchups_reorient_and_scale_symmetrically(self):
        data = fixture()
        rows = dc.two_sided_rows(data)
        n = len(data)
        np.testing.assert_allclose(rows.own_off_pass_ypp.iloc[:n], data.away_off_pass_ypp_z * 1.2)
        np.testing.assert_allclose(rows.own_def_pass_ypp.iloc[:n], data.away_def_pass_ypp_z * .9)
        np.testing.assert_allclose(rows.own_off_pass_ypp.iloc[n:], -rows.own_def_pass_ypp.iloc[:n])
        np.testing.assert_allclose(rows.own_def_pass_ypp.iloc[n:], -rows.own_off_pass_ypp.iloc[:n])
        np.testing.assert_allclose(rows.weather_wind_mph.iloc[:n], rows.weather_wind_mph.iloc[n:])

    def test_preparation_does_not_use_target_scores_or_future_rows(self):
        data = fixture()
        first = dc.prepare_two_sided(data, 2026, 1)
        data.loc[data.season == 2026, ['away_score', 'home_score']] = 9999.
        future = data.iloc[-1:].assign(week=2, away_off_pass_ypp_z=9999.)
        second = dc.prepare_two_sided(pd.concat([data, future]), 2026, 1)
        for a, b in zip(first[1:], second[1:]):
            np.testing.assert_array_equal(a, b)

    def test_weather_requires_coverage_and_overrides_indoor(self):
        data = fixture().drop(columns=dc.HISTORICAL_WEATHER + ['weather_indoor'])
        data.loc[0, 'roof'] = 'closed'
        weather = fixture()[['game_id'] + dc.HISTORICAL_WEATHER]
        with patch.object(pd, 'read_parquet', return_value=weather):
            result = dc.attach_historical_weather(data, 'unused')
            self.assertEqual(result.feels_like_f.iloc[0], 72.)
            self.assertEqual(result.wind_mph.iloc[0], 0.)
        with patch.object(pd, 'read_parquet', return_value=weather.iloc[1:]):
            with self.assertRaisesRegex(ValueError, 'Historical weather missing'):
                dc.attach_historical_weather(data, 'unused')

    def test_forecast_fills_only_the_game_missing_historical_weather(self):
        data = fixture().drop(columns=dc.HISTORICAL_WEATHER + ['weather_indoor'])
        full = fixture()[['game_id'] + dc.HISTORICAL_WEATHER]
        historical = full.iloc[1:]  # game 0 (the not-yet-played target) has no reanalysis row
        forecast = full.iloc[[0]].copy()
        forecast[dc.HISTORICAL_WEATHER] = forecast[dc.HISTORICAL_WEATHER] + 100.  # visibly different values
        with patch.object(pd, 'read_parquet', side_effect=[historical, forecast]):
            result = dc.attach_historical_weather(data, 'unused-historical', 'unused-forecast')
        # Select columns first, then rows -- .loc[row, columns] on a frame
        # with other, non-numeric columns (game_id, roof, ...) can coerce
        # the row slice to object dtype, unrelated to the fallback logic itself.
        weather_cols = result[dc.HISTORICAL_WEATHER]
        np.testing.assert_allclose(weather_cols.iloc[0], forecast[dc.HISTORICAL_WEATHER].iloc[0])
        np.testing.assert_allclose(weather_cols.iloc[1], full[dc.HISTORICAL_WEATHER].iloc[1])

    def test_forecast_fallback_still_raises_if_game_missing_from_both(self):
        data = fixture().drop(columns=dc.HISTORICAL_WEATHER + ['weather_indoor'])
        full = fixture()[['game_id'] + dc.HISTORICAL_WEATHER]
        with patch.object(pd, 'read_parquet', side_effect=[full.iloc[1:], full.iloc[2:]]):
            with self.assertRaisesRegex(ValueError, 'Historical weather missing'):
                dc.attach_historical_weather(data, 'unused-historical', 'unused-forecast')

    def test_small_fit_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(utils, 'cache_path', return_value=Path(folder) / 'test.parquet'):
            result, importance = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1)
            np.testing.assert_allclose(result.prediction, result.away_points - result.home_points, atol=1e-5)
            np.testing.assert_allclose(result.total_prediction, result.away_points + result.home_points, atol=1e-5)
            self.assertTrue(np.isfinite(result.variance).all())
            self.assertIn('attr_away_off_qb_elo', result)
            self.assertIn('attr_away_def_qb_elo', result)
            self.assertIn('own_def_qb_elo', importance.feature.values)
            with patch.object(ss, 'fit_member', side_effect=AssertionError('cache miss')):
                cached_result, cached_importance = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1)
                pd.testing.assert_frame_equal(result, cached_result)
                # Parquet round-trips drop the pre-sort index; compare values, not the index itself.
                pd.testing.assert_frame_equal(importance.reset_index(drop=True), cached_importance.reset_index(drop=True))

    def test_attribution_schema_matches_summarize_and_explains_the_prediction(self):
        # The real correctness check for fit_two_sided's attribution
        # derivation (see its docstring): if the cross-slot signs were
        # wrong, prediction - baseline - sum(attributions) would NOT be
        # ~0. This is the same check test_differentials_and_explanations
        # runs against summarize()'s (differently-derived) output.
        with tempfile.TemporaryDirectory() as folder, patch.object(utils, 'cache_path', return_value=Path(folder) / 'test.parquet'):
            details, _ = ss.fit_two_sided(fixture(), 2026, 1, iterations=2, epochs=1, jobs=1)
        self.assertTrue((details.integration_residual.abs() < .05).all())
        self.assertTrue((details.total_integration_residual.abs() < .05).all())
        spread = ss.market_details(details, 'spread')
        total = ss.market_details(details, 'total')
        np.testing.assert_allclose(spread.prediction, details.prediction)
        np.testing.assert_allclose(total.prediction, details.total_prediction)
        # market_details' 'total' view drops attr_ entirely in favor of
        # total_attr_ (renamed to attr_ on the way out) -- matches
        # summarize()'s existing convention, already tested elsewhere.
        self.assertIn('attr_away_off_qb_elo', spread.columns)
        self.assertIn('attr_away_off_qb_elo', total.columns)
