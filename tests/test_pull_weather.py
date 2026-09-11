import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pull_weather as weather


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.game = dict(game_id='test', kickoff='2026-09-13T13:00:00-04:00',
                         latitude=44.5013, longitude=-88.0622)
        self.now = pd.Timestamp('2026-09-10T12:00:00Z')

    def test_requires_timezone_and_valid_coordinates(self):
        with self.assertRaises(ValueError):
            weather.utc('2026-09-13T13:00')
        with self.assertRaises(ValueError):
            weather.request_plan(dict(self.game, latitude=100), 'live', self.now)

    def test_archive_cycle_precedes_decision_with_publication_delay(self):
        _, params, timing = weather.request_plan(self.game, 'archive', pd.Timestamp('2026-09-14', tz='UTC'))
        self.assertEqual(params['run'], '2026-09-12T06:00')
        self.assertLessEqual(timing['available'], timing['cutoff'])
        self.assertEqual(timing['kickoff'], pd.Timestamp('2026-09-13T17:00Z'))
        with self.assertRaises(ValueError):
            weather.request_plan(dict(self.game, kickoff='2025-09-13T13:00-04:00'), 'archive', self.now)

    def test_live_never_backdates(self):
        with self.assertRaises(ValueError):
            weather.request_plan(self.game, 'live', pd.Timestamp('2026-09-13T00:00Z'))

    def test_hourly_choice_and_model_contract(self):
        _, _, timing = weather.request_plan(self.game, 'live', self.now)
        frame = pd.DataFrame(dict(valid_at=pd.to_datetime(['2026-09-13T17:00Z']),
                                  temperature_f=[70.], wind_mph=[10.], precip_probability=[.3],
                                  retrieved_at=[self.now.isoformat()]))
        result = weather.game_forecast(self.game, frame, timing, 'live')
        self.assertEqual(result['issued_at'], self.now.isoformat())
        self.assertEqual(result['precip_probability'], .3)
        from joint_scoring import weather_features
        game = pd.DataFrame(dict(game_id=['test'], gameday=['2026-09-13'],
                                 gametime=['13:00:00'], roof=['outdoors']))
        output = weather_features(game, forecasts=pd.DataFrame([result]))
        self.assertEqual(output.weather_temperature_f.iloc[0], 70.)
        self.assertEqual(output.weather_precip_probability.iloc[0], .3)

    def test_response_cache_preserves_capture_time_and_probability_units(self):
        payload = dict(latitude=44.5, longitude=-88.06,
                       hourly=dict(time=['2026-09-13T17:00'], temperature_2m=[70],
                                   wind_speed_10m=[10], precipitation_probability=[30]))
        with tempfile.TemporaryDirectory() as folder, patch.object(weather.requests, 'Session') as session:
            session.return_value.__enter__.return_value.get.return_value.json.return_value = payload
            first = weather.fetch('https://api.open-meteo.com/v1/forecast', {}, 'live', folder, self.now)
            second = weather.fetch('https://api.open-meteo.com/v1/forecast', {}, 'live', folder, self.now)
            self.assertEqual(session.call_count, 1)
            self.assertEqual(first.precip_probability.iloc[0], .3)
            pd.testing.assert_frame_equal(first, second)


if __name__ == '__main__':
    unittest.main()

