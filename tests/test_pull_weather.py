import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pull_weather as weather


class WeatherTests(unittest.TestCase):
    def test_weather_gradients(self):
        self.assertEqual(weather.weather_color('Temp F', 32), (150, 225, 255))
        self.assertEqual(weather.weather_color('Temp F', 100), (255, 55, 55))
        self.assertEqual(weather.weather_color('Temp F', -100), (15, 30, 90))
        self.assertGreater(weather.weather_color('Time (ET)', 12)[0], weather.weather_color('Time (ET)', 3)[0])
        for column, maximum in [('Precip %', 100), ('Precip in', .5)]:
            low, high = weather.weather_color(column, 0), weather.weather_color(column, maximum)
            self.assertGreater(high[2] - high[0], low[2] - low[0])
        self.assertEqual(len(set(weather.weather_color('Wind mph', 20))), 1)

    def test_colored_table_keeps_alignment_and_small_precipitation(self):
        import re
        table = pd.DataFrame({'Time (ET)': ['Fri 09/11 07:00 PM EDT'], 'Temp F': [32.],
                              'Precip in': [.012], 'Precip %': [float('nan')]})
        plain = weather.weather_table(table, [19], color=False)
        colored = weather.weather_table(table, [19], color=True)
        self.assertIn('\033[38;2;', colored)
        self.assertEqual(re.sub(r'\x1b\[[0-9;]*m', '', colored), plain)
        self.assertIn('0.012', plain)
        self.assertIn('--', plain)

    def test_no_color_environment(self):
        table = pd.DataFrame({'Time (ET)': ['Fri 09/11 07:00 PM EDT'], 'Temp F': [32.]})
        with patch.object(weather.sys.stdout, 'isatty', return_value=True), patch.dict(weather.os.environ, {'NO_COLOR': '1'}):
            self.assertNotIn('\033', weather.weather_table(table, [19]))

    def test_ip_location_then_weather(self):
        from unittest.mock import Mock
        location = Mock()
        location.json.return_value = dict(loc='40.7143,-74.0060', city='New York City', region='New York')
        forecast = Mock()
        forecast.json.return_value = dict(timezone='America/New_York', hourly=dict(
            time=[int((pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=1)).timestamp())], temperature_2m=[70.]))
        with patch.object(weather.requests, 'get', side_effect=[location, forecast]) as get, patch('builtins.print'):
            weather.local_weather()
            self.assertEqual(get.call_args_list[0].args[0], 'https://ipinfo.io/json')
            self.assertEqual(get.call_args.kwargs['params']['latitude'], 40.7143)
            self.assertEqual(get.call_args.kwargs['params']['longitude'], -74.0060)

    def test_bad_ip_location_does_not_fetch_weather(self):
        with patch.object(weather.requests, 'get') as get, patch('builtins.print'):
            for value in ['', 'unknown', 'nan,0', '91,0']:
                get.reset_mock()
                get.return_value.text = value
                with self.assertRaises(ValueError):
                    weather.local_weather()
                self.assertEqual(get.call_count, 1)
            get.reset_mock()
            with self.assertRaises(ValueError):
                weather.local_weather(latitude=40.)
            get.assert_not_called()

    def test_local_weather_uses_future_window_without_game_inputs_or_writes(self):
        now = pd.Timestamp.now(tz='UTC')
        times = [now - pd.Timedelta(hours=1), now + pd.Timedelta(hours=1), now + pd.Timedelta(hours=73)]
        payload = dict(timezone='America/Chicago', hourly=dict(
            time=[int(t.timestamp()) for t in times], temperature_2m=[60., 70., 80.],
            wind_speed_10m=[10.] * 3, precipitation_probability=[30.] * 3))
        with patch.object(weather.requests, 'get') as get, patch.object(weather.utils, 'save_parquet') as save, patch('builtins.print'):
            get.return_value.json.return_value = payload
            table = weather.local_weather(44.5, -88.06)
            self.assertEqual(get.call_args.kwargs['params']['timezone'], 'America/New_York')
            self.assertEqual(table['Time (ET)'].iloc[0], times[1].tz_convert('America/New_York').strftime('%a %m/%d %I:%M %p %Z'))
            self.assertEqual(get.call_args.kwargs['params']['forecast_days'], 4)
            self.assertEqual(get.call_args.kwargs['params']['timeformat'], 'unixtime')
            self.assertEqual(len(table), 1)
            self.assertEqual(table['Temp F'].iloc[0], 70.)
            self.assertEqual(table['Precip %'].iloc[0], 30.)
            save.assert_not_called()

    def test_parse_interval(self):
        self.assertEqual(weather.parse_interval('30min'), pd.Timedelta(minutes=30))
        self.assertEqual(weather.parse_interval('3 hrs'), pd.Timedelta(hours=3))
        self.assertEqual(weather.parse_interval('1d'), pd.Timedelta(days=1))
        for bad in ['10min', '0h', 'h', '2 weeks', '1.5h']:
            with self.assertRaises(ValueError):
                weather.parse_interval(bad)

    def test_local_weather_aggregates_multi_hour_intervals(self):
        now = pd.Timestamp.now(tz='UTC').ceil('h')
        times = [now + pd.Timedelta(hours=h) for h in range(7)]
        payload = dict(hourly=dict(time=[int(t.timestamp()) for t in times],
                                   temperature_2m=[60., 63., 66., 70., 70., 70., 99.],
                                   precipitation=[.1, .2, 0., 0., .05, 0., 0.],
                                   wind_gusts_10m=[10., 30., 20., 5., 5., 5., 5.]))
        with patch.object(weather.requests, 'get') as get, patch('builtins.print'):
            get.return_value.json.return_value = payload
            table = weather.local_weather(44.5, -88.06, interval='3h', count=3)
            self.assertIn('hourly', get.call_args.kwargs['params'])
        # Third 3h row only has one hour of data, so it is dropped.
        self.assertEqual(list(table['Temp F']), [63., 70.])
        self.assertEqual([round(v, 3) for v in table['Precip in']], [.3, .05])
        self.assertEqual(list(table['Gust mph']), [30., 5.])

    def test_local_weather_sub_hour_uses_15_minute_feed(self):
        now = pd.Timestamp.now(tz='UTC').ceil('15min')
        times = [now + pd.Timedelta(minutes=15 * i) for i in range(4)]
        payload = dict(minutely_15=dict(time=[int(t.timestamp()) for t in times], temperature_2m=[60., 62., 64., 66.]))
        with patch.object(weather.requests, 'get') as get, patch('builtins.print'):
            get.return_value.json.return_value = payload
            table = weather.local_weather(44.5, -88.06, interval='30min', count=2)
            self.assertIn('minutely_15', get.call_args.kwargs['params'])
        self.assertEqual(list(table['Temp F']), [61., 65.])

    def test_local_weather_rejects_window_past_horizon(self):
        with patch.object(weather.requests, 'get') as get:
            with self.assertRaises(ValueError):
                weather.local_weather(44.5, -88.06, interval='1d', count=20)
            get.assert_not_called()

    def test_local_weather_rejects_invalid_coordinates_before_request(self):
        with patch.object(weather.requests, 'get') as get:
            for latitude, longitude in [(91, 0), (0, 181), (float('nan'), 0)]:
                with self.assertRaises(ValueError):
                    weather.local_weather(latitude, longitude)
            get.assert_not_called()

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
