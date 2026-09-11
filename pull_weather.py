"""Pull timestamped Open-Meteo GFS forecasts, separately from model training."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import utils

MODEL = 'gfs_global'
ARCHIVE_START = pd.Timestamp('2026-04-02', tz='UTC')
VARIABLES = {'temperature_2m': 'temperature_f', 'wind_speed_10m': 'wind_mph',
             'precipitation_probability': 'precip_probability',
             'wind_gusts_10m': 'gust_mph', 'precipitation': 'precip_inches',
             'wind_direction_10m': 'wind_direction_deg'}


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f'Timestamp needs a timezone offset: {value}')
    return stamp.tz_convert('UTC')


def request_plan(game, mode, now, decision_hours=24, publication_hours=8):
    kickoff = utc(game['kickoff'])
    cutoff = kickoff - pd.Timedelta(hours=decision_hours)
    latitude, longitude = float(game['latitude']), float(game['longitude'])
    if not np.isfinite([latitude, longitude]).all() or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError('Invalid latitude/longitude')
    params = dict(latitude=latitude, longitude=longitude, models=MODEL,
                  hourly=','.join(VARIABLES), timezone='UTC',
                  temperature_unit='fahrenheit', wind_speed_unit='mph',
                  precipitation_unit='inch', forecast_days=16)
    if mode == 'archive':
        run = (cutoff - pd.Timedelta(hours=publication_hours)).floor('6h')
        available = run + pd.Timedelta(hours=publication_hours)
        if run < ARCHIVE_START:
            raise ValueError('Exact GFS runs are archived from 2026-04-02; no observed-weather fallback')
        if available > now:
            raise ValueError('Requested archive run is not yet expected to be available')
        params['run'] = run.strftime('%Y-%m-%dT%H:%M')
        endpoint = 'https://single-runs-api.open-meteo.com/v1/forecast'
        basis = 'run_plus_conservative_publication_delay'
    else:
        if now > cutoff:
            raise ValueError('Live pull is after the decision cutoff; use archive for past decisions')
        if kickoff + pd.Timedelta(hours=3) > now.normalize() + pd.Timedelta(days=16):
            raise ValueError('Game is beyond the 16-day forecast horizon')
        run, available = pd.NaT, now
        endpoint = 'https://api.open-meteo.com/v1/forecast'
        basis = 'captured_at'
    return endpoint, params, dict(kickoff=kickoff, cutoff=cutoff, run=run,
                                 available=available, availability_basis=basis)


def fetch(endpoint, params, mode, cache_dir, now, refresh=False):
    # Archive runs are immutable; live snapshots refresh each UTC hour.
    identity = [endpoint, params, now.floor('h').isoformat() if mode == 'live' else None]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir) / (key + '.parquet')
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    with requests.Session() as session:
        retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504],
                      respect_retry_after_header=False)
        session.mount('https://', HTTPAdapter(max_retries=retry))
        response = session.get(endpoint, params=params, timeout=(10, 45))
        if mode == 'archive' and response.status_code == 400:
            reason = response.json().get('reason', '')
            if 'ncep_gefs' in reason and 'not available' in reason:
                print('Archive probability ensemble unavailable; probability will remain missing.', flush=True)
                core = dict(params, hourly=','.join(k for k in VARIABLES if k != 'precipitation_probability'))
                response = session.get(endpoint, params=core, timeout=(10, 45))
        if not response.ok:
            raise ValueError(f'Open-Meteo HTTP {response.status_code}: {response.text[:400]}')
        response.raise_for_status()
        payload = response.json()
    if payload.get('error'):
        raise ValueError(payload.get('reason', 'Open-Meteo error'))
    hourly = payload.get('hourly', {})
    if not hourly.get('time'):
        raise ValueError('Open-Meteo returned no hourly forecast')
    frame = pd.DataFrame(hourly).rename(columns=VARIABLES)
    frame['valid_at'] = pd.to_datetime(frame.pop('time'), utc=True)
    for name in VARIABLES.values():
        frame[name] = pd.to_numeric(frame.get(name, np.nan), errors='coerce')
    frame['precip_probability'] /= 100.
    if frame['precip_probability'].dropna().lt(0).any() or frame['precip_probability'].dropna().gt(1).any():
        raise ValueError('Invalid precipitation probability')
    frame['retrieved_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    frame['grid_latitude'], frame['grid_longitude'] = payload.get('latitude'), payload.get('longitude')
    utils.save_parquet(frame, path)
    return frame


def game_forecast(game, frame, timing, mode):
    # One row per game/run: use the nearest hourly forecast to kickoff.
    # Raw hourly cache retains the rest of the game window for future features.
    delta = (frame.valid_at - timing['kickoff']).abs()
    row = frame.loc[delta.idxmin()].copy()
    if delta.min() > pd.Timedelta(minutes=30):
        raise ValueError('No forecast within 30 minutes of kickoff')
    if not np.isfinite([row.temperature_f, row.wind_mph]).all():
        raise ValueError('Missing forecast temperature or wind')
    retrieved = utc(row.retrieved_at)
    available = retrieved if mode == 'live' else timing['available']
    if available > timing['cutoff']:
        raise ValueError('Forecast was captured/estimated available after the decision cutoff')
    result = row.to_dict()
    result.update(game_id=str(game['game_id']), kickoff=timing['kickoff'].isoformat(),
                  issued_at=available.isoformat(), valid_at=row.valid_at.isoformat(),
                  run_initialized_at=None if pd.isna(timing['run']) else timing['run'].isoformat(),
                  decision_at=timing['cutoff'].isoformat(),
                  latitude=float(game['latitude']), longitude=float(game['longitude']),
                  source='open-meteo', weather_model=MODEL, mode=mode,
                  availability_basis=timing['availability_basis'])
    return result


def scheduled_games(season, week, mode, decision_hours):
    schedule = pd.read_parquet('data/sched.parquet')
    games = schedule[(schedule.season == season) & (schedule.week == week)].copy()
    if games.empty:
        raise ValueError(f'No scheduled games for {season} week {week}')
    kickoff = pd.to_datetime(games.gameday.astype(str).str[:10] + ' ' + games.gametime.astype(str), errors='raise')
    games['kickoff'] = kickoff.dt.tz_localize('America/New_York', ambiguous='raise', nonexistent='raise').dt.tz_convert('UTC')
    if mode == 'live':
        eligible = games.kickoff - pd.Timedelta(hours=decision_hours) >= pd.Timestamp.now(tz='UTC')
        print(f'Schedule: {len(games)} games; skipping {int((~eligible).sum())} past the forecast decision cutoff.', flush=True)
        games = games[eligible].copy()
    if games.empty:
        raise ValueError('No games remain before the decision cutoff; use --mode archive for past decisions')
    path = Path('data/weather/stadium_coordinates.parquet')
    if path.exists():
        venues = pd.read_parquet(path)
    else:
        print('Loading stadium coordinates (cached after first pull)...', flush=True)
        url = 'https://raw.githubusercontent.com/greerreNFL/stadiums/master/data/stadiums.csv'
        response = requests.get(url, timeout=(10, 45))
        response.raise_for_status()
        venues = pd.read_csv(io.StringIO(response.text))[['stadium_id', 'lat', 'lon']]
        venues = venues.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
        utils.save_parquet(venues, path)
    games = games.merge(venues, on='stadium_id', how='left', validate='many_to_one')
    missing = games.latitude.isna() | games.longitude.isna()
    if missing.any():
        raise ValueError('Missing stadium coordinates: ' + ', '.join(games.loc[missing, 'stadium_id'].astype(str)))
    games['kickoff'] = games.kickoff.map(lambda d: d.isoformat())
    print(games[['away_team', 'home_team', 'stadium', 'kickoff']].to_string(index=False), flush=True)
    return games


def pull(games, args):
    if games.empty or games.game_id.isna().any() or games.game_id.duplicated().any():
        raise ValueError('Require nonempty, unique game_id values')
    now = pd.Timestamp.now(tz='UTC')
    plans = [(game, request_plan(game, args.mode, now, args.decision_hours, args.publication_hours))
             for game in games.to_dict('records')]
    output = Path(args.output)
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    summaries = []
    for index, (game, (endpoint, params, timing)) in enumerate(plans, 1):
        print(f'[{index}/{len(plans)}] {game["game_id"]}: fetching {args.mode} forecast...', flush=True)
        frame = fetch(endpoint, params, args.mode, args.cache_dir, now, args.refresh)
        result = game_forecast(game, frame, timing, args.mode)
        for name in ['away_team', 'home_team', 'stadium', 'roof']:
            if name in game:
                result[name] = game[name]
        summaries.append(result)
        existing = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
        existing = existing.drop_duplicates(['game_id', 'issued_at', 'valid_at', 'weather_model'], keep='last')
        utils.save_parquet(existing, output)
        print(f'{game["game_id"]}: {result["temperature_f"]:.1f} F, '
              f'wind {result["wind_mph"]:.1f} mph; valid {result["valid_at"]}', flush=True)
    summary = pd.DataFrame(summaries)
    columns = [c for c in ['game_id', 'stadium', 'temperature_f', 'wind_mph', 'gust_mph', 'precip_probability', 'valid_at'] if c in summary]
    print('\n' + summary[columns].to_string(index=False, float_format=lambda n: f'{n:.1f}'))
    print('Outdoor forecasts; probability is 0–1. Roof overrides are applied by the model.')
    print(f'Forecast file: {output}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, help='Read games from existing data/sched.parquet')
    p.add_argument('--week', type=int)
    p.add_argument('--games', help='CSV/parquet with game_id,kickoff,latitude,longitude')
    p.add_argument('--game-id')
    p.add_argument('--kickoff', help='ISO timestamp with offset, e.g. 2026-09-13T13:00:00-04:00')
    p.add_argument('--latitude', type=float)
    p.add_argument('--longitude', type=float)
    p.add_argument('--mode', choices=['live', 'archive'], default='live')
    p.add_argument('--decision-hours', type=float, default=24)
    p.add_argument('--publication-hours', type=float, default=8,
                   help='Archive availability assumption after GFS initialization (default: 8)')
    p.add_argument('--output', default='data/weather/forecasts.parquet')
    p.add_argument('--cache-dir', default='data/cache/open_meteo')
    p.add_argument('--refresh', action='store_true')
    return p


if __name__ == '__main__':
    p = parser()
    args = p.parse_args()
    if args.decision_hours < 0 or args.publication_hours < 6:
        p.error('Require nonnegative decision-hours and publication-hours >= 6')
    if args.season is not None or args.week is not None:
        if args.season is None or args.week is None:
            p.error('Provide both --season and --week')
        if args.games or any(v is not None for v in [args.game_id, args.kickoff, args.latitude, args.longitude]):
            p.error('Use season/week OR custom game inputs')
        try:
            games = scheduled_games(args.season, args.week, args.mode, args.decision_hours)
        except (ValueError, OSError, requests.RequestException) as error:
            p.exit(1, f'Weather schedule failed: {error}\n')
    elif args.games:
        if any(v is not None for v in [args.game_id, args.kickoff, args.latitude, args.longitude]):
            p.error('Use --games OR individual location/game arguments')
        path = Path(args.games)
        if not path.exists():
            p.error(f'{path} does not exist. Use --season YEAR --week WEEK to read data/sched.parquet instead.')
        games = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    else:
        if any(v is None for v in [args.game_id, args.kickoff, args.latitude, args.longitude]):
            p.error('Supply --games or --game-id, --kickoff, --latitude and --longitude')
        games = pd.DataFrame([dict(game_id=args.game_id, kickoff=args.kickoff,
                                   latitude=args.latitude, longitude=args.longitude)])
    missing = {'game_id', 'kickoff', 'latitude', 'longitude'} - set(games)
    if missing:
        p.error(f'Missing columns: {sorted(missing)}')
    try:
        pull(games, args)
    except (ValueError, requests.RequestException) as error:
        p.exit(1, f'Weather pull failed: {error}\n')
