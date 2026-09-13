"""Pull timestamped Open-Meteo GFS forecasts, separately from model training."""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import utils

MODEL = 'gfs_global'
ARCHIVE_START = pd.Timestamp('2026-04-02', tz='UTC')
# apparent_temperature/rain/snowfall/snow_depth added so a live/archive pull
# produces every column data_crunchski_3.HISTORICAL_WEATHER needs (matching
# HISTORICAL_VARIABLES' naming below) -- lets attach_historical_weather fall
# back to a forecast for a game that hasn't been played yet, not just extend
# it with forecast-only extras (precip_probability, gust_mph, wind_direction).
VARIABLES = {'temperature_2m': 'temperature_f', 'apparent_temperature': 'feels_like_f',
            'wind_speed_10m': 'wind_mph', 'precipitation_probability': 'precip_probability',
            'wind_gusts_10m': 'gust_mph', 'precipitation': 'precip_inches',
            'rain': 'rain_inches', 'snowfall': 'snowfall_inches', 'snow_depth': 'snow_depth_inches',
            'wind_direction_10m': 'wind_direction_deg'}

# Separate product from everything above: actual OBSERVED conditions (ERA5-
# family reanalysis), not a forecast of any kind -- issued_at/decision_hours/
# publication_hours don't apply, because there's no "before kickoff" version
# of a fact that already happened. Answers "did weather correlate with
# outcomes," never "could this have been bet on in advance" -- that question
# still needs the live/archive forecast modes above, run going forward.
HISTORICAL_ENDPOINT = 'https://archive-api.open-meteo.com/v1/archive'
# Own dict, not derived from VARIABLES: no precipitation_probability
# (probability is a forecast concept -- observed weather either rained or it
# didn't) and no wind_direction_10m (not requested), but with a few extra
# ground-conditions variables the live/forecast side never asked for.
HISTORICAL_VARIABLES = {'temperature_2m': 'temperature_f', 'apparent_temperature': 'feels_like_f',
                        'relative_humidity_2m': 'humidity_pct',
                        'wind_speed_10m': 'wind_mph', 'wind_gusts_10m': 'gust_mph',
                        'precipitation': 'precip_inches', 'rain': 'rain_inches',
                        'snowfall': 'snowfall_inches', 'snow_depth': 'snow_depth_inches'}
# API returns these two in fixed units regardless of the unit params below
# (snowfall: cm always, snow_depth: meters always) -- converted to inches
# after the fact so every precip-family column shares one unit.
_HISTORICAL_RAW_UNITS = {'snowfall_inches': ('cm', 1 / 2.54), 'snow_depth_inches': ('m', 39.3701)}


def fetch_historical(latitude, longitude, date, cache_dir, refresh=False):
    """One calendar day's hourly reanalysis at one location. A past, closed
    date's reanalysis is immutable -- cached forever once fetched, refetched
    only on --refresh."""
    # Variable set is part of the cache key on purpose: adding/removing a
    # variable from HISTORICAL_VARIABLES must invalidate old cache entries
    # rather than silently returning a frame missing the new columns.
    identity = ['historical', round(latitude, 4), round(longitude, 4), date.isoformat(),
               sorted(HISTORICAL_VARIABLES)]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir) / (key + '.parquet')
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    params = dict(latitude=latitude, longitude=longitude,
                  start_date=date.isoformat(), end_date=date.isoformat(),
                  hourly=','.join(HISTORICAL_VARIABLES), timezone='UTC',
                  temperature_unit='fahrenheit', wind_speed_unit='mph',
                  precipitation_unit='inch')
    # A bulk pull fires thousands of these sequentially; a small pace here
    # (only on an actual cache miss -- cache hits above return before this
    # line) keeps it under Open-Meteo's rate limit instead of tripping 429s.
    time.sleep(0.1)
    with requests.Session() as session:
        # 429 (rate limited) is now retried, not just 5xx -- discovered the
        # hard way: a ~6700-request unpaced bulk pull silently dropped two
        # entire seasons (2021-2022) to unretried 429s, each one caught by
        # pull_historical's per-game try/except and logged as a "failure"
        # instead of surfacing as the systematic, retryable issue it was.
        retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                      respect_retry_after_header=True)
        session.mount('https://', HTTPAdapter(max_retries=retry))
        response = session.get(HISTORICAL_ENDPOINT, params=params, timeout=(10, 45))
        if not response.ok:
            raise ValueError(f'Open-Meteo historical HTTP {response.status_code}: {response.text[:400]}')
        payload = response.json()
    if payload.get('error'):
        raise ValueError(payload.get('reason', 'Open-Meteo historical error'))
    hourly = payload.get('hourly', {})
    if not hourly.get('time'):
        raise ValueError('Open-Meteo historical returned no hourly data')
    frame = pd.DataFrame(hourly).rename(columns=HISTORICAL_VARIABLES)
    frame['valid_at'] = pd.to_datetime(frame.pop('time'), utc=True)
    for name in HISTORICAL_VARIABLES.values():
        frame[name] = pd.to_numeric(frame.get(name, np.nan), errors='coerce')
    for name, (_, factor) in _HISTORICAL_RAW_UNITS.items():
        frame[name] = frame[name] * factor
    frame['retrieved_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    utils.save_parquet(frame, path)
    return frame


def historical_games(seasons):
    """Every completed game in `seasons`, with stadium coordinates and a real
    UTC kickoff -- same construction as scheduled_games(), for the whole
    history instead of one week."""
    schedule = pd.read_parquet('data/sched.parquet')
    games = schedule[schedule.season.isin(list(seasons)) & schedule.away_score.notna()].copy()
    if games.empty:
        raise ValueError(f'No completed games for seasons {list(seasons)}')
    kickoff = pd.to_datetime(games.gameday.astype(str).str[:10] + ' ' + games.gametime.astype(str), errors='coerce')
    games['kickoff'] = kickoff.dt.tz_localize('America/New_York', ambiguous='NaT',
                                              nonexistent='NaT').dt.tz_convert('UTC')
    missing_kickoff = games.kickoff.isna()
    if missing_kickoff.any():
        print(f'Skipping {int(missing_kickoff.sum())} games with unparseable kickoff time.', flush=True)
    games = games[~missing_kickoff]
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
    missing_coords = games.latitude.isna() | games.longitude.isna()
    if missing_coords.any():
        print(f'Skipping {int(missing_coords.sum())} games with unknown stadium coordinates: '
              f'{sorted(games.loc[missing_coords, "stadium_id"].dropna().unique())}', flush=True)
    return games[~missing_coords].reset_index(drop=True)


def pull_historical(seasons, output='data/weather/historical.parquet',
                    cache_dir='data/cache/open_meteo_historical', refresh=False):
    """Bulk pull, in the spirit of data_pullson.pull_sched/pull_pbp: actual
    OBSERVED conditions (temperature, wind, gusts, direction, precipitation)
    for every completed game across `seasons`, from Open-Meteo's ERA5-based
    historical/reanalysis archive -- a different endpoint and a different kind
    of data from everything else in this file. See HISTORICAL_ENDPOINT's
    comment for the observed-vs-forecast distinction before using this to
    justify anything about real-time betting value.

    Checkpointed every 25 games and resumable: games already present in
    `output` are skipped on a re-run, matching every other cache in this
    codebase. A single game's failure (bad coordinates, no matching hour,
    a transient API error) is logged and skipped, not fatal to the whole pull.
    """
    games = historical_games(seasons)
    output = Path(output)
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    have = set(existing.game_id) if 'game_id' in existing else set()
    rows = existing.to_dict('records') if not existing.empty else []
    todo = games[~games.game_id.isin(have)] if have else games
    print(f'{len(games)} completed games in range; {len(games) - len(todo)} already pulled, '
          f'{len(todo)} to fetch.', flush=True)
    failures = []
    for i, game in enumerate(todo.to_dict('records'), 1):
        try:
            lat, lon = float(game['latitude']), float(game['longitude'])
            # A single day's fetch only spans 00:00-23:00 UTC of that
            # calendar date. Late US kickoffs routinely cross the UTC
            # midnight boundary (e.g. a 6:40pm ET kickoff is 23:40 UTC, and
            # the truly-nearest hour, 00:00 UTC next day, would otherwise
            # never get fetched at all) -- so pull the next day too. Cheap:
            # fetch_historical caches per single day, and this next-day
            # fetch is often already cached from some other game anyway.
            next_day = (pd.Timestamp(game['kickoff']) + pd.Timedelta(days=1)).date()
            frame = pd.concat([
                fetch_historical(lat, lon, game['kickoff'].date(), cache_dir, refresh),
                fetch_historical(lat, lon, next_day, cache_dir, refresh),
            ], ignore_index=True)
            # Prefer the hour at or before kickoff (pregame conditions, not
            # postgame) over a nominally-closer hour after it; fall back to
            # nearest overall only if nothing before kickoff was fetched.
            before = frame[frame.valid_at <= game['kickoff']]
            candidate = before.loc[before.valid_at.idxmax()] if not before.empty else None
            delta = (frame.valid_at - game['kickoff']).abs()
            if candidate is None or (game['kickoff'] - candidate.valid_at) > pd.Timedelta(hours=2):
                candidate = frame.loc[delta.idxmin()]
            if delta.loc[candidate.name] > pd.Timedelta(hours=2):
                raise ValueError('No reanalysis hour within 2 hours of kickoff')
            row = candidate.to_dict()
        except (ValueError, requests.RequestException) as error:
            print(f'[{i}/{len(todo)}] {game["game_id"]}: FAILED -- {error}', flush=True)
            failures.append(dict(game_id=game['game_id'], error=str(error)))
            continue
        row.update(game_id=str(game['game_id']), season=int(game['season']), week=int(game['week']),
                   away_team=game['away_team'], home_team=game['home_team'],
                   stadium=game.get('stadium'), roof=game.get('roof'),
                   kickoff=game['kickoff'].isoformat(), source='open-meteo', mode='reanalysis')
        rows.append(row)
        if i % 25 == 0 or i == len(todo):
            checkpoint = pd.DataFrame(rows).drop_duplicates('game_id', keep='last')
            utils.save_parquet(checkpoint, output)
            print(f'[{i}/{len(todo)}] checkpoint saved ({len(checkpoint)} total rows)', flush=True)
    result = pd.DataFrame(rows).drop_duplicates('game_id', keep='last')
    utils.save_parquet(result, output)
    print(f"\nDone: {len(result)} games with observed weather -> {output}", flush=True)
    if failures:
        print(f'{len(failures)} game(s) failed and were skipped -- see returned failures list.', flush=True)
    return result, pd.DataFrame(failures)


# The subset of HISTORICAL_VARIABLES' output columns actually worth handing
# to a model -- drops temperature_f (feels_like_f supersedes it), humidity_pct
# and gust_mph (kept in the raw pull for later inspection, left out here as
# redundant/low-signal for now). Optional input pattern: pass a different
# list to try a different feature set without touching the function.
WEATHER_FEATURE_COLUMNS = ['feels_like_f', 'precip_inches', 'rain_inches',
                           'snowfall_inches', 'snow_depth_inches', 'wind_mph']
# Roof values (from sched.parquet) under which weather has zero effect on
# play. 'open' (a retractable roof left open) is deliberately NOT included --
# that's a real outdoor game.
INDOOR_ROOFS = {'dome', 'closed'}


def build_weather_features(source='data/weather/historical.parquet',
                           output='data/weather/historical_features.parquet',
                           feature_columns=None):
    """Collapse the raw observed-weather pull down to modeling features, with
    indoor games overridden to a fixed controlled-environment reading (72F,
    calm, dry) instead of whatever ambient outdoor conditions happened to
    exist at the stadium's coordinates that day -- weather cannot affect a
    game once the roof is shut, and leaving the real outdoor reading in would
    just be noise attributed to a game it had no way to touch.

    feature_columns: optional override of WEATHER_FEATURE_COLUMNS, so a
    different feature subset can be tried without editing this function.
    """
    feature_columns = feature_columns or WEATHER_FEATURE_COLUMNS
    df = pd.read_parquet(source)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f'{source} is missing {missing} -- rerun --historical with the '
                         f'current HISTORICAL_VARIABLES first')
    keys = ['game_id', 'season', 'week', 'away_team', 'home_team', 'roof']
    features = df[keys + feature_columns].copy()
    indoor = features.roof.isin(INDOOR_ROOFS)
    if 'feels_like_f' in feature_columns:
        features.loc[indoor, 'feels_like_f'] = 72.0
    for col in feature_columns:
        if col != 'feels_like_f':
            features.loc[indoor, col] = 0.0
    utils.save_parquet(features, output)
    print(f'{len(features)} games -> {output}  ({int(indoor.sum())} indoor games overridden '
         f'to controlled-environment readings)', flush=True)
    return features


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
    # Same fixed-unit quirk as the historical archive (snowfall: cm, snow_depth:
    # meters, regardless of the unit params above) -- see _HISTORICAL_RAW_UNITS.
    for name, (_, factor) in _HISTORICAL_RAW_UNITS.items():
        if name in frame:
            frame[name] = frame[name] * factor
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


def weather_color(column, value):
    """Fixed scales keep colors comparable across locations and forecast days."""
    scales = {
        'Time (ET)': [(0, (165, 165, 165)), (4, (150, 150, 150)),
                      (12, (255, 255, 255)), (16, (255, 255, 255)), (24, (165, 165, 165))],
        'Temp F': [(-30, (15, 30, 90)), (0, (35, 80, 185)), (32, (150, 225, 255)),
                   (60, (245, 225, 100)), (80, (255, 155, 45)), (100, (255, 55, 55))],
        'Wind mph': [(0, (145, 145, 145)), (40, (250, 250, 250))],
        'Gust mph': [(0, (145, 145, 145)), (40, (250, 250, 250))],
        'Precip %': [(0, (155, 165, 180)), (50, (100, 185, 255)), (100, (35, 100, 240))],
        'Precip in': [(0, (155, 165, 180)), (.1, (100, 185, 255)), (.5, (35, 100, 240))],
        'Cloud %': [(0, (255, 255, 255)), (100, (75, 75, 75))],
        'Snow in': [(0, (155, 165, 180)), (.1, (190, 220, 255)), (1.0, (110, 165, 255))],
    }
    scales['Feels Like F'] = scales['Temp F']
    if pd.isna(value):
        return (130, 130, 130)
    stops, colors = zip(*scales[column])
    return tuple(int(round(np.interp(value, stops, channel))) for channel in zip(*colors))


def weather_table(table, hours, color=None):
    """Pad before coloring so ANSI escape sequences do not shift columns."""
    if color is None:
        color = sys.stdout.isatty() and 'NO_COLOR' not in os.environ and os.environ.get('TERM') != 'dumb'
    def _cell(column, v):
        if column == 'Time (ET)':
            return str(v)
        if pd.isna(v):
            return '--'
        if column == 'Snow in':
            # It's snowing -- say so, don't make someone scan a column of
            # decimals to notice. Blank (not 0.000) when it isn't.
            return f'❄{v:.2f}"' if v > 0 else '--'
        decimals = 3 if column == 'Precip in' else 0 if column in ('Precip %', 'Cloud %') else 1
        return f'{v:.{decimals}f}'

    rows = []
    for row in table.itertuples(index=False, name=None):
        rows.append([_cell(table.columns[i], v) for i, v in enumerate(row)])
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(table.columns)]
    lines = ['  '.join(c.ljust(w) if i == 0 else c.rjust(w)
                       for i, (c, w) in enumerate(zip(table.columns, widths)))]
    for row, values, hour in zip(rows, table.itertuples(index=False, name=None), hours):
        cells = []
        for i, (label, value, width) in enumerate(zip(row, values, widths)):
            cell = label.ljust(width) if i == 0 else label.rjust(width)
            if color:
                rgb = weather_color(table.columns[i], hour if i == 0 else value)
                cell = f'\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{cell}\033[0m'
            cells.append(cell)
        lines.append('  '.join(cells))
    return '\n'.join(lines)


GEOCODING_ENDPOINT = 'https://geocoding-api.open-meteo.com/v1/search'
US_STATES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC',
}


def geocode_city(query, count=8):
    """Resolve free-text like 'Pittsburgh, PA' (a bare city name works too)
    to coordinates via Open-Meteo's own geocoding API -- prints every
    candidate it actually considered, so a wrong match is visible, not
    silent. US results are preferred; an optional ', ST' or ', State Name'
    narrows among same-named cities (there are ~40 US "Springfield"s)."""
    parts = [p.strip() for p in query.split(',')]
    name, state_hint = parts[0], (parts[1] if len(parts) > 1 else None)
    print(f'Looking up "{query}"...', flush=True)
    response = requests.get(GEOCODING_ENDPOINT, timeout=(10, 20),
                            params=dict(name=name, count=count, language='en', format='json'))
    response.raise_for_status()
    results = response.json().get('results') or []
    if not results:
        raise ValueError(f'No location found for "{query}"')
    us = [r for r in results if r.get('country_code') == 'US']
    candidates = us or results
    if state_hint:
        hint = state_hint.lower()
        narrowed = [r for r in candidates if hint in (r.get('admin1') or '').lower()
                   or hint.upper() == US_STATES.get(r.get('admin1'), '')]
        if narrowed:
            candidates = narrowed
        else:
            print(f'  (no candidate matched state/region "{state_hint}" -- showing all matches for "{name}")')
    print(f'Found {len(candidates)} candidate(s):', flush=True)
    for r in candidates:
        region = r.get('admin1', '?')
        print(f'  {r["name"]}, {region}, {r.get("country", "?")}  '
             f'({r["latitude"]:.4f}, {r["longitude"]:.4f})  pop={r.get("population") or "?"}')
    best = max(candidates, key=lambda r: r.get('population') or 0)
    print(f'-> Using: {best["name"]}, {best.get("admin1", "?")}, {best.get("country", "?")} '
         f'at ({best["latitude"]:.5f}, {best["longitude"]:.5f})', flush=True)
    return float(best['latitude']), float(best['longitude']), best


def local_weather(latitude=None, longitude=None, city=None):
    """Display the next 72 hours at supplied coordinates, a geocoded city, or
    IP location; no NFL writes."""
    if city is not None and (latitude is not None or longitude is not None):
        raise ValueError('Provide --city, or --latitude/--longitude, or neither for IP location -- not both')
    place = None
    if city is not None:
        latitude, longitude, resolved = geocode_city(city)
        place = f'{resolved["name"]}, {resolved.get("admin1", "")}'.rstrip(', ')
    elif (latitude is None) != (longitude is None):
        raise ValueError('Provide both --latitude and --longitude, or neither for IP location')
    elif latitude is None:
        print('Locating via IPinfo (approximate; VPNs can change the result)...', flush=True)
        response = requests.get('https://ipinfo.io/json', timeout=(10, 15))
        response.raise_for_status()
        info = response.json()
        try:
            latitude, longitude = map(float, info['loc'].split(','))
        except (KeyError, ValueError):
            raise ValueError('IPinfo returned no valid coordinates; supply --latitude and --longitude') from None
        place = ', '.join(p for p in [info.get('city'), info.get('region')] if p) or None
    if not np.isfinite([latitude, longitude]).all() or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError('Invalid latitude/longitude')
    where = f'{place} ({latitude:.5f}, {longitude:.5f})' if place else f'{latitude:.5f}, {longitude:.5f}'
    print(f'Weather at {where}: fetching next 72 hours...', flush=True)
    params = dict(latitude=latitude, longitude=longitude, models=MODEL,
                  hourly=','.join(list(VARIABLES) + ['cloud_cover', 'apparent_temperature', 'snowfall']),
                  timezone='America/New_York', forecast_days=4, timeformat='unixtime',
                  temperature_unit='fahrenheit', wind_speed_unit='mph', precipitation_unit='inch')
    response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=(10, 45))
    response.raise_for_status()
    payload = response.json()
    if payload.get('error'):
        raise ValueError(payload.get('reason', 'Open-Meteo error'))
    hourly = payload.get('hourly', {})
    if not hourly.get('time'):
        raise ValueError('Open-Meteo returned no hourly forecast')
    frame = pd.DataFrame(hourly)
    valid = pd.to_datetime(frame['time'], unit='s', utc=True)
    now = pd.Timestamp.now(tz='UTC')
    upcoming = (valid >= now) & (valid <= now + pd.Timedelta(hours=72))
    frame = frame.loc[upcoming].copy()
    if frame.empty:
        raise ValueError('Open-Meteo returned no forecasts in the next 72 hours')
    frame['time'] = valid.loc[upcoming].dt.tz_convert('America/New_York').dt.strftime('%a %m/%d %I:%M %p %Z')
    # snowfall has no unit override on this endpoint either -- always cm,
    # same as the historical archive API. Converted here so it reads in the
    # same inches as Precip in.
    if 'snowfall' in frame:
        frame['snowfall'] = pd.to_numeric(frame['snowfall'], errors='coerce') / 2.54
    columns = {'time': 'Time (ET)', 'temperature_2m': 'Temp F', 'apparent_temperature': 'Feels Like F',
               'precipitation_probability': 'Precip %', 'precipitation': 'Precip in',
               'snowfall': 'Snow in', 'cloud_cover': 'Cloud %',
               'wind_speed_10m': 'Wind mph', 'wind_gusts_10m': 'Gust mph'}
    table = frame.reindex(columns=columns).rename(columns=columns)
    print(f'Next 72 hours | Eastern Time (ET) | {MODEL}')
    hours = valid.loc[upcoming].dt.tz_convert('America/New_York').dt.hour
    print(weather_table(table, hours))
    print('Hourly model forecast, not a measurement at your exact spot. No NFL files changed.')
    return table


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--here', action='store_true', help='Next 72 hours at your IP location, or supplied --latitude/--longitude')
    p.add_argument('--historical', action='store_true',
                   help='Bulk-pull OBSERVED weather (reanalysis, not a forecast) for every completed '
                        'game in --start-season..--end-season. See pull_historical()\'s docstring for '
                        'the observed-vs-forecast distinction.')
    p.add_argument('--start-season', type=int, help='With --historical: first season, inclusive')
    p.add_argument('--end-season', type=int, help='With --historical: last season, inclusive')
    p.add_argument('--build-features', action='store_true',
                   help='Collapse data/weather/historical.parquet to the modeling feature set '
                        '(WEATHER_FEATURE_COLUMNS), with indoor games zeroed to a controlled '
                        'reading. Run after --historical.')
    p.add_argument('--season', type=int, help='Read games from existing data/sched.parquet')
    p.add_argument('--week', type=int)
    p.add_argument('--games', help='CSV/parquet with game_id,kickoff,latitude,longitude')
    p.add_argument('--game-id')
    p.add_argument('--kickoff', help='ISO timestamp with offset, e.g. 2026-09-13T13:00:00-04:00')
    p.add_argument('--latitude', type=float)
    p.add_argument('--longitude', type=float)
    p.add_argument('--city', help='Free-text "City, ST" (or just a city name) to geocode, '
                                  'used with --here instead of --latitude/--longitude')
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
    if args.here or args.city:
        if any(v is not None for v in [args.season, args.week, args.games, args.game_id, args.kickoff]) or args.mode != 'live':
            p.error('--here/--city is a standalone daily forecast; do not combine with game inputs or archive mode')
        try:
            local_weather(args.latitude, args.longitude, city=args.city)
        except (ValueError, requests.RequestException) as error:
            p.exit(1, f'Local weather failed: {error}\n')
        p.exit()
    if args.historical:
        other = [args.season, args.week, args.games, args.game_id, args.kickoff, args.latitude, args.longitude]
        if any(v is not None for v in other) or args.mode != 'live':
            p.error('--historical is a standalone bulk pull; do not combine with forecast-mode game inputs')
        if args.start_season is None or args.end_season is None or args.start_season > args.end_season:
            p.error('--historical requires --start-season <= --end-season')
        try:
            _, failures = pull_historical(
                range(args.start_season, args.end_season + 1),
                output='data/weather/historical.parquet' if args.output == 'data/weather/forecasts.parquet' else args.output,
                cache_dir='data/cache/open_meteo_historical' if args.cache_dir == 'data/cache/open_meteo' else args.cache_dir,
                refresh=args.refresh)
        except (ValueError, requests.RequestException) as error:
            p.exit(1, f'Historical weather pull failed: {error}\n')
        if not failures.empty:
            print(failures.to_string(index=False))
        p.exit()
    if args.build_features:
        other = [args.season, args.week, args.games, args.game_id, args.kickoff,
                args.latitude, args.longitude, args.start_season, args.end_season]
        if any(v is not None for v in other) or args.historical or args.mode != 'live':
            p.error('--build-features is standalone; run it by itself after --historical')
        try:
            build_weather_features(
                output='data/weather/historical_features.parquet' if args.output == 'data/weather/forecasts.parquet' else args.output)
        except (ValueError, FileNotFoundError) as error:
            p.exit(1, f'Building weather features failed: {error}\n')
        p.exit()
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
