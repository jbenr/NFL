# Open-Meteo forecasts

Standalone downloader; does not train models or change existing backtests.

For a weekly batch:

    python pull_weather.py --season 2026 --week 1

Reads data/sched.parquet, joins cached stadium coordinates by stadium_id, and
prints schedule, request progress and weather results. Live mode skips games
past the decision cutoff. Coordinates are cached from
https://github.com/greerreNFL/stadiums in data/weather/stadium_coordinates.parquet.
Custom CSV/parquet inputs remain available with --games PATH.

## Next 72 hours where you are

    python pull_weather.py --here

Automatically looks up approximate coordinates via https://ipinfo.io/loc,
prints the location used, and pulls the next 72 hours. VPNs can change the result.
IPinfo sees your public IP; the returned coordinates are sent to Open-Meteo.
To override IP location with map/GPS coordinates:

    python pull_weather.py --here --latitude YOUR_LATITUDE --longitude YOUR_LONGITUDE

Replace the placeholders with coordinates from your map/GPS (negative longitude
for locations west of Greenwich). No game ID, kickoff or schedule is needed.
Explicit coordinates bypass IPinfo and are sent to Open-Meteo.
Prints only future hourly temperature, wind, gusts and
precipitation in Eastern Time (America/New_York, automatic daylight saving),
using dates and AM/PM. Hours before now or beyond 72 hours are excluded.
These are model forecasts, not exact
on-site measurements. This mode does not write to NFL caches or forecast files.

## Custom games

The --here terminal table uses true-color gradients: clock hours fade from gray
overnight to white at midday (not calculated sunrise/sunset); temperature runs
from dark blue at -30 F through icy blue at 32 F, yellow/orange, and red at 100 F.
Precipitation becomes more blue toward 100% or 0.5 inches per hour; wind and
gusts brighten from gray toward white at 40 mph. Scales are fixed and clamped.
Precipitation inches display three decimals so small amounts stay visible.
Redirected output stays plain; NO_COLOR=1 disables color in a terminal.

Input CSV (or parquet) columns: game_id,kickoff,latitude,longitude.
Use the schedule's exact game_id and actual stadium coordinates, including
neutral/international sites and historical relocations. Kickoff must include
its timezone offset, e.g. 2026-09-13T13:00:00-04:00. No city/team geocoding guesses.

For one location (replace example game ID/time with your scheduled game):

    python pull_weather.py --game-id YOUR_GAME_ID --latitude 44.5013 --longitude -88.0622 --kickoff 2026-09-13T13:00:00-04:00

This example location is Lambeau Field; it does not assert a scheduled game there.

Exact archived GFS runs:

    python pull_weather.py --games data/weather/games.csv --mode archive

GFS exact-run coverage starts 2026-04-02. Earlier dates fail explicitly.
This is NOT a way to fill the 2017–2025 backtest. No reanalysis, observed weather
or hindcasts are silently substituted.

## Consistent inputs and timing

- Model pinned to gfs_global for both live and archive.
- Default decision cutoff: kickoff minus 24 hours (--decision-hours).
- Live issued_at means capture time, not an invented publication time. A live
  request after the decision cutoff is rejected.
- Archive picks a 6-hourly initialization plus an assumed 8-hour publication
  delay, never later than the decision cutoff. This is a conservative assumption,
  NOT verified historical publication time. Stored availability_basis makes that
  distinction explicit. --publication-hours cannot be less than 6.
- Nearest hourly forecast within 30 minutes of kickoff; ties choose earlier.
  No artificial minute-level precision or precipitation interpolation.
- Temperature F, wind/gusts mph, precipitation probability 0–1, precipitation
  inches and wind direction degrees. Provider-null probability remains missing,
  never replaced with precipitation amount or zero.
  Exact-run probability depends on a separate GEFS archive. If that run is
  unavailable, the puller reports it and retries core GFS variables only.
- The current model consumes temperature, wind and probability only. Additional
  fields are saved for later feature work, not silently added to production.
- Outdoor forecasts are preserved. Existing model preprocessing handles
  closed/domed roof overrides separately; the puller does not guess roof status.

## Caches and output

Raw hourly forecast tables: data/cache/open_meteo/.
Live requests reuse the current UTC hour's capture; use --refresh to fetch anew.
Archive requests reuse immutable run/location responses.
Capture timestamps survive cache reuse. HTTP failures are not cached as weather.

Forecast snapshots append to data/weather/forecasts.parquet, preserving past
captures. Completed games are written atomically, one at a time. An interrupted
batch can reuse its completed request caches.

Use with the existing models:

    python joint_scoring.py --season 2026 --week 1 --weather-file data/weather/forecasts.parquet
    python joint_feature_selection.py --run --weather-file data/weather/forecasts.parquet

Only matching game IDs and forecasts available by the configured cutoff are
eligible. A 2026 file does not supply weather for a 2017–2025 experiment.
Use matching --decision-hours on downloader and model.

Check Open-Meteo's current usage/license requirements for your use:
https://open-meteo.com/en/terms
https://open-meteo.com/en/docs
https://open-meteo.com/en/docs/single-runs-api

Tests:

    python -m unittest tests.test_pull_weather
