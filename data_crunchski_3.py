"""Reusable matchup preparation; historical team-stat math stays in crunchski_2."""
import numpy as np
import pandas as pd
import polars as pl

from edge_scan import PRODUCTION_FEATURES

METRICS = [f[len('away_off_'):] for f in PRODUCTION_FEATURES if f.startswith('away_off_')]
FEATURES = [f'{unit}_{metric}' for metric in METRICS for unit in ['off', 'def']]
GROUPS = {'stadium': ['stadium_id'], 'field': ['surface', 'roof'], 'referee': ['referee']}
BASE_CONTEXT = ['home_field', 'rest_advantage']
INPUT_MODES = ['separate', 'differential', 'percentile', 'zscore']
HISTORICAL_WEATHER = ['feels_like_f', 'wind_mph', 'precip_inches', 'rain_inches',
                      'snowfall_inches', 'snow_depth_inches']


def attach_historical_weather(panel, path, forecast_path=None):
    """Explicit retrospective input for games already played. forecast_path
    (pull_weather.py --season ... --week ... --mode live, written to
    data/weather/forecasts.parquet by default) is an opt-in fallback used
    ONLY for games with no historical row at all -- i.e. games that haven't
    been played yet. Training rows always keep real reanalysis; only a
    genuinely future target game ever falls back to a forecast, and never
    silently: if forecast_path is omitted, a missing game still raises
    exactly as before."""
    weather = pd.read_parquet(path)
    columns = ['game_id'] + HISTORICAL_WEATHER
    result = panel.merge(weather[columns], on='game_id', how='left', validate='one_to_one')
    missing = ~np.isfinite(result[HISTORICAL_WEATHER].to_numpy(dtype=float)).all(axis=1)
    if missing.any() and forecast_path is not None:
        forecast = pd.read_parquet(forecast_path)
        available = [c for c in HISTORICAL_WEATHER if c in forecast.columns]
        # A game can have multiple recorded forecast issuances (one per
        # pull_weather.py run, e.g. across --decision-hours retries) --
        # keep the most complete row per game, not whichever happened to
        # come first (which could predate a since-added weather column).
        # Ties (equally complete) break on the latest issuance.
        forecast = forecast.assign(_complete=forecast[available].notna().sum(axis=1))
        sort_cols = ['_complete'] + (['issued_at'] if 'issued_at' in forecast else [])
        fallback = (forecast.sort_values(sort_cols).drop_duplicates('game_id', keep='last')
                            [['game_id'] + available].set_index('game_id'))
        for column in available:
            result.loc[missing, column] = result.loc[missing, 'game_id'].map(fallback[column])
        missing = ~np.isfinite(result[HISTORICAL_WEATHER].to_numpy(dtype=float)).all(axis=1)
    if missing.any():
        bad = result.loc[missing, 'game_id']
        raise ValueError(f'Historical weather missing for {len(bad)} games, including {bad.iloc[0]}')
    indoor = result.roof.fillna('').str.lower().isin(['dome', 'closed'])
    result.loc[indoor, 'feels_like_f'] = 72.
    result.loc[indoor, HISTORICAL_WEATHER[1:]] = 0.
    result['weather_indoor'] = indoor.astype(float)
    return result


def two_sided_rows(games):
    """Requires UNSCALED league-z columns from the existing snapshot builder."""
    blocks = []
    for side, opponent in [('away', 'home'), ('home', 'away')]:
        values = {}
        for metric in METRICS:
            away_off = games[f'away_off_{metric}_z'].to_numpy()
            away_def = games[f'away_def_{metric}_z'].to_numpy()
            own_off, own_def = (away_off, away_def) if side == 'away' else (-away_def, -away_off)
            usage = 'pass' if 'pass' in metric else 'run' if 'run' in metric else None
            if usage:
                own_off = own_off * (games[f'{side}_raw_off_{usage}_%'].to_numpy() + .5)
                own_def = own_def * (games[f'{opponent}_raw_off_{usage}_%'].to_numpy() + .5)
            values[f'own_off_{metric}'] = own_off
            values[f'own_def_{metric}'] = own_def
        for column in HISTORICAL_WEATHER + ['weather_indoor']:
            values['weather_' + column.removeprefix('weather_')] = games[column].to_numpy()
        values['home_field'] = games.home_field_adv.to_numpy() if side == 'home' else np.zeros(len(games))
        values['rest_advantage'] = games[f'{side}_rest'].to_numpy() - games[f'{opponent}_rest'].to_numpy()
        blocks.append(pd.DataFrame(values))
    return pd.concat(blocks, ignore_index=True)


def prepare_two_sided(panel, season, week):
    prior = (panel.season < season) | ((panel.season == season) & (panel.week < week))
    train = panel.loc[prior].copy()
    target = panel.loc[(panel.season == season) & (panel.week == week)].copy()
    if train.empty or target.empty:
        raise ValueError('Need earlier training games and a target week')
    x, xp = two_sided_rows(train), two_sided_rows(target)
    features = [c for c in x if c not in BASE_CONTEXT]
    x = x.replace([np.inf, -np.inf], np.nan)
    xp = xp.replace([np.inf, -np.inf], np.nan)
    fill = x.median().fillna(0.)
    x, xp = x.fillna(fill), xp.fillna(fill)
    center, scale = x[features].mean(), x[features].std(ddof=0)
    scale = scale.mask(scale < 1e-6, 1.)
    x[features], xp[features] = (x[features] - center) / scale, (xp[features] - center) / scale
    y = np.r_[train.away_score, train.home_score].astype('float32')
    if not np.isfinite(y).all():
        raise ValueError('Training scores must be finite')
    offset = float(y.mean())
    target.attrs['model_features'] = features
    target.attrs['context_names'] = BASE_CONTEXT.copy()
    return target, x.to_numpy('float32'), y - offset, xp.to_numpy('float32'), offset


def referee_tendencies(games, history, prior_games=20):
    """Pregame referee scoring averages, shrunk toward the prior league average."""
    result = games.reset_index(drop=True).copy()
    history = history.copy()
    history['ref_total'] = history.away_score + history.home_score
    history = history[np.isfinite(history.ref_total)]
    for (season, week), block in result.groupby(['season', 'week']):
        prior = history[(history.season >= season - 2) &
                        ((history.season < season) | ((history.season == season) & (history.week < week)))]
        league = float(prior.ref_total.mean()) if len(prior) else np.nan
        grouped = prior.dropna(subset=['referee']).groupby('referee').ref_total.agg(['sum', 'count'])
        counts = block.referee.map(grouped['count']).fillna(0)
        sums = block.referee.map(grouped['sum']).fillna(0)
        average = (sums + prior_games * league) / (counts + prior_games)
        result.loc[block.index, 'referee_prior_games'] = counts
        result.loc[block.index, 'referee_avg_total'] = average
        result.loc[block.index, 'referee_league_total'] = league
        result.loc[block.index, 'referee_total_delta'] = (average - league).fillna(0)
    return result


def feature_names(inputs, metrics=None):
    """metrics: optional override of METRICS (e.g. a feature-selected subset
    from optimus_prime's shared-model track) -- None (default, every existing
    caller) uses the full production metric list, unchanged."""
    if inputs not in INPUT_MODES:
        raise ValueError(f'inputs must be one of {INPUT_MODES}')
    metrics = metrics if metrics is not None else METRICS
    return ['diff_' + metric for metric in metrics] if inputs != 'separate' \
        else [f'{unit}_{metric}' for metric in metrics for unit in ['off', 'def']]


def scoring_rows(games, inputs='separate', metrics=None):
    """Away rows followed by home rows; each sees its offense and opposing defense."""
    feature_names(inputs, metrics)
    if inputs in ['percentile', 'zscore']:
        raise ValueError('Percentile/zscore require training references; use prepare()')
    metrics = metrics if metrics is not None else METRICS
    # Existing callers supply pandas. Avoid a pandas -> Polars -> pandas
    # round trip, which costs more than this small amount of arithmetic.
    if isinstance(games, pd.DataFrame):
        blocks = []
        for side, opponent in [('away', 'home'), ('home', 'away')]:
            values = {}
            for metric in metrics:
                offense = games[f'{side}_raw_off_{metric}'].to_numpy()
                defense = games[f'{opponent}_raw_def_{metric}'].to_numpy()
                if inputs == 'differential':
                    values[f'diff_{metric}'] = offense - defense
                else:
                    values[f'off_{metric}'], values[f'def_{metric}'] = offense, defense
            values['home_field'] = games.home_field_adv.to_numpy() if side == 'home' else np.zeros(len(games))
            values['rest_advantage'] = games[f'{side}_rest'].to_numpy() - games[f'{opponent}_rest'].to_numpy()
            blocks.append(pd.DataFrame(values))
        return pd.concat(blocks, ignore_index=True)
    columns = [f'{side}_raw_{unit}_{metric}' for side in ['away', 'home']
               for unit in ['off', 'def'] for metric in metrics]
    frame = games.select(columns + ['home_field_adv', 'away_rest', 'home_rest'])
    blocks = []
    for side, opponent in [('away', 'home'), ('home', 'away')]:
        expressions = []
        for metric in metrics:
            offense = pl.col(f'{side}_raw_off_{metric}')
            defense = pl.col(f'{opponent}_raw_def_{metric}')
            if inputs == 'differential':
                expressions.append((offense - defense).alias(f'diff_{metric}'))
            else:
                expressions.extend([offense.alias(f'off_{metric}'), defense.alias(f'def_{metric}')])
        expressions.extend([
            (pl.col('home_field_adv').cast(pl.Float64) if side == 'home' else pl.lit(0.)).alias('home_field'),
            (pl.col(f'{side}_rest') - pl.col(f'{opponent}_rest')).alias('rest_advantage'),
        ])
        blocks.append(frame.select(expressions))
    return pl.concat(blocks).to_pandas()


def matchup_representation(train, target, inputs, metrics):
    """Use stored production percentile/z-score differences, or raw shared
    inputs. percentile and zscore are parallel representations built the
    same way by data_crunchski_2.comp_stats -- normalize each team's stat
    against every team in that same week's league snapshot (never against
    its own history, never across concatenated training rows), then
    difference, then apply comp_stats' usage scaling post-normalization.
    zscore just reads the '_z' suffixed columns rather than the unsuffixed
    (rank-based) ones -- same off/def sign conventions, same scaling.

    Negate away-defense differences for the home offense perspective. Retain
    production's rank directions and its asymmetric usage scaling exactly.
    """
    if inputs in ['percentile', 'zscore']:
        suffix = '' if inputs == 'percentile' else '_z'
        results = []
        for games in [train, target]:
            blocks = []
            for side, unit, sign in [('away', 'off', 1.), ('home', 'def', -1.)]:
                values = {f'diff_{metric}': sign * games[f'away_{unit}_{metric}{suffix}'].to_numpy()
                          for metric in metrics}
                opponent = 'home' if side == 'away' else 'away'
                values['home_field'] = games.home_field_adv.to_numpy() if side == 'home' else np.zeros(len(games))
                values['rest_advantage'] = games[f'{side}_rest'].to_numpy() - games[f'{opponent}_rest'].to_numpy()
                blocks.append(pd.DataFrame(values))
            results.append(pd.concat(blocks, ignore_index=True))
        return tuple(results)
    if inputs in ['separate', 'differential']:
        return scoring_rows(train, inputs, metrics), scoring_rows(target, inputs, metrics)
    raise ValueError(f'Unknown input mode: {inputs}')


def prepare(games, season, week, groups=(), inputs='separate', metrics=None):
    """metrics: optional override of METRICS -- see feature_names' docstring."""
    features = feature_names(inputs, metrics)
    if 'referee' in groups and 'referee_total_delta' not in games:
        games = referee_tendencies(games, games)
    prior = (games.season < season) | ((games.season == season) & (games.week < week))
    train = games.loc[prior].copy()
    target = games.loc[(games.season == season) & (games.week == week)].copy()
    if len(train) < 2 or target.empty:
        raise ValueError('Need completed training games and a nonempty target week')
    y = np.r_[train.away_score, train.home_score].astype('float32')
    if not np.isfinite(y).all():
        raise ValueError('Shared scoring requires finite completed training scores')
    x, xp = matchup_representation(train, target, inputs, metrics if metrics is not None else METRICS)
    context_names = BASE_CONTEXT.copy()
    for group in groups:
        if group == 'referee':
            name = 'referee:prior_total_delta'
            for frame, source in [(x, train), (xp, target)]:
                frame[name] = np.tile(source.referee_total_delta.to_numpy(), 2)
            context_names.append(name)
            continue
        for column in GROUPS[group]:
            # Categories are learned from completed training games only.
            for value in sorted(train[column].dropna().astype(str).unique()):
                name = f'{group}:{column}={value}'
                for frame, source in [(x, train), (xp, target)]:
                    active = source[column].astype(str).eq(value).to_numpy(dtype='float32')
                    frame[name] = (np.r_[np.zeros(len(source)), active * source.home_field_adv]
                                   if group == 'stadium' else np.r_[active, active])
                context_names.append(name)
    fill = x.replace([np.inf, -np.inf], np.nan).median().fillna(0)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(fill)
    xp = xp.replace([np.inf, -np.inf], np.nan).fillna(fill)
    center, scale = x[features].mean(), x[features].std(ddof=0)
    scale = scale.mask(scale < 1e-6, 1)
    x[features], xp[features] = (x[features] - center) / scale, (xp[features] - center) / scale
    # Same role-specific transform for both sides; context is a separate additive term.
    offset = float(y.mean())
    columns = features + context_names
    target.attrs['context_names'] = context_names
    target.attrs['model_features'] = features
    return target, x[columns].to_numpy('float32'), y - offset, xp[columns].to_numpy('float32'), offset
