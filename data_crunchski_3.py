"""Reusable matchup preparation; historical team-stat math stays in crunchski_2."""
import numpy as np
import pandas as pd
import polars as pl

from edge_scan import PRODUCTION_FEATURES

METRICS = [f[len('away_off_'):] for f in PRODUCTION_FEATURES if f.startswith('away_off_')]
FEATURES = [f'{unit}_{metric}' for metric in METRICS for unit in ['off', 'def']]
# Model 2.1's additions (model_spec selects them): EPA per play and success
# rate, split by play type because overall EPA is ~0.97 correlated with its
# passing half. Each one becomes an own-offense and an own-defense input, and
# the pass/run naming is what earns them usage scaling in two_sided_rows.
EPA_METRICS = ['pass_epa_pp', 'run_epa_pp', 'pass_success_%', 'run_success_%']


def use_epa(enabled=True):
    """Add (or drop) EPA_METRICS in the shared METRICS/FEATURES lists, in
    place, so every module that imported them sees the same set."""
    base = [m for m in METRICS if m not in EPA_METRICS]
    METRICS[:] = base + (EPA_METRICS if enabled else [])
    FEATURES[:] = [f'{unit}_{metric}' for metric in METRICS for unit in ['off', 'def']]
    return list(METRICS)
GROUPS = {'stadium': ['stadium_id'], 'field': ['surface', 'roof'], 'referee': ['referee']}
BASE_CONTEXT = ['home_field', 'rest_advantage']
# Model 2.2's addition (model_spec selects it): how much further this team
# travelled to the game than its opponent did, from travel.attach_travel's
# per-game miles. Measured in thousands of miles so it sits in the same range
# as a rest-day difference -- context inputs skip standardization and feed a
# plain linear layer, where a raw 2,700 would swamp the other two.
TRAVEL_CONTEXT = 'travel_advantage'
TRAVEL_SCALE = 1000.


# ---------------------------------------------------------- roster shrinkage
# Which unit's turnover makes a metric untrustworthy. This map is asserted,
# not derived: an attempt to derive it from stat persistence failed because
# the units do not turn over independently (41% of their variance is one
# common "this team is stable" factor), and the joint fit returned impossible
# signs -- keeping your backs made rushing stats persist LESS. So it is a
# judgement, written in one place to be argued with.
#
# The quarterback is deliberately absent: qb_elo already carries him, and
# folding him in here would discount the one position the model handles well.
METRIC_UNITS = {
    'pass_ypp': 'receivers', 'pass_completion_%': 'receivers', 'explosive_pass_%': 'receivers',
    'pass_epa_pp': 'receivers', 'pass_success_%': 'receivers',
    'sack_%': 'oline', 'qb_hit_%': 'oline',
    'run_ypp': 'backs', 'explosive_run_%': 'backs', 'run_epa_pp': 'backs',
    'run_success_%': 'backs', 'stuff_%': 'oline',
    'first_down_pp': 'receivers', 'series_success_%': 'receivers',
    'third_down_%': 'receivers', 'fourth_down_%': 'receivers',
    'turnovers_pp': 'receivers', 'penalties_pp': 'oline',
}
# A defensive input is trusted according to the defence that produced it.
DEFENSIVE_UNITS = {'receivers': 'pass_def', 'oline': 'run_def', 'backs': 'run_def'}
ROSTER_SHRINK = {}          # (season, team) -> {unit: retention}; empty = feature off
SHRINK_FADE_WEEKS = 6.      # by when the current season's own games have taken over


def use_roster_shrink(table=None, fade_weeks=6.):
    """Shrink a team's carried-forward stats toward the league average in
    proportion to how much of the relevant unit left.

    The inputs are league z-scores, so the league average is exactly zero and
    shrinking is a multiply: a team returning 60% of its receivers carries 60%
    of its measured passing edge and 40% of nothing.

    It fades with the season. In week 1 every stat is last year's and the full
    discount applies; by `fade_weeks` the lookback is mostly games this roster
    actually played, so there is nothing stale to discount. Pass None to turn
    it off."""
    global ROSTER_SHRINK, SHRINK_FADE_WEEKS
    SHRINK_FADE_WEEKS = float(fade_weeks)
    if table is None:
        ROSTER_SHRINK = {}
        return {}
    ROSTER_SHRINK = {(int(r.season), r.team): {u: getattr(r, u) for u in
                                               ['receivers', 'backs', 'oline', 'pass_def', 'run_def']}
                     for r in table.itertuples()}
    return ROSTER_SHRINK


def shrink_factor(season, team, metric, unit_side, week):
    """How much of this team's measured edge in `metric` to keep."""
    if not ROSTER_SHRINK:
        return 1.
    units = ROSTER_SHRINK.get((int(season), team))
    if units is None:
        return 1.
    unit = METRIC_UNITS.get(metric, 'receivers')
    if unit_side == 'def':
        unit = DEFENSIVE_UNITS.get(unit, 'pass_def')
    retention = units.get(unit)
    if retention is None or not np.isfinite(retention):
        return 1.
    # Full discount in week 1, none once the current season has taken over.
    stale = max(0., 1. - (float(week) - 1.) / SHRINK_FADE_WEEKS)
    return 1. - (1. - float(retention)) * stale


def use_travel(enabled=True):
    """Add (or drop) the travel input in the shared BASE_CONTEXT list, in
    place, so every module that imported it sees the same set."""
    BASE_CONTEXT[:] = [c for c in BASE_CONTEXT if c != TRAVEL_CONTEXT] + ([TRAVEL_CONTEXT] if enabled else [])
    return list(BASE_CONTEXT)
# Two independent choices, not one list. COMBINATION decides whether a
# team's offense and the opposing defense stay apart or get differenced;
# NORMALIZATION decides what scale they are on. The old names conflated them
# ('zscore' meant z-scored AND differenced), which left the cell the
# production model actually uses -- separate + zscore, see two_sided_rows --
# unreachable from every other architecture.
INPUT_MODES = ['separate', 'differential']
NORMALIZATIONS = ['raw', 'zscore', 'percentile']
LEGACY_INPUT_MODES = {'percentile': ('differential', 'percentile'), 'zscore': ('differential', 'zscore')}
HISTORICAL_WEATHER = ['feels_like_f', 'wind_mph', 'precip_inches', 'rain_inches',
                      'snowfall_inches', 'snow_depth_inches']
# The subset the two-sided model actually takes as inputs. Rain and snowfall
# are parts of precip_inches, snow depth was nearly always zero, and indoor
# games already carry fixed 72°F / calm / dry readings (attach_historical_weather),
# so no separate indoor flag -- a dome now looks like a mild, calm, dry day.
# HISTORICAL_WEATHER stays the loading/coverage contract for the weather files.
MODEL_WEATHER = ['feels_like_f', 'wind_mph', 'precip_inches']


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
        if ROSTER_SHRINK:
            # Same treatment for both rows, each against its own roster.
            for metric in METRICS:
                for unit_side in ['off', 'def']:
                    owner = games[f'{side}_team'] if unit_side == 'off' else games[f'{opponent}_team']
                    factor = np.array([shrink_factor(s, t, metric, unit_side, w)
                                       for s, t, w in zip(games.season, owner, games.week)])
                    values[f'own_{unit_side}_{metric}'] = values[f'own_{unit_side}_{metric}'] * factor
        for column in MODEL_WEATHER:
            values['weather_' + column] = games[column].to_numpy()
        values['home_field'] = games.home_field_adv.to_numpy() if side == 'home' else np.zeros(len(games))
        values['rest_advantage'] = games[f'{side}_rest'].to_numpy() - games[f'{opponent}_rest'].to_numpy()
        if TRAVEL_CONTEXT in BASE_CONTEXT:
            values[TRAVEL_CONTEXT] = (games[f'{side}_travel_miles'].to_numpy()
                                      - games[f'{opponent}_travel_miles'].to_numpy()) / TRAVEL_SCALE
        blocks.append(pd.DataFrame(values))
    return pd.concat(blocks, ignore_index=True)


def prepare_two_sided(panel, season, week, objective='points'):
    """objective='points' (the two-sided model): each row's label is that
    team's score, and the spread comes from subtracting the two predictions.

    objective='margin' (sided-spread): each row's label is the margin from
    that team's own point of view, so both rows are estimating the same
    quantity with opposite signs and the spread is their AVERAGE. That is the
    point of it -- subtracting two score estimates adds their errors, while
    averaging two margin estimates cancels them, which halves the variance of
    the only number a spread bet cares about. The cost is the scoreline
    anchor: nothing forces the model to commit to 24-17, and no total can be
    recovered, so this objective predicts spreads only."""
    prior = (panel.season < season) | ((panel.season == season) & (panel.week < week))
    train = panel.loc[prior].copy()
    target = panel.loc[(panel.season == season) & (panel.week == week)].copy()
    if train.empty or target.empty:
        raise ValueError('Need earlier training games and a target week')
    # "Prior" is a week/season boundary, not a completion check -- e.g.
    # generating this week's packet before last week's Monday-night game
    # has finished means that one row still has no final score. Drop it
    # from training rather than failing the whole run over one game.
    incomplete = train.away_score.isna() | train.home_score.isna()
    if incomplete.any():
        print(f'{int(incomplete.sum())} prior game(s) missing a final score -- excluding from training: '
             f'{", ".join(train.loc[incomplete, "game_id"])}', flush=True)
        train = train.loc[~incomplete].copy()
        if train.empty:
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
    if objective == 'margin':
        margin = (train.away_score - train.home_score).to_numpy()
        y = np.r_[margin, -margin].astype('float32')
    elif objective == 'points':
        y = np.r_[train.away_score, train.home_score].astype('float32')
    else:
        raise ValueError(f"objective must be 'points' or 'margin', got {objective!r}")
    if not np.isfinite(y).all():
        raise ValueError('Training scores must be finite')
    offset = float(y.mean())
    target.attrs['model_features'] = features
    target.attrs['context_names'] = BASE_CONTEXT.copy()
    target.attrs['objective'] = objective
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
            if TRAVEL_CONTEXT in BASE_CONTEXT and f'{side}_travel_miles' in games:
                values[TRAVEL_CONTEXT] = (games[f'{side}_travel_miles'].to_numpy()
                                          - games[f'{opponent}_travel_miles'].to_numpy()) / TRAVEL_SCALE
            blocks.append(pd.DataFrame(values))
        return pd.concat(blocks, ignore_index=True)
    columns = [f'{side}_raw_{unit}_{metric}' for side in ['away', 'home']
               for unit in ['off', 'def'] for metric in metrics]
    travel_columns = (['away_travel_miles', 'home_travel_miles']
                      if TRAVEL_CONTEXT in BASE_CONTEXT and 'away_travel_miles' in games.columns else [])
    frame = games.select(columns + ['home_field_adv', 'away_rest', 'home_rest'] + travel_columns)
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
        if travel_columns:
            expressions.append(((pl.col(f'{side}_travel_miles') - pl.col(f'{opponent}_travel_miles'))
                                / TRAVEL_SCALE).alias(TRAVEL_CONTEXT))
        blocks.append(frame.select(expressions))
    return pl.concat(blocks).to_pandas()


def normalized_rows(games, inputs, metrics, suffix):
    """League-normalized matchup inputs: z-scores ('_z') or percentiles ('').

    comp_stats stores these already differenced and already usage-scaled --
    away_off_{metric} is this week's away offense measured against the whole
    league and then set against the home defense. So 'differential' takes one
    number per metric, and 'separate' keeps the offense-vs-defense and
    defense-vs-offense matchups as two inputs, which is what two_sided_rows
    does and what the two-sided model's results rest on."""
    blocks = []
    for side, opponent in [('away', 'home'), ('home', 'away')]:
        values = {}
        for metric in metrics:
            away_off = games[f'away_off_{metric}{suffix}'].to_numpy()
            away_def = games[f'away_def_{metric}{suffix}'].to_numpy()
            own_off, own_def = (away_off, away_def) if side == 'away' else (-away_def, -away_off)
            if inputs == 'differential':
                values[f'diff_{metric}'] = own_off
            else:
                values[f'off_{metric}'], values[f'def_{metric}'] = own_off, own_def
        values['home_field'] = games.home_field_adv.to_numpy() if side == 'home' else np.zeros(len(games))
        values['rest_advantage'] = games[f'{side}_rest'].to_numpy() - games[f'{opponent}_rest'].to_numpy()
        blocks.append(pd.DataFrame(values))
    return pd.concat(blocks, ignore_index=True)


def matchup_representation(train, target, inputs, metrics, normalize='raw'):
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
    if inputs in LEGACY_INPUT_MODES:            # 'zscore'/'percentile' as a single word
        inputs, normalize = LEGACY_INPUT_MODES[inputs]
    if inputs not in INPUT_MODES:
        raise ValueError(f'Unknown input mode: {inputs}; have {", ".join(INPUT_MODES)}')
    if normalize not in NORMALIZATIONS:
        raise ValueError(f'Unknown normalization: {normalize}; have {", ".join(NORMALIZATIONS)}')
    if normalize == 'raw':
        return scoring_rows(train, inputs, metrics), scoring_rows(target, inputs, metrics)
    suffix = '_z' if normalize == 'zscore' else ''
    return (normalized_rows(train, inputs, metrics, suffix),
            normalized_rows(target, inputs, metrics, suffix))


def prepare(games, season, week, groups=(), inputs='separate', metrics=None, normalize='raw'):
    """metrics: optional override of METRICS -- see feature_names' docstring.
    normalize: 'raw', 'zscore' or 'percentile' -- the scale the matchup inputs
    are measured on, independent of whether they are differenced."""
    if inputs in LEGACY_INPUT_MODES:
        inputs, normalize = LEGACY_INPUT_MODES[inputs]
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
    x, xp = matchup_representation(train, target, inputs, metrics if metrics is not None else METRICS,
                                   normalize)
    context_names = [name for name in BASE_CONTEXT if name in x.columns]
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
