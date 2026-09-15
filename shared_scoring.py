"""Shared team-points challenger: one scoring function, applied to both teams."""
import inspect
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_crunchski_3 import (
    METRICS, FEATURES, GROUPS, BASE_CONTEXT, referee_tendencies,
    feature_names, scoring_rows, prepare,
)


def build_model(n_context=2, n_features=len(FEATURES)):
    from tensorflow import keras
    from model_shredski import create_model
    inputs = keras.Input(shape=(n_features + n_context,))
    matchup = keras.layers.Lambda(lambda x: x[:, :n_features])(inputs)
    context = keras.layers.Lambda(lambda x: x[:, n_features:n_features+2])(inputs)
    points = create_model(n_features)(matchup)
    adjustment = keras.layers.Dense(1, use_bias=False, name='venue_and_rest')(context)
    terms = [points, adjustment]
    if n_context > 2:
        extra = keras.layers.Lambda(lambda x: x[:, n_features+2:])(inputs)
        terms.append(keras.layers.Dense(1, use_bias=False, kernel_initializer='zeros',
                     kernel_regularizer=keras.regularizers.l2(.1), name='context_groups')(extra))
    model = keras.Model(inputs, keras.layers.Add()(terms))
    model.compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss='mse')
    return model


def fit_member(i, x, y, xp, seed, epochs, n_features=len(FEATURES)):
    from modelo_workers import initialize_worker
    initialize_worker()
    # Must precede TensorFlow's first import in each process. Python exceptions
    # still propagate; TF_CPP_MIN_LOG_LEVEL=0 restores native diagnostics.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    from model_shredski import integrated_gradients, permutation_importance, squared_error
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed + i)
    model = build_model(x.shape[1] - n_features, n_features)
    try:
        model.fit(x, y[:, None], epochs=epochs, verbose=0, callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=5)])
        prediction = np.asarray(model(xp, training=False)).reshape(-1)
        attrs, baseline, _ = integrated_gradients(
            model, tf.constant(xp), tf.zeros((1, x.shape[1])), m_steps=64)
        sample = np.random.default_rng(seed).choice(len(x), min(128, len(x)), replace=False)
        importance = permutation_importance(model, x[sample], y[sample], squared_error, seed + i)
        if not all(np.isfinite(a).all() for a in [prediction, attrs, baseline, importance]):
            raise ValueError(f'Non-finite shared-scoring member {i}')
        return prediction, attrs, baseline, importance
    finally:
        del model
        tf.keras.backend.clear_session()


def summarize(target, runs, offset):
    features = target.attrs.get('model_features', FEATURES)
    count = len(target)
    scores = np.stack([r[0] for r in runs]) + offset
    attrs = np.mean([r[1] for r in runs], axis=0)
    bases = np.mean([r[2] for r in runs], axis=0) + offset
    margins = scores[:, :count] - scores[:, count:]
    totals = scores[:, :count] + scores[:, count:]
    result = target[['away_team', 'home_team']].reset_index(drop=True).copy()
    result['away_points'], result['home_points'] = scores[:, :count].mean(0), scores[:, count:].mean(0)
    result['prediction'] = margins.mean(0)
    result['variance'] = margins.var(0, ddof=1) if len(runs) > 1 else 0.
    result['baseline'] = bases[:count] - bases[count:]
    result['total_prediction'] = totals.mean(0)
    result['total_variance'] = totals.var(0, ddof=1) if len(runs) > 1 else 0.
    result['total_baseline'] = bases[:count] + bases[count:]
    # Group the offense and opponent-defense effects for each matchup/metric.
    # Only metrics actually in `features` -- METRICS is the full universe,
    # but a narrowed subset (optimus_prime's shared-model track) means most
    # calls now see fewer than that.
    active_metrics = [m for m in METRICS if f'off_{m}' in features or f'diff_{m}' in features]
    for metric in active_metrics:
        columns = ([features.index('diff_' + metric)] if 'diff_' + metric in features else
                   [features.index(f'off_{metric}'), features.index(f'def_{metric}')])
        result[f'attr_away_off_{metric}'] = attrs[:count, columns].sum(axis=1)
        result[f'attr_away_def_{metric}'] = -attrs[count:, columns].sum(axis=1)
        result[f'total_attr_away_off_{metric}'] = result[f'attr_away_off_{metric}']
        result[f'total_attr_away_def_{metric}'] = -result[f'attr_away_def_{metric}']
    context_names = target.attrs.get('context_names', BASE_CONTEXT)
    for j, name in enumerate(context_names, len(features)):
        feature = {'home_field': 'home_field_adv', 'rest_advantage': 'away_rest_adv'}.get(name, 'context_' + name.split(':')[0])
        for prefix, sign in [('', -1), ('total_', 1)]:
            key = f'{prefix}attr_{feature}'
            result[key] = result.get(key, 0) + attrs[:count, j] + sign * attrs[count:, j]
    result['integration_residual'] = result.prediction - result.baseline - result.filter(regex='^attr_').sum(axis=1)
    result['total_integration_residual'] = result.total_prediction - result.total_baseline - result.filter(regex='^total_attr_').sum(axis=1)
    imps = np.stack([r[3] for r in runs])
    importance = pd.DataFrame({'feature': features + context_names,
                               'importance': imps.mean(0), 'std': imps.std(0)})
    return result, importance.sort_values('importance', ascending=False)


def fit_panel(panel, season, week, iterations=100, epochs=100, seed=1337, jobs=None, groups=(), inputs='separate',
             metrics=None, progress_position=0):
    """metrics: optional override of data_crunchski_3.METRICS -- a feature-
    selected subset from optimus_prime's shared-model track. None (default,
    every existing caller) uses the full production metric list, unchanged.
    n_features for the model is derived from target.attrs['model_features']
    below, so it adapts to a narrowed subset automatically.

    progress_position: tqdm `position` for the ensemble-member bar below --
    0 (default, every existing caller) is plain single-bar behavior; a
    caller running its own stacked dashboard of outer bars (optimus_prime's
    shared-model track) passes a position below its own bars so this one
    nests underneath instead of fighting them for the same terminal line."""
    import optimize_picks as op
    import utils
    from joblib import Parallel, delayed, parallel_config
    from modelo_workers import initialize_worker
    from tqdm import tqdm
    if iterations < 2 or epochs < 1:
        raise ValueError('Use at least two ensemble members and one epoch')
    jobs = min(iterations, int(os.environ.get('NFL_MODEL_JOBS', '8'))) if jobs is None else jobs
    if jobs < 1:
        raise ValueError('jobs must be positive')
    if groups:
        sched = pd.read_parquet('data/sched.parquet').replace(
            {'away_team': op.dc.RELOCATED_TEAMS, 'home_team': op.dc.RELOCATED_TEAMS})
        columns = list(dict.fromkeys(c for g in groups for c in GROUPS[g]))
        panel = panel.drop(columns=columns, errors='ignore').merge(sched[op.KEY + columns], on=op.KEY, validate='one_to_one')
        if 'referee' in groups:
            panel = referee_tendencies(panel, sched)
    target, x, y, xp, offset = prepare(panel, season, week, groups, inputs, metrics)
    fingerprint = [a.tobytes().hex() for a in [x, y, xp]]
    identity = [fingerprint, target[op.KEY].to_dict('list'), offset, iterations, epochs, seed, list(groups),
                target.attrs['context_names'], inputs, target.attrs['model_features']]
    cached = utils.cache_path('shared_scoring', identity, [__file__, 'data_crunchski_3.py', 'model_shredski.py', 'modelo_workers.py'])
    importance_path = cached.with_suffix('.importance.parquet')
    engine = inspect.getsource(build_model) + inspect.getsource(fit_member)
    raw_cache = utils.cache_path('shared_scoring_members', identity + [engine],
                                ['model_shredski.py', 'modelo_workers.py']).with_suffix('.npz')
    if cached.exists() and importance_path.exists():
        details, importance = pd.read_parquet(cached), pd.read_parquet(importance_path)
        tqdm.write('Shared scoring: cached')
    else:
        if raw_cache.exists():
            with np.load(raw_cache) as saved:
                runs = list(zip(*(saved[name] for name in ['scores', 'attrs', 'bases', 'importance'])))
            tqdm.write('Shared scoring: rebuilding report from cached members')
        else:
            with parallel_config(backend='loky', inner_max_num_threads=1):
                with Parallel(n_jobs=jobs, return_as='generator', batch_size=1, initializer=initialize_worker) as pool:
                    runs = list(tqdm(pool(delayed(fit_member)(i, x, y, xp, seed, epochs, len(target.attrs['model_features']))
                                         for i in range(iterations)), total=iterations, position=progress_position,
                                     leave=False, desc=f'Shared scoring ({season} wk{week}, {jobs} CPU workers)'))
            temporary = raw_cache.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **{name: np.stack([r[i] for r in runs])
                                for i, name in enumerate(['scores', 'attrs', 'bases', 'importance'])})
            os.replace(temporary, raw_cache)
        details, importance = summarize(target, runs, offset)
        utils.save_parquet(details, cached)
        utils.save_parquet(importance, importance_path)
    return target, details, importance


def fit_two_sided(panel, season, week, iterations=100, epochs=100, seed=1337, jobs=8):
    """Same team-score network, now with both matchups and weather as inputs.

    Returns (details, importance), schema-compatible with summarize()'s
    output (attr_away_off_{metric}/attr_away_def_{metric}/baseline/
    integration_residual, plus total_ counterparts) -- market_details and
    weekly_packet.matchup_attribution work on it unchanged.

    away_score and home_score are two independent evaluations of the same
    shared-weight network, each on its own input row -- not one function
    composed with itself, so there's no chain rule linking the two rows'
    slots to each other. Integrated gradients' completeness axiom
    guarantees sum(attributions) == f(input) - f(baseline) separately for
    each row. margin = away_score - home_score, so margin's attribution
    for ANY single input slot is just that slot's away-row attribution
    minus its home-row attribution -- full stop, no cross-referencing
    between different slots (e.g. own_off_{metric} and own_def_{metric}
    each get their own independent slot-level attribution; they are not
    combined with each other). total = away_score + home_score sums
    instead of subtracting. Every slot (metric or context/weather) uses
    this identical away-minus/plus-home rule; only the display label
    differs (own_off_{metric} -> away_off_{metric}, etc.). Verified
    empirically: prediction - baseline - sum(attr_) is ~0 (see
    tests/test_two_sided.py) -- an earlier, over-engineered version of
    this that tried to cross-reference off_{metric} with def_{metric}
    broke that identity; this simpler version is the one that's actually
    correct."""
    import hashlib
    import utils
    from data_crunchski_3 import prepare_two_sided
    from joblib import Parallel, delayed, parallel_config
    from modelo_workers import initialize_worker
    from tqdm import tqdm
    if iterations < 2 or jobs < 1 or epochs < 1:
        raise ValueError('Require iterations >= 2, jobs >= 1 and epochs >= 1')
    target, x, y, xp, offset = prepare_two_sided(panel, season, week)
    names = target.attrs['model_features'] + BASE_CONTEXT
    digest = hashlib.sha256(b''.join(a.tobytes() for a in [x, y, xp])).hexdigest()
    identity = [digest, names, offset, season, week, iterations, epochs, seed,
               target[['away_team', 'home_team']].to_dict('list')]
    cached = utils.cache_path('two_sided_scores', identity,
                              [__file__, 'data_crunchski_3.py', 'model_shredski.py', 'modelo_workers.py'])
    importance_path = cached.with_suffix('.importance.parquet')
    if cached.exists() and importance_path.exists():
        print(f'  {season} wk{week}: two-sided scores cached', flush=True)
        return pd.read_parquet(cached), pd.read_parquet(importance_path)
    n_features = len(target.attrs['model_features'])
    with parallel_config(backend='loky', n_jobs=min(jobs, iterations), inner_max_num_threads=1):
        runs = list(tqdm(Parallel(return_as='generator', initializer=initialize_worker)(
            delayed(fit_member)(i, x, y, xp, seed, epochs, n_features) for i in range(iterations)),
            total=iterations, desc=f'Two-sided scores ({season} wk{week})'))
    count = len(target)
    scores = np.stack([r[0] for r in runs]) + offset
    away, home = scores[:, :count], scores[:, count:]
    result = target[['away_team', 'home_team']].reset_index(drop=True).copy()
    result['away_points'], result['home_points'] = away.mean(0), home.mean(0)
    result['prediction'], result['variance'] = (away - home).mean(0), (away - home).var(0, ddof=1)
    result['total_prediction'], result['total_variance'] = (away + home).mean(0), (away + home).var(0, ddof=1)
    attrs = np.mean([r[1] for r in runs], axis=0)
    bases = np.mean([r[2] for r in runs], axis=0) + offset
    away_base, home_base = bases[:count], bases[count:]
    result['baseline'], result['total_baseline'] = away_base - home_base, away_base + home_base

    # IG completeness guarantees sum(attributions) == f(input) - f(baseline)
    # for the away row and, separately, for the home row -- so margin's
    # (= away_score - home_score) attribution for ANY single input slot is
    # just that slot's away-row attribution minus its home-row attribution,
    # full stop, no cross-referencing between different slots. (Total sums
    # instead of subtracting.) own_off_{metric}/own_def_{metric} each get
    # their own slot-level attribution, labeled away_off_{metric}/
    # away_def_{metric} for display -- NOT combined with each other.
    features = target.attrs['model_features']
    names_all = features + BASE_CONTEXT
    def label(name):
        if name.startswith('own_off_'):
            return f"away_off_{name[len('own_off_'):]}"
        if name.startswith('own_def_'):
            return f"away_def_{name[len('own_def_'):]}"
        return {'home_field': 'home_field_adv', 'rest_advantage': 'away_rest_adv'}.get(name, f'context_{name}')
    for j, name in enumerate(names_all):
        feature = label(name)
        result[f'attr_{feature}'] = attrs[:count, j] - attrs[count:, j]
        result[f'total_attr_{feature}'] = attrs[:count, j] + attrs[count:, j]
    result['integration_residual'] = result.prediction - result.baseline - result.filter(regex='^attr_').sum(axis=1)
    result['total_integration_residual'] = (result.total_prediction - result.total_baseline
                                            - result.filter(regex='^total_attr_').sum(axis=1))
    imps = np.stack([r[3] for r in runs])
    importance = pd.DataFrame({'feature': names, 'importance': imps.mean(0),
                               'std': imps.std(0)}).sort_values('importance', ascending=False)
    utils.save_parquet(result, cached)
    utils.save_parquet(importance, importance_path)
    return result, importance


def market_details(details, market):
    if market == 'spread':
        return details.copy()
    result = details.drop(columns=[c for c in details if c.startswith('attr_')]).copy()
    for column in ['prediction', 'variance', 'baseline', 'integration_residual']:
        result[column] = result['total_' + column]
    for column in details.filter(regex='^total_attr_'):
        result[column[len('total_'):]] = details[column]
    return result


def two_sided_packet(season, week, lookback=20, train_window=100, iterations=100, epochs=100,
                     seed=1337, jobs=None, weather_file=None, forecast_file=None):
    """Single-week two-sided packet -- same model/inputs as backtester.py's
    --model two-sided (league z-scores, symmetric usage scaling, historical
    weather), but one target week instead of a season-long backtest, written
    to data/results/{season}_{week}_{lookback}/packet_shared/ via
    weekly_packet.write_packets. This is the ongoing production path for
    that packet; fit_panel/preview (the older percentile-diff
    representation) are retired.

    forecast_file: falls back to a pulled forecast (pull_weather.py --season
    ... --week ... --mode live) for the target week if it hasn't been played
    yet and so has no historical/reanalysis weather -- see
    data_crunchski_3.attach_historical_weather. Defaults to
    data/weather/forecasts.parquet if that file exists, else omitted. If
    games are still missing weather (no forecast pulled yet, or a stale one
    that doesn't cover this week), this pulls a live forecast for the
    target week itself instead of failing outright -- see the ValueError
    handler below. Also refreshes data/sched.parquet and the affected
    data/pbp/pbp_{season}.parquet (same as main.py --refresh) if any prior
    (already-should-be-final) week is missing scores in the cached
    schedule -- that's staleness, not a game still in progress, and
    training on it silently would mean skipping real recent games."""
    from types import SimpleNamespace
    import data_crunchski_3 as dc3
    from backtester import KEY, build_panel, history_weeks, market_panel, settle
    from weekly_packet import write_packets
    sched = pd.read_parquet('data/sched.parquet')
    prior = (sched.season < season) | ((sched.season == season) & (sched.week < week))
    stale = sched.loc[prior & sched.away_score.isna()]
    if not stale.empty:
        stale_seasons = sorted(stale.season.unique().tolist())
        print(f'{len(stale)} prior game(s) in data/sched.parquet missing a score (season(s) {stale_seasons}) -- '
             'this should already be final, so refreshing schedule + play-by-play before continuing...', flush=True)
        import data_pullson
        data_pullson.pull_sched(stale_seasons)
        data_pullson.pull_pbp(stale_seasons)
    weather_file = Path(weather_file or 'data/weather/historical_features.parquet')
    if not weather_file.exists():
        raise ValueError(f'Missing historical weather: {weather_file}')
    if forecast_file is None:
        default_forecast = Path('data/weather/forecasts.parquet')
        forecast_file = default_forecast if default_forecast.exists() else None
    span = history_weeks(SimpleNamespace(start_season=season, season=season, week=week))
    panel = build_panel(season, week, span + train_window - 20, lookback, 'steep', use_scaling=False)
    try:
        panel = dc3.attach_historical_weather(panel, weather_file, forecast_file)
    except ValueError as error:
        if 'Historical weather missing' not in str(error):
            raise
        # Training-window games always have real reanalysis (see
        # attach_historical_weather's docstring) -- this only ever fires
        # for the target week itself, i.e. it hasn't been played yet and
        # has no forecast pulled (or a stale one) covering it. Pull one now
        # instead of making that a manual "go run pull_weather.py first" step.
        print(f'{error}\nPulling a live weather forecast for {season} wk{week} to fill the gap...', flush=True)
        import pull_weather
        forecast_file = Path(forecast_file or 'data/weather/forecasts.parquet')
        try:
            games = pull_weather.scheduled_games(season, week, 'live', decision_hours=24)
            pull_args = SimpleNamespace(mode='live', decision_hours=24, publication_hours=8,
                                        output=str(forecast_file), cache_dir='data/cache/open_meteo', refresh=False)
            pull_weather.pull(games, pull_args)
        except Exception as pull_error:
            raise ValueError(f'{error} (auto-pull also failed: {pull_error})') from pull_error
        panel = dc3.attach_historical_weather(panel, weather_file, forecast_file)
    target_rows = panel[(panel.season == season) & (panel.week == week)]
    if target_rows.empty:
        raise ValueError(f'{season} wk{week}: not present in the prepared panel')
    wid = int(target_rows.week_id.iloc[0])
    history = panel[panel.week_id < wid]
    regular = sorted(history.loc[history.game_type.eq('REG'), 'week_id'].unique())
    if len(regular) < train_window:
        raise ValueError(f'{season} wk{week}: need {train_window} prior regular training weeks; got {len(regular)}')
    data = pd.concat([history[history.week_id >= regular[-train_window]], target_rows]).sort_values(KEY)
    details, importance = fit_two_sided(data, season, week, iterations, epochs, seed, jobs or min(iterations, 8))
    config = dict(model=f'two-sided-team-points-v1 / {iterations} members', calculation='two-sided-team-points-v1',
                  lookback=lookback, train_window=train_window, status='PASS', attribution_schema=2,
                  reason='Experimental two-sided scoring with historical weather; no validated betting cutoffs',
                  importance_note='Each bar is a paired refit/drop test: remove one feature (from both the '
                                  'away-attacking and home-attacking sides of this shared-weight network), '
                                  'retrain, and see how prediction error on the training sample changed. '
                                  'Positive means removing it made the model worse (it was pulling weight); '
                                  'negative means removing it made the model better (it was actively hurting '
                                  'predictions). This is a training-sample diagnostic, not held-out betting '
                                  'evidence, and correlated features can substitute for one another -- a low '
                                  "score doesn't mean a feature is useless, just that something else covers it.",
                  baseline_note='Both team scores share one network; each metric sums the away-attacking and '
                                'home-attacking matchups (see fit_two_sided\'s docstring for the derivation).')
    # write_packets needs its usual index/spread/total/importance/stats
    # pages + CSVs on disk to cross-reference each other and to bundle into
    # one portable file (bundle_single_file, called from inside
    # write_packets) -- only packet.html and the small, prediction-free
    # {market}_config.json (model/notes metadata, already visible as text
    # on the page -- kept only because refresh_packet() reads it back) are
    # worth keeping afterward; the rest is built in a scratch directory and
    # discarded. NOTE: refresh_packet(shared=True) can no longer restyle a
    # saved packet without refitting -- the *_details.csv/*_importance.csv
    # it needs for that no longer get kept on disk. That's an accepted
    # tradeoff for not cluttering data/results/, not an oversight.
    final_folder = Path(f'data/results/{season}_{week}_{lookback}/packet_shared') / f'{season}_{week:02d}'
    with tempfile.TemporaryDirectory() as scratch:
        scratch_output = Path(scratch)
        for market in ['spread', 'total']:
            predictions = market_panel(target_rows, market).merge(
                market_details(details, market), on=['away_team', 'home_team'], validate='one_to_one')
            predictions['edge'] = predictions.prediction - predictions.market_base
            write_packets(settle(predictions), panel, importance, dict(config, market=market), scratch_output)
        scratch_folder = scratch_output / f'{season}_{week:02d}'
        bundled = scratch_folder / 'packet.html'
        if not bundled.exists():
            raise ValueError(f'{season} wk{week}: packet bundling failed -- no pages were written')
        # Wipe rather than merge -- a folder from before this scratch-dir
        # change existed would otherwise keep its old index.html/*.csv/etc
        # forever (this move only ever adds files, never removes stale
        # ones). final_folder is exclusively owned by this function.
        shutil.rmtree(final_folder, ignore_errors=True)
        final_folder.mkdir(parents=True, exist_ok=True)
        final_path = final_folder / 'packet.html'
        shutil.move(str(bundled), str(final_path))
        for market in ['spread', 'total']:
            saved_config = scratch_folder / f'{market}_config.json'
            if saved_config.exists():
                shutil.move(str(saved_config), str(final_folder / saved_config.name))
    print(f'Packet: {final_path}')
    return details


def preview(season, week, lookback=20, iterations=100, epochs=100, seed=1337, jobs=None, groups=(), inputs='differential'):
    import optimize_picks as op
    from weekly_packet import write_packets
    groups = tuple(sorted(set(groups)))
    panel = op.build_panel(season, week, lookback, lookback, 'mean')
    target, details, importance = fit_panel(panel, season, week, iterations, epochs, seed, jobs, groups, inputs)
    suffix = ('_differential' if inputs == 'differential' else '') + ('_' + '_'.join(groups) if groups else '')
    output = Path(f'data/results/{season}_{week}_{lookback}/packet_shared{suffix}')
    config = dict(model=f'shared-team-points / {inputs} / {iterations} members / seed {seed}',
                  market='spread', lookback=lookback, calculation='shared-scoring-v1', status='PASS', attribution_schema=2,
                  context_groups=list(groups), input_mode=inputs,
                  reason='Experimental shared scoring; no validated betting cutoffs',
                  headline_href=f'../../html_{season}_{week}_{lookback}.html',
                  baseline_note='Both scores use the same neutral, mean-input reference; it cancels in the margin. Venue and rest are separate learned adjustments. Each metric groups offense and opposing-defense contributions.',
                  importance_note='Shared team-score training-sample permutation MSE diagnostic; not held-out betting evidence.')
    if inputs == 'differential':
        config['baseline_note'] = ('Both scores use the same mean-differential neutral reference, which cancels in the margin. '
                                   'Each metric is raw offense minus opposing defense, then standardized. Context remains separate.')
    for market in ['spread', 'total']:
        predictions = op.market_panel(target, market).merge(market_details(details, market),
                        on=['away_team', 'home_team'], validate='one_to_one')
        predictions['edge'] = predictions.prediction - predictions.market_base
        config = dict(config, market=market)
        if market == 'total':
            config['baseline_note'] = 'The two shared neutral scoring baselines add for the total. Contributions add rather than subtract.'
        write_packets(op.settle(predictions), panel, importance, config, output)
    shown = details[['away_team', 'home_team', 'away_points', 'home_points']].copy()
    shown['away_spread'], shown['sd'] = -details.prediction, np.sqrt(details.variance)
    shown['total'], shown['total_sd'] = details.total_prediction, np.sqrt(details.total_variance)
    print(shown.to_string(index=False, float_format=lambda value: f'{value:.1f}'))
    print(f'Packet: {output / f"{season}_{week:02d}" / "index.html"}')
    return details


# No CLI here -- weekly_packet.py is the canonical entry point for building a
# packet (`python weekly_packet.py --model two-sided --season ... --week ...`),
# parameterized by which model to fit. This file just holds the model code.
