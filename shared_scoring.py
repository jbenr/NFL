"""Shared team-points challenger: one scoring function, applied to both teams."""
import argparse
import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd

from edge_scan import PRODUCTION_FEATURES

METRICS = [f[len('away_off_'):] for f in PRODUCTION_FEATURES if f.startswith('away_off_')]
FEATURES = [f'{unit}_{metric}' for metric in METRICS for unit in ['off', 'def']]
GROUPS = {'stadium': ['stadium_id'], 'field': ['surface', 'roof'], 'referee': ['referee']}
BASE_CONTEXT = ['home_field', 'rest_advantage']


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


def feature_names(inputs):
    if inputs not in ['separate', 'differential']:
        raise ValueError('inputs must be separate or differential')
    return ['diff_' + metric for metric in METRICS] if inputs == 'differential' else FEATURES


def scoring_rows(games, inputs='separate'):
    """Away rows followed by home rows; each sees its offense and opposing defense."""
    blocks = []
    for side, opponent in [('away', 'home'), ('home', 'away')]:
        block = pd.DataFrame(index=games.index)
        for metric in METRICS:
            offense = games[f'{side}_raw_off_{metric}']
            defense = games[f'{opponent}_raw_def_{metric}']
            if inputs == 'differential':
                block[f'diff_{metric}'] = offense - defense
            else:
                block[f'off_{metric}'] = offense
                block[f'def_{metric}'] = defense
        block['home_field'] = games.home_field_adv if side == 'home' else 0.
        block['rest_advantage'] = games[f'{side}_rest'] - games[f'{opponent}_rest']
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def prepare(games, season, week, groups=(), inputs='separate'):
    features = feature_names(inputs)
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
    x, xp = scoring_rows(train, inputs), scoring_rows(target, inputs)
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
    for metric in METRICS:
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


def fit_panel(panel, season, week, iterations=100, epochs=100, seed=1337, jobs=None, groups=(), inputs='separate'):
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
    target, x, y, xp, offset = prepare(panel, season, week, groups, inputs)
    fingerprint = [a.tobytes().hex() for a in [x, y, xp]]
    identity = [fingerprint, target[op.KEY].to_dict('list'), offset, iterations, epochs, seed, list(groups),
                target.attrs['context_names'], inputs, target.attrs['model_features']]
    cached = utils.cache_path('shared_scoring', identity, [__file__, 'model_shredski.py', 'modelo_workers.py'])
    importance_path = cached.with_suffix('.importance.parquet')
    engine = inspect.getsource(build_model) + inspect.getsource(fit_member)
    raw_cache = utils.cache_path('shared_scoring_members', identity + [engine],
                                ['model_shredski.py', 'modelo_workers.py']).with_suffix('.npz')
    if cached.exists() and importance_path.exists():
        details, importance = pd.read_parquet(cached), pd.read_parquet(importance_path)
        print('Shared scoring: cached')
    else:
        if raw_cache.exists():
            with np.load(raw_cache) as saved:
                runs = list(zip(*(saved[name] for name in ['scores', 'attrs', 'bases', 'importance'])))
            print('Shared scoring: rebuilding report from cached members')
        else:
            with parallel_config(backend='loky', inner_max_num_threads=1):
                with Parallel(n_jobs=jobs, return_as='generator', batch_size=1, initializer=initialize_worker) as pool:
                    runs = list(tqdm(pool(delayed(fit_member)(i, x, y, xp, seed, epochs, len(target.attrs['model_features']))
                                         for i in range(iterations)), total=iterations,
                                     desc=f'Shared scoring ({season} wk{week}, {jobs} CPU workers)'))
            temporary = raw_cache.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **{name: np.stack([r[i] for r in runs])
                                for i, name in enumerate(['scores', 'attrs', 'bases', 'importance'])})
            os.replace(temporary, raw_cache)
        details, importance = summarize(target, runs, offset)
        utils.save_parquet(details, cached)
        utils.save_parquet(importance, importance_path)
    return target, details, importance


def market_details(details, market):
    if market == 'spread':
        return details.copy()
    result = details.drop(columns=[c for c in details if c.startswith('attr_')]).copy()
    for column in ['prediction', 'variance', 'baseline', 'integration_residual']:
        result[column] = result['total_' + column]
    for column in details.filter(regex='^total_attr_'):
        result[column[len('total_'):]] = details[column]
    return result


def preview(season, week, lookback=20, iterations=100, epochs=100, seed=1337, jobs=None, groups=(), inputs='differential'):
    import optimize_picks as op
    from weekly_packet import write_packets
    groups = tuple(sorted(set(groups)))
    panel = op.build_panel(season, week, lookback, lookback, 'legacy')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--week', type=int, default=1)
    parser.add_argument('--lookback', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--groups', nargs='*', choices=list(GROUPS), default=[])
    parser.add_argument('--inputs', choices=['differential', 'separate'], default='differential')
    preview(**vars(parser.parse_args()))
