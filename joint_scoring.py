"""Direct margin and total models with shared, two-sided matchup functions."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

import shared_scoring as ss
import data_crunchski_3 as dc
import utils
from playoff_importance import FEATURES as IMPORTANCE, game_importance


GROUPS = ['stadium', 'field', 'referee', 'weather', 'importance']


def weather_features(games, source='forecast', forecasts=None, decision_hours=24):
    """Forecast values must have been issued by the fixed pregame decision time."""
    if source not in ['forecast', 'recorded']:
        raise ValueError('weather source must be forecast or recorded')
    result = games.copy()
    values = pd.DataFrame(np.nan, index=result.index,
                          columns=['temperature_f', 'wind_mph', 'precip_probability'])
    if source == 'recorded':
        for target, column in [('temperature_f', 'temp'), ('wind_mph', 'wind')]:
            values[target] = pd.to_numeric(result[column], errors='coerce')
    elif forecasts is not None:
        required = ['game_id', 'issued_at', 'valid_at'] + list(values.columns)
        missing = set(required) - set(forecasts)
        if missing:
            raise ValueError(f'Forecast file missing columns: {sorted(missing)}')
        f = forecasts[required].copy()
        # Schedule gametime is US Eastern; forecast timestamps must carry offsets.
        kickoff = pd.to_datetime(result.gameday.astype(str).str[:10] + ' ' +
                                 result.gametime.astype(str), errors='coerce')
        kickoff = kickoff.dt.tz_localize('America/New_York', ambiguous='NaT',
                                        nonexistent='NaT').dt.tz_convert('UTC')
        for column in ['issued_at', 'valid_at']:
            if not f[column].astype(str).str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', regex=True).all():
                raise ValueError(f'{column} must include a timezone offset')
            f[column] = pd.to_datetime(f[column], utc=True, errors='raise')
        f = f.merge(pd.DataFrame({'game_id': result.game_id, 'kickoff': kickoff}),
                    on='game_id', validate='many_to_one')
        f = f[(f.issued_at <= f.kickoff - pd.Timedelta(hours=decision_hours)) &
              ((f.valid_at - f.kickoff).abs() <= pd.Timedelta(hours=1))]
        f = f.sort_values('issued_at').drop_duplicates('game_id', keep='last').set_index('game_id')
        for column in values:
            values[column] = pd.to_numeric(result.game_id.map(f[column]), errors='coerce')
    values.loc[~values.temperature_f.between(-80, 140), 'temperature_f'] = np.nan
    values.loc[~values.wind_mph.between(0, 200), 'wind_mph'] = np.nan
    values.loc[~values.precip_probability.between(0, 1), 'precip_probability'] = np.nan
    indoor = result.roof.fillna('').str.lower().isin(['dome', 'closed'])
    # Explicit modeling convention, not a measured indoor temperature.
    values.loc[indoor, :] = [72., 0., 0.]
    for column in values:
        result['weather_' + column] = values[column]
        result['weather_' + column + '_missing'] = values[column].isna().astype(float)
    result['weather_indoor'] = indoor.astype(float)
    result['weather_roof_missing'] = result.roof.isna().astype(float)
    result['weather_source'] = source
    return result


def context_panel(panel, groups, weather_source='forecast', weather_file=None, decision_hours=24):
    import optimize_picks as op
    sched = pd.read_parquet('data/sched.parquet').replace(
        {'away_team': op.dc.RELOCATED_TEAMS, 'home_team': op.dc.RELOCATED_TEAMS})
    columns = ['stadium_id', 'stadium', 'gametime', 'roof', 'surface', 'referee', 'temp', 'wind']
    result = panel.drop(columns=columns, errors='ignore').merge(
        sched[op.KEY + columns], on=op.KEY, validate='one_to_one')
    if 'referee' in groups:
        result = dc.referee_tendencies(result, sched)
    if 'importance' in groups:
        seasons = sorted(result.season.unique().tolist())
        weeks = result[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
        path = utils.cache_path('joint_importance', weeks.values.tolist(), [__file__, 'playoff_importance.py', 'data_crunchski_2.py', 'data/sched.parquet'])
        if path.exists():
            importance = pd.read_parquet(path)
        else:
            importance = game_importance(sched[sched.season.isin(seasons)], weeks=weeks)
            utils.save_parquet(importance, path)
        columns = [f'{side}_{name}' for side in ['away', 'home'] for name in IMPORTANCE]
        columns += [c for c in importance if c.endswith(('_if_win', '_if_loss'))]
        columns += ['importance_method', 'importance_samples']
        result = result.drop(columns=columns, errors='ignore').merge(
            importance[op.KEY + columns], on=op.KEY, validate='one_to_one')
    if 'weather' in groups:
        forecasts = None
        if weather_file:
            path = Path(weather_file)
            forecasts = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        result = weather_features(result, weather_source, forecasts, decision_hours)
        outside = result.weather_indoor.eq(0)
        missing = result.loc[outside, 'weather_temperature_f_missing'].sum()
        print(f'Weather: {weather_source}; outdoor temperature missing {int(missing)}/{int(outside.sum())}', flush=True)
    return result


def prepare(panel, season, week, groups):
    base_groups = [g for g in groups if g in ss.GROUPS]
    target, x, _, xp, _ = dc.prepare(panel, season, week, base_groups, 'differential')
    prior = panel[(panel.season < season) | ((panel.season == season) & (panel.week < week))]
    extras, future, names = [], [], []
    if 'importance' in groups:
        for feature in IMPORTANCE:
            names.append('importance:' + feature)
            extras.append(np.r_[prior['away_' + feature], prior['home_' + feature]])
            future.append(np.r_[target['away_' + feature], target['home_' + feature]])
    if 'weather' in groups:
        for column in [c for c in panel if c.startswith('weather_') and c != 'weather_source']:
            names.append('weather:' + column[len('weather_'):])
            extras.append(np.tile(prior[column], 2))
            future.append(np.tile(target[column], 2))
    if extras:
        train = pd.DataFrame(np.array(extras, dtype=float).T).replace([np.inf, -np.inf], np.nan)
        test = pd.DataFrame(np.array(future, dtype=float).T).replace([np.inf, -np.inf], np.nan)
        fill = train.median().fillna(0)
        train, test = train.fillna(fill), test.fillna(fill)
        center, scale = train.mean(), train.std(ddof=0).replace(0, 1)
        x = np.c_[x, (train - center) / scale]
        xp = np.c_[xp, (test - center) / scale]
    names = dc.feature_names('differential') + target.attrs['context_names'] + names
    n, m = len(prior), len(target)
    # Each game is one training row. Reversal swaps entire side blocks.
    x, xp = np.c_[x[:n], x[n:]], np.c_[xp[:m], xp[m:]]
    targets = {'spread': (prior.away_score - prior.home_score).to_numpy('float32'),
               'total': (prior.away_score + prior.home_score).to_numpy('float32')}
    return target, x.astype('float32'), targets, xp.astype('float32'), names


def build_model(side_width, market):
    from tensorflow import keras
    from model_shredski import create_model
    inputs = keras.Input(shape=(side_width * 2,))
    reverse = keras.layers.Concatenate()([inputs[:, side_width:], inputs[:, :side_width]])
    shared = create_model(side_width * 2)
    first, second = shared(inputs), shared(reverse)
    output = (keras.layers.Subtract()([first, second]) if market == 'spread' else
              keras.layers.Average()([first, second]))
    model = keras.Model(inputs, output)
    model.compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss='mse')
    return model


def fit_member(i, x, y, xp, seed, epochs, market):
    from modelo_workers import initialize_worker
    initialize_worker()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    from model_shredski import integrated_gradients, permutation_importance, squared_error
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed + i)
    model = build_model(x.shape[1] // 2, market)
    try:
        model.fit(x, y[:, None], epochs=epochs, verbose=0, callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=5)])
        prediction = np.asarray(model(xp, training=False)).reshape(-1)
        attrs, baseline, _ = integrated_gradients(model, tf.constant(xp),
                                                tf.zeros((1, x.shape[1])), m_steps=64)
        sample = np.random.default_rng(seed).choice(len(x), min(128, len(x)), replace=False)
        importance = permutation_importance(model, x[sample], y[sample], squared_error, seed + i)
        if not all(np.isfinite(a).all() for a in [prediction, attrs, baseline, importance]):
            raise ValueError(f'Non-finite joint {market} member {i}')
        return prediction, attrs, baseline, importance
    finally:
        del model
        tf.keras.backend.clear_session()


def attribution_name(name, side):
    if name.startswith('diff_'):
        return ('away_off_' if side == 0 else 'away_def_') + name[5:]
    return {'home_field': 'home_field_adv', 'rest_advantage': 'away_rest_adv'}.get(
        name, 'context_' + name.split(':')[0])


def fit_panel(panel, season, week, iterations=100, epochs=100, seed=1337, jobs=8, groups=()):
    from joblib import Parallel, delayed, parallel_config
    from tqdm import tqdm
    from modelo_workers import initialize_worker
    if iterations < 2 or epochs < 1 or jobs < 1:
        raise ValueError('Require iterations >= 2, epochs >= 1, jobs >= 1')
    target, x, targets, xp, names = prepare(panel, season, week, groups)
    details, importances = {}, {}
    for market, y in targets.items():
        offset = 0. if market == 'spread' else float(y.mean())
        y = y - offset
        digest = hashlib.sha256(b''.join(a.tobytes() for a in [x, y, xp])).hexdigest()
        identity = [market, digest, names, offset, iterations, epochs, seed, list(groups)]
        path = utils.cache_path('joint_members', identity,
                                [__file__, 'model_shredski.py', 'modelo_workers.py'])
        path = path.with_suffix('.npz')
        if path.exists():
            with np.load(path) as saved:
                runs = list(zip(*(saved[k] for k in ['scores', 'attrs', 'bases', 'importance'])))
            print(f'Joint {market}: cached')
        else:
            with parallel_config(backend='loky', inner_max_num_threads=1):
                with Parallel(n_jobs=min(jobs, iterations), return_as='generator',
                              batch_size=1, initializer=initialize_worker) as pool:
                    runs = list(tqdm(pool(delayed(fit_member)(i, x, y, xp, seed, epochs, market)
                                    for i in range(iterations)), total=iterations,
                                    desc=f'Joint {market} ({season} wk{week})'))
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **{k: np.stack([r[i] for r in runs])
                for i, k in enumerate(['scores', 'attrs', 'bases', 'importance'])})
            os.replace(temporary, path)
        scores = np.stack([r[0] for r in runs]) + offset
        attrs = np.mean([r[1] for r in runs], axis=0)
        row = target[['away_team', 'home_team']].reset_index(drop=True).copy()
        row['model_family'] = 'joint'
        row['prediction'], row['variance'] = scores.mean(0), scores.var(0, ddof=1)
        row['baseline'] = np.mean([r[2] for r in runs], axis=0) + offset
        for side in range(2):
            for j, name in enumerate(names):
                key = 'attr_' + attribution_name(name, side)
                row[key] = row.get(key, 0) + attrs[:, side * len(names) + j]
        row['integration_residual'] = row.prediction - row.baseline - row.filter(regex='^attr_').sum(axis=1)
        imps = np.stack([r[3] for r in runs])
        importances[market] = pd.DataFrame({
            'feature': [attribution_name(n, side) if n.startswith('diff_') else
                        ('Away ' if side == 0 else 'Home ') + n for side in range(2) for n in names],
            'importance': imps.mean(0), 'std': imps.std(0)}).sort_values('importance', ascending=False)
        details[market] = row
    # Display-only implied scores from independently trained margin and total.
    for row in details.values():
        row['away_points'] = (details['total'].prediction + details['spread'].prediction) / 2
        row['home_points'] = (details['total'].prediction - details['spread'].prediction) / 2
        row['scores_implied'] = True
    return target, details, importances


def main(args):
    import optimize_picks as op
    from weekly_packet import write_packets
    groups = tuple(sorted(set(args.groups)))
    panel = op.build_panel(args.season, args.week, args.lookback, args.lookback, 'mean')
    panel = context_panel(panel, groups, args.weather_source, args.weather_file, args.decision_hours)
    target, details, importance = fit_panel(panel, args.season, args.week, args.iterations,
                                            args.epochs, args.seed, args.jobs, groups)
    suffix = '_'.join(groups) or 'baseline'
    output = Path(args.output or f'data/results/{args.season}_{args.week}_{args.lookback}/packet_joint_{suffix}_{args.weather_source}')
    config = dict(model=f'joint matchup / {args.iterations} members / seed {args.seed}',
                  lookback=args.lookback, calculation='joint-matchup-v1', status='PASS',
                  context_groups=list(groups), input_mode='differential', attribution_schema=2,
                  weather_source=args.weather_source, decision_hours=args.decision_hours,
                  reason='Experimental direct-target models; no validated betting cutoffs.',
                  headline_href=f'../../html_{args.season}_{args.week}_{args.lookback}.html',
                  baseline_note='Both matchups interact through shared weights. Margin and total are separate training targets. Scores are implied from those targets, not directly trained.',
                  importance_note='Training-sample permutation MSE diagnostic, not held-out feature evidence. Each side is shuffled separately; weights remain shared.',
                  context_note='Playoff leverage compares winning versus losing this game across paired remaining-season scenarios using only earlier-week results. Tracks berth, division, bye, top seed and seeding changes; fair-coin future games, no future ties, approximate later tiebreaks. Not official clinching probabilities. Stored roof/referee assignments are not decision-time verified. Recorded weather is retrospective only; forecast mode requires archived forecasts. Missing weather is flagged; closed roofs use nominal 72 F and zero wind/precipitation.')
    for market in ['spread', 'total']:
        predictions = op.market_panel(target, market).merge(details[market],
                        on=['away_team', 'home_team'], validate='one_to_one')
        predictions['edge'] = predictions.prediction - predictions.market_base
        write_packets(op.settle(predictions), panel, importance[market],
                      dict(config, market=market), output)
    shown = details['spread'][['away_team', 'home_team']].copy()
    shown['away_spread'] = -details['spread'].prediction
    shown['spread_sd'] = np.sqrt(details['spread'].variance)
    shown['total'] = details['total'].prediction
    shown['total_sd'] = np.sqrt(details['total'].variance)
    print(shown.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    print(f'Packet: {output / f"{args.season}_{args.week:02d}" / "index.html"}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, default=2026)
    p.add_argument('--week', type=int, default=1)
    p.add_argument('--lookback', type=int, default=20)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--groups', nargs='*', choices=GROUPS, default=GROUPS)
    p.add_argument('--weather-source', choices=['forecast', 'recorded'], default='forecast')
    p.add_argument('--weather-file', help='Archived CSV/parquet forecasts; see readmes/JOINT_MODELS.md')
    p.add_argument('--decision-hours', type=float, default=24)
    p.add_argument('--output')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.decision_hours < 0:
        raise SystemExit('--decision-hours must be nonnegative')
    main(args)
