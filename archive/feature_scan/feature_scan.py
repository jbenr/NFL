"""Offline add/remove/retrain feature screen. Run: python feature_scan.py"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_scan_data import KEYS, build_candidates, load_game_stats, normalize_schedule, pregame_context


def chronological_folds(data, holdout_season):
    """Whole seasons; training always precedes evaluation."""
    years = sorted(data.loc[data.season < holdout_season, 'season'].unique())
    folds = []
    for year in years[1:]:
        train = np.flatnonzero(data.season.to_numpy() < year)
        val = np.flatnonzero(data.season.to_numpy() == year)
        if len(train) >= 200 and len(val) >= 100:
            folds.append((int(year), train, val))
    if len(folds) < 2:
        raise ValueError('Need three sufficiently populated seasons before the reserved season')
    return folds


def build_variants(catalog):
    full, baseline = catalog.feature.tolist(), ['home_field_adv']
    variants = {'baseline': baseline, 'full': full,
                'without_home_control': [f for f in full if f not in baseline]}
    tests = [(f'feature:{r.feature}', r.label, 'feature', [r.feature])
             for r in catalog.itertuples() if r.feature not in baseline]
    tests += [(f'group:{group}', group, 'group', rows.feature.tolist())
              for group, rows in catalog.groupby('group', sort=False) if group != 'Home field']
    for key, _, _, features in tests:
        variants[f'add:{key}'] = list(dict.fromkeys(baseline + features))
        variants[f'drop:{key}'] = [f for f in full if f not in features]
    return variants, tests


def fit_variant(data, train, val, features, learner, track, year, variant):
    features = list(features) + (['spread_line'] if track == 'market_residual' else [])
    if learner == 'ridge':
        estimator = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                  StandardScaler(), Ridge(alpha=20.0))
    else:
        estimator = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
            HistGradientBoostingRegressor(max_iter=80, learning_rate=.05, max_leaf_nodes=7,
                min_samples_leaf=30, l2_regularization=10, early_stopping=False, random_state=1337))
    xtr, xval = data.iloc[train][features], data.iloc[val][features]
    estimator.fit(xtr, data.iloc[train][track])
    prediction = estimator.predict(xval)
    observed = data.iloc[val][track].to_numpy()
    if not np.isfinite(prediction).all():
        raise ValueError(f'Non-finite predictions: {learner}/{track}/{variant}/{year}')
    cover = prediction if track == 'market_residual' else prediction + data.iloc[val].spread_line.to_numpy()
    actual = data.iloc[val].market_residual.to_numpy()
    selected = (cover != 0) & (actual != 0)
    wins = (np.sign(cover) == np.sign(actual)) & selected
    meta = dict(learner=learner, track=track, year=year, variant=variant,
                mae=float(np.abs(observed - prediction).mean()),
                rmse=float(np.sqrt(np.square(observed - prediction).mean())), n=len(val),
                ats_n=int(selected.sum()), ats_wins=int(wins.sum()),
                ats_rate=float(wins.sum() / selected.sum()) if selected.any() else np.nan)
    if variant in ['full', 'baseline']:
        home, neutral = xval.copy(), xval.copy()
        home['home_field_adv'], neutral['home_field_adv'] = 1.0, 0.0
        meta['home_minus_neutral_points'] = float((estimator.predict(home) - estimator.predict(neutral)).mean())
        meta['training_neutral_games'] = int(data.iloc[train].home_field_adv.eq(0).sum())
        meta['training_home_games'] = int(data.iloc[train].home_field_adv.eq(1).sum())
        meta['training_home_mean_away_margin'] = float(data.iloc[train].loc[data.iloc[train].home_field_adv.eq(1), 'margin'].mean())
        meta['mean_fitted_home_margin'] = float(estimator.predict(home).mean())
    result = data.iloc[val][KEYS + ['game_id', 'margin', 'market_residual', 'spread_line']].copy()
    result['prediction'] = prediction
    result['abs_error'] = np.abs(observed - prediction)
    result['predicted_cover_margin'] = cover
    return meta, result


def paired_improvement(reference, trial, seed=1337):
    """Positive means trial wins; bootstrap paired games in whole-week blocks."""
    if not reference[KEYS].equals(trial[KEYS]):
        raise ValueError('Paired comparisons require identical games in identical order')
    paired = reference[KEYS].copy()
    paired['gain'] = reference.abs_error.to_numpy() - trial.abs_error.to_numpy()
    weekly = paired.groupby(['season', 'week']).gain.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(weekly), (1000, len(weekly)))
    estimates = weekly['sum'].to_numpy()[sampled].sum(axis=1) / weekly['count'].to_numpy()[sampled].sum(axis=1)
    yearly = paired.groupby('season').gain.mean()
    return dict(gain=float(paired.gain.mean()), low=float(np.quantile(estimates, .025)),
                high=float(np.quantile(estimates, .975)), positive_seasons=int(yearly.gt(0).sum()),
                seasons=len(yearly), n=len(paired))


def summarize_comparisons(outputs, tests):
    rows = []
    for learner in ['ridge', 'boosted_trees']:
        for track in ['margin', 'market_residual']:
            combined = {}
            for meta, pred in outputs:
                if meta['learner'] == learner and meta['track'] == track:
                    combined.setdefault(meta['variant'], []).append(pred)
            combined = {k: pd.concat(v, ignore_index=True) for k, v in combined.items()}
            for key, label, level, _ in tests:
                addition = paired_improvement(combined['baseline'], combined[f'add:{key}'])
                removal = paired_improvement(combined[f'drop:{key}'], combined['full'])
                row = dict(learner=learner, track=track, feature=key, label=label, level=level)
                row.update({f'add_{k}': v for k, v in addition.items()})
                row.update({f'drop_{k}': v for k, v in removal.items()})
                row['assessment'] = ('Conditional support' if removal['low'] > 0 else
                    'Removal candidate' if removal['high'] < 0 else
                    'Adds to baseline; conditional value uncertain' if addition['low'] > 0 else
                    'Uncertain / possibly redundant')
                rows.append(row)
    return pd.DataFrame(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--start-season', type=int, default=2021)
    parser.add_argument('--holdout-season', type=int, default=2025)
    parser.add_argument('--asof-season', type=int, default=2025)
    parser.add_argument('--asof-week', type=int, default=20)
    parser.add_argument('--lookback-games', type=int, default=20)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.start_season < 2021 or args.holdout_season > args.asof_season or args.lookback_games < 1 or args.jobs < 1:
        parser.error('Use start-season >= 2021, holdout <= asof, and positive lookback/jobs')
    outdir = args.output or args.root / 'data' / 'feature_scan' / datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.root / 'data' / 'sched.parquet'
    schedule = normalize_schedule(pd.read_parquet(schedule_path))
    schedule = schedule[(schedule.season >= args.start_season - 1) & (schedule.season <= args.asof_season)]
    # Fixture dates can inform games remaining; current/future results cannot.
    future = (schedule.season == args.asof_season) & (schedule.week >= args.asof_week)
    schedule.loc[future, ['away_score', 'home_score']] = np.nan
    game_stats = load_game_stats(args.root, range(args.start_season - 1, args.asof_season + 1))
    game_stats = game_stats[(game_stats.season < args.asof_season) | (game_stats.week < args.asof_week)]
    # Exclude canceled/unfinished games from rolling histories (e.g. BUF-CIN 2022).
    completed_ids = schedule.loc[schedule.away_score.notna() & schedule.home_score.notna(), 'game_id']
    game_stats = game_stats[game_stats.game_id.isin(completed_ids)]
    data, snapshots, catalog = build_candidates(schedule, game_stats, args.lookback_games)
    # Identical eligible games for every feature; train-only imputation for gaps.
    scan = data[(data.season >= args.start_season) & (data.season < args.holdout_season)
                & data.eligible_history & data.margin.notna() & data.spread_line.notna()
                & data.home_field_adv.notna()].sort_values(KEYS).reset_index(drop=True)
    folds = chronological_folds(scan, args.holdout_season)
    variants, tests = build_variants(catalog)
    print(f'Scan: {len(scan)} games, {len(catalog)} candidates, {len(folds)} chronological folds; '
          f'{args.holdout_season} reserved.', flush=True)
    tasks = [(learner, track, year, variant, features, train, val)
             for learner in ['ridge', 'boosted_trees'] for track in ['margin', 'market_residual']
             for year, train, val in folds for variant, features in variants.items()]
    unique, task_lookup = {}, []
    for learner, track, year, variant, features, train, val in tasks:
        key = (learner, track, year, tuple(features))
        if key not in unique:
            unique[key] = (len(unique), (learner, track, year, variant, features, train, val))
        task_lookup.append((unique[key][0], variant))
    from tqdm import tqdm
    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = list(tqdm(Parallel(n_jobs=args.jobs, return_as='generator', batch_size=4)(
            delayed(fit_variant)(scan, tr, va, fs, learner, track, year, variant)
            for _, (learner, track, year, variant, fs, tr, va) in unique.values()),
            total=len(unique), desc='Add / remove / retrain'))
    outputs = []
    for index, variant in task_lookup:
        meta, prediction = results[index]
        outputs.append((dict(meta, variant=variant), prediction))
    scores = pd.DataFrame([meta for meta, _ in outputs])
    for year, _, val in folds:
        residual = scan.iloc[val].market_residual.to_numpy()
        scores = pd.concat([scores, pd.DataFrame([dict(learner='market', track='market_residual', year=year,
            variant='market_only', mae=float(np.abs(residual).mean()),
            rmse=float(np.sqrt(np.square(residual).mean())), n=len(val), ats_n=0)])], ignore_index=True)
    summary = summarize_comparisons(outputs, tests)
    home = scores[scores.variant.isin(['baseline', 'full'])][[
        'learner', 'track', 'year', 'variant', 'home_minus_neutral_points', 'training_neutral_games', 'training_home_games',
        'training_home_mean_away_margin', 'mean_fitted_home_margin']]
    from feature_scan_report import team_summary, write_report
    teams = team_summary(snapshots, args.asof_season, args.asof_week)
    context = pregame_context(schedule)
    context = context[(context.season == args.asof_season) & (context.week == args.asof_week)]
    data.to_parquet(outdir / 'candidate_dataset.parquet', index=False)
    predictions = [pred.assign(**{k: meta[k] for k in ['learner', 'track', 'year', 'variant']})
                   for meta, pred in outputs]
    pd.concat(predictions, ignore_index=True).to_parquet(outdir / 'scan_predictions.parquet', index=False)
    import sklearn
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), scan_games=len(scan),
        validation_seasons=[f[0] for f in folds], holdout_season=args.holdout_season,
        asof_season=args.asof_season, asof_week=args.asof_week, lookback_games=args.lookback_games,
        unique_fits=len(unique), sklearn_version=sklearn.__version__,
        candidate_dataset_sha256=file_hash(outdir / 'candidate_dataset.parquet'),
        schedule_sha256=file_hash(schedule_path),
        source_sha256={p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name('feature_scan_data.py'),
                                                   Path(__file__).with_name('feature_scan_report.py')]},
        pbp_files=[dict(path=str(args.root / 'data' / 'pbp' / f'pbp_{s}.parquet'),
                       size=(args.root / 'data' / 'pbp' / f'pbp_{s}.parquet').stat().st_size,
                       mtime_ns=(args.root / 'data' / 'pbp' / f'pbp_{s}.parquet').stat().st_mtime_ns)
                   for s in range(args.start_season - 1, args.asof_season + 1)],
        model_settings=dict(ridge_alpha=20, tree_iterations=80, tree_leaves=7, tree_l2=10,
                            tree_learning_rate=.05, tree_min_leaf=30, seed=1337))
    write_report(outdir, summary, scores, home, teams, context, catalog, metadata)
    print(f'Report: {outdir / "report.html"}', flush=True)
    print(summary[(summary.level == 'group') & (summary.track == 'market_residual')]
          [['label', 'learner', 'add_gain', 'drop_gain', 'assessment']].sort_values('drop_gain', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
