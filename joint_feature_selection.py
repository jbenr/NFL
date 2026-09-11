"""Budgeted chronological feature selection for the actual joint neural models."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import joint_scoring as js
import optimize_picks as op
import shared_research as research
import utils


def feature_key(name):
    return name.split(':')[0]


def mask_features(x, names, selected):
    """Zero excluded standardized inputs on BOTH sides; keep network size fixed."""
    keep = np.array([feature_key(n) in selected for n in names], dtype=bool)
    return x * np.tile(keep, 2)


def weekly_folds(panel, start, stop, stride=1):
    """One-week tests, always fit on the preceding 20 regular weeks + playoffs."""
    weeks = panel[(panel.season >= start) & (panel.season < stop)]
    for _, season in weeks.groupby('season', sort=True):
        for (year, week), test in list(season.groupby(['season', 'week'], sort=True))[::stride]:
            prior = panel[(panel.season < year) | ((panel.season == year) & (panel.week < week))]
            regular = prior[prior.game_type.eq('REG')][['season', 'week']].drop_duplicates()
            regular = regular.sort_values(['season', 'week'])
            if len(regular) < 20:
                raise ValueError(f'Insufficient training history before {year} wk{week}')
            sy, sw = regular.iloc[-20]
            train = prior[(prior.season > sy) | ((prior.season == sy) & (prior.week >= sw))]
            yield int(year), int(week), pd.concat([train, test]).sort_values(op.KEY)


def predict_member(i, x, y, xp, market, epochs, seed):
    from modelo_workers import initialize_worker
    initialize_worker()
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed + i)
    model = js.build_model(x.shape[1] // 2, market)
    try:
        model.fit(x, y[:, None], epochs=epochs, verbose=0, callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=.5, patience=5)])
        prediction = np.asarray(model(xp, training=False)).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError(f'Non-finite {market} predictions')
        return prediction
    finally:
        tf.keras.backend.clear_session()


def metrics(rows):
    error = rows.prediction - rows.actual
    return dict(games=len(rows), mae=float(error.abs().mean()),
                rmse=float(np.sqrt((error**2).mean())))


def paired_summary(baseline, chosen, seed):
    paired = baseline.merge(chosen, on=op.KEY, suffixes=('_base', '_chosen'),
                            validate='one_to_one')
    paired['gain'] = ((paired.prediction_base - paired.actual_base).abs() -
                      (paired.prediction_chosen - paired.actual_chosen).abs())
    seasons = paired.groupby('season').gain.agg(['sum', 'count', 'mean'])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(seasons), size=(2000, len(seasons)))
    gains = seasons['sum'].to_numpy()[draws].sum(1) / seasons['count'].to_numpy()[draws].sum(1)
    return dict(mae_improvement=float(paired.gain.mean()),
                season_block_interval_95=np.quantile(gains, [.025, .975]).tolist(),
                seasons_improved=int((seasons['mean'] > 0).sum()), seasons=len(seasons),
                note='Descriptive interval, not adjusted for feature search; one season cannot measure year-to-year uncertainty.')


def search_subsets(features, evaluate, budget, beam):
    """Backward beam search; not exhaustive and not a global-optimum guarantee."""
    full = tuple(sorted(features))
    scores = {full: evaluate(full)}
    frontier = [full]
    while frontier and len(scores) < budget:
        candidates = sorted({tuple(f for f in parent if f != removed)
                             for parent in frontier for removed in parent if len(parent) > 1}
                            - scores.keys())
        if not candidates:
            break
        # Complete each removal layer or stop: do not favor alphabetical features.
        if len(scores) + len(candidates) > budget:
            break
        for candidate in candidates:
            scores[candidate] = evaluate(candidate)
        frontier = sorted(candidates, key=lambda s: (scores[s], len(s), s))[:beam]
    best = min(scores, key=lambda s: (scores[s], len(s), s))
    return best, scores


class Evaluator:
    def __init__(self, panel, args, root):
        self.panel, self.args, self.root = panel, args, root
        self.prepared = {}

    def forecasts(self, selected, market, start, stop, members, stride=1):
        from joblib import Parallel, delayed, parallel_config
        from modelo_workers import initialize_worker
        rows = []
        for year, week, data in weekly_folds(self.panel, start, stop, stride):
            if (year, week) not in self.prepared:
                self.prepared[year, week] = js.prepare(data, year, week, self.args.groups)
            target, x, targets, xp, names = self.prepared[year, week]
            x, xp = mask_features(x, names, selected), mask_features(xp, names, selected)
            y = targets[market]
            if not all(np.isfinite(a).all() for a in [x, y, xp]):
                raise ValueError(f'Non-finite input/target at {year} wk{week}')
            offset = float(y.mean()) if market == 'total' else 0.
            y = y - offset
            digest = hashlib.sha256(b''.join(a.tobytes() for a in [x, y, xp])).hexdigest()
            path = utils.cache_path('joint_feature_predictions',
                [market, year, week, x.shape, xp.shape, digest, names, list(selected),
                 offset, members, self.args.epochs, self.args.seed],
                [__file__, 'joint_scoring.py', 'shared_scoring.py',
                 'model_shredski.py', 'modelo_workers.py'])
            if path.exists():
                forecast = pd.read_parquet(path)
            else:
                print(f'  {market} {year} wk{week}: {len(selected)} features, {members} members', flush=True)
                with parallel_config(backend='loky', inner_max_num_threads=1):
                    runs = Parallel(n_jobs=min(self.args.jobs, members), batch_size=1,
                                    initializer=initialize_worker)(
                        delayed(predict_member)(i, x, y, xp, market, self.args.epochs, self.args.seed)
                        for i in range(members))
                scores = np.stack(runs) + offset
                forecast = target[op.KEY].reset_index(drop=True).copy()
                forecast['actual'] = ((target.away_score - target.home_score) if market == 'spread'
                                      else (target.away_score + target.home_score)).to_numpy()
                forecast['prediction'] = scores.mean(0)
                forecast['sd'] = scores.std(0, ddof=1)
                utils.save_parquet(forecast, path)
            rows.append(forecast)
        if not rows:
            raise ValueError('No completed games in evaluation period')
        return pd.concat(rows, ignore_index=True)


def run(args):
    os.environ.setdefault('NFL_WORKERS', '1')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    sources = ['data/sched.parquet', 'data_crunchski_2.py', 'utils.py', 'optimize_picks.py',
               'shared_research.py', 'joint_scoring.py', 'shared_scoring.py', 'playoff_importance.py']
    sources += sorted(Path('data/pbp').glob('pbp_*.parquet'))
    if args.weather_file:
        sources.append(args.weather_file)
    config = vars(args).copy()
    # A single cached context-enriched panel; no data pulls inside the search.
    path = utils.cache_path('joint_selection_panel',
        [args.start_season, args.season, args.week, args.groups, args.weather_source,
         args.weather_file, args.decision_hours], sources)
    if path.exists():
        panel = pd.read_parquet(path)
        print('Prepared panel: cached', flush=True)
    else:
        panel = op.build_panel(args.season, args.week, research.history_weeks(args), 20, 'legacy')
        panel = js.context_panel(panel, args.groups, args.weather_source,
                                 args.weather_file, args.decision_hours)
        utils.save_parquet(panel, path)
    panel = panel[np.isfinite(panel.away_score) & np.isfinite(panel.home_score)]
    features = js.ss.feature_names('differential') + js.ss.BASE_CONTEXT + args.groups
    features = tuple(sorted(set(features)))
    if args.max_subsets < len(features) + 1:
        raise ValueError(f'max-subsets must be at least {len(features) + 1} for all single removals')
    config['source_sha256'] = hashlib.sha256(b''.join(Path(p).read_bytes() for p in
        [__file__, 'joint_scoring.py', 'shared_scoring.py', 'model_shredski.py', 'modelo_workers.py'])).hexdigest()
    stamp = hashlib.sha256(json.dumps(config, sort_keys=True).encode() +
                           pd.util.hash_pandas_object(panel, index=False).values.tobytes()).hexdigest()[:12]
    root = root / stamp
    root.mkdir(exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    evaluator = Evaluator(panel, args, root)
    for market in args.markets:
        board, saved = [], {}
        def evaluate(selected):
            forecast = evaluator.forecasts(selected, market, args.start_season,
                       args.validation_season, args.search_members, args.week_stride)
            score = metrics(forecast)
            comparison = paired_summary(saved[tuple(sorted(features))], forecast, args.seed) if saved else {}
            board.append(dict(features=' | '.join(selected), count=len(selected), **score,
                              mae_gain_vs_full=comparison.get('mae_improvement', 0),
                              seasons_improved=comparison.get('seasons_improved', 0)))
            pd.DataFrame(board).sort_values('mae').to_csv(root / f'{market}_search.csv', index=False)
            saved[selected] = forecast
            print(f'{market}: {len(board)} subsets evaluated; MAE {score["mae"]:.3f}', flush=True)
            return score['mae']
        best, scores = search_subsets(features, evaluate, args.max_subsets, args.beam)
        full = tuple(sorted(features))
        # Recheck the selected subset against full at production ensemble size
        # on ALL development weeks before opening the final evaluation period.
        baseline = evaluator.forecasts(full, market, args.start_season,
                                        args.validation_season, args.iterations)
        chosen = evaluator.forecasts(best, market, args.start_season,
                                      args.validation_season, args.iterations)
        selected = best if metrics(chosen)['mae'] < metrics(baseline)['mae'] else full
        result = dict(selected_features=list(selected), removed_features=sorted(set(full)-set(selected)),
                      subsets_evaluated=len(scores), search_method='backward beam; not exhaustive',
                      development_baseline=metrics(baseline), development_candidate=metrics(chosen),
                      development_comparison=paired_summary(baseline, chosen, args.seed),
                      status='EXPLORATORY: final period may have been inspected in earlier research')
        (root / f'{market}_selection.json').write_text(json.dumps(result, indent=2))
        # Selection is frozen above. Holdout outcomes never feed the search.
        hold_base = evaluator.forecasts(full, market, args.validation_season, args.season + 1, args.iterations)
        hold_chosen = evaluator.forecasts(selected, market, args.validation_season, args.season + 1, args.iterations)
        utils.save_parquet(hold_base, root / f'{market}_holdout_baseline.parquet')
        utils.save_parquet(hold_chosen, root / f'{market}_holdout_selected.parquet')
        result.update(holdout_baseline=metrics(hold_base), holdout_selected=metrics(hold_chosen),
                      holdout_comparison=paired_summary(hold_base, hold_chosen, args.seed))
        (root / f'{market}_selection.json').write_text(json.dumps(result, indent=2))
        folds = hold_chosen.groupby('season').apply(lambda d: pd.Series(metrics(d)), include_groups=False)
        folds.to_csv(root / f'{market}_holdout_by_season.csv')
        print(f'{market}: {root / (market + "_selection.json")}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', help='Execute; otherwise show the compute plan only')
    p.add_argument('--markets', nargs='+', choices=['spread', 'total'], default=['spread', 'total'])
    p.add_argument('--start-season', type=int, default=2017)
    p.add_argument('--validation-season', type=int, default=2025)
    p.add_argument('--season', type=int, default=2025)
    p.add_argument('--week', type=int, default=22)
    p.add_argument('--groups', nargs='*', choices=js.GROUPS, default=js.GROUPS.copy())
    p.add_argument('--max-subsets', type=int, default=64)
    p.add_argument('--beam', type=int, default=2)
    p.add_argument('--week-stride', type=int, default=1, help='Development screening only; final comparisons use every week')
    p.add_argument('--search-members', type=int, default=10)
    p.add_argument('--iterations', type=int, default=100, help='Final comparison ensemble members')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--weather-source', choices=['forecast', 'recorded'], default='forecast')
    p.add_argument('--weather-file')
    p.add_argument('--decision-hours', type=float, default=24)
    p.add_argument('--output', default='data/optimize_picks/joint_feature_selection')
    return p


if __name__ == '__main__':
    p = parser()
    args = p.parse_args()
    if not args.start_season < args.validation_season <= args.season:
        p.error('Require start-season < validation-season <= season')
    if min(args.search_members, args.iterations) < 2 or min(args.epochs, args.jobs, args.beam, args.week_stride, args.max_subsets) < 1:
        p.error('Require >=2 members and positive budgets/workers/stride')
    if args.decision_hours < 0:
        p.error('decision-hours must be nonnegative')
    args.groups = sorted(set(args.groups))
    schedule = pd.read_parquet('data/sched.parquet')
    weeks = schedule[['season', 'week']].drop_duplicates()
    dev = weeks[(weeks.season >= args.start_season) & (weeks.season < args.validation_season)]
    count = sum(len(s.iloc[::args.week_stride]) for _, s in dev.groupby('season'))
    print(f'Plan: {len(args.markets)} markets; up to {args.max_subsets} subsets each; '
          f'{count} weekly screening folds; {args.search_members} members per fit.')
    print(f'Screening upper bound: {len(args.markets) * args.max_subsets * count * args.search_members:,} neural fits '
          f'plus {args.iterations}-member finalist comparisons. Cached fits are reused.')
    print('Objective: game-weighted out-of-sample MAE; RMSE also reported. No betting-rule optimization.')
    if args.run:
        run(args)
    else:
        print('Add --run to execute. Do not run alongside another large neural backtest.')
