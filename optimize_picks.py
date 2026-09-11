"""Cached, chronological spread/total research using the normal feature pipeline.

Feature screening -> price/filter calibration -> fixed retrospective validation.
This is a reproducible Ridge challenger, NOT the production neural ensemble.
Its variance and cutoffs are valid only for this exact fitted model family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data_crunchski_2 as dc
import utils
from edge_scan import PRODUCTION_FEATURES

KEY = ['season', 'week', 'away_team', 'home_team']
DIFF_CUTOFFS = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
SD_QUANTILES = [1, .75, .5, .25]
MODEL = 'weekly-block-bagged-ridge-v1'


def build_panel(season, week, history_weeks, lookback, calculation):
    data = dc.prep_test_train(season, week, lookback, history_weeks=history_weeks,
                             calculation=calculation)
    data = dc.additional_features(data)
    sched = pd.read_parquet('data/sched.parquet').replace(
        {'away_team': dc.RELOCATED_TEAMS, 'home_team': dc.RELOCATED_TEAMS})
    columns = KEY + ['game_id', 'gameday', 'spread_line', 'total_line',
                     'away_spread_odds', 'home_spread_odds', 'over_odds', 'under_odds']
    data = data.drop(columns=[c for c in columns if c not in KEY], errors='ignore')
    data = data.merge(sched[columns], on=KEY, validate='one_to_one')
    data['margin'] = data.away_score - data.home_score
    data['points'] = data.away_score + data.home_score
    weeks = data[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    weeks['week_id'] = range(len(weeks))
    return data.merge(weeks, on=['season', 'week']).sort_values(KEY).reset_index(drop=True)


def symmetric_features(panel):
    """Remove away-only usage multipliers; retain legacy rates and rank differences."""
    result = panel.copy()
    for feature in PRODUCTION_FEATURES:
        if not feature.startswith('away_off_') or feature not in result:
            continue
        metric = feature[len('away_off_'):]
        usage = 'pass' if 'pass' in metric else 'run' if 'run' in metric else None
        if usage:
            denominator = result[f'away_raw_off_{usage}_%'] + .5
            if not np.isfinite(denominator).all() or denominator.le(0).any():
                raise ValueError(f'Invalid usage rates for symmetric feature {feature}')
            result[feature] = result[feature] / denominator
    return result


def market_panel(panel, market):
    data = panel.copy()
    if market == 'spread':
        data['market_base'] = -data.spread_line  # away minus home points
        data['actual'] = data.margin
        data['positive_odds'], data['negative_odds'] = data.away_spread_odds, data.home_spread_odds
    else:
        data['market_base'] = data.total_line
        data['actual'] = data.points
        data['positive_odds'], data['negative_odds'] = data.over_odds, data.under_odds
    data['residual'] = data.actual - data.market_base
    data['market'] = market
    return data.dropna(subset=['market_base'])


def feature_names(panel, market):
    if market == 'spread':
        return [f for f in PRODUCTION_FEATURES if f in panel]
    # Symmetric scoring levels, not away/home differences. Weather from final
    # game records is deliberately excluded: it is not a pregame forecast.
    return [f for f in panel if f.startswith('total_') and f != 'total_line'] + ['home_field_adv']


def weekly_predict(panel, features, min_train_weeks=30, *, bags=1, seed=1337, alpha=20):
    """Refit strictly before each week; retain exact additive explanations."""
    columns = list(dict.fromkeys(KEY + features + ['week_id', 'residual', 'actual',
                       'market_base', 'positive_odds', 'negative_odds', 'market']))
    data = panel[columns].sort_values(KEY).reset_index(drop=True)
    fingerprint = pd.util.hash_pandas_object(data, index=False).values.tobytes().hex()
    cached = utils.cache_path('predictions', [fingerprint, features, min_train_weeks,
                                             bags, seed, alpha, MODEL], [__file__])
    if cached.exists():
        return pd.read_parquet(cached)
    rows = []
    for wid, test in data.groupby('week_id', sort=True):
        train = data[(data.week_id < wid) & data.residual.notna()]
        weeks = train.week_id.unique()
        if len(weeks) < min_train_weeks:
            continue
        rng = np.random.default_rng(seed + int(wid))
        predictions, contributions, intercepts = [], [], []
        for _ in range(bags):
            sample = train if bags == 1 else pd.concat(
                [train[train.week_id == w] for w in rng.choice(weeks, len(weeks))])
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                  StandardScaler(), Ridge(alpha=alpha))
            model.fit(sample[features], sample.residual)
            transformed = model[:-1].transform(test[features])
            attr = transformed * model[-1].coef_
            contributions.append(attr)
            intercepts.append(float(model[-1].intercept_))
            predictions.append(attr.sum(axis=1) + model[-1].intercept_)
        out = test.drop(columns=features, errors='ignore').copy()
        out['edge'] = np.mean(predictions, axis=0)
        out['prediction'] = out.market_base + out.edge
        out['variance'] = np.var(predictions, axis=0, ddof=1) if bags > 1 else 0.0
        out['baseline'] = out.market_base + np.mean(intercepts)
        for j, feature in enumerate(features):
            out['attr_' + feature] = np.mean(contributions, axis=0)[:, j]
        rows.append(out)
    if not rows:
        raise ValueError('Not enough completed training weeks for walk-forward predictions')
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result[['edge', 'prediction', 'variance']]).all().all():
        raise ValueError('Non-finite research predictions')
    utils.save_parquet(result, cached)
    return result


def policy_sd_cutoff(policy):
    """Read SD policies and legacy variance policies without changing selections."""
    value = policy.get('sd_cutoff') if 'sd_cutoff' in policy else policy.get('var_cutoff')
    if value is None or pd.isna(value):
        return None
    if not np.isfinite(value) or value < 0:
        raise ValueError('Uncertainty cutoff must be finite and nonnegative')
    return float(value if 'sd_cutoff' in policy else np.sqrt(value))


def neural_spec(features, lookback, calculation, iterations=100, seed=1337, market='spread'):
    code = b''.join(Path(p).read_bytes() for p in ['model_shredski.py', 'modelo_workers.py', 'data_crunchski_2.py'])
    return dict(model='neural', market=market, features=list(features), lookback=lookback,
                calculation=calculation, iterations=iterations, seed=seed, epochs=100,
                train_weeks=20, code_sha256=hashlib.sha256(code).hexdigest())


def settle(predictions, diff_cutoff=0, sd_cutoff=None, *, var_cutoff=None):
    data = predictions.copy()
    if var_cutoff is not None:
        if sd_cutoff is not None:
            raise ValueError('Specify SD or legacy variance, not both')
        sd_cutoff = policy_sd_cutoff({'var_cutoff': var_cutoff})
    sd_cutoff = policy_sd_cutoff({'sd_cutoff': sd_cutoff})
    data['sd'] = np.sqrt(data.variance.where(data.variance >= 0))
    qualifies = data.edge.abs().ge(diff_cutoff) & data.edge.ne(0) & np.isfinite(data.edge) & np.isfinite(data.sd)
    if sd_cutoff is not None:
        qualifies &= data.sd.le(sd_cutoff)
    data['qualifies'] = qualifies
    odds = np.where(data.edge > 0, data.positive_odds, data.negative_odds).astype(float)
    missing = ~np.isfinite(odds) | (np.abs(odds) < 100)
    odds[missing] = -110
    data['odds'], data['assumed_odds'] = odds, missing
    payoff = np.where(odds > 0, odds / 100, 100 / np.maximum(np.abs(odds), 1))
    result = np.sign(data.edge) * np.sign(data.residual)
    data['win'] = result.gt(0)
    data['push'] = result.eq(0)
    data['pnl'] = np.where(data.residual.isna(), np.nan,
                         np.where(qualifies, np.where(result > 0, payoff, np.where(result == 0, 0, -1)), 0))
    return data


def score(predictions, diff_cutoff=0, sd_cutoff=None, *, var_cutoff=None):
    data = settle(predictions, diff_cutoff, sd_cutoff, var_cutoff=var_cutoff)
    bets = data[data.qualifies & data.residual.notna()]
    n, wins, pushes = len(bets), int(bets.win.sum()), int(bets.push.sum())
    return dict(n=n, wins=wins, losses=n-wins-pushes, pushes=pushes,
                win_rate=wins / (n-pushes) if n > pushes else None,
                pnl_units=float(bets.pnl.sum()), roi=float(bets.pnl.mean()) if n else None,
                assumed_odds=int(bets.assumed_odds.sum()))


def roi_interval(predictions, diff_cutoff=0, sd_cutoff=None, reps=2000):
    data = settle(predictions, diff_cutoff, sd_cutoff)
    data['bet'] = data.qualifies & data.residual.notna()
    weekly = data.groupby('week_id').agg(pnl=('pnl', 'sum'), bets=('bet', 'sum'))
    if len(weekly) < 2 or weekly.bets.sum() == 0:
        return [None, None]
    idx = np.random.default_rng(1337).integers(0, len(weekly), (reps, len(weekly)))
    count = weekly.bets.to_numpy()[idx].sum(axis=1)
    values = weekly.pnl.to_numpy()[idx].sum(axis=1) / np.maximum(count, 1)
    return np.quantile(values[count > 0], [.025, .975]).tolist()


def feature_scan(panel, features, min_train_weeks, end_week):
    """Paired refit/drop tests on discovery weeks only, never the final window."""
    discovery = panel[panel.week_id < end_week]
    full = weekly_predict(discovery, features, min_train_weeks)
    base_error = (full.prediction - full.actual) ** 2
    base_pnl = settle(full).pnl
    rows = []
    for feature in features:
        reduced = weekly_predict(discovery, [f for f in features if f != feature], min_train_weeks)
        delta = (reduced.prediction - reduced.actual) ** 2 - base_error
        paired = pd.DataFrame({'week_id': full.week_id, 'delta': delta,
                               'pnl': base_pnl - settle(reduced).pnl})
        weekly = paired.groupby('week_id').agg(total=('delta', 'sum'), n=('delta', 'count'))
        idx = np.random.default_rng(1337).integers(0, len(weekly), (4000, len(weekly)))
        boot = weekly.total.to_numpy()[idx].sum(axis=1) / weekly.n.to_numpy()[idx].sum(axis=1)
        lo, hi = np.quantile(boot, [.025 / len(features), 1 - .025 / len(features)])
        rows.append(dict(feature=feature, mse_contribution=float(delta.mean()),
                         ci_low=lo, ci_high=hi, pnl_contribution=float(paired.pnl.sum())))
    return pd.DataFrame(rows).sort_values('mse_contribution', ascending=False)


def cutoff_grid(predictions, min_bets):
    rows = []
    for diff in DIFF_CUTOFFS:
        for quantile in SD_QUANTILES:
            # Transform the old threshold exactly; interpolated SD quantiles can
            # differ slightly from sqrt(variance quantile) on small samples.
            sd = None if quantile == 1 else float(np.sqrt(predictions.variance.quantile(quantile)))
            result = score(predictions, diff, sd)
            result.update(diff_cutoff=diff, sd_cutoff=sd, sd_quantile=quantile,
                          eligible=result['n'] >= min_bets)
            rows.append(result)
    return pd.DataFrame(rows)


def research(args):
    from weekly_packet import write_packets, write_research_report
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidates, panels, importance = [], {}, {}
    boundaries = None
    for lookback in args.lookbacks:
        for calculation in args.calculations:
            label = f'{lookback}_{calculation}'
            print(f'\nPreparing {label} (historical features cached)', flush=True)
            panel = build_panel(args.season, args.week, args.history_weeks, lookback, calculation)
            panels[label] = panel
            completed = panel[panel.margin.notna()].week_id.unique()
            scored = sorted(w for w in completed if w >= args.min_train_weeks)
            if len(scored) < 40:
                raise ValueError('Need at least 40 scoring weeks after training burn-in')
            split = (scored[len(scored) // 2], scored[3 * len(scored) // 4])
            if boundaries is not None and split != boundaries:
                raise ValueError('Calculation variants must evaluate identical weeks')
            boundaries = split
            for market in args.markets:
                data = market_panel(panel, market)
                features = feature_names(data, market)
                print(f'  {market}: paired scan of {len(features)} features', flush=True)
                scan = feature_scan(data, features, args.min_train_weeks, split[0])
                scan.to_csv(output / f'features_{market}_{label}.csv', index=False)
                importance[(market, label)] = scan
                ranked = scan.loc[scan.mse_contribution > 0, 'feature'].tolist()
                subsets = [features, list(dict.fromkeys(['home_field_adv'] + ranked[:8])),
                           list(dict.fromkeys(['home_field_adv'] + ranked[:16])), ['home_field_adv']]
                seen = set()
                for subset in subsets:
                    if tuple(subset) in seen:
                        continue
                    seen.add(tuple(subset))
                    # Cutoffs and feature/calculation choices see calibration only.
                    preds = weekly_predict(data[data.week_id < split[1]], subset,
                                           args.min_train_weeks, bags=args.bags, seed=args.seed)
                    calibration = preds[preds.week_id >= split[0]]
                    grid = cutoff_grid(calibration, args.min_bets)
                    grid['market'], grid['calculation'], grid['lookback'] = market, calculation, lookback
                    grid['features'] = json.dumps(subset)
                    candidates.append(grid)
                    pd.concat(candidates).to_csv(output / 'cutoff_grid.csv', index=False)
                    print(f'    {len(subset):2d} features; best calibration PnL {grid.pnl_units.max():+.2f}', flush=True)
    grid = pd.concat(candidates, ignore_index=True)
    summaries = {}
    for market in args.markets:
        eligible = grid[(grid.market == market) & grid.eligible]
        if eligible.empty:
            summaries[market] = {'status': 'PASS', 'reason': 'Insufficient calibration bets'}
            continue
        chosen = eligible.sort_values(['pnl_units', 'n'], ascending=[False, False]).iloc[0]
        label = f'{int(chosen.lookback)}_{chosen.calculation}'
        features = json.loads(chosen.features)
        sd = policy_sd_cutoff(chosen)
        config = dict(model=MODEL, market=market, features=features,
                      lookback=int(chosen.lookback), calculation=chosen.calculation,
                      history_weeks=args.history_weeks, min_train_weeks=args.min_train_weeks,
                      bags=args.bags, seed=args.seed, alpha=20,
                      diff_cutoff=float(chosen.diff_cutoff), sd_cutoff=sd,
                      calibration_start=int(boundaries[0]), validation_start=int(boundaries[1]))
        data = market_panel(panels[label], market)
        preds = weekly_predict(data, features, args.min_train_weeks, bags=args.bags, seed=args.seed)
        calibration = preds[(preds.week_id >= boundaries[0]) & (preds.week_id < boundaries[1])]
        validation = preds[(preds.week_id >= boundaries[1]) & preds.residual.notna()]
        stats = score(validation, config['diff_cutoff'], sd)
        interval = roi_interval(validation, config['diff_cutoff'], sd)
        # A positive point estimate alone is not high confidence. Validation is
        # retrospective (these seasons have been explored), not a profit promise.
        supported = (chosen.pnl_units > 0 and stats['n'] >= args.min_bets
                     and interval[0] is not None and interval[0] > 0)
        config['status'] = 'PAPER QUALIFIED' if supported else 'PASS'
        config['reason'] = ('Positive retrospective ROI lower bound; prospective paper validation required'
                            if supported else 'No adequately supported positive validation ROI')
        config['calibration'] = score(calibration, config['diff_cutoff'], sd)
        config['validation'] = stats
        config['validation_roi_95'] = interval
        config['validation_unfiltered'] = score(validation)
        config['variants_tested'] = int((grid.market == market).sum())
        config['training_start'] = data[['season', 'week']].iloc[0].astype(int).tolist()
        config['validation_dates'] = validation[['season', 'week']].drop_duplicates().astype(int).values.tolist()
        summaries[market] = config
        settled = settle(preds, config['diff_cutoff'], sd)
        settled['phase'] = np.select([settled.week_id < boundaries[0], settled.week_id < boundaries[1]],
                                     ['discovery', 'calibration'], default='validation')
        settled.to_parquet(output / f'predictions_{market}.parquet', index=False)
        settled.drop(columns=[c for c in settled if c.startswith('attr_')]).to_csv(
            output / f'picks_{market}.csv', index=False)
        write_packets(settled[settled.week_id >= boundaries[1]], panels[label],
                      importance[(market, label)], config, output / 'weeks')
        print(f"\n{market.upper()}: {config['status']} — {stats['n']} validation bets, "
              f"{stats['pnl_units']:+.2f} units; ROI CI {interval}", flush=True)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    write_research_report(summaries, grid, output)
    print(f'\nResearch and weekly packets: {output / "report.html"}', flush=True)
    return summaries


def confirm_neural(args):
    """Recalibrate on actual neural predictions; never reuse Ridge variance."""
    import model_shredski as ms
    output = Path(args.output)
    summaries = json.loads((output / 'summary.json').read_text())
    reports = {}
    for market, config in summaries.items():
        if 'features' not in config:
            continue
        panel = build_panel(args.season, args.week, args.history_weeks,
                            config['lookback'], config['calculation'])
        features = config['features']
        rows = []
        for (season, week), test in panel[panel.week_id >= config['calibration_start']].groupby(['season', 'week']):
            wid = int(test.week_id.iloc[0])
            history = panel[panel.week_id < wid]
            regular = history.loc[history.game_type == 'REG', 'week_id'].unique()
            history = history[history.week_id >= sorted(regular)[-min(20, len(regular))]]
            data = pd.concat([history, test]).copy()
            # Match the research missing-data policy, using prior games only.
            fill = history[features].replace([np.inf, -np.inf], np.nan).median().fillna(0)
            data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(fill)
            identity = pd.util.hash_pandas_object(data[KEY + features + ['away_score', 'home_score']], index=False).values.tobytes().hex()
            cached = utils.cache_path('neural_confirmation', [identity, market, features, args.neural_iterations, args.seed],
                                      ['model_shredski.py', 'modelo_workers.py'])
            if cached.exists():
                forecast = pd.read_parquet(cached)
            else:
                forecast = ms.modelo(data, int(season), int(week), output, bt=True,
                                     features=features, market=market, iterations=args.neural_iterations,
                                     random_state=args.seed, round_predictions=False)
                utils.save_parquet(forecast, cached)
            result = market_panel(test, market).merge(forecast, on=['away_team', 'home_team'], validate='one_to_one')
            result['edge'] = result.prediction - result.market_base
            rows.append(result)
            pd.concat(rows).to_parquet(output / f'neural_predictions_{market}.parquet', index=False)
        predictions = pd.concat(rows, ignore_index=True)
        calibration = predictions[predictions.week_id < config['validation_start']]
        validation = predictions[predictions.week_id >= config['validation_start']]
        grid = cutoff_grid(calibration, args.min_bets)
        grid.to_csv(output / f'neural_cutoffs_{market}.csv', index=False)
        eligible = grid[grid.eligible]
        if eligible.empty:
            reports[market] = dict(status='PASS', reason='Insufficient calibration bets')
            continue
        chosen = eligible.sort_values('pnl_units', ascending=False).iloc[0]
        sd = policy_sd_cutoff(chosen)
        stats = score(validation, float(chosen.diff_cutoff), sd)
        interval = roi_interval(validation, float(chosen.diff_cutoff), sd)
        supported = chosen.pnl_units > 0 and stats['n'] >= args.min_bets and interval[0] is not None and interval[0] > 0
        reports[market] = dict(model=f'neural-{args.neural_iterations}-members', features=features,
                               lookback=config['lookback'], calculation=config['calculation'], train_weeks=20,
                               seed=args.seed, epochs=100, diff_cutoff=float(chosen.diff_cutoff), sd_cutoff=sd,
                               model_spec=neural_spec(features, config['lookback'], config['calculation'], args.neural_iterations, args.seed, market),
                               evaluation='weekly_walk_forward',
                               validated_through=validation.loc[validation.residual.notna(), ['season', 'week']].max().astype(int).tolist(),
                               status='PAPER QUALIFIED' if supported else 'PASS',
                               validation=stats, validation_roi_95=interval,
                               calibration=score(calibration, float(chosen.diff_cutoff), sd),
                               reason='Ridge-screened shortlist, neural-specific recalibration; not the unchanged 100-member production model')
        (output / 'neural_summary.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
        print(f'NEURAL {market}: {reports[market]["status"]}; {stats}', flush=True)
    from weekly_packet import write_research_report
    write_research_report(summaries, pd.read_csv(output / 'cutoff_grid.csv'), output)
    return reports


def production_rule_scan(args):
    """Freeze the packet models; test only edge/SD rules on weekly forecasts."""
    import model_shredski as ms
    from weekly_packet import page
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    panel = build_panel(args.season, args.week, args.history_weeks, 20, 'legacy')
    symmetric = getattr(args, 'symmetric', False)
    if symmetric:
        panel = symmetric_features(panel)
    calculation = 'legacy-symmetric-v1' if symmetric else 'legacy'
    summaries, cards = {}, []
    for market in args.markets:
        features = [f for f in feature_names(panel, market) if f != 'away_game_importance']
        spec = neural_spec(features, 20, calculation, 100, args.seed, market)
        rows = []
        weeks = panel[(panel.season >= args.start_season) & panel.margin.notna()]
        for (season, week), test in weeks.groupby(['season', 'week']):
            wid = int(test.week_id.iloc[0])
            history = panel[panel.week_id < wid]
            regular = sorted(history.loc[history.game_type == 'REG', 'week_id'].unique())
            if len(regular) < 20:
                raise ValueError('Increase history-weeks: each forecast needs 20 regular training weeks')
            history = history[history.week_id >= regular[-20]]
            data = pd.concat([history, test]).sort_values(KEY).copy()
            identity = pd.util.hash_pandas_object(data[KEY + features + ['away_score', 'home_score']], index=False).values.tobytes().hex()
            sources = ['model_shredski.py', 'modelo_workers.py']
            cached = utils.cache_path('neural_confirmation', [identity, market, features, 100, args.seed], sources)
            packet_cache = utils.cache_path('neural_predictions', [identity, market, features, 100, args.seed, 100], sources)
            if cached.exists() or packet_cache.exists():
                forecast = pd.read_parquet(cached if cached.exists() else packet_cache)
                forecast = forecast[['away_team', 'home_team', 'prediction', 'variance']]
                print(f'{market} {season} wk{week}: cached', flush=True)
            else:
                print(f'{market} {season} wk{week}: actual 100-member production model', flush=True)
                forecast = ms.modelo(data, int(season), int(week), output, bt=True,
                                     features=features, market=market, iterations=100,
                                     random_state=args.seed, round_predictions=False)
                utils.save_parquet(forecast, cached)
            result = market_panel(test, market).merge(forecast, on=['away_team', 'home_team'], validate='one_to_one')
            result['edge'] = result.prediction - result.market_base
            result['sd'] = np.sqrt(result.variance)
            result['training_games'] = len(history)
            rows.append(result)
            utils.save_parquet(pd.concat(rows, ignore_index=True), output / f'predictions_{market}.parquet')
            if len(rows) % 5 == 0:
                # TensorFlow retains native allocations across fits. Recycle
                # processes, not models/seeds, to bound memory on long scans.
                from joblib.externals.loky import get_reusable_executor
                get_reusable_executor().shutdown(wait=True)
        predictions = pd.concat(rows, ignore_index=True)
        calibration = predictions[predictions.season < args.validation_season]
        validation = predictions[predictions.season >= args.validation_season]
        if calibration.empty or validation.empty:
            raise ValueError('Both calibration and later validation seasons are required')
        rules = []
        for edge in [0, .5, 1, 1.5, 2, 3, 4, 5, 6, 8]:
            for sd in [None, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4]:
                earlier = score(calibration, edge, sd)
                later = score(validation, edge, sd)
                lo, hi = roi_interval(validation, edge, sd)
                rules.append(dict(edge_cutoff=edge, sd_cutoff=sd,
                                  calibration_bets=earlier['n'], calibration_pnl=earlier['pnl_units'],
                                  validation_bets=later['n'], validation_wins=later['wins'],
                                  validation_losses=later['losses'], validation_pushes=later['pushes'],
                                  validation_hit_rate=later['win_rate'], validation_pnl=later['pnl_units'],
                                  validation_roi=later['roi'], roi_low=lo, roi_high=hi))
        grid = pd.DataFrame(rules)
        grid.to_csv(output / f'rules_{market}.csv', index=False)
        eligible = grid[grid.calibration_bets >= args.min_bets]
        if eligible.empty:
            summaries[market] = dict(status='PASS', reason='Insufficient calibration bets', model_spec=spec)
            continue
        chosen = eligible.sort_values(['calibration_pnl', 'calibration_bets'], ascending=False).iloc[0]
        sd = policy_sd_cutoff(chosen)
        edge = float(chosen.edge_cutoff)
        stats = score(validation, edge, sd)
        interval = roi_interval(validation, edge, sd)
        supported = chosen.calibration_pnl > 0 and stats['n'] >= args.min_bets and interval[0] is not None and interval[0] > 0
        last = validation[['season', 'week']].sort_values(['season', 'week']).iloc[-1].astype(int).tolist()
        summaries[market] = dict(model='neural-100-members', model_spec=spec, features=features,
                                 diff_cutoff=edge, sd_cutoff=sd, evaluation='weekly_walk_forward',
                                 validated_through=last, calibration=score(calibration, edge, sd),
                                 validation=stats, validation_roi_95=interval,
                                 status='PAPER QUALIFIED' if supported else 'PASS',
                                 reason='Fixed production features/architecture; cutoff selected on earlier seasons only')
        settled = settle(predictions, edge, sd)
        settled.to_csv(output / f'picks_{market}.csv', index=False)
        blocks = []
        for (season, week), block in settled.groupby(['season', 'week']):
            blocks.append(dict(season=int(season), week=int(week), **score(block, edge, sd)))
        pd.DataFrame(blocks).to_csv(output / f'weekly_{market}.csv', index=False)
        (output / 'neural_summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
        print(f'FIXED MODEL {market}: edge >= {edge}, SD <= {sd}; {stats}; ROI CI {interval}', flush=True)
        shown = eligible.sort_values('calibration_pnl', ascending=False).head(15)[
            ['edge_cutoff', 'sd_cutoff', 'calibration_bets', 'calibration_pnl',
             'validation_bets', 'validation_hit_rate', 'validation_pnl', 'validation_roi']]
        cards.append(f'<section class="card"><h2>{market.title()} — {summaries[market]["status"]}</h2><p>Frozen rule: edge ≥ {edge} points; maximum SD {sd if sd is not None else "none"}. Validation PnL {stats["pnl_units"]:+.2f} units on {stats["n"]} bets. ROI interval {interval}.</p><p>Rows ordered by calibration PnL, never validation PnL. Other rows are exploratory comparisons, not independently confirmed winners.</p>{shown.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0)}<p><a href="rules_{market}.csv">All 100 rules</a> · <a href="weekly_{market}.csv">Weekly results</a> · <a href="picks_{market}.csv">Every game</a></p></section>')
        note = f'<h1>Fixed-model edge / SD test</h1><p>Actual 100-member models; 100 epochs; seed {args.seed}; 20-week {calculation} features. Calibration: {args.start_season}–{args.validation_season-1}; validation: {args.validation_season}–{args.season}. Weekly refits use earlier games only. Stored odds, one-unit risk, pushes returned. Historical prices/QBs are not decision-time-verified snapshots. Previously explored seasons make this retrospective evidence, not a future-profit guarantee. SD measures model disagreement.</p>'
        (output / 'report.html').write_text(page('Fixed-model cutoff test', note + ''.join(cards)), encoding='utf-8')
    return summaries


def rescore_saved(output):
    """Convert saved policies/grids to SD and rescore without any model fitting."""
    from weekly_packet import write_packets, write_research_report
    output = Path(output)
    summaries = json.loads((output / 'summary.json').read_text())
    for filename in ['summary.json', 'neural_summary.json']:
        path = output / filename
        if not path.exists():
            continue
        configs = json.loads(path.read_text())
        for market, config in configs.items():
            if 'diff_cutoff' not in config:
                continue
            config['sd_cutoff'] = policy_sd_cutoff(config)
            config.pop('var_cutoff', None)
            prefix = 'neural_' if filename.startswith('neural') else ''
            saved = output / f'{prefix}predictions_{market}.parquet'
            data = pd.read_parquet(saved)
            data = settle(data, config['diff_cutoff'], config['sd_cutoff'])
            boundary = summaries[market]['validation_start']
            validation = data[data.week_id >= boundary]
            calibration = data[(data.week_id >= summaries[market]['calibration_start']) & (data.week_id < boundary)]
            config['validation'] = score(validation, config['diff_cutoff'], config['sd_cutoff'])
            config['validation_roi_95'] = roi_interval(validation, config['diff_cutoff'], config['sd_cutoff'])
            config['calibration'] = score(calibration, config['diff_cutoff'], config['sd_cutoff'])
            utils.save_parquet(data, saved)
            data.drop(columns=[c for c in data if c.startswith('attr_')]).to_csv(output / f'{prefix}picks_{market}.csv', index=False)
            if not prefix:
                season, week = data[['season', 'week']].sort_values(['season', 'week']).iloc[-1].astype(int)
                panel = build_panel(int(season), int(week), config['history_weeks'], config['lookback'], config['calculation'])
                scan = pd.read_csv(output / f'features_{market}_{config["lookback"]}_{config["calculation"]}.csv')
                write_packets(validation, panel, scan, config, output / 'weeks')
            print(f'{prefix}{market}: SD <= {config["sd_cutoff"]}; validation PnL {config["validation"]["pnl_units"]:+.2f}; {config["status"]}')
        path.write_text(json.dumps(configs, indent=2, allow_nan=False))
    for path in [output / 'cutoff_grid.csv', *output.glob('neural_cutoffs_*.csv')]:
        grid = pd.read_csv(path)
        if 'var_cutoff' in grid:
            grid['sd_cutoff'] = np.sqrt(grid.pop('var_cutoff'))
        grid.rename(columns={'var_quantile': 'sd_quantile'}).to_csv(path, index=False)
    write_research_report(json.loads((output / 'summary.json').read_text()), pd.read_csv(output / 'cutoff_grid.csv'), output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2025)
    parser.add_argument('--week', type=int, default=22)
    parser.add_argument('--history-weeks', '--lookback-weeks', type=int, default=150)
    parser.add_argument('--lookbacks', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--calculations', nargs='+', choices=['legacy', 'weighted'], default=['legacy', 'weighted'])
    parser.add_argument('--markets', nargs='+', choices=['spread', 'total'], default=['spread', 'total'])
    parser.add_argument('--min-train-weeks', type=int, default=30)
    parser.add_argument('--min-bets', type=int, default=60)
    parser.add_argument('--bags', type=int, default=15)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--output', default='data/optimize_picks/current')
    parser.add_argument('--confirm-neural', action='store_true', help='Confirm the saved shortlist with actual neural fits (cached per week)')
    parser.add_argument('--rescore', action='store_true', help='Rescore saved predictions in SD units, with no fitting')
    parser.add_argument('--rule-scan', action='store_true', help='Test edge/SD rules on the actual fixed 100-member packet models')
    parser.add_argument('--symmetric', action='store_true', help='Rule-scan challenger: remove away-only usage multipliers; spread only')
    parser.add_argument('--start-season', type=int, default=2024)
    parser.add_argument('--validation-season', type=int, default=2025)
    parser.add_argument('--neural-iterations', type=int, default=20, help='Exact ensemble size for neural confirmation, not a variance proxy')
    args = parser.parse_args(argv)
    if args.symmetric and (not args.rule_scan or args.rescore or args.confirm_neural):
        parser.error('--symmetric requires --rule-scan')
    if args.symmetric:
        args.markets = ['spread']
        if args.output == 'data/optimize_picks/current':
            args.output = 'data/optimize_picks/symmetric_rules'
        if Path(args.output).resolve() == Path('data/optimize_picks/production_rules').resolve():
            parser.error('Use a separate output directory for the symmetric experiment')
    if args.bags < 2 or min(args.lookbacks) < 1 or args.min_bets < 1:
        parser.error('Use at least two ensemble members and positive lookbacks/minimum bets')
    if args.rescore:
        return rescore_saved(args.output)
    if args.rule_scan:
        if args.output == 'data/optimize_picks/current':
            args.output = 'data/optimize_picks/production_rules'
        if not args.start_season < args.validation_season <= args.season:
            parser.error('Require start-season < validation-season <= season')
        return production_rule_scan(args)
    return confirm_neural(args) if args.confirm_neural else research(args)


if __name__ == '__main__':
    main()
