"""Shared backtesting engine (panel construction, walk-forward fitting,
settlement/scoring, cutoff grids) plus its own CLI: chronological, cached
backtests comparing shared and joint context-group models. No live betting.

The primitives here (build_panel, market_panel, feature_names, weekly_predict,
feature_scan, settle/score/roi_interval, cutoff_grid, apply_representation,
symmetrize_training) used to live in optimize_picks.py; that file now imports
them back (`from backtester import ...`) for its own research()/confirm_neural()/
production_rule_scan()/rescore_saved() CLI, which is a distinct thing this
file doesn't do -- shared_scoring.py, joint_scoring.py, weekly_packet.py,
joint_feature_selection.py, and optimus_prime.py depend on the primitives
directly (import backtester), not on optimize_picks' own CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data_crunchski_2 as dc
import shared_scoring as ss
import utils
from edge_scan import PRODUCTION_FEATURES

KEY = ['season', 'week', 'away_team', 'home_team']
DIFF_CUTOFFS = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
SD_QUANTILES = [1, .75, .5, .25]
MODEL = 'weekly-block-bagged-ridge-v1'


def build_panel(season, week, history_weeks, lookback, calculation, use_scaling=True):
    data = dc.prep_test_train(season, week, lookback, history_weeks=history_weeks,
                             calculation=calculation, use_scaling=use_scaling)
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


def _raw_pair(feature):
    """(forward_a, forward_b, mirror_a, mirror_b) raw column names for an
    off/def opponent-adjusted production feature, or None if `feature` isn't
    one. forward_* is what the away-perspective row currently reads;
    mirror_* is the column -- already present in the same row, comp_stats
    merges every raw stat for both teams' full off+def profiles into every
    game -- that holds the equivalent quantity from the home team's
    perspective on the SAME game."""
    if feature.startswith('away_off_'):
        metric = feature[len('away_off_'):]
        return (f'away_raw_off_{metric}', f'home_raw_def_{metric}',
               f'home_raw_off_{metric}', f'away_raw_def_{metric}')
    if feature.startswith('away_def_'):
        metric = feature[len('away_def_'):]
        return (f'away_raw_def_{metric}', f'home_raw_off_{metric}',
               f'home_raw_def_{metric}', f'away_raw_off_{metric}')
    return None


def apply_representation(panel, features, mode):
    """Rebuild `features` in a different representation, using columns
    comp_stats already computed -- no new crunch for any mode.
      'percentile' (default): unchanged, current production columns.
      'raw_diff':   the same off/def opponent-adjusted pairing, on raw
                    (unranked, unscaled) team levels instead of percentile
                    ranks.
      'z_diff':     raw_diff, with each raw column standardized across the
                    panel first (removes cross-metric scale differences).
      'raw_levels': each side's raw level fed as its own separate feature
                    (two columns) instead of one differenced column -- lets
                    the model weight offense and defense independently
                    instead of assuming a strict 1:1 tradeoff.
    Features with no off/def raw pair (home_field_adv, away_rest_adv,
    away_game_importance, or anything with an incomplete raw pair) pass
    through unchanged in every mode -- they aren't part of comp_stats'
    rank/raw duality.

    Returns (data, new_feature_list, mirror_columns). mirror_columns is
    {feature: same-row column holding the other perspective's equivalent
    value}, used by symmetrize_training -- populated only for 'raw_levels'.
    The other three representations have no clean, unambiguous linear swap
    available without rebuilding the crunch from a hypothetical
    swapped-home/away schedule (a plain a-b difference does not negate
    under swap in general; only raw_levels' two-separate-columns form has
    an unambiguous same-row mirror) -- documented gap, not a silent one.
    """
    if mode == 'percentile':
        return panel, list(features), {}
    if mode not in ('raw_diff', 'z_diff', 'raw_levels'):
        raise ValueError(f'Unknown representation mode: {mode}')
    data = panel.copy()
    new_features, mirror_columns = [], {}
    for feature in features:
        pair = _raw_pair(feature)
        if pair is None or not set(pair).issubset(data.columns):
            new_features.append(feature)  # no/incomplete raw pair; keep as-is
            continue
        raw_a, raw_b, mirror_a, mirror_b = pair
        a, b = data[raw_a], data[raw_b]
        if mode == 'z_diff':
            a = (a - a.mean()) / a.std(ddof=0)
            b = (b - b.mean()) / b.std(ddof=0)
        if mode in ('raw_diff', 'z_diff'):
            name = f'{mode}_{feature}'
            data[name] = a - b
            new_features.append(name)
        else:  # raw_levels
            name_a, name_b = f'raw_levels_{raw_a}', f'raw_levels_{raw_b}'
            data[name_a], data[name_b] = data[raw_a], data[raw_b]
            new_features += [name_a, name_b]
            mirror_columns[name_a] = mirror_a
            mirror_columns[name_b] = mirror_b
    return data, list(dict.fromkeys(new_features)), mirror_columns


def symmetrize_training(train, mirror_columns, structure):
    """structure='asymmetric' (current production default): no-op.
    structure='symmetric': the linear analogue of the neural joint model's
    F(A,B)/F(B,A) trick, done via data augmentation instead of architecture
    so it works for a linear model -- appends one mirrored row per training
    game (home team's own perspective on the same game, using the same-row
    columns mirror_columns points at) with the target negated, then fits one
    model on both. A no-op whenever mirror_columns is empty (i.e. every
    representation except 'raw_levels' -- see apply_representation)."""
    if structure == 'asymmetric' or not mirror_columns:
        return train
    if structure != 'symmetric':
        raise ValueError(f'Unknown structure: {structure}')
    mirror = train.copy()
    for feature, source in mirror_columns.items():
        mirror[feature] = train[source]
    mirror['residual'] = -train['residual']
    mirror['actual'] = 2 * train['market_base'] - train['actual']
    return pd.concat([train, mirror], ignore_index=True)


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


def weekly_predict(panel, features, min_train_weeks=30, *, bags=1, seed=1337, alpha=20,
                   mirror_columns=None, structure='asymmetric'):
    """Refit strictly before each week; retain exact additive explanations.

    mirror_columns/structure: optional hook for optimus_prime's representation/
    structure audit -- see symmetrize_training's docstring. Both are plain,
    JSON-serializable values (a dict of strings, a string), so they fold
    cleanly into the cache key below; default leaves every existing caller
    byte-identical (structure='asymmetric' is a no-op)."""
    mirror_columns = mirror_columns or {}
    # total_game_importance rides along even when it isn't a fitted feature
    # (e.g. every spread config) -- score/cutoff_grid's importance-cutoff
    # filter needs it post-hoc regardless of whether the model used it.
    extra = ['total_game_importance'] if 'total_game_importance' in panel.columns else []
    columns = list(dict.fromkeys(KEY + features + list(mirror_columns.values()) + extra +
                       ['week_id', 'residual', 'actual', 'market_base',
                        'positive_odds', 'negative_odds', 'market']))
    data = panel[columns].sort_values(KEY).reset_index(drop=True)
    fingerprint = pd.util.hash_pandas_object(data, index=False).values.tobytes().hex()
    cached = utils.cache_path('predictions', [fingerprint, features, min_train_weeks,
                                             bags, seed, alpha, MODEL, mirror_columns, structure], [__file__])
    if cached.exists():
        return pd.read_parquet(cached)
    rows = []
    from tqdm import tqdm
    # Later weeks train on more history, so this slows down as it goes --
    # the bar's ETA will drift upward through the run, that's expected, not stuck.
    groups = tqdm(data.groupby('week_id', sort=True), desc='weekly_predict',
                 total=data.week_id.nunique(), unit='week')
    for wid, test in groups:
        train = data[(data.week_id < wid) & data.residual.notna()]
        train = symmetrize_training(train, mirror_columns, structure)
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
        # total_game_importance is a fitted feature for the total market --
        # keep it in the output anyway (same reason it was force-included
        # above): it's also a post-hoc cutoff_grid filter column.
        out = test.drop(columns=[f for f in features if f != 'total_game_importance'], errors='ignore').copy()
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


def settle(predictions, diff_cutoff=0, sd_cutoff=None, *, var_cutoff=None,
          importance_cutoff=None, importance_col='total_game_importance'):
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
    if importance_cutoff is not None:
        if importance_col not in data:
            raise ValueError(f'{importance_col} not present in predictions')
        qualifies &= data[importance_col].ge(importance_cutoff)
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


def score(predictions, diff_cutoff=0, sd_cutoff=None, *, var_cutoff=None,
         importance_cutoff=None, importance_col='total_game_importance'):
    data = settle(predictions, diff_cutoff, sd_cutoff, var_cutoff=var_cutoff,
                  importance_cutoff=importance_cutoff, importance_col=importance_col)
    bets = data[data.qualifies & data.residual.notna()]
    n, wins, pushes = len(bets), int(bets.win.sum()), int(bets.push.sum())
    return dict(n=n, wins=wins, losses=n-wins-pushes, pushes=pushes,
                win_rate=wins / (n-pushes) if n > pushes else None,
                pnl_units=float(bets.pnl.sum()), roi=float(bets.pnl.mean()) if n else None,
                assumed_odds=int(bets.assumed_odds.sum()))


def roi_interval(predictions, diff_cutoff=0, sd_cutoff=None, reps=2000, *,
                 importance_cutoff=None, importance_col='total_game_importance'):
    data = settle(predictions, diff_cutoff, sd_cutoff,
                  importance_cutoff=importance_cutoff, importance_col=importance_col)
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


def cutoff_grid(predictions, min_bets, importance_fractions=None, importance_col='total_game_importance'):
    """importance_fractions: optional "keep only the top X% most important
    games" sweep (e.g. [1.0, .75, .5, .25]; 1.0 = no filter). None (default,
    every existing caller) leaves behavior byte-identical to before this
    axis existed -- a single pass with no importance filtering."""
    importance_fractions = importance_fractions or [1.0]
    rows = []
    for diff in DIFF_CUTOFFS:
        for quantile in SD_QUANTILES:
            # Transform the old threshold exactly; interpolated SD quantiles can
            # differ slightly from sqrt(variance quantile) on small samples.
            sd = None if quantile == 1 else float(np.sqrt(predictions.variance.quantile(quantile)))
            for keep in importance_fractions:
                importance_cutoff = (None if keep >= 1.0 else
                                     float(predictions[importance_col].quantile(1 - keep)))
                result = score(predictions, diff, sd, importance_cutoff=importance_cutoff,
                              importance_col=importance_col)
                result.update(diff_cutoff=diff, sd_cutoff=sd, sd_quantile=quantile,
                              importance_keep_fraction=keep, importance_cutoff=importance_cutoff,
                              eligible=result['n'] >= min_bets)
                rows.append(result)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# backtester's own CLI: compare shared/joint context-group models.
# --------------------------------------------------------------------------
def evaluate(predictions, validation_season, min_bets):
    calibration = predictions[predictions.season < validation_season]
    validation = predictions[predictions.season >= validation_season]
    grid = cutoff_grid(calibration, min_bets)
    eligible = grid[grid.eligible].sort_values(['pnl_units', 'n'], ascending=False)
    error = validation.prediction - validation.actual
    result = dict(games=len(validation), mae=float(error.abs().mean()), mse=float((error**2).mean()),
                  status='EXPLORATORY')
    if eligible.empty:
        return dict(result, reason='Insufficient calibration bets'), grid
    rule = eligible.iloc[0]
    edge, sd = float(rule.diff_cutoff), policy_sd_cutoff(rule)
    return dict(result, diff_cutoff=edge, sd_cutoff=sd,
                calibration=score(calibration, edge, sd),
                validation=score(validation, edge, sd),
                roi_95=roi_interval(validation, edge, sd)), grid


def history_weeks(args):
    """Cover the requested evaluation seasons plus the 20-week training burn-in."""
    weeks = pd.read_parquet('data/sched.parquet', columns=['season', 'week', 'game_type'])
    weeks = weeks[weeks.game_type.eq('REG')].drop_duplicates(['season', 'week'])
    prior = weeks[weeks.season < args.start_season]
    if prior.shape[0] < 20:
        raise ValueError(f'Need 20 regular training weeks before {args.start_season}; schedule has {len(prior)}')
    evaluation = weeks[(weeks.season >= args.start_season) &
                       ((weeks.season < args.season) |
                        ((weeks.season == args.season) & (weeks.week <= args.week)))]
    # A regular-season target itself is included without consuming a backward
    # step; postseason targets enter the preceding regular week on the way back.
    target_regular = ((weeks.season == args.season) & (weeks.week == args.week)).any()
    return len(evaluation) + 20 - int(target_regular)


def compare(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    panel = build_panel(args.season, args.week, history_weeks(args), 20, 'mean')
    evaluation = panel[(panel.season >= args.start_season) & panel.margin.notna()]
    if evaluation.empty:
        raise ValueError('No completed games in the requested evaluation period')
    first = evaluation.week_id.min()
    training = panel[(panel.week_id < first) & panel.game_type.eq('REG')]
    if training.week_id.nunique() < 20:
        raise ValueError('Loaded panel lacks 20 regular training weeks; stopped before context calculations')
    joint = getattr(args, 'model', 'shared') == 'joint'
    if joint:
        import joint_scoring as js
        panel = js.context_panel(panel, args.groups, args.weather_source,
                                 args.weather_file, args.decision_hours)
    variants = [()] + [(g,) for g in args.groups]
    if args.combined:
        variants = list(itertools.chain.from_iterable(itertools.combinations(args.groups, n)
                        for n in range(len(args.groups) + 1)))
    if not args.combined and not args.individual:
        variants = [()] + ([tuple(args.groups)] if args.groups else [])
    report, saved = {}, {}
    for groups in variants:
        name = '_'.join(groups) or 'baseline'
        rows = {'spread': [], 'total': []}
        weeks = panel[(panel.season >= args.start_season) & panel.margin.notna()]
        for index, ((season, week), test) in enumerate(weeks.groupby(['season', 'week'])):
            history = panel[panel.week_id < test.week_id.iloc[0]]
            regular = sorted(history.loc[history.game_type.eq('REG'), 'week_id'].unique())
            if len(regular) < 20:
                raise ValueError(f'Missing training history before {season} week {week}')
            data = pd.concat([history[history.week_id >= regular[-20]], test]).sort_values(KEY)
            print(f'{name}: {season} wk{week}', flush=True)
            if joint:
                target, details, _ = js.fit_panel(data, int(season), int(week), args.iterations,
                                                100, args.seed, args.jobs or 8, groups)
            else:
                target, details, _ = ss.fit_panel(data, int(season), int(week), args.iterations,
                                                 100, args.seed, args.jobs, groups, args.inputs)
            for market in rows:
                forecast = market_panel(target, market).merge(
                           details[market] if joint else ss.market_details(details, market),
                           on=['away_team', 'home_team'], validate='one_to_one')
                forecast['edge'] = forecast.prediction - forecast.market_base
                rows[market].append(forecast)
                utils.save_parquet(pd.concat(rows[market], ignore_index=True),
                                   root / f'{name}_{market}.parquet')
            if (index + 1) % 5 == 0:
                from joblib.externals.loky import get_reusable_executor
                get_reusable_executor().shutdown(wait=True)
        report[name] = {}
        for market, blocks in rows.items():
            predictions = pd.concat(blocks, ignore_index=True)
            result, grid = evaluate(predictions, args.validation_season, args.min_bets)
            grid.to_csv(root / f'{name}_{market}_cutoffs.csv', index=False)
            saved[name, market] = predictions
            if groups:
                baseline = saved['baseline', market]
                paired = predictions.merge(baseline[KEY + ['prediction']], on=KEY,
                                           suffixes=('', '_baseline'), validate='one_to_one')
                paired = paired[paired.season >= args.validation_season]
                delta = ((paired.prediction_baseline - paired.actual)**2 -
                         (paired.prediction - paired.actual)**2)
                result['mse_improvement_vs_baseline'] = float(delta.mean())
                base_rule = report['baseline'][market]
                if 'diff_cutoff' in base_rule:
                    result['validation_at_baseline_rule'] = score(
                        predictions[predictions.season >= args.validation_season],
                        base_rule['diff_cutoff'], base_rule['sd_cutoff'])
            report[name][market] = result
        (root / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        flat = [dict(group=g, market=m, games=v['games'], mae=v['mae'], mse=v['mse'],
                     mse_improvement=v.get('mse_improvement_vs_baseline', 0),
                     bets=v.get('validation', {}).get('n', 0),
                     pnl=v.get('validation', {}).get('pnl_units', 0))
                for g, markets in report.items() for m, v in markets.items()]
        pd.DataFrame(flat).to_csv(root / 'comparison.csv', index=False)
        from weekly_packet import page
        note = ('<h1>Matchup models · context tests</h1><p>Retrospective exploratory evidence. '
                'Each fit uses earlier games; cutoff selection uses calibration seasons only. '
                'Positive MSE improvement means lower error than the baseline. '
                'Roof status and referee assignment availability at betting time is unverified. '
                'No production model or betting policy is changed.</p>')
        if joint:
            note += f'<p>Direct margin/total models. Weather source: {args.weather_source}; forecast lead: {args.decision_hours:g} hours. Missing weather is flagged and imputed. Recorded weather is retrospective, not forecast-validated. Importance uses paired win/loss playoff scenarios: fair-coin remaining games, no future ties, approximate later tiebreaks.</p>'
        (root / 'report.html').write_text(page('Context tests', note +
            pd.DataFrame(flat).to_html(index=False, float_format=lambda x: f'{x:.3f}') +
            '<p><a href="summary.json">Rules, fixed-baseline-rule comparison, and ROI intervals</a></p>'),
            encoding='utf-8')
    print(f'Results: {root / "report.html"}')
    return report


def two_sided_season(args):
    """Fixed retrospective experiment; no model/feature/cutoff selection.
    Evaluates every week from args.start_season's week 1 through
    args.season/args.week, one continuous walk-forward -- args.start_season
    defaults to args.season itself (one season) when not given."""
    import data_crunchski_3 as dc3
    model_version = getattr(args, 'model_version', 'legacy')
    if model_version == 'model_2.0':
        import model_spec
        model_name = model_spec.ID
        model_label = model_spec.LABEL
        calculation = 'weighted'
        output_root = model_spec.RESULTS.name
    elif model_version == 'legacy':
        model_name = 'two-sided-team-points-v1'
        model_label = 'Two-sided team scores'
        calculation = 'steep'
        output_root = 'two_sided'
    else:
        raise ValueError(f'Unknown two-sided model version: {model_version}')
    if args.lookback < 1 or args.train_window < 1 or args.iterations < 2 or args.jobs < 1 or args.epochs < 1:
        raise ValueError('Lookback, training window, workers and epochs must be positive; members >= 2')
    weather_file = Path(args.weather_file or 'data/weather/historical_features.parquet')
    if not weather_file.exists():
        raise ValueError(f'Missing historical weather: {weather_file}')
    schedule = pd.read_parquet('data/sched.parquet')
    scheduled = schedule[(schedule.season >= args.start_season) &
                         ((schedule.season < args.season) | ((schedule.season == args.season) & (schedule.week <= args.week)))]
    if scheduled.empty or scheduled[['away_score', 'home_score']].isna().any().any():
        raise ValueError('Need a completed season/window for this retrospective run')
    end_week = int(scheduled[scheduled.season == args.season].week.max())
    config = dict(model=model_name, model_version=model_version, start_season=args.start_season,
                  season=args.season, end_week=end_week,
                  lookback=args.lookback, train_window=args.train_window, calculation=calculation,
                  inputs='league_snapshot_zscore', usage_scaling='symmetric_post_normalization',
                  weather_source='historical_reanalysis', weather_file=str(weather_file),
                  weather_features=dc3.MODEL_WEATHER, iterations=args.iterations,
                  epochs=args.epochs, seed=args.seed, status='RETROSPECTIVE — NOT PREGAME VALIDATION',
                  qb_decay='existing QB Elo decay unchanged')
    fingerprint = utils.cache_path('two_sided_runs', config, [__file__, 'data_crunchski_3.py',
        'data_crunchski_2.py', 'shared_scoring.py', 'model_shredski.py', 'modelo_workers.py',
        'data/sched.parquet', weather_file]).stem
    # Keep the existing legacy single-season path name unchanged (data/bt/two_sided/{season}/...)
    # so this isn't a breaking rename for the common case; multi-season runs
    # get their own {start_season}-{season} folder instead.
    season_label = str(args.season) if args.start_season == args.season else f'{args.start_season}-{args.season}'
    output = Path(args.output or f'data/bt/{output_root}/{season_label}/{fingerprint}')
    span_label = f'{args.season} weeks 1–{end_week}' if args.start_season == args.season else f'{args.start_season} wk1 – {args.season} wk{end_week}'
    print(f'{model_label}: {span_label}, {len(scheduled)} games\n'
          f'  {calculation} / league z-scores / symmetric usage scaling / historical weather\n'
          f'  feature lookback {args.lookback}; training window {args.train_window} REG weeks\n'
          f'  {args.iterations} members, {args.prep_jobs} preparation / {args.jobs} training workers; pre-season warmup included\n'
          f'  RETROSPECTIVE WEATHER EXPERIMENT — not pregame betting validation\n'
          f'  Output: {output}', flush=True)
    if args.plan:
        return config
    from types import SimpleNamespace
    # end_week, not the raw args.week -- args.week can be past whatever's
    # actually scheduled/complete for args.season (end_week is already
    # clamped to that above), and history_weeks counts scheduled weeks
    # regardless of whether they're played, so an uncapped args.week would
    # over-count the required history span.
    span = history_weeks(SimpleNamespace(start_season=args.start_season, season=args.season, week=end_week)) + max(args.train_window - 20, 0)
    panel = build_panel(args.season, end_week, span, args.lookback, calculation, use_scaling=False)
    panel = dc3.attach_historical_weather(panel, weather_file)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    results, importances = [], []
    evaluation = panel[(panel.season >= args.start_season) &
                       ((panel.season < args.season) | ((panel.season == args.season) & (panel.week <= end_week)))]
    if set(evaluation.game_id) != set(scheduled.game_id):
        raise ValueError('Prepared evaluation games differ from the requested schedule')
    for (season, week), target in evaluation.groupby(['season', 'week'], sort=True):
        history = panel[panel.week_id < target.week_id.iloc[0]]
        regular = sorted(history.loc[history.game_type.eq('REG'), 'week_id'].unique())
        if len(regular) < args.train_window:
            raise ValueError(f'{season} wk{week}: insufficient pregame training history')
        data = pd.concat([history[history.week_id >= regular[-args.train_window]], target]).sort_values(KEY)
        details, importance = ss.fit_two_sided(data, int(season), int(week), args.iterations, args.epochs, args.seed, args.jobs)
        importances.append(importance.set_index('feature')['importance'])
        result = target.merge(details, on=['away_team', 'home_team'], validate='one_to_one')
        results.append(result)
        utils.save_parquet(result, output / f'{season}_wk{week:02d}.parquet')
        shown = result[['away_team', 'home_team', 'away_points', 'home_points']].copy()
        shown['away_spread'], shown['spread_sd'] = -result.prediction, np.sqrt(result.variance)
        shown['total'], shown['total_sd'] = result.total_prediction, np.sqrt(result.total_variance)
        print(shown.to_string(index=False, float_format=lambda v: f'{v:.1f}'), flush=True)
        if int(week) % 5 == 0:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
    all_games = pd.concat(results, ignore_index=True)
    all_games.to_csv(output / 'predictions.csv', index=False)
    mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    mean_importance.rename('importance').to_csv(output / 'feature_importance.csv')
    report = dict(config, games=len(all_games), markets={})
    for market in ['spread', 'total']:
        data = market_panel(all_games, market)
        if market == 'total':
            data['prediction'], data['variance'] = data.total_prediction, data.total_variance
        data['edge'] = data.prediction - data.market_base
        error = data.prediction - data.actual
        report['markets'][market] = dict(mae=float(error.abs().mean()), rmse=float(np.sqrt((error**2).mean())),
                                         all_picks=score(data))
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['markets'], indent=2))
    print(f'Results: {output / "predictions.csv"}')
    return report


def configure_workers(args):
    """Bridge CLI preparation workers to the unchanged production loader."""
    if args.prep_jobs < 1 or args.jobs < 1:
        raise ValueError('--prep-jobs and --jobs must be positive integers')
    os.environ['NFL_WORKERS'] = str(args.prep_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['shared', 'joint', 'two-sided'], default='shared')
    parser.add_argument('--model-version', choices=['legacy', 'model_2.0'], default='legacy',
                        help='Two-sided: legacy steep experiment or current Model 2.0 weighted production settings')
    parser.add_argument('--lookback', type=int, default=20, help='Two-sided experiment stat lookback')
    parser.add_argument('--train-window', type=int, default=100, help='Two-sided experiment training REG weeks')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--plan', action='store_true', help='Print two-sided experiment settings without training')
    parser.add_argument('--season', type=int, default=2025)
    parser.add_argument('--week', type=int, default=22)
    parser.add_argument('--start-season', type=int, default=None,
                        help='Two-sided: first season\'s week 1 to evaluate from (default: --season itself, '
                             'i.e. one season). Shared/joint: 2024.')
    parser.add_argument('--validation-season', type=int, default=2025)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--jobs', type=int, default=8, help='Training workers (default: 8)')
    parser.add_argument('--prep-jobs', type=int, default=1,
                        help='Data-preparation workers (default: 1; overrides NFL_WORKERS)')
    parser.add_argument('--min-bets', type=int, default=60)
    parser.add_argument('--groups', nargs='+', choices=list(ss.GROUPS) + ['weather', 'importance'], default=list(ss.GROUPS))
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument('--combined', action='store_true', help='Test every subset of context groups')
    comparison.add_argument('--individual', action='store_true', help='Compare baseline with each group separately')
    comparison.add_argument('--full-only', action='store_true', help='Baseline versus all requested groups together (default)')
    parser.add_argument('--weather-source', choices=['forecast', 'recorded'], default='forecast')
    parser.add_argument('--weather-file')
    parser.add_argument('--decision-hours', type=float, default=24)
    parser.add_argument('--inputs', choices=['differential', 'separate'], default='differential')
    parser.add_argument('--output')
    args = parser.parse_args()
    try:
        configure_workers(args)
    except ValueError as error:
        parser.error(str(error))
    if args.model == 'two-sided':
        # Multi-season by request: --start-season's week 1 through --season/
        # --week, one continuous walk-forward. Default (no --start-season)
        # stays a single season, matching the old behavior.
        if args.start_season is None:
            args.start_season = args.season
        if args.start_season > args.season:
            parser.error('--start-season must be <= --season')
        # validation-season/groups/combined/individual/full-only/inputs/
        # min-bets belong to the older shared/joint compare() path below and
        # two_sided_season() never reads them -- error instead of a silent
        # no-op if one of those got set alongside --model two-sided.
        irrelevant = [('--validation-season', args.validation_season, 2025),
                      ('--groups', args.groups, list(ss.GROUPS)), ('--combined', args.combined, False),
                      ('--individual', args.individual, False), ('--full-only', args.full_only, False),
                      ('--inputs', args.inputs, 'differential'), ('--min-bets', args.min_bets, 60)]
        set_but_unused = [flag for flag, value, default in irrelevant if value != default]
        if set_but_unused:
            parser.error(f'{", ".join(set_but_unused)} have no effect on --model two-sided.')
        try:
            two_sided_season(args)
        except ValueError as error:
            parser.error(str(error))
        raise SystemExit(0)
    if args.plan:
        parser.error('--plan is currently supported for --model two-sided only')
    if args.model_version != 'legacy':
        parser.error('--model-version is currently supported for --model two-sided only')
    if args.start_season is None:
        args.start_season = 2024
    if args.output is None:
        args.output = 'data/optimize_picks/shared_context' + ('_differential' if args.inputs == 'differential' else '')
        if args.model == 'joint':
            args.output = 'data/optimize_picks/joint_context_' + args.weather_source
    if args.model == 'shared' and any(g not in ss.GROUPS for g in args.groups):
        parser.error('Weather and importance groups require --model joint')
    if args.model == 'joint' and args.inputs != 'differential':
        parser.error('Joint models require differential inputs')
    if args.decision_hours < 0:
        parser.error('--decision-hours must be nonnegative')
    if not args.start_season < args.validation_season <= args.season:
        parser.error('Require start-season < validation-season <= season')
    args.groups = list(dict.fromkeys(args.groups))
    compare(args)
