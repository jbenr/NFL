"""Find the most predictive model structure for spread and for total, separately.

Six things get tested, none of them assumed:
  1. Feature combinations       -- which subset is actually predictive
  2. Lookback window            -- how many recent games define "current form"
  3. Recency weighting          -- weighted/steep decay taper vs. plain mean vs. median
  4. Usage-scaling multiplier   -- comp_stats' pass_%/run_% weighting, on or off
  5. Calculation type           -- percentile-rank diff / raw diff / z-scored
     diff / raw levels fed separately (undifferenced)
  6. Model structure            -- one model with all features weighted
     independently, vs. one shared function applied to both sides of the ball
     (the linear analogue of the neural joint model's F(A,B)/F(B,A) trick)
  7. Cutoffs                    -- diff / SD / game-importance filters

Two-phase design, because running the full weekly walk-forward for every
combination of the above would be hundreds of backtests:

  PHASE 0: materialize and cache every panel needed for axes 2-4 (lookback x
  recency x usage-scaling) -- these three change what happens INSIDE the
  crunch (data_crunchski_2.prep_test_train / comp_stats), not a transform on
  top of it, so each combination needs a real crunch. Axes 1, 5, 6 are
  transforms on top of whichever panel already exists and don't need their
  own crunch -- comp_stats already saves every team's raw (unranked,
  undifferenced) level as away_raw_*/home_raw_* alongside the percentile-rank
  differences, so representation and structure are pure column arithmetic on
  a Phase 0 panel that's already sitting on disk.

  PHASE 1 (this file, today): season-blocked cross-validation (hold out one
  season, fit once on the rest, repeat for a handful of recent seasons,
  pool) to rank combinations of representation x structure on top of every
  Phase 0 panel (phase1_config_scan.csv), then a cheap paired feature scan
  on top of the single best config per market (phase1_feature_scan_*.csv).
  ~5 fits per configuration instead of ~300 weekly refits -- the efficiency
  trade the whole two-phase design exists for. This RANKS candidates; it
  does not confirm them. Season-blocked folds train on data that includes
  weeks after the held-out season (real leakage), which is fine for ranking
  -- every candidate gets the same advantage -- but not a claim of real
  performance, and with only a handful of season-blocks the feature-scan
  confidence intervals are wide/noisy: a screen, not a verdict.

  PHASE 2 (next): the one winning configuration per market gets the real,
  leak-free weekly walk-forward (reusing backtester.weekly_predict/
  feature_scan/cutoff_grid exactly as built) for an honest confirmation
  number, plus the diff/SD/importance cutoff grid.

Everything reuses backtester.py's existing functions -- build_panel,
market_panel, feature_names, score -- rather than reimplementing any of
them. Every place a methodology choice needs testing, the affected function
gets one optional parameter (use_scaling on comp_stats/prep_test_train/
build_panel; representation/structure here), not a separate copy per variant.

Usage:
    python optimus_prime.py --start-season 2010

Runs Phase 0 (idempotent -- cached panels reload instantly on a rerun) then
immediately continues into Phase 1 against that manifest.

Output: auto-named under data/bt/<track>/<start_season>-<season>wk<week>_<timestamp>/,
<track> is optimus_ridge (default), optimus_full (--shared), or shared_sweep
(--shared-only). --output overrides this outright.
    phase0_data_manifest.csv
    phase1_config_scan.csv
    phase1_feature_scan_spread.csv
    phase1_feature_scan_total.csv
    phase1_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Everything this file calls (build_panel, market_panel, feature_names,
# weekly_predict, score/settle/roi_interval, cutoff_grid, apply_representation,
# symmetrize_training) is a primitive, not part of optimize_picks' own CLI --
# so this depends on backtester.py directly, where those primitives now live.
import backtester as op

# The real, testable "team form window" axis -- distinct from history_weeks
# (total data available, which should just be as large as possible; there's
# nothing to optimize there). main.py's own production runs use lookback=20;
# this range brackets it on both sides.
LOOKBACK_WINDOWS = [10, 15, 20, 30, 50, 75, 100]

# data_crunchski_2.calc_stats' three cross-game aggregation modes: 'mean'
# (plain average across games in the lookback window -- the clean
# no-recency-weighting baseline), 'weighted' (recency-weighted average,
# game-level not play-level -- see _combine_games), 'median' (per-game
# median, robust to a single explosive game). Replaces the old 'legacy'
# mode, which only partially applied its own weighting (fixed numerator,
# untouched count-based denominator for most ratio features) rather than
# being a genuine unweighted baseline. Genuinely sweeping the decay
# function's own parameters (steepness/floor in gradual_acceleration_with_floor)
# is a deeper follow-up, not done here -- this tests which of the three
# aggregation modes wins, not "which decay shape is best."
RECENCY_MODES = ['weighted', 'steep', 'mean', 'median']

# comp_stats' pass_%/run_% usage multiplier, on (matches all prior runs) or off.
USE_SCALING_OPTIONS = [True, False]

# Phase 1: representation and structure, tested only for spread. Total's
# features (total_*) are already raw, symmetric, un-differenced averages
# across both teams' off+def profiles -- there's no "away minus home" to
# re-represent and no asymmetric off/home-def split to symmetrize, so
# sweeping these for total would just repeat the identical fit four times.
REPRESENTATIONS = ['percentile', 'raw_diff', 'z_diff', 'raw_levels']
STRUCTURES = ['asymmetric', 'symmetric']
CV_SEASONS_BACK = 5  # most-recent seasons, each held out once, for Phase 1's cheap CV


# --------------------------------------------------------------------------
# Phase 0: materialize the panels axes 2-4 actually need a crunch for.
# --------------------------------------------------------------------------
def history_for_start(start_season, season, week):
    """Translate calendar bounds into the existing loader's REG-week counter."""
    weeks = pd.read_parquet('data/sched.parquet')[['season', 'week', 'game_type']].drop_duplicates()
    start = (weeks.season == start_season) & weeks.week.eq(1)
    end = (weeks.season == season) & weeks.week.eq(week)
    if start_season > season or not start.any() or not end.any():
        raise ValueError('Require scheduled start-season week 1 and an end week on or after it')
    regular = weeks.game_type.eq('REG')
    if int((regular & (weeks.season < start_season)).sum()) < max(LOOKBACK_WINDOWS):
        raise ValueError(f'Not enough schedule history before {start_season} for the longest lookback')
    before_end = (weeks.season < season) | ((weeks.season == season) & (weeks.week < week))
    return int((regular & (weeks.season >= start_season) & before_end).sum())


def load_phase0_panel(season, week, history_weeks, start_season, lookback, calculation, use_scaling):
    """Build (or, after the first call, cache-hit load) one Phase 0 panel and
    trim it to the requested calendar research window -- shared by Phase 0's
    own generation loop and Phase 1, so both see identical data."""
    panel = op.build_panel(season, week, history_weeks, lookback, calculation, use_scaling)
    panel = panel.loc[panel.season >= start_season].copy()
    bounds = panel[['season', 'week']].sort_values(['season', 'week'])
    if tuple(bounds.iloc[0]) != (start_season, 1) or tuple(bounds.iloc[-1]) != (season, week):
        raise ValueError('Prepared panel does not match requested calendar bounds')
    return panel


def generate_data_grid(season, week, start_season, output: Path) -> pd.DataFrame:
    """Materialize every (lookback, recency, use_scaling) panel. Idempotent:
    op.build_panel's own caching (utils.cache_path, keyed on every argument
    that affects the result) means a panel already built for a given
    combination loads from disk instead of recrunching -- safe to re-run this
    after adding a lookback value without redoing the ones already done.
    """
    history_weeks = history_for_start(start_season, season, week)
    print(f'Research span: {start_season} wk1 through {season} wk{week}; feature lookback history loaded automatically.', flush=True)
    grid = list(itertools.product(LOOKBACK_WINDOWS, RECENCY_MODES, USE_SCALING_OPTIONS))
    print(f"Phase 0: materializing {len(grid)} panels "
          f"({len(LOOKBACK_WINDOWS)} lookbacks x {len(RECENCY_MODES)} recency modes x "
          f"{len(USE_SCALING_OPTIONS)} scaling options)\n", flush=True)

    rows = []
    for i, (lookback, calculation, use_scaling) in enumerate(grid, 1):
        label = f"lookback={lookback:3d}  recency={calculation:8s}  scaling={str(use_scaling):5s}"
        t0 = time.perf_counter()
        try:
            panel = load_phase0_panel(season, week, history_weeks, start_season, lookback, calculation, use_scaling)
            elapsed = time.perf_counter() - t0
            rows.append(dict(lookback=lookback, calculation=calculation, use_scaling=use_scaling,
                             status='ok', start_season=start_season, end_season=season, end_week=week,
                             n_games=len(panel), n_weeks=int(panel.week_id.nunique()),
                             elapsed_sec=round(elapsed, 1), error=''))
            print(f"  [{i:2d}/{len(grid)}] {label}  ->  {len(panel):4d} games, "
                  f"{int(panel.week_id.nunique()):3d} weeks  ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            rows.append(dict(lookback=lookback, calculation=calculation, use_scaling=use_scaling,
                             status='FAILED', n_games=0, n_weeks=0,
                             elapsed_sec=round(elapsed, 1), error=str(e)))
            print(f"  [{i:2d}/{len(grid)}] {label}  ->  FAILED: {e}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path = output / 'phase0_data_manifest.csv'
    manifest.to_csv(manifest_path, index=False)

    n_ok = int((manifest.status == 'ok').sum())
    total_time = manifest.elapsed_sec.sum()
    print(f"\nPhase 0 complete: {n_ok}/{len(grid)} panels ready "
          f"({total_time/60:.1f} min total, mostly cache-driven on any re-run).")
    if n_ok < len(grid):
        failed = manifest[manifest.status == 'FAILED'][['lookback', 'calculation', 'use_scaling', 'error']]
        print(f"\n{len(failed)} combination(s) failed -- these need attention before Phase 1 can use them:")
        print(failed.to_string(index=False))
    print(f"\nManifest -> {manifest_path}")
    return manifest


# --------------------------------------------------------------------------
# Phase 1: representation and structure as transforms on a Phase 0 panel.
# apply_representation/symmetrize_training live in backtester.py now --
# weekly_predict (Phase 2's real walk-forward) needs symmetrize_training
# directly, so the shared home for both is the shared infrastructure module.
# --------------------------------------------------------------------------
def season_blocked_cv(data_market, features, seasons=None, alpha=20, structure='asymmetric', mirror_columns=None):
    """Cheap ranking signal, NOT a performance claim: hold out one season at
    a time, fit ONCE on every other week (including weeks after the held-out
    season -- real leakage, but identical for every candidate compared this
    way, so relative ranking stays valid), predict the held-out season, pool
    across folds. ~len(seasons) fits per configuration instead of the ~300
    weekly refits weekly_predict does -- the whole point of Phase 1. The
    pooled result has the same shape backtester.score/settle expect, so
    they're reused as-is rather than reimplemented."""
    mirror_columns = mirror_columns or {}
    available = sorted(data_market.season.unique())
    seasons = seasons or available[-CV_SEASONS_BACK:]
    folds = []
    for held_out in seasons:
        train = data_market[(data_market.season != held_out) & data_market.residual.notna()]
        test = data_market[data_market.season == held_out]
        if train.empty or test.empty:
            continue
        train = op.symmetrize_training(train, mirror_columns, structure)
        model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=alpha))
        model.fit(train[features], train.residual)
        pred = test.copy()
        pred['edge'] = model.predict(test[features])
        pred['prediction'] = pred.market_base + pred.edge
        pred['variance'] = 0.0
        folds.append(pred)
    if not folds:
        raise ValueError('No season had both training and held-out data')
    return pd.concat(folds, ignore_index=True)


def season_blocked_feature_scan(data_market, features, seasons=None, alpha=20, structure='asymmetric', mirror_columns=None):
    """Cheap analogue of backtester.feature_scan: paired drop-one-feature
    test, using season_blocked_cv instead of the weekly walk-forward. With
    only a handful of season-blocks the bootstrap CI here is wide -- a
    screen to narrow Phase 2's feature list, not a confident final answer."""
    seasons = seasons or sorted(data_market.season.unique())[-CV_SEASONS_BACK:]
    full = season_blocked_cv(data_market, features, seasons, alpha, structure, mirror_columns)
    base_error = ((full.prediction - full.actual) ** 2).to_numpy()
    rows = []
    for feature in features:
        reduced = season_blocked_cv(data_market, [f for f in features if f != feature],
                                    seasons, alpha, structure, mirror_columns)
        if len(reduced) != len(full):
            raise ValueError('Fold row counts changed between full and reduced fits')
        delta = ((reduced.prediction - reduced.actual) ** 2).to_numpy() - base_error
        paired = pd.DataFrame({'season': full.season.to_numpy(), 'delta': delta})
        by_season = paired.groupby('season').agg(total=('delta', 'sum'), n=('delta', 'count'))
        idx = np.random.default_rng(1337).integers(0, len(by_season), (4000, len(by_season)))
        boot = by_season.total.to_numpy()[idx].sum(axis=1) / by_season.n.to_numpy()[idx].sum(axis=1)
        lo, hi = np.quantile(boot, [.025 / len(features), 1 - .025 / len(features)])
        rows.append(dict(feature=feature, mse_contribution=float(delta.mean()), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows).sort_values('mse_contribution', ascending=False)


# Candidate subset sizes, as a fraction of the ranked feature list (rounded,
# minimum 1). Deliberately several sizes, not just "drop the worst one" --
# the leave-one-out ranking above only tells you each feature's marginal
# value holding everything else fixed; it says nothing about whether a much
# smaller or much larger set actually predicts better, which is what this
# tests for real (each candidate gets its own season-blocked CV score).
FEATURE_SUBSET_FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
FEATURE_SUBSET_ANCHOR = 'home_field_adv'  # sane floor feature, unioned into every candidate if present


def build_feature_subsets(fscan, all_features, fractions=None, anchor=FEATURE_SUBSET_ANCHOR):
    """Candidate feature subsets built from season_blocked_feature_scan's
    per-feature ranking (already sorted best-to-worst by mse_contribution):
    several sizes, plus the "everything with a positive contribution" natural
    cutoff optimize_picks.research() already uses, plus the full set --
    each unioned with `anchor` if it exists. Deduplicated."""
    fractions = fractions or FEATURE_SUBSET_FRACTIONS
    ranked = fscan.feature.tolist()
    positive = fscan.loc[fscan.mse_contribution > 0, 'feature'].tolist()
    anchor_set = [anchor] if anchor in all_features else []
    candidates = [anchor_set + ranked[:max(1, round(f * len(ranked)))] for f in fractions]
    candidates.append(anchor_set + positive if positive else anchor_set + ranked[:1])
    candidates.append(list(all_features))
    seen, unique = set(), []
    for subset in candidates:
        subset = list(dict.fromkeys(subset))
        key = tuple(sorted(subset))
        if subset and key not in seen:
            seen.add(key)
            unique.append(subset)
    return unique


def season_blocked_subset_search(data_market, all_features, fscan, structure='asymmetric',
                                 mirror_columns=None, seasons=None, alpha=20):
    """The actual feature-selection step: score every candidate subset (see
    build_feature_subsets) with the same cheap season-blocked CV used
    everywhere else in Phase 1, and rank them. Cheap: a handful of subsets x
    a handful of season folds, not a combinatorial search -- extensive in
    that it tests real subsets at several sizes, not exhaustive (2^n_features
    is not attemptable for 63 features, or for that matter 18)."""
    subsets = build_feature_subsets(fscan, all_features)
    rows = []
    for subset in subsets:
        try:
            preds = season_blocked_cv(data_market, subset, seasons, alpha, structure, mirror_columns)
            stats = op.score(preds)
            mse = float(((preds.prediction - preds.actual) ** 2).mean())
            rows.append(dict(n_features=len(subset), features=json.dumps(subset), status='ok', mse=mse, **stats))
        except Exception as e:
            rows.append(dict(n_features=len(subset), features=json.dumps(subset), status='FAILED', error=str(e)))
    return pd.DataFrame(rows).sort_values('pnl_units', ascending=False, na_position='last')


def config_scan(manifest, season, week, history_weeks, start_season, output: Path) -> pd.DataFrame:
    """Phase 1a: for every Phase 0 panel that succeeded, every market, every
    representation, every structure -- season-blocked CV on that market's
    full current feature set. Cheap (a handful of Ridge fits per row), so
    the whole grid runs in a couple of minutes even though it's ~180 rows."""
    ok = manifest[manifest.status == 'ok']
    grid = []
    for _, spec in ok.iterrows():
        for market in ['spread', 'total']:
            representations = REPRESENTATIONS if market == 'spread' else ['percentile']
            structures = STRUCTURES if market == 'spread' else ['asymmetric']
            for mode, structure in itertools.product(representations, structures):
                grid.append((spec, market, mode, structure))
    print(f'\nPhase 1a: {len(grid)} (panel x market x representation x structure) configs, '
          f'season-blocked CV over the last {CV_SEASONS_BACK} seasons\n', flush=True)

    rows, panel_cache = [], {}
    for i, (spec, market, mode, structure) in enumerate(grid, 1):
        key = (int(spec.lookback), spec.calculation, bool(spec.use_scaling))
        label = (f"lookback={key[0]:3d} recency={key[1]:8s} scaling={str(key[2]):5s} "
                f"{market:6s} {mode:11s} {structure}")
        row = dict(lookback=key[0], recency=key[1], use_scaling=key[2],
                   market=market, representation=mode, structure=structure)
        try:
            if key not in panel_cache:
                panel_cache[key] = load_phase0_panel(season, week, history_weeks, start_season, *key)
            panel = panel_cache[key]
            data = op.market_panel(panel, market)
            base_features = op.feature_names(data, market)
            rep_data, features, mirror_columns = op.apply_representation(data, base_features, mode)
            preds = season_blocked_cv(rep_data, features, structure=structure, mirror_columns=mirror_columns)
            stats = op.score(preds)
            mse = float(((preds.prediction - preds.actual) ** 2).mean())
            row.update(status='ok', n_features=len(features), mse=mse, **stats)
            print(f"  [{i:3d}/{len(grid)}] {label}  ->  n={stats['n']:4d} "
                  f"win_rate={stats['win_rate']:.3f} pnl={stats['pnl_units']:+7.2f} "
                  f"mse={mse:.2f}" if stats['n'] else
                  f"  [{i:3d}/{len(grid)}] {label}  ->  0 qualifying bets", flush=True)
        except Exception as e:
            row.update(status='FAILED', error=str(e))
            print(f"  [{i:3d}/{len(grid)}] {label}  ->  FAILED: {e}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'phase1_config_scan.csv', index=False)

    scan = pd.DataFrame(rows)
    print(f"\nPhase 1a complete -> {output / 'phase1_config_scan.csv'}")
    return scan


def run_phase1(manifest, season, week, history_weeks, start_season, output: Path):
    scan = config_scan(manifest, season, week, history_weeks, start_season, output)
    summary = {}
    for market in ['spread', 'total']:
        candidates = scan[(scan.market == market) & (scan.status == 'ok') & (scan.n >= 20)]
        if candidates.empty:
            print(f'\n{market.upper()}: no config produced enough qualifying bets to rank -- skipping feature scan.')
            summary[market] = {'status': 'INSUFFICIENT_DATA'}
            continue
        best = candidates.sort_values(['pnl_units', 'win_rate'], ascending=False).iloc[0]
        print(f"\n{market.upper()} best config: lookback={int(best.lookback)} recency={best.recency} "
              f"scaling={best.use_scaling} representation={best.representation} structure={best.structure} "
              f"-- pnl={best.pnl_units:+.2f} units on {int(best.n)} bets, win_rate={best.win_rate:.3f}", flush=True)
        panel = load_phase0_panel(season, week, history_weeks, start_season,
                                  int(best.lookback), best.recency, bool(best.use_scaling))
        data = op.market_panel(panel, market)
        base_features = op.feature_names(data, market)
        rep_data, features, mirror_columns = op.apply_representation(data, base_features, best.representation)
        print(f'  Feature scan (screen, not a verdict -- only {CV_SEASONS_BACK} season-blocks): '
              f'{len(features)} features', flush=True)
        fscan = season_blocked_feature_scan(rep_data, features, structure=best.structure, mirror_columns=mirror_columns)
        fscan.to_csv(output / f'phase1_feature_scan_{market}.csv', index=False)

        print(f'  Feature subset search: scoring {len(FEATURE_SUBSET_FRACTIONS) + 2} candidate subsets '
             f'built from that ranking', flush=True)
        subset_scan = season_blocked_subset_search(rep_data, features, fscan, structure=best.structure,
                                                    mirror_columns=mirror_columns)
        subset_scan.to_csv(output / f'phase1_feature_subsets_{market}.csv', index=False)
        eligible_subsets = subset_scan[(subset_scan.status == 'ok') & (subset_scan.n >= 20)]
        if eligible_subsets.empty:
            chosen_features = features  # fall back to the full set; nothing else qualified
            subset_note = 'no candidate subset had enough bets; kept the full feature set'
        else:
            winner = eligible_subsets.iloc[0]
            chosen_features = json.loads(winner.features)
            subset_note = (f'{int(winner.n_features)} features beat the full {len(features)}-feature set '
                          f'in this screen' if winner.n_features < len(features) else
                          'the full feature set won this screen')
        print(f'  -> {subset_note} ({len(chosen_features)} features chosen)', flush=True)

        summary[market] = dict(status='SCREENED', lookback=int(best.lookback), recency=best.recency,
                               use_scaling=bool(best.use_scaling), representation=best.representation,
                               structure=best.structure, n_features_available=len(features),
                               cv_seasons_back=CV_SEASONS_BACK, calibration_metrics=dict(
                                   n=int(best.n), win_rate=float(best.win_rate), pnl_units=float(best.pnl_units),
                                   roi=float(best.roi) if pd.notna(best.roi) else None, mse=float(best.mse)),
                               top_features=fscan.head(10).feature.tolist(),
                               features=chosen_features, n_features=len(chosen_features))
    (output / 'phase1_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"\nPhase 1 complete -> {output / 'phase1_summary.json'}")
    return summary


# --------------------------------------------------------------------------
# Phase 2: real leak-free weekly walk-forward confirmation of the single
# Phase 1 winner per market, plus the diff/SD/importance cutoff grid.
# --------------------------------------------------------------------------
# "Keep only the top X% most important games" -- 1.0 is no filter (every
# existing cutoff_grid caller still gets exactly that by default; this list
# is Phase 2-specific opt-in).
IMPORTANCE_KEEP_FRACTIONS = [1.0, .75, .5, .25]


def run_phase2(phase1_summary, season, week, history_weeks, start_season, output: Path,
              min_train_weeks=30, min_bets=40, bags=15, seed=1337):
    """The expensive step the whole two-phase design exists to defer, run
    exactly once per market on the single Phase 1 winner: optimize_picks'
    real weekly walk-forward (strictly-prior-weeks-only refits), a
    discovery/calibration/validation split identical in spirit to
    optimize_picks.research (cutoffs chosen on calibration, validation
    touched once), and the diff/SD/importance cutoff grid."""
    summary = {}
    for market in ['spread', 'total']:
        config = phase1_summary.get(market, {})
        if config.get('status') != 'SCREENED':
            print(f'\n{market.upper()}: no Phase 1 winner to confirm (status={config.get("status")}) -- skipping.')
            summary[market] = {'status': 'PASS', 'reason': 'No Phase 1 winner'}
            continue
        print(f"\n{market.upper()}: confirming lookback={config['lookback']} recency={config['recency']} "
              f"scaling={config['use_scaling']} representation={config['representation']} "
              f"structure={config['structure']} with the real weekly walk-forward...", flush=True)
        panel = load_phase0_panel(season, week, history_weeks, start_season, config['lookback'],
                                  config['recency'], config['use_scaling'])
        data = op.market_panel(panel, market)
        base_features = op.feature_names(data, market)
        rep_data, all_rep_features, mirror_columns = op.apply_representation(data, base_features, config['representation'])
        # Phase 1's feature subset search already picked the winning subset
        # (phase1_feature_subsets_{market}.csv) -- confirm that one, not the
        # full set, unless an older phase1_summary.json predates that step.
        features = config.get('features') or all_rep_features
        mirror_columns = {k: v for k, v in mirror_columns.items() if k in features}
        print(f'  Using {len(features)}/{len(all_rep_features)} features selected in Phase 1.', flush=True)

        scored = sorted(w for w in rep_data.loc[rep_data.residual.notna(), 'week_id'].unique() if w >= min_train_weeks)
        if len(scored) < 40:
            summary[market] = {'status': 'PASS', 'reason': 'Insufficient scored weeks for a 3-way split'}
            continue
        split = (scored[len(scored) // 2], scored[3 * len(scored) // 4])

        preds = op.weekly_predict(rep_data, features, min_train_weeks, bags=bags, seed=seed,
                                  mirror_columns=mirror_columns, structure=config['structure'])
        calibration = preds[(preds.week_id >= split[0]) & (preds.week_id < split[1])]
        validation = preds[(preds.week_id >= split[1]) & preds.residual.notna()]
        grid = op.cutoff_grid(calibration, min_bets, importance_fractions=IMPORTANCE_KEEP_FRACTIONS)
        grid.to_csv(output / f'phase2_cutoff_grid_{market}.csv', index=False)

        eligible = grid[grid.eligible]
        if eligible.empty:
            summary[market] = {'status': 'PASS', 'reason': 'Insufficient calibration bets at any cutoff'}
            continue
        chosen = eligible.sort_values(['pnl_units', 'n'], ascending=[False, False]).iloc[0]
        sd = op.policy_sd_cutoff(chosen)
        importance_cutoff = None if pd.isna(chosen.importance_cutoff) else float(chosen.importance_cutoff)
        diff = float(chosen.diff_cutoff)
        stats = op.score(validation, diff, sd, importance_cutoff=importance_cutoff)
        interval = op.roi_interval(validation, diff, sd, importance_cutoff=importance_cutoff)
        supported = (chosen.pnl_units > 0 and stats['n'] >= min_bets
                    and interval[0] is not None and interval[0] > 0)
        summary[market] = dict(config, status='PAPER QUALIFIED' if supported else 'PASS',
                               reason=('Positive retrospective ROI lower bound; prospective paper '
                                       'validation required' if supported else
                                       'No adequately supported positive validation ROI'),
                               n_features=len(features), diff_cutoff=diff, sd_cutoff=sd,
                               importance_keep_fraction=float(chosen.importance_keep_fraction),
                               importance_cutoff=importance_cutoff,
                               calibration_start=int(split[0]), validation_start=int(split[1]),
                               calibration=op.score(calibration, diff, sd, importance_cutoff=importance_cutoff),
                               validation=stats, validation_roi_95=interval,
                               validation_unfiltered=op.score(validation))
        settled = op.settle(preds, diff, sd, importance_cutoff=importance_cutoff)
        settled.drop(columns=[c for c in settled if c.startswith('attr_')]).to_csv(
            output / f'phase2_picks_{market}.csv', index=False)
        print(f"  {market.upper()}: {summary[market]['status']} -- {stats['n']} validation bets, "
              f"{stats['pnl_units']:+.2f} units; ROI 95% CI {interval}", flush=True)

    import json
    (output / 'phase2_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"\nPhase 2 complete -> {output / 'phase2_summary.json'}")
    return summary


# --------------------------------------------------------------------------
# Shared model track: the "PRIME" architecture requested directly -- one
# scoring function (shared_scoring.py), applied to both teams' own-offense-
# vs-opponent-defense perspective with genuinely shared weights (the same
# model object evaluated on both rows, not a data-augmentation approximation
# of symmetry). Predicts each team's own points; spread = away - home,
# total = away + home, both from ONE fit per week.
#
# Feature selection here works differently than the Ridge track on purpose:
# a real neural fit isn't cheap enough to screen with season-blocked CV or
# leave-one-out refits. Instead this uses permutation importance, which
# shared_scoring.fit_panel already computes as part of every normal fit (free,
# not an extra N+1 refits) -- and it selects at the METRIC level (both the
# off_ and def_ half of a metric are always in or out together, since that's
# the actual unit data_crunchski_3.feature_names/scoring_rows operate on),
# which structurally can't produce the "kept one side of a matchup, dropped
# the other" problem the Ridge raw_levels subset search had.
# --------------------------------------------------------------------------
SHARED_TRAIN_WINDOWS = [20, 30, 50, 100]  # weeks of recent history the neural model itself trains on
SHARED_SCREEN_ITERATIONS = 50
SHARED_CONFIRM_ITERATIONS = 100
SHARED_METRIC_FRACTIONS = [0.35, 0.5, 0.7, 1.0]  # candidate metric-subset sizes to confirm
SHARED_RESULTS_DIR = Path('data/bt')  # every shared-model week's forecast lands here, never just in memory
_RECYCLE_EVERY = 5  # weeks between worker-pool recycles -- see shared_walk_forward's docstring


def load_shared_panel(season, week, start_season, lookback, train_window, calculation, use_scaling):
    """Keep pre-start training rows; each row also needs its own stat lookback."""
    sched = pd.read_parquet('data/sched.parquet', columns=['season', 'week', 'game_type'])
    prior = sched[(sched.season < start_season) & sched.game_type.eq('REG')].drop_duplicates(['season', 'week'])
    if len(prior) < train_window + lookback:
        raise ValueError(f'Need {train_window} training weeks plus {lookback} feature-history weeks before {start_season}')
    history = history_for_start(start_season, season, week) + train_window
    return op.build_panel(season, week, history, lookback, calculation, use_scaling)


def _shared_run_dir(market, lookback, train_window, calculation, use_scaling, inputs, metrics,
                    iterations=50, seed=1337, epochs=100, groups=(), start_season=2010, season=2025, week=22):
    """One directory per exact (market, lookback, train_window, recency mode,
    usage-scaling, inputs mode, metric subset) combination -- the path alone
    identifies exactly which model variant and parameters produced every
    file inside it. metrics=None (full set) is labeled distinctly from any
    narrowed subset, which gets a short content hash so two different
    subsets of the same size never collide. calculation/use_scaling/inputs
    are included because they change what the panel or the fit actually
    computes -- without them here, two different recipes sharing a
    (lookback, train_window, metrics) would silently overwrite each other's
    per-week files on disk."""
    import data_crunchski_3 as dc3
    universe = len(dc3.METRICS)
    metric_label = f'full{universe}' if metrics is None else \
        f'{len(metrics)}of{universe}-{hashlib.sha256(",".join(sorted(metrics)).encode()).hexdigest()[:8]}'
    scaling_label = 'scaled' if use_scaling else 'unscaled'
    return (SHARED_RESULTS_DIR / 'shared_model' / market /
           f'lookback{lookback:03d}_trainwindow{train_window:03d}_{calculation}_{scaling_label}_{inputs}_{metric_label}' /
           f'members{iterations}_seed{seed}_epochs{epochs}_{"-".join(sorted(groups)) or "base"}_{start_season}-{season}wk{week:02d}')


def _write_shared_config(run_dir, **params):
    """config.json, written once per run directory, spelling out every
    parameter explicitly -- any file under data/bt/ is self-describing
    without reverse-engineering it from the directory name alone."""
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        config_path.write_text(json.dumps(
            dict(model='shared_scoring (real shared-weight neural net, not the Ridge approximation)', **params),
            indent=2, default=str))


def _estimate_weeks_per_config(season, week, start_season, train_window):
    """Schedule-only estimate; training windows do not shorten evaluation."""
    sched = pd.read_parquet('data/sched.parquet', columns=['season', 'week', 'game_type'])
    completed = sched[(sched.season > start_season) | (sched.season == start_season)]
    completed = completed[(completed.season < season) | ((completed.season == season) & (completed.week <= week))]
    n_weeks = completed[['season', 'week']].drop_duplicates().shape[0]
    return n_weeks  # training history comes from before start_season


class BacktestDashboard:
    """Compact, in-place-updating progress display for the shared-model
    track: overall week-fits done vs. an upfront (approximate) estimate,
    which exact stage/config/subset is currently running, and weeks
    remaining within that stage. fit_panel's own ensemble-member bar nests
    directly below these three (progress_position=3) instead of fighting
    them for the same terminal lines. Everything except the overall bar
    uses leave=False, so each stage's/week's line is replaced by the next
    instead of scrolling -- exactly the "compact and legible" ask."""
    POSITION_STAGE, POSITION_WEEKS, POSITION_MEMBERS = 1, 2, 3

    def __init__(self, total_estimate):
        from tqdm import tqdm
        self._tqdm = tqdm
        self.overall = tqdm(total=total_estimate, position=0, desc='Overall backtest (est.)',
                            unit='wk-fit', leave=True)
        self.stage = tqdm(total=1, position=self.POSITION_STAGE, bar_format='{desc}', leave=False)
        self.weeks = tqdm(total=0, position=self.POSITION_WEEKS, desc='weeks in this stage', unit='wk', leave=False)

    def start_stage(self, label, n_weeks):
        self.stage.set_description_str(label)
        self.weeks.reset(total=n_weeks)

    def week_done(self):
        self.weeks.update(1)
        self.overall.update(1)

    def note(self, message):
        """Print a real message (e.g. a config's final pnl) above the bars
        without corrupting them -- tqdm.write, not print."""
        self._tqdm.write(message)

    def close(self):
        for bar in [self.weeks, self.stage, self.overall]:
            bar.close()


def shared_walk_forward(panel, start_season, season, week, lookback, train_window, iterations,
                        calculation='mean', use_scaling=True, metrics=None, groups=(), inputs='separate',
                        seed=1337, jobs=None, epochs=100, progress_label='', dashboard=None,
                        recycle_every=_RECYCLE_EVERY):
    """Real weekly walk-forward for the shared scoring model: each week
    refits on only the most recent `train_window` regular weeks of history
    (matching backtester.compare()'s own windowing), predicting spread and
    total simultaneously from one fit (ss.fit_panel returns both markets at
    once -- a real structural advantage over the Ridge track's two separate
    models).

    Every week's SETTLED forecast is written immediately to
    data/bt/shared_model/<market>/<exact config>/<season>_wk<week>.parquet --
    never accumulated in a growing in-memory list -- so peak memory stays
    flat across a long walk-forward instead of climbing with progress.
    ss.fit_panel has its own disk cache too, so a rerun (or resuming after a
    crash) is fast regardless; this is specifically about not holding
    hundreds of weeks' results in RAM at once during a single run.

    Also recycles the loky worker pool every 5 weeks -- TensorFlow leaks
    native memory across many fits inside the same persistent worker
    process, a known issue this codebase already works around the same way
    in optimize_picks.production_rule_scan and backtester.compare.

    calculation/use_scaling: purely for labeling the output directory and
    config.json -- `panel` must already have been built with these exact
    settings (via load_phase0_panel), since this function doesn't rebuild
    the panel itself. Get this wrong and the label lies about what's inside.

    Returns (predictions_by_market, mean_importance) -- mean_importance is
    the average permutation importance across every week's fit, indexed by
    raw feature name (off_X/def_X/context names)."""
    import shared_scoring as ss
    run_dirs = {market: _shared_run_dir(market, lookback, train_window, calculation, use_scaling, inputs, metrics,
                                      iterations, seed, epochs, groups, start_season, season, week)
               for market in ['spread', 'total']}
    for market, run_dir in run_dirs.items():
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_shared_config(run_dir, market=market, lookback=lookback, train_window=train_window,
                             calculation=calculation, use_scaling=use_scaling,
                             metrics=metrics, groups=list(groups), inputs=inputs, iterations=iterations,
                             epochs=epochs, seed=seed, start_season=start_season, season=season, week=week)
    weeks = panel[(panel.season >= start_season) & panel.margin.notna()]
    importances = []
    week_list = list(weeks.groupby(['season', 'week']))
    if dashboard is not None:
        dashboard.start_stage(progress_label, len(week_list))
    fit_position = BacktestDashboard.POSITION_MEMBERS if dashboard is not None else 0
    for i, ((wk_season, wk_week), test) in enumerate(week_list, 1):
        wid = int(test.week_id.iloc[0])
        history = panel[panel.week_id < wid]
        regular = sorted(history.loc[history.game_type.eq('REG'), 'week_id'].unique())
        if len(regular) < train_window:
            raise ValueError(f'{wk_season} wk{wk_week}: need {train_window} prior regular training weeks; got {len(regular)}')
        data = pd.concat([history[history.week_id >= regular[-train_window]], test]).sort_values(op.KEY)
        if dashboard is None:
            tag = f'{progress_label}  ' if progress_label else ''
            print(f'    {tag}[{i:3d}/{len(week_list)}] {int(wk_season)} wk{int(wk_week)}', flush=True)
        target, details, importance = ss.fit_panel(data, int(wk_season), int(wk_week), iterations, epochs,
                                                    seed, jobs, groups, inputs, metrics, progress_position=fit_position)
        if dashboard is not None:
            dashboard.week_done()
        importances.append(importance.set_index('feature')['importance'])
        for market, run_dir in run_dirs.items():
            forecast = op.market_panel(test, market).merge(
                ss.market_details(details, market), on=['away_team', 'home_team'], validate='one_to_one')
            forecast['edge'] = forecast.prediction - forecast.market_base
            forecast['week_id'] = wid
            utils.save_parquet(forecast, run_dir / f'{int(wk_season)}_wk{int(wk_week):02d}.parquet')
        del target, details, data
        if recycle_every > 0 and i % recycle_every == 0:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
    if not importances:
        raise ValueError(f'No week had {train_window} regular training weeks available')
    mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    # Read back only the weeks THIS call processed (respects start_season) --
    # not a blind glob, which could also pick up another run's leftover files
    # sharing this same (lookback, train_window, metrics) directory but a
    # different start_season.
    expected = [f'{int(s)}_wk{int(w):02d}.parquet' for (s, w), _ in week_list]
    predictions = {}
    for market, run_dir in run_dirs.items():
        files = [run_dir / name for name in expected if (run_dir / name).exists()]
        if files:
            predictions[market] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return predictions, mean_importance


def rank_metrics(mean_importance, metrics_universe):
    """Collapse off_X/def_X permutation importance into one score per metric
    (sum of both halves) -- the selection unit for the metric-subset search
    below is a whole metric, not an individual off_X/def_X feature, so a
    matchup pair can never be split the way the Ridge raw_levels search
    accidentally split them."""
    scores = {metric: mean_importance.get(f'off_{metric}', 0.0) + mean_importance.get(f'def_{metric}', 0.0)
                     + mean_importance.get(f'diff_{metric}', 0.0)
             for metric in metrics_universe}
    return pd.Series(scores).sort_values(ascending=False)


def shared_config_scan(panel_specs, season, week, start_season, output: Path, iterations, groups=(), jobs=None,
                       dashboard=None, recycle_every=_RECYCLE_EVERY):
    """Sweep (comp-stat lookback x neural train-window x recency mode x
    usage-scaling x inputs mode), small screening ensemble, full metric set.
    Scores each config on both markets and keeps every week's permutation
    importance for the metric-subset search that follows on the winner.
    Deliberately does NOT keep the panel or predictions from every combo in
    memory -- with a wide grid that adds up across combos on top of
    shared_walk_forward's own per-week savings. load_phase0_panel is
    disk-cached, so reloading the winner's panel later (in run_shared_track)
    is cheap, not a repeat of the crunch.

    panel_specs: list of (lookback, train_window, calculation, use_scaling, inputs)."""
    say = dashboard.note if dashboard is not None else (lambda m: print(m, flush=True))
    rows, importances = [], {}
    say(f'\nShared track config scan: {len(panel_specs)} configs '
       f'(lookback x train-window x recency x scaling x inputs), '
       f'{iterations}-member screening ensemble, {jobs or "default"} CPU workers\n')
    for i, (lookback, train_window, calculation, use_scaling, inputs) in enumerate(panel_specs, 1):
        key = (lookback, train_window, calculation, use_scaling, inputs)
        label = (f'comp-lookback={lookback:3d}  train-window={train_window:3d} weeks  '
                f'recency={calculation:8s}  scaling={str(use_scaling):5s}  inputs={inputs}')
        combo_tag = (f'CONFIG SCAN [{i}/{len(panel_specs)}] lookback={lookback} train_window={train_window} '
                    f'{calculation} scaling={use_scaling} {inputs}')
        say(f'  [{i}/{len(panel_specs)}] {label}')
        row = dict(lookback=lookback, train_window=train_window, calculation=calculation,
                  use_scaling=use_scaling, inputs=inputs)
        try:
            panel = load_shared_panel(season, week, start_season, lookback, train_window, calculation, use_scaling)
            predictions, importance = shared_walk_forward(panel, start_season, season, week, lookback, train_window,
                                                           iterations, calculation=calculation, use_scaling=use_scaling,
                                                           inputs=inputs, groups=groups, jobs=jobs,
                                                           progress_label=combo_tag, dashboard=dashboard,
                                                           recycle_every=recycle_every)
            importances[key] = importance
            for market in ['spread', 'total']:
                stats = op.score(predictions[market])
                row.update({f'{market}_n': stats['n'], f'{market}_win_rate': stats['win_rate'],
                           f'{market}_pnl_units': stats['pnl_units']})
            row['status'] = 'ok'
            say(f"    -> spread pnl={row.get('spread_pnl_units', float('nan')):+.2f} "
               f"(n={row.get('spread_n', 0)})  total pnl={row.get('total_pnl_units', float('nan')):+.2f} "
               f"(n={row.get('total_n', 0)})")
        except Exception as e:
            row['status'] = 'FAILED'
            row['error'] = str(e)
            say(f'    -> FAILED: {e}')
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'shared_config_scan.csv', index=False)
    return pd.DataFrame(rows), importances


def shared_metric_subset_search(panel, start_season, season, week, lookback, train_window, calculation,
                                use_scaling, inputs, iterations, mean_importance, market, groups=(), jobs=None,
                                dashboard=None, recycle_every=_RECYCLE_EVERY):
    """Score candidate metric subsets (built from mean_importance's ranking,
    several sizes) with the same screening walk-forward. One market's cutoff-
    free score decides the winner; the OTHER market's predictions from the
    exact same fits are free (fit_panel predicts both at once), so both get
    reported even though only one market drives the subset choice here."""
    import data_crunchski_3 as dc3
    say = dashboard.note if dashboard is not None else (lambda m: print(m, flush=True))
    ranked = rank_metrics(mean_importance, dc3.METRICS).index.tolist()
    subsets = list(dict.fromkeys(
        tuple(ranked[:max(1, round(f * len(ranked)))]) for f in SHARED_METRIC_FRACTIONS))
    rows = []
    for i, subset in enumerate(subsets, 1):
        subset_tag = f'SUBSET SEARCH {market.upper()} [{i}/{len(subsets)}] {len(subset)}/{len(ranked)} metrics'
        say(f'    subset [{i}/{len(subsets)}]: {len(subset)}/{len(ranked)} metrics')
        try:
            predictions, _ = shared_walk_forward(panel, start_season, season, week, lookback, train_window,
                                                 iterations, calculation=calculation, use_scaling=use_scaling,
                                                 inputs=inputs, metrics=list(subset), groups=groups, jobs=jobs,
                                                 progress_label=subset_tag, dashboard=dashboard,
                                                 recycle_every=recycle_every)
            row = dict(n_metrics=len(subset), metrics=json.dumps(list(subset)), status='ok')
            for m in ['spread', 'total']:
                stats = op.score(predictions[m])
                row.update({f'{m}_n': stats['n'], f'{m}_win_rate': stats['win_rate'], f'{m}_pnl_units': stats['pnl_units']})
        except Exception as e:
            row = dict(n_metrics=len(subset), metrics=json.dumps(list(subset)), status='FAILED', error=str(e))
        rows.append(row)
    scan = pd.DataFrame(rows).sort_values(f'{market}_pnl_units', ascending=False, na_position='last')
    return scan


def run_shared_track(season, week, start_season, output: Path, lookbacks=None, train_windows=None,
                     calculations=None, use_scaling_options=None, input_modes=None,
                     screen_iterations=SHARED_SCREEN_ITERATIONS, confirm_iterations=SHARED_CONFIRM_ITERATIONS,
                     min_bets=40, groups=(), jobs=None, recycle_every=_RECYCLE_EVERY):
    """Orchestrates the whole shared-model track: config scan (lookback x
    train-window x recency mode x usage-scaling x inputs mode, full metrics,
    small ensemble) -> metric-subset search on the winning config per market
    -> one real confirmation fit per market with a larger ensemble -> the
    same diff/SD/importance cutoff grid used everywhere else. Every fit is
    disk-cached by shared_scoring.fit_panel, so this is fully safe to
    re-run/resume after an interruption.

    calculations defaults to sweeping all four data_crunchski_2.calc_stats
    cross-game aggregation modes (weighted, steep, mean, median -- see
    DECAY_PRESETS/_combine_games' docstrings); use_scaling_options defaults to sweeping
    both values. input_modes defaults to all four modes (separate,
    differential, percentile, zscore) -- percentile and zscore are parallel
    representations, both built by data_crunchski_2.comp_stats from the same
    per-week, all-teams-in-the-league normalization (see
    data_crunchski_3.matchup_representation's docstring).

    recycle_every: weeks between loky worker-pool recycles (TensorFlow
    leaks native memory across many fits in the same persistent worker
    process -- see shared_walk_forward's docstring). Lower this if memory
    is still climbing faster than you'd like with the default cadence;
    0 disables recycling entirely."""
    lookbacks = lookbacks or [20, 30, 50]
    train_windows = train_windows or SHARED_TRAIN_WINDOWS
    calculations = calculations or ['weighted', 'steep', 'mean', 'median']
    use_scaling_options = use_scaling_options if use_scaling_options is not None else [True, False]
    input_modes = input_modes or ['separate', 'differential', 'percentile', 'zscore']
    specs = list(itertools.product(lookbacks, train_windows, calculations, use_scaling_options, input_modes))

    # Overall bar total: an upfront, schedule-only ESTIMATE (config scan +
    # 4 subset-search sizes x 2 markets + 1 confirmation x 2 markets, all
    # using each config's representative week count) -- not exact, since
    # the real per-config week count depends on the crunched panel, not
    # just the schedule. Good enough for "how far along am I," not a claim
    # of precision.
    representative_weeks = {tw: _estimate_weeks_per_config(season, week, start_season, tw) for tw in train_windows}
    scan_total = sum(representative_weeks[tw] for _, tw, *_ in specs)
    per_market_followup = (len(SHARED_METRIC_FRACTIONS) + 1) * max(representative_weeks.values())
    total_estimate = scan_total + 2 * per_market_followup
    dashboard = BacktestDashboard(total_estimate)

    try:
        scan, importances = shared_config_scan(specs, season, week, start_season, output, screen_iterations,
                                               groups, jobs, dashboard=dashboard, recycle_every=recycle_every)
        summary = {}
        for market in ['spread', 'total']:
            col = f'{market}_pnl_units'
            candidates = scan[(scan.status == 'ok') & (scan[f'{market}_n'] >= 20)]
            if candidates.empty:
                dashboard.note(f'\nSHARED {market.upper()}: no config produced enough qualifying bets -- skipping.')
                summary[market] = {'status': 'INSUFFICIENT_DATA'}
                continue
            best = candidates.sort_values(col, ascending=False).iloc[0]
            lookback, train_window = int(best.lookback), int(best.train_window)
            calculation, use_scaling, inputs = best.calculation, bool(best.use_scaling), best.inputs
            dashboard.note(f"\nSHARED {market.upper()} best config: lookback={lookback} train_window={train_window} "
                          f"recency={calculation} scaling={use_scaling} inputs={inputs} "
                          f"-- pnl={best[col]:+.2f} units on {int(best[f'{market}_n'])} bets, "
                          f"win_rate={best[f'{market}_win_rate']:.3f}")
            mean_importance = importances[(lookback, train_window, calculation, use_scaling, inputs)]
            panel = load_shared_panel(season, week, start_season, lookback, train_window, calculation, use_scaling)
            dashboard.note(f'  Metric subset search for {market} ({screen_iterations}-member screens):')
            subset_scan = shared_metric_subset_search(
                panel, start_season, season, week, lookback, train_window, calculation, use_scaling, inputs,
                screen_iterations, mean_importance, market, groups, jobs, dashboard=dashboard,
                recycle_every=recycle_every)
            subset_scan.to_csv(output / f'shared_metric_subsets_{market}.csv', index=False)
            eligible = subset_scan[(subset_scan.status == 'ok') & (subset_scan[f'{market}_n'] >= 20)]
            chosen_metrics = json.loads(eligible.iloc[0].metrics) if not eligible.empty else list(dc3_metrics_fallback())

            dashboard.note(f'  Confirming with {confirm_iterations}-member ensemble, {len(chosen_metrics)} metrics...')
            confirm_tag = f'CONFIRM {market.upper()} lookback={lookback} train_window={train_window}'
            preds, _ = shared_walk_forward(panel, start_season, season, week, lookback, train_window, confirm_iterations,
                                           calculation=calculation, use_scaling=use_scaling, inputs=inputs,
                                           metrics=chosen_metrics, groups=groups, jobs=jobs,
                                           progress_label=confirm_tag, dashboard=dashboard, recycle_every=recycle_every)
            data_market = preds[market]
            # Both markets' predictions come from the same underlying
            # away_points/home_points (fit_panel predicts each team's own
            # score; spread and total are just two views of it -- see
            # shared_scoring.summarize/market_details). Save the confirmed
            # run's full per-game predictions -- actual scores, both
            # markets' derived values, and the raw away/home points they
            # come from -- so a derived-spread-vs-derived-total comparison
            # doesn't require re-fitting anything.
            diagnostics = data_market.copy()
            diagnostics['derived_spread'] = diagnostics.away_points - diagnostics.home_points
            diagnostics['derived_total'] = diagnostics.away_points + diagnostics.home_points
            diagnostics.to_csv(output / f'shared_predictions_{market}.csv', index=False)
            scored = sorted(data_market.week_id.unique())
            if len(scored) < 40:
                summary[market] = {'status': 'PASS', 'reason': 'Insufficient scored weeks for a 3-way split'}
                continue
            split = (scored[len(scored) // 2], scored[3 * len(scored) // 4])
            calibration = data_market[(data_market.week_id >= split[0]) & (data_market.week_id < split[1])]
            validation = data_market[data_market.week_id >= split[1]]
            grid = op.cutoff_grid(calibration, min_bets, importance_fractions=IMPORTANCE_KEEP_FRACTIONS)
            grid.to_csv(output / f'shared_cutoff_grid_{market}.csv', index=False)
            eligible_grid = grid[grid.eligible]
            if eligible_grid.empty:
                summary[market] = {'status': 'PASS', 'reason': 'Insufficient calibration bets at any cutoff'}
                continue
            chosen = eligible_grid.sort_values(['pnl_units', 'n'], ascending=[False, False]).iloc[0]
            sd = op.policy_sd_cutoff(chosen)
            importance_cutoff = None if pd.isna(chosen.importance_cutoff) else float(chosen.importance_cutoff)
            diff = float(chosen.diff_cutoff)
            stats = op.score(validation, diff, sd, importance_cutoff=importance_cutoff)
            interval = op.roi_interval(validation, diff, sd, importance_cutoff=importance_cutoff)
            supported = chosen.pnl_units > 0 and stats['n'] >= min_bets and interval[0] is not None and interval[0] > 0
            summary[market] = dict(status='PAPER QUALIFIED' if supported else 'PASS', lookback=lookback,
                                   train_window=train_window, calculation=calculation, use_scaling=use_scaling,
                                   inputs=inputs, metrics=chosen_metrics, n_metrics=len(chosen_metrics),
                                   diff_cutoff=diff, sd_cutoff=sd, importance_cutoff=importance_cutoff,
                                   calibration=op.score(calibration, diff, sd, importance_cutoff=importance_cutoff),
                                   validation=stats, validation_roi_95=interval,
                                   validation_unfiltered=op.score(validation))
            settled = op.settle(data_market, diff, sd, importance_cutoff=importance_cutoff)
            settled.to_csv(output / f'shared_picks_{market}.csv', index=False)
            dashboard.note(f"  SHARED {market.upper()}: {summary[market]['status']} -- {stats['n']} validation bets, "
                          f"{stats['pnl_units']:+.2f} units; ROI 95% CI {interval}")
        (output / 'shared_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        dashboard.note(f"\nShared track complete -> {output / 'shared_summary.json'}")
        return summary
    finally:
        dashboard.close()


def dc3_metrics_fallback():
    import data_crunchski_3 as dc3
    return dc3.METRICS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=2025)
    ap.add_argument('--week', type=int, default=22)
    ap.add_argument('--start-season', type=int, default=2010,
                    help='First research season, beginning week 1 (default: 2010)')
    ap.add_argument('--output', default=None,
                    help='Where result CSVs/JSON land (default: auto-named under data/bt/ by track, '
                         'research window, and a run timestamp -- see main() for the exact scheme). Set '
                         'this explicitly to resume a specific run, e.g. with --phase2-only.')
    ap.add_argument('--phase0-only', action='store_true', help='Stop after materializing panels')
    ap.add_argument('--phase1-only', action='store_true', help='Stop after the cheap CV screen')
    ap.add_argument('--phase2-only', action='store_true',
                    help="Skip straight to Phase 2 (the expensive real walk-forward) using --output's "
                         'existing phase1_summary.json -- Phase 0/1 are cache-hit-cheap to redo, but '
                         "Phase 2 isn't, and there's no reason to touch it again if you already have a "
                         'Phase 1 winner picked out.')
    ap.add_argument('--min-bets', type=int, default=40, help='Phase 2: minimum calibration bets for a cutoff to be eligible')
    ap.add_argument('--shared', action='store_true',
                    help='Also run the shared-model track (the real symmetric-weight architecture, '
                         'shared_scoring.py) -- a real TensorFlow ensemble refit every week for every '
                         '(lookback x train-window) combo, much heavier than the Ridge track. Runs '
                         'after Phase 0/1/2 by default; pair with --shared-only to run just this.')
    ap.add_argument('--shared-only', action='store_true',
                    help='Run only the shared-model track, skipping Phase 0/1/2 entirely (implies --shared)')
    # No --shared- prefix below: the Ridge track exposes no --lookbacks/
    # --train-windows/--jobs/--screen-iterations/--confirm-iterations of its
    # own to collide with, so the prefix was pure noise, not disambiguation.
    ap.add_argument('--lookbacks', type=int, nargs='+', default=None,
                    help='Comp-stat lookbacks to sweep for the shared track (default: 20 30 50 -- a '
                         "deliberately smaller default than the Ridge track's full LOOKBACK_WINDOWS, "
                         'since each value here costs a full real walk-forward, not a cheap CV fold)')
    ap.add_argument('--train-windows', type=int, nargs='+', default=None,
                    help=f'Weeks of recent history the neural model itself trains on (default: {SHARED_TRAIN_WINDOWS})')
    ap.add_argument('--calculations', nargs='+', choices=['weighted', 'steep', 'mean', 'median'], default=None,
                    help='calc_stats cross-game aggregation modes to sweep: weighted (recency-weighted '
                         'average across games), steep (same idea, much faster decay/lower floor -- '
                         'recent games dominate far more), mean (plain average), median (per-game '
                         'median) (default: all four)')
    ap.add_argument('--use-scaling-options', nargs='+', type=int, choices=[0, 1], default=None,
                    help="comp_stats' pass/run usage-scaling multiplier, as 0/1 (default: both 1 and 0)")
    ap.add_argument('--input-modes', nargs='+', choices=['separate', 'differential', 'percentile', 'zscore'], default=None,
                    help='Shared inputs: separate raw levels, differential raw differences, or '
                         'production percentile/z-score differences, both normalized per-week against '
                         'every team in that week league-wide, never against own history (default: all four)')
    ap.add_argument('--screen-iterations', type=int, default=SHARED_SCREEN_ITERATIONS,
                    help='Ensemble size while screening configs/metric subsets (default: %(default)s -- '
                         'production uses 100; this trades ensemble stability for being able to afford '
                         'the sweep at all)')
    ap.add_argument('--confirm-iterations', type=int, default=SHARED_CONFIRM_ITERATIONS,
                    help='Ensemble size for the one final confirmed config per market (default: %(default)s)')
    ap.add_argument('--jobs', type=int, default=None,
                    help="CPU workers for the shared model's ensemble fits (default: min(iterations, 8) via "
                         'shared_scoring.fit_panel/NFL_MODEL_JOBS -- raise this toward your core count to '
                         'cut wall-clock time roughly proportionally; each worker is pinned to 1 internal '
                         'thread, so this scales cleanly up to your real core count)')
    ap.add_argument('--recycle-every', type=int, default=_RECYCLE_EVERY,
                    help='Weeks between loky worker-pool recycles, to bound TensorFlow\'s native memory '
                         'growth across many fits in the same persistent worker process (default: '
                         '%(default)s). Lower this if memory keeps climbing faster than you want; 0 disables '
                         'recycling entirely.')
    args = ap.parse_args()

    if args.shared_only:
        args.shared = True
    if args.phase2_only and not args.output:
        ap.error('--phase2-only requires --output pointing at the run directory with an existing phase1_summary.json')
    # Auto-name by track (what kind of run this is) and research window (what
    # it covers), not just a bare timestamp -- makes old runs identifiable
    # from the directory name alone. --output still overrides this outright,
    # which --phase2-only requires (it resumes a specific prior run).
    track = 'shared_sweep' if args.shared_only else ('optimus_full' if args.shared else 'optimus_ridge')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = Path(args.output or f'data/bt/{track}/{args.start_season}-{args.season}wk{args.week:02d}_{stamp}')
    output.mkdir(parents=True, exist_ok=True)

    try:
        if not args.shared_only:
            if args.phase2_only:
                summary_path = output / 'phase1_summary.json'
                if not summary_path.exists():
                    raise ValueError(f'{summary_path} not found -- run Phase 1 first (or without --phase2-only)')
                phase1_summary = json.loads(summary_path.read_text())
                history_weeks = history_for_start(args.start_season, args.season, args.week)
                run_phase2(phase1_summary, args.season, args.week, history_weeks, args.start_season, output,
                          min_bets=args.min_bets)
            else:
                manifest = generate_data_grid(args.season, args.week, args.start_season, output)
                if args.phase0_only:
                    return
                if (manifest.status == 'ok').sum() == 0:
                    raise ValueError('No Phase 0 panel succeeded; nothing for Phase 1 to build on')
                history_weeks = history_for_start(args.start_season, args.season, args.week)
                phase1_summary = run_phase1(manifest, args.season, args.week, history_weeks, args.start_season, output)
                if not args.phase1_only:
                    run_phase2(phase1_summary, args.season, args.week, history_weeks, args.start_season, output,
                              min_bets=args.min_bets)
        if args.shared:
            use_scaling_options = [bool(v) for v in args.use_scaling_options] if args.use_scaling_options else None
            run_shared_track(args.season, args.week, args.start_season, output,
                             lookbacks=args.lookbacks, train_windows=args.train_windows,
                             calculations=args.calculations, use_scaling_options=use_scaling_options,
                             input_modes=args.input_modes, screen_iterations=args.screen_iterations,
                             confirm_iterations=args.confirm_iterations, min_bets=args.min_bets,
                             jobs=args.jobs, recycle_every=args.recycle_every)
    except ValueError as error:
        ap.error(str(error))


if __name__ == '__main__':
    main()
