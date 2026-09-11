"""Sequential methodology audit: which of your own design choices actually help?

Every one of these was a real call made while building the production pipeline,
never individually tested against an alternative:

  1. Recency weighting  -- data_crunchski_2's cubic-decay taper (calculation=
     'weighted') vs. no decay (calculation='legacy').
  2. Stat representation -- percentile-rank differencing (comp_stats' rank_it/
     rev_rank_it, what the production model trains on) vs. raw values, z-scored
     (what the newer joint/shared-scoring models train on). Built once here so
     this can be tested with the fast Ridge screen instead of needing the full
     neural ensemble to answer a representation question.
  3. Feature selection -- paired drop-test (reusing optimize_picks.feature_scan),
     on top of whichever recency+representation combo won stages 1-2.
  4. Cutoff selection, INCLUDING game importance as a filter -- diff_cutoff and
     sd_cutoff already existed in optimize_picks.py; this adds an
     importance_cutoff dimension (minimum |away_game_importance| to bet) as a
     genuine third axis, not just eyeballed.

Each stage is DECIDED on calibration only (seasons before --validation-season)
and holds every earlier stage's winner fixed -- this is staged/greedy, not a
joint search over every combination, for the same reason optimize_picks.py's
own docstring gives: exhaustive search over this many axes would overfit the
calibration window far worse than a staged heuristic that gets to explain each
choice. The validation season is touched exactly once, at the very end, per
market -- if the calibration and validation numbers disagree, the validation
number is the true one.

Spread and totals are audited independently and can land on different answers
at every stage -- they are different statistical questions (who wins by how
much, vs. how much scoring environment exists), and optimize_picks.py's own
feature_names() already reflects that split (differential features for spread,
level/sum features for total).

A fast bagged Ridge stands in for the real 100-model neural ensemble at every
stage, for the same reason optimize_picks.py uses one: refitting per week per
grid point needs to run in seconds, not hours. Nothing here is a claim about
the production ensemble's behavior -- it's a screen for which choices are
worth confirming against the real thing.

Usage:
    python optimus_prime.py --markets spread total
    python optimus_prime.py --markets spread --validation-season 2025 --history-weeks 150

Output: data/optimus_prime/<timestamp>/summary.json, stage CSVs, report.html.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimize_picks as op
from edge_scan import PRODUCTION_FEATURES

RECENCY_MODES = ['legacy', 'weighted']
REPRESENTATIONS = ['percentile', 'raw']
FEATURE_SUBSET_SIZES = [8, 16, None]  # None = every feature that survived discovery
DIFF_CUTOFFS = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
SD_QUANTILES = [1, .75, .5, .25]
IMPORTANCE_QUANTILES = [1, .75, .5, .25]  # 1 = no importance filter, tested explicitly
MIN_CALIBRATION_BETS = 40


def raw_diff_features(panel: pd.DataFrame) -> pd.DataFrame:
    """The joint/shared-scoring representation (raw team rates, not percentile
    ranks), reframed into PRODUCTION_FEATURES' own away_off_X/away_def_X naming
    and sign convention so swapping representations is a single-axis change --
    it does NOT also swap sign convention or add the pass_%/run_% usage
    multiplier production's percentile version applies; those are separate
    methodology questions, not part of "does raw vs. percentile help."
    """
    result = panel.copy()
    metrics = [f[len('away_off_'):] for f in PRODUCTION_FEATURES if f.startswith('away_off_')]
    built = []
    for metric in metrics:
        off, def_ = f'away_raw_off_{metric}', f'home_raw_def_{metric}'
        if off in result and def_ in result:
            result[f'raw_away_off_{metric}'] = result[off] - result[def_]
            built.append(f'raw_away_off_{metric}')
        off2, def2 = f'away_raw_def_{metric}', f'home_raw_off_{metric}'
        if off2 in result and def2 in result:
            result[f'raw_away_def_{metric}'] = result[off2] - result[def2]
            built.append(f'raw_away_def_{metric}')
    result.attrs['raw_features'] = built
    return result


def representation_features(panel: pd.DataFrame, representation: str) -> tuple[pd.DataFrame, list[str]]:
    context = [f for f in ['away_rest_adv', 'home_field_adv'] if f in panel]
    if representation == 'percentile':
        return panel, [f for f in PRODUCTION_FEATURES if f in panel]
    panel = raw_diff_features(panel)
    return panel, panel.attrs['raw_features'] + context


def split_calibration(data: pd.DataFrame, validation_season: int):
    calibration = data[data.season < validation_season]
    validation = data[data.season == validation_season]
    return calibration, validation


def stage_recency(season, week, history_weeks, market, validation_season, min_train_weeks):
    print(f'\n[{market}] Stage 1: recency weighting', flush=True)
    rows = []
    for calculation in RECENCY_MODES:
        panel = op.build_panel(season, week, history_weeks, history_weeks, calculation)
        data = op.market_panel(panel, market)
        features = [f for f in PRODUCTION_FEATURES if f in data] if market == 'spread' else \
            op.feature_names(data, market)
        preds = op.weekly_predict(data, features, min_train_weeks, bags=15)
        calibration, _ = split_calibration(preds, validation_season)
        score = op.score(calibration)
        rows.append(dict(calculation=calculation, **score))
        print(f'  {calculation:10s} calibration n={score["n"]:4d} '
              f'win_rate={score["win_rate"]} pnl={score["pnl_units"]}', flush=True)
    grid = pd.DataFrame(rows)
    winner = grid.sort_values('pnl_units', ascending=False).iloc[0]['calculation']
    print(f'  -> winner: {winner}', flush=True)
    return winner, grid


def stage_representation(season, week, history_weeks, calculation, market, validation_season, min_train_weeks):
    print(f'\n[{market}] Stage 2: stat representation (recency={calculation})', flush=True)
    panel = op.build_panel(season, week, history_weeks, history_weeks, calculation)
    data = op.market_panel(panel, market)
    rows, cache = [], {}
    for representation in (REPRESENTATIONS if market == 'spread' else ['percentile']):
        # Totals already use a level/sum framing (feature_names' total_* columns),
        # which has no percentile-vs-raw analogue -- the representation question
        # is specific to differential (spread) features.
        rep_data, features = representation_features(data, representation)
        preds = op.weekly_predict(rep_data, features, min_train_weeks, bags=15)
        calibration, _ = split_calibration(preds, validation_season)
        score = op.score(calibration)
        rows.append(dict(representation=representation, n_features=len(features), **score))
        cache[representation] = (rep_data, features)
        print(f'  {representation:10s} ({len(features)} features) calibration n={score["n"]:4d} '
              f'win_rate={score["win_rate"]} pnl={score["pnl_units"]}', flush=True)
    grid = pd.DataFrame(rows)
    winner = grid.sort_values('pnl_units', ascending=False).iloc[0]['representation']
    print(f'  -> winner: {winner}', flush=True)
    return winner, cache[winner][0], cache[winner][1], grid


def stage_feature_selection(data, features, market, validation_season, min_train_weeks):
    print(f'\n[{market}] Stage 3: feature selection ({len(features)} candidates)', flush=True)
    calibration, _ = split_calibration(data, validation_season)
    weeks = calibration.week_id.unique()
    discovery_end = sorted(weeks)[len(weeks) // 2]  # first half of calibration: discovery only
    scan = op.feature_scan(calibration, features, min_train_weeks, discovery_end)
    ranked = scan.loc[scan.mse_contribution > 0, 'feature'].tolist()
    context = [f for f in ['away_rest_adv', 'home_field_adv'] if f in features]
    subsets = {'all': features}
    for k in FEATURE_SUBSET_SIZES:
        if k is None:
            subsets[f'top_{len(ranked)}_survivors'] = list(dict.fromkeys(context + ranked)) or features
        elif k < len(ranked):
            subsets[f'top_{k}'] = list(dict.fromkeys(context + ranked[:k]))
    rows = []
    for name, subset in subsets.items():
        preds = op.weekly_predict(calibration[calibration.week_id >= discovery_end], subset, min_train_weeks, bags=15)
        score = op.score(preds)
        rows.append(dict(subset=name, n_features=len(subset), **score))
        print(f'  {name:24s} ({len(subset):2d} features) n={score["n"]:4d} '
              f'win_rate={score["win_rate"]} pnl={score["pnl_units"]}', flush=True)
    grid = pd.DataFrame(rows)
    winner = grid.sort_values('pnl_units', ascending=False).iloc[0]['subset']
    print(f'  -> winner: {winner} ({subsets[winner]})', flush=True)
    return subsets[winner], scan, grid


def stage_cutoffs(preds_calibration, market):
    """diff_cutoff x sd_quantile x importance_quantile. importance_quantile=1
    means no importance filter -- a real grid point, not assumed away."""
    print(f'\n[{market}] Stage 4: cutoffs, including game importance as a filter', flush=True)
    has_importance = 'away_game_importance' in preds_calibration and preds_calibration.away_game_importance.notna().any()
    rows = []
    for diff in DIFF_CUTOFFS:
        for sdq in SD_QUANTILES:
            sd = None if sdq == 1 else float(np.sqrt(preds_calibration.variance.quantile(sdq)))
            for iq in (IMPORTANCE_QUANTILES if has_importance else [1]):
                frame = preds_calibration
                if iq < 1:
                    cut = frame.away_game_importance.abs().quantile(iq)
                    frame = frame[frame.away_game_importance.abs() >= cut]
                score = op.score(frame, diff, sd)
                rows.append(dict(diff_cutoff=diff, sd_quantile=sdq, sd_cutoff=sd,
                                 importance_quantile=iq, eligible=score['n'] >= MIN_CALIBRATION_BETS,
                                 **score))
    grid = pd.DataFrame(rows)
    eligible = grid[grid.eligible]
    if eligible.empty:
        print('  no cutoff combination reached the minimum bet count', flush=True)
        return dict(diff_cutoff=0, sd_cutoff=None, importance_quantile=1), grid
    best = eligible.sort_values(['pnl_units', 'n'], ascending=[False, False]).iloc[0]
    no_importance_best = eligible[eligible.importance_quantile == 1].sort_values('pnl_units', ascending=False)
    baseline_pnl = no_importance_best.iloc[0]['pnl_units'] if not no_importance_best.empty else float('nan')
    print(f'  best without importance filter: pnl={baseline_pnl:+.2f}', flush=True)
    print(f'  best overall (importance_quantile={best.importance_quantile}): '
          f'diff>={best.diff_cutoff}, sd<={best.sd_cutoff}, pnl={best.pnl_units:+.2f}', flush=True)
    if best.importance_quantile < 1 and best.pnl_units > baseline_pnl:
        print(f'  -> importance filter helps on calibration (+{best.pnl_units - baseline_pnl:.2f} units)', flush=True)
    else:
        print('  -> importance filter does not beat diff/SD cutoffs alone on calibration', flush=True)
    return dict(diff_cutoff=float(best.diff_cutoff), sd_cutoff=None if pd.isna(best.sd_cutoff) else float(best.sd_cutoff),
               importance_quantile=float(best.importance_quantile)), grid


def audit_market(market, season, week, history_weeks, validation_season, min_train_weeks, output):
    winner_calc, recency_grid = stage_recency(season, week, history_weeks, market, validation_season, min_train_weeks)
    winner_rep, data, features, rep_grid = stage_representation(
        season, week, history_weeks, winner_calc, market, validation_season, min_train_weeks)
    chosen_features, scan, subset_grid = stage_feature_selection(
        data, features, market, validation_season, min_train_weeks)

    print(f'\n[{market}] Stage 4 prep: refitting chosen config on full calibration window', flush=True)
    calibration, validation = split_calibration(data, validation_season)
    preds_calibration = op.weekly_predict(calibration, chosen_features, min_train_weeks, bags=15)
    cutoffs, cutoff_grid_df = stage_cutoffs(preds_calibration, market)

    print(f'\n[{market}] FINAL: applying the fully-chosen configuration to the untouched '
          f'{validation_season} validation season...', flush=True)
    preds_validation = op.weekly_predict(pd.concat([calibration, validation]), chosen_features, min_train_weeks, bags=15)
    preds_validation = preds_validation[preds_validation.season == validation_season]
    val_frame = preds_validation
    if cutoffs['importance_quantile'] < 1 and 'away_game_importance' in val_frame:
        cut = preds_calibration.away_game_importance.abs().quantile(cutoffs['importance_quantile'])
        val_frame = val_frame[val_frame.away_game_importance.abs() >= cut]
    validation_score = op.score(val_frame, cutoffs['diff_cutoff'], cutoffs['sd_cutoff'])
    interval = op.roi_interval(val_frame, cutoffs['diff_cutoff'], cutoffs['sd_cutoff'])
    print(f'  VALIDATION ({validation_season}): n={validation_score["n"]} '
          f'win_rate={validation_score["win_rate"]} pnl={validation_score["pnl_units"]:+.2f} '
          f'roi_95={interval}', flush=True)

    recency_grid.to_csv(output / f'{market}_stage1_recency.csv', index=False)
    rep_grid.to_csv(output / f'{market}_stage2_representation.csv', index=False)
    subset_grid.to_csv(output / f'{market}_stage3_features.csv', index=False)
    scan.to_csv(output / f'{market}_stage3_feature_scan.csv', index=False)
    cutoff_grid_df.to_csv(output / f'{market}_stage4_cutoffs.csv', index=False)

    return dict(
        market=market, recency=winner_calc, representation=winner_rep,
        features=chosen_features, cutoffs=cutoffs,
        calibration=op.score(preds_calibration, cutoffs['diff_cutoff'], cutoffs['sd_cutoff']),
        validation=validation_score, validation_roi_95=interval,
        validation_season=validation_season,
        supported=bool(validation_score['n'] >= MIN_CALIBRATION_BETS and validation_score['pnl_units'] > 0
                      and interval[0] is not None and interval[0] > 0),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=2025)
    ap.add_argument('--week', type=int, default=22)
    ap.add_argument('--history-weeks', type=int, default=150)
    ap.add_argument('--validation-season', type=int, default=2025)
    ap.add_argument('--min-train-weeks', type=int, default=40)
    ap.add_argument('--markets', nargs='+', choices=['spread', 'total'], default=['spread', 'total'])
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    output = Path(args.output or f'data/optimus_prime/{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}')
    output.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for market in args.markets:
        summaries[market] = audit_market(market, args.season, args.week, args.history_weeks,
                                         args.validation_season, args.min_train_weeks, output)

    (output / 'summary.json').write_text(json.dumps(summaries, indent=2, default=str))
    print(f'\nSaved -> {output}/summary.json (+ per-market stage CSVs)')


if __name__ == '__main__':
    main()
