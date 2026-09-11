"""Chronological, cached group tests for the shared scoring model; no live betting."""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_picks as op
import shared_scoring as ss
import utils


def evaluate(predictions, validation_season, min_bets):
    calibration = predictions[predictions.season < validation_season]
    validation = predictions[predictions.season >= validation_season]
    grid = op.cutoff_grid(calibration, min_bets)
    eligible = grid[grid.eligible].sort_values(['pnl_units', 'n'], ascending=False)
    error = validation.prediction - validation.actual
    result = dict(games=len(validation), mae=float(error.abs().mean()), mse=float((error**2).mean()),
                  status='EXPLORATORY')
    if eligible.empty:
        return dict(result, reason='Insufficient calibration bets'), grid
    rule = eligible.iloc[0]
    edge, sd = float(rule.diff_cutoff), op.policy_sd_cutoff(rule)
    return dict(result, diff_cutoff=edge, sd_cutoff=sd,
                calibration=op.score(calibration, edge, sd),
                validation=op.score(validation, edge, sd),
                roi_95=op.roi_interval(validation, edge, sd)), grid


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
    os.environ.setdefault('NFL_WORKERS', '1')
    panel = op.build_panel(args.season, args.week, history_weeks(args), 20, 'legacy')
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
            data = pd.concat([history[history.week_id >= regular[-20]], test]).sort_values(op.KEY)
            print(f'{name}: {season} wk{week}', flush=True)
            if joint:
                target, details, _ = js.fit_panel(data, int(season), int(week), args.iterations,
                                                100, args.seed, args.jobs or 8, groups)
            else:
                target, details, _ = ss.fit_panel(data, int(season), int(week), args.iterations,
                                                 100, args.seed, args.jobs, groups, args.inputs)
            for market in rows:
                forecast = op.market_panel(target, market).merge(
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
                paired = predictions.merge(baseline[op.KEY + ['prediction']], on=op.KEY,
                                           suffixes=('', '_baseline'), validate='one_to_one')
                paired = paired[paired.season >= args.validation_season]
                delta = ((paired.prediction_baseline - paired.actual)**2 -
                         (paired.prediction - paired.actual)**2)
                result['mse_improvement_vs_baseline'] = float(delta.mean())
                base_rule = report['baseline'][market]
                if 'diff_cutoff' in base_rule:
                    result['validation_at_baseline_rule'] = op.score(
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['shared', 'joint'], default='shared')
    parser.add_argument('--season', type=int, default=2025)
    parser.add_argument('--week', type=int, default=22)
    parser.add_argument('--start-season', type=int, default=2024)
    parser.add_argument('--validation-season', type=int, default=2025)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--jobs', type=int, default=8, help='Training workers (default: 8)')
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
