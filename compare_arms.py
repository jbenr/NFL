"""Score two backtest runs against each other on the same games.

Built for the taper A/B -- 'weighted' (which floors last season at 0.05 by
September) against 'solved' (which prices it at 0.40) -- but it takes any
two run directories.

Only games BOTH arms scored are compared, so a difference is a difference
in the model and not in which weeks happened to finish.

Sign conventions, checked against a completed run rather than assumed:
  margin      = away_score - home_score
  prediction  = the model on that same scale
  spread_line = home-relative, i.e. the OPPOSITE sign
so the market's implied margin is -spread_line, the model's edge is
`prediction + spread_line`, and the away side covers when
`margin + spread_line > 0`.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load(folder):
    files = sorted(glob.glob(str(Path(folder) / '*.parquet')))
    if not files:
        raise SystemExit(f'no scored weeks in {folder}')
    keep = ['game_id', 'season', 'week', 'prediction', 'total_prediction',
            'variance', 'total_variance', 'margin', 'points',
            'spread_line', 'total_line']
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frames.append(frame[[c for c in keep if c in frame.columns]])
    return pd.concat(frames, ignore_index=True)


def graded(frame):
    """Add the market-relative quantities every metric below is built on."""
    out = frame.dropna(subset=['prediction', 'margin', 'spread_line']).copy()
    out['spread_edge'] = out.prediction + out.spread_line
    out['away_covered'] = out.margin + out.spread_line
    out['spread_sd'] = np.sqrt(out.variance) if 'variance' in out else np.nan
    if 'total_prediction' in out and 'total_line' in out:
        out['total_edge'] = out.total_prediction - out.total_line
        out['total_over'] = out.points - out.total_line
        out['total_sd'] = np.sqrt(out.total_variance) if 'total_variance' in out else np.nan
    return out


def hit_rate(edge, outcome, threshold):
    """Share of non-push picks that landed, at an absolute edge cut.

    Pushes are dropped rather than counted as half: a push returns the
    stake, so it is not a result, and folding it in moves every rate
    toward 50% by an amount that depends on how many there were."""
    taken = np.abs(edge) >= threshold
    live = taken & (outcome != 0)
    if not live.any():
        return np.nan, 0
    right = np.sign(edge[live]) == np.sign(outcome[live])
    return float(right.mean()), int(live.sum())


def common_games(runs):
    """Restrict every run to the games all of them scored.

    Two runs that stopped at different weeks would otherwise be compared
    on different football, and the gap would be the schedule rather than
    the model."""
    shared = set.intersection(*(set(frame.game_id) for frame in runs.values()))
    return {name: frame[frame.game_id.isin(shared)].sort_values('game_id').reset_index(drop=True)
            for name, frame in runs.items()}, shared


def accuracy(runs):
    rows = []
    for name, frame in runs.items():
        rows.append(dict(
            run=name,
            spread_mae=float((frame.prediction - frame.margin).abs().mean()),
            total_mae=float((frame.total_prediction - frame.points).abs().mean()),
            spread_ic=float(frame.prediction.corr(frame.margin)),
            total_ic=float(frame.total_prediction.corr(frame.points))))
    return pd.DataFrame(rows)


def market_table(runs, edge, outcome, thresholds):
    """Hit rate at each absolute-edge cut: [{threshold, {run: (rate, n)}}].

    Plain rows rather than a DataFrame -- run names become column labels
    here, and anything with a space in it stops being reachable once
    pandas turns the row into a namedtuple."""
    return [dict(threshold=threshold,
                 cells={name: hit_rate(frame[edge].to_numpy(), frame[outcome].to_numpy(), threshold)
                        for name, frame in runs.items()})
            for threshold in thresholds]


def report(runs, thresholds=(0., 1., 2., 3., 4., 5., 6., 7.)):
    """Score any number of runs against each other on their shared games."""
    runs, shared = common_games(runs)
    any_frame = next(iter(runs.values()))
    print(f'{len(shared)} games scored by all {len(runs)} runs, '
          f'{any_frame.season.min():.0f}-{any_frame.season.max():.0f} weeks '
          f'{any_frame.week.min():.0f}-{any_frame.week.max():.0f}\n')

    names = list(runs)
    width = max(len(n) for n in names) + 2
    print('ACCURACY')
    print(f"{'':>{width}} {'spread MAE':>11} {'total MAE':>10} {'spread IC':>10} {'total IC':>9}")
    for row in accuracy(runs).itertuples():
        print(f'{row.run:>{width}} {row.spread_mae:>11.3f} {row.total_mae:>10.3f} '
              f'{row.spread_ic:>10.3f} {row.total_ic:>9.3f}')

    for market, edge, outcome in (('SPREAD', 'spread_edge', 'away_covered'),
                                  ('TOTAL', 'total_edge', 'total_over')):
        if edge not in any_frame.columns:
            continue
        print(f'\n{market} PICKS BY EDGE   (hit rate, n)')
        print(f"{'edge >=':>8} " + ' '.join(f'{n:>16}' for n in names))
        table = market_table(runs, edge, outcome, thresholds)
        for row in table:
            cells = [f'{rate:>9.1%} ({count:>4})' if np.isfinite(rate) else f'{"-":>16}'
                     for rate, count in (row['cells'][name] for name in names)]
            print(f"{row['threshold']:>8.0f} " + ' '.join(cells))
        # Ranked on the cut the tier buckets actually use, and only where
        # there is enough volume for the number to mean anything.
        at_four = next((r for r in table if r['threshold'] == 4.), None)
        if at_four:
            ranked = sorted(((rate, count, name) for name, (rate, count) in at_four['cells'].items()
                             if np.isfinite(rate) and count >= 100), reverse=True)
            if ranked:
                print('  at edge>=4: ' + ', '.join(f'{n} {r:.1%} ({c})' for r, c, n in ranked))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+', help='one or more backtest run directories')
    parser.add_argument('--names', nargs='+', default=None,
                        help='a label per folder (defaults to the folder name)')
    args = parser.parse_args()
    names = args.names or [Path(f).name[:10] for f in args.folders]
    if len(names) != len(args.folders):
        raise SystemExit(f'{len(names)} names for {len(args.folders)} folders')
    report({name: graded(load(folder)) for name, folder in zip(names, args.folders)})


if __name__ == '__main__':
    main()
