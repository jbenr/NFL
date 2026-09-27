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


def report(left, right, names, thresholds=(0., 1., 2., 3., 4., 5., 6., 7.)):
    shared = set(left.game_id) & set(right.game_id)
    left = left[left.game_id.isin(shared)].sort_values('game_id').reset_index(drop=True)
    right = right[right.game_id.isin(shared)].sort_values('game_id').reset_index(drop=True)
    print(f'{len(shared)} games scored by both arms, '
          f'{left.season.min():.0f}-{left.season.max():.0f} weeks '
          f'{left.week.min():.0f}-{left.week.max():.0f}\n')

    print('ACCURACY')
    print(f"{'':>12} {'spread MAE':>11} {'total MAE':>10} {'spread IC':>10} {'total IC':>9}")
    for frame, name in ((left, names[0]), (right, names[1])):
        smae = float((frame.prediction - frame.margin).abs().mean())
        tmae = float((frame.total_prediction - frame.points).abs().mean())
        sic = float(frame.prediction.corr(frame.margin))
        tic = float(frame.total_prediction.corr(frame.points))
        print(f'{name:>12} {smae:>11.3f} {tmae:>10.3f} {sic:>10.3f} {tic:>9.3f}')

    for market, edge, outcome in (('SPREAD', 'spread_edge', 'away_covered'),
                                  ('TOTAL', 'total_edge', 'total_over')):
        if edge not in left.columns:
            continue
        print(f'\n{market} PICKS BY EDGE')
        print(f"{'edge >=':>8} " + ' '.join(f'{n:>16}' for n in names) + '   delta')
        for threshold in thresholds:
            a, na = hit_rate(left[edge].to_numpy(), left[outcome].to_numpy(), threshold)
            b, nb = hit_rate(right[edge].to_numpy(), right[outcome].to_numpy(), threshold)
            if not (na or nb):
                continue
            gap = f'{(b - a) * 100:+.1f}' if np.isfinite(a) and np.isfinite(b) else ''
            cells = [f'{rate:>9.1%} ({count:>4})' if np.isfinite(rate) else f'{"-":>16}'
                     for rate, count in ((a, na), (b, nb))]
            print(f'{threshold:>8.0f} ' + ' '.join(cells) + f'   {gap:>6}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--names', nargs=2, default=['left', 'right'])
    args = parser.parse_args()
    report(graded(load(args.left)), graded(load(args.right)), args.names)


if __name__ == '__main__':
    main()
