#!/usr/bin/env python3
"""Where does a model actually predict well, and is that region real?

The pick search asks "which slice has the best hit rate", which is a
binary outcome on a few hundred games and therefore mostly noise -- the
shuffled-outcome demo produced 67% rules from thin air. This asks a
better-powered question of the same grid: where is the model's ERROR
lowest? MAE averages a few hundred real numbers instead of a few hundred
coin flips, which is why the MAE ordering across eight backtest arms was
stable while their hit-rate ordering inverted.

Then it applies the test that actually separates signal from search
noise: a genuine region is a PLATEAU. If a cell is good because the model
is good there, its neighbours -- one notch up or down on each threshold
-- are good too, because nothing about football changes when an edge cut
moves from 4.0 to 4.5. If a cell is good because 300,000 draws produced
one lucky sample, its neighbours are ordinary. So every cell is scored by
its own error AND its neighbourhood's, and only cells sitting on a broad
low-error shelf are reported.

Thresholds are nested (edge >= e, SD <= s, leverage >= l, week >= w) so
that "adjacent" is well defined and a neighbour really is the same rule
loosened slightly.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# Ordered ladders: position in the list is what makes two cells adjacent.
EDGE = [0., 1., 2., 3., 4., 5., 6., 7., 8.]
SD_PCT = [.2, .33, .5, .67, .8, 1.0]
LEVERAGE = [0., .2, .4, .55, .7, .85]
WEEK = [1, 3, 5, 7, 9, 11, 13]
# How far the posted total sits from the middle of the market's range. The
# installed totals buckets are all built on "outside 42-46", so a grid
# without this axis cannot see the dimension those rules live in. Measured
# as distance from 44, which makes it an ordered ladder like the others.
LINE_DISTANCE = [0., 1., 2., 3., 4., 6.]
AXES = ['edge', 'sd', 'leverage', 'week']


def graded(folder, market='spread'):
    """Every scored game with the sliceable quantities attached.

    `covered` is signed so that sign(edge) == sign(covered) means the pick
    landed, in both markets: the away side on a spread, the over on a
    total. Spread predictions are away-minus-home while the posted line is
    home-relative, which is why one adds and the other subtracts."""
    files = sorted(Path(folder).glob('*wk*.parquet'))
    if not files:
        raise SystemExit(f'no scored weeks in {folder}')
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if market == 'total':
        need = ['total_prediction', 'points', 'total_line']
        if any(c not in d.columns for c in need):
            raise SystemExit(f'{folder} has no totals -- sided-spread forecasts none')
        d = d.dropna(subset=need).copy()
        d['abs_error'] = (d.total_prediction - d.points).abs()
        d['edge'] = d.total_prediction - d.total_line
        d['covered'] = d.points - d.total_line
        d['sd'] = np.sqrt(d.total_variance.clip(lower=0)) if 'total_variance' in d else np.nan
    else:
        d = d.dropna(subset=['prediction', 'margin', 'spread_line']).copy()
        d['abs_error'] = (d.prediction - d.margin).abs()
        d['edge'] = d.prediction + d.spread_line
        d['covered'] = d.margin + d.spread_line
        d['sd'] = np.sqrt(d.variance.clip(lower=0))
    d['abs_edge'] = d.edge.abs()
    d['leverage'] = d.total_game_importance
    # Only meaningful for totals; on a spread the posted number is a
    # team-strength statement rather than a scoring one, so the axis is
    # collapsed to a single always-true option below.
    d['line_distance'] = ((d.total_line - 44.).abs() if market == 'total'
                          and 'total_line' in d.columns else 0.)
    return d[d.covered != 0].reset_index(drop=True)


def week_ladder(data):
    """Opening weeks to try, inside whatever span the data covers.

    The axis is `week >= w`, so restricting the data to weeks 1-10 first
    makes every cell a bounded window -- otherwise every rule runs to the
    playoffs and early-season football can never be isolated."""
    lo, hi = int(data.week.min()), int(data.week.max())
    return [w for w in range(lo, hi + 1, max(1, (hi - lo) // 6 or 1))] or [lo]


def cells(data, min_n, weeks=None, lines=None):
    """Every nested threshold combination, with its error and its record."""
    weeks = weeks or week_ladder(data)
    lines = lines if lines is not None else (
        LINE_DISTANCE if data.line_distance.abs().max() > 0 else [0.])
    sd_cuts = [data.sd.quantile(q) for q in SD_PCT]
    out = {}
    for i, e in enumerate(EDGE):
        by_edge = data.abs_edge >= e
        for j, s in enumerate(sd_cuts):
            by_sd = by_edge & (data.sd <= s)
            for k, l in enumerate(LEVERAGE):
                by_lev = by_sd & (data.leverage >= l)
                for m, w in enumerate(weeks):
                    by_week = by_lev & (data.week >= w)
                    for q, dist in enumerate(lines):
                        sub = data[by_week & (data.line_distance >= dist)]
                        if len(sub) < min_n:
                            continue
                        won = np.sign(sub.edge) == np.sign(sub.covered)
                        out[(i, j, k, m, q)] = dict(
                            n=len(sub), mae=float(sub.abs_error.mean()),
                            rate=float(won.mean()), edge=e, sd_pct=SD_PCT[j],
                            sd=float(s), leverage=l, week=w, line_distance=dist)
    return out


def plateau(out, radius=1):
    """Each cell's error alongside its neighbourhood's.

    `own` is the cell; `shelf` averages it with every cell within one
    notch on each axis. A spike has a good `own` and an ordinary `shelf`;
    a real region has both. `neighbours` says how much of the
    neighbourhood actually existed -- a cell on the edge of the grid or
    surrounded by under-volume cells has little support and is reported
    with that caveat rather than silently flattered."""
    scored = []
    for key, cell in out.items():
        near = []
        width = len(next(iter(out)))
        for shift in itertools.product(*[range(-radius, radius + 1)] * width):
            if not any(shift):
                continue
            neighbour = out.get(tuple(a + b for a, b in zip(key, shift)))
            if neighbour:
                near.append(neighbour['mae'])
        if not near:
            continue
        scored.append(dict(cell, key=key, shelf=float(np.mean([cell['mae'], *near])),
                           neighbours=len(near), spread=float(np.std(near))))
    return sorted(scored, key=lambda c: c['shelf'])


def describe(cell):
    bits = [f"edge>={cell['edge']:g}", f"SD<=p{int(cell['sd_pct'] * 100)} ({cell['sd']:.2f})",
            f"leverage>={cell['leverage']:g}", f"wk{cell['week']}+"]
    if cell.get('line_distance'):
        bits.append(f"total {cell['line_distance']:g}+ from 44")
    return ' · '.join(bits)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder')
    p.add_argument('--min-n', type=int, default=150)
    p.add_argument('--top', type=int, default=12)
    p.add_argument('--min-neighbours', type=int, default=20,
                   help='how much of the neighbourhood must exist for a cell to be trusted')
    p.add_argument('--market', choices=['spread', 'total'], default='spread')
    p.add_argument('--weeks', default=None, metavar='LO-HI',
                   help='restrict to a span first, e.g. 1-10, so the week axis gives '
                        'bounded windows instead of rules that all run to the playoffs')
    args = p.parse_args(argv)

    data = graded(args.folder, args.market)
    if args.weeks:
        lo, hi = (int(x) for x in args.weeks.split('-'))
        data = data[(data.week >= lo) & (data.week <= hi)].reset_index(drop=True)
    grid = cells(data, args.min_n)
    ranked = [c for c in plateau(grid) if c['neighbours'] >= args.min_neighbours]
    whole = float(data.abs_error.mean())
    base = float((np.sign(data.edge) == np.sign(data.covered)).mean())
    span = f' · weeks {args.weeks}' if args.weeks else ''
    print(f'{args.market} · {len(data)} graded games{span} · whole-sample MAE {whole:.3f} · hit rate {base:.1%}')
    print(f'{len(grid):,} cells at n>={args.min_n}; {len(ranked):,} with a full enough neighbourhood\n')
    print('LOWEST-ERROR SHELVES  (ranked by the neighbourhood, not the cell)')
    print(f"{'shelf MAE':>10} {'own MAE':>9} {'vs all':>8} {'hit':>7} {'n':>6} {'nbrs':>5}  rule")
    print('-' * 104)
    for cell in ranked[:args.top]:
        print(f"{cell['shelf']:>10.3f} {cell['mae']:>9.3f} {cell['mae'] - whole:>+8.3f} "
              f"{cell['rate']:>7.1%} {cell['n']:>6} {cell['neighbours']:>5}  {describe(cell)}")
    return ranked


if __name__ == '__main__':
    main()
