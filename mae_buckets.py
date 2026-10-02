#!/usr/bin/env python3
"""Which model, in which part of the season, at what disagreement with the
market, what ensemble spread and what playoff leverage, misses by the
least -- and do those cells pick better?

Two deliberate choices.

RAW CUTOFFS, not percentiles. A percentile is computed against a
distribution, and the live packet sees one week of games. It is also a
moving target: this model's ensemble SD drifted from ~2.9 to ~2.6 across
the sample, so a fixed percentile is a different threshold in 2010 than in
2025 and the era counts come out lopsided. Absolute numbers mean the same
thing every week, which is what a rule has to do to be bettable. The cost
is that a cut of 4.0 selects differently on two-sided (SD ~4.2) than on
sided-spread (SD ~2.8) -- fine, because the model is part of the cell.

DISJOINT WEEK BUCKETS. 1-3, 4-10, 11-18 and the playoffs, so a cell cannot
borrow strength from a window another cell already covers and every pick
is counted once.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mae_landscape as ml

WEEK_BUCKETS = [('wk1-3', 1, 3), ('wk4-10', 4, 10), ('wk11-18', 11, 18), ('playoffs', 19, 22)]
EDGE = [0., 2., 3., 4., 5., 6., 8.]
SD = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 99.]
LEVERAGE = [0., .4, .55, .70, .85]
MIN_N = 100


def models(market):
    """Every finished full-season backtest, labelled by what defines it."""
    found = {}
    for folder in sorted(Path('data/bt').glob('model_*/2010-2025/*')) + \
                  sorted(Path('data/bt').glob('model_*_sided_spread/2010-2025/*')):
        if len(list(folder.glob('*wk*.parquet'))) < 341:
            continue
        cfg = json.loads((folder / 'config.json').read_text())
        arch = cfg.get('architecture') or 'two-sided'
        if market == 'total' and arch == 'sided-spread':
            continue          # sided-spread forecasts no total at all
        label = (f"{cfg['model_version'].replace('model_', '')}-{cfg.get('calculation', '?')[:9]}"
                 f"{'-ss' if arch == 'sided-spread' else ''}")
        found[label] = folder
    return found


def cells(market, min_n=MIN_N):
    rows = []
    for label, folder in models(market).items():
        d = ml.graded(folder, market)
        for name, lo, hi in WEEK_BUCKETS:
            window = d[(d.week >= lo) & (d.week <= hi)]
            if len(window) < min_n:
                continue
            for e in EDGE:
                for s in SD:
                    for lev in LEVERAGE:
                        sub = window[(window.abs_edge >= e) & (window.sd <= s)
                                     & (window.leverage >= lev)]
                        if len(sub) < min_n:
                            continue
                        won = np.sign(sub.edge) == np.sign(sub.covered)
                        rows.append(dict(
                            model=label, weeks=name, edge=e, sd=s, lev=lev, n=len(sub),
                            mae=float(sub.abs_error.mean()), hit=float(won.mean()),
                            units=float(won.sum() * (100 / 110) - (~won).sum()),
                            seasons=int(sub.season.nunique()),
                            recent=float((np.sign(sub[sub.season >= 2018].edge)
                                          == np.sign(sub[sub.season >= 2018].covered)).mean())
                            if (sub.season >= 2018).sum() >= 40 else np.nan))
    return pd.DataFrame(rows)


def report(market, top=12, min_n=MIN_N):
    table = cells(market, min_n)
    if table.empty:
        print(f'{market}: nothing clears n>={min_n}')
        return table
    print(f'=== {market.upper()} === {len(table):,} cells '
          f'({table.model.nunique()} models x 4 week buckets x raw edge/SD/leverage), n>={min_n}\n')
    print('DOES LOW MAE STILL MEAN BETTER PICKS, WITHIN EACH PART OF THE SEASON?')
    print(f"{'weeks':>10} {'cells':>7} {'corr':>7}   lowest-MAE third -> highest, mean hit rate")
    print('-' * 86)
    for name, _, _ in WEEK_BUCKETS:
        part = table[table.weeks == name]
        if len(part) < 30:
            continue
        thirds = pd.qcut(part.mae, 3, labels=False, duplicates='drop')
        rates = [f'{part.hit[thirds == t].mean():.1%}' for t in sorted(set(thirds))]
        print(f'{name:>10} {len(part):>7} {part.mae.corr(part.hit):>+7.3f}   ' + '  ->  '.join(rates))
    for name, _, _ in WEEK_BUCKETS:
        part = table[table.weeks == name]
        if part.empty:
            continue
        print(f'\n--- {name}: lowest-MAE cells ---')
        print(f"{'model':>17} {'edge':>5} {'SD':>5} {'lev':>5} {'MAE':>7} {'hit':>7} "
              f"{'recent':>7} {'n':>5} {'u/szn':>7}")
        for r in part.nsmallest(top, 'mae').itertuples():
            recent = f'{r.recent:.1%}' if np.isfinite(r.recent) else '  -  '
            print(f'{r.model:>17} {r.edge:>5.0f} {r.sd:>5.1f} {r.lev:>5.2f} {r.mae:>7.2f} '
                  f'{r.hit:>7.1%} {recent:>7} {r.n:>5} {r.units / r.seasons:>7.2f}')
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--market', choices=['spread', 'total', 'both'], default='both')
    p.add_argument('--top', type=int, default=12)
    p.add_argument('--min-n', type=int, default=MIN_N)
    p.add_argument('--out', default=None, help='write every cell to this CSV')
    args = p.parse_args(argv)
    frames = []
    for market in (['spread', 'total'] if args.market == 'both' else [args.market]):
        frames.append(report(market, args.top, args.min_n).assign(market=market))
        print()
    if args.out:
        pd.concat(frames).to_csv(args.out, index=False)
        print(f'all cells -> {args.out}')


if __name__ == '__main__':
    main()
