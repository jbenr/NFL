#!/usr/bin/env python3
"""One standardized, gated performance check for every betting rule we test.

The problem this solves: search enough combinations of week, edge, ensemble SD,
playoff leverage and line size and something always clears 60%. So this scores
the whole grid twice -- once on the real outcomes, once on coin-flip outcomes
that keep every rule's sample size -- and reports both. A search is only
interesting when it clears far more rules than chance clears on the same grid.

    python rule_search.py data/bt/model_2.0/2010-2025/9ad3386489380923c826
    python rule_search.py <folder> --market total --draws 500

A rule is a conjunction of five conditions and must pass all of GATE to count.
Quantile bands (SD, leverage) are taken inside each week window, so "tight SD"
means the same thing in September as in January.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GATE = dict(min_picks=150,     # ~10 a season over a 16-season backtest; thinner cannot be sized
            min_rate=.540,     # 1.6 points clear of the -110 break-even at 52.4%
            min_era=.500,      # no four-season stretch under water
            min_half=.524,     # both halves of the record at least break-even
            min_era_n=25)      # ...whenever that slice has enough games to mean anything
ERAS = [(2010, 2013), (2014, 2017), (2018, 2021), (2022, 2025)]
HALVES = [(2010, 2017), (2018, 2025)]
WEEK_WINDOWS = [(1, 22, 'all weeks'), (1, 4, 'wk1-4'), (5, 8, 'wk5-8'), (9, 12, 'wk9-12'),
                (13, 14, 'wk13-14'), (15, 18, 'wk15-18'), (19, 22, 'playoffs'), (1, 12, 'wk1-12'),
                (5, 12, 'wk5-12'), (5, 18, 'wk5-18'), (9, 14, 'wk9-14'), (13, 22, 'wk13+'),
                (15, 22, 'wk15+'), (5, 22, 'wk5+')]
EDGES = [2., 3., 4., 5., 6., 7.]
SD_BANDS = [(0., 1., 'any sd'), (0., .25, 'sd<=p25'), (0., .5, 'sd<=p50'), (0., .75, 'sd<=p75'),
            (.25, 1., 'sd>p25'), (.5, 1., 'sd>p50')]
IMP_BANDS = [(0., 1., 'any imp'), (.5, 1., 'imp>=p50'), (.75, 1., 'imp>=p75'), (0., .5, 'imp<p50')]
LINE_BANDS = {'spread': [(0., 99., 'any line'), (0., 3.5, '|line|<=3.5'), (4., 9.5, '|line| 4-9.5'),
                         (10., 99., '|line|>=10')],
              'total': [(0., 99., 'any line'), (0., 41.5, 'total<42'), (42., 46., 'total 42-46'),
                        (46.5, 99., 'total>46')]}


def load(folder, market):
    """A backtest folder's weekly parquets, graded for one market."""
    from backtester import market_panel, settle
    files = sorted(glob.glob(str(Path(folder) / '*wk*.parquet')))
    if not files:
        raise SystemExit(f'No weekly parquets in {folder}')
    games = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    data = market_panel(games, market)
    if market == 'total':
        data['prediction'], data['variance'] = data.total_prediction, data.total_variance
    data['edge'] = data.prediction - data.market_base
    data = settle(data, 0.)
    data['sd'] = np.sqrt(data.variance.clip(lower=0))
    data['abs_edge'] = data.edge.abs()
    return data.dropna(subset=['residual']).reset_index(drop=True)


def rules(d, market):
    """(labels, mask matrix) for the full cross product."""
    labels, masks = [], []
    week, edge, sd = d.week.to_numpy(), d.abs_edge.to_numpy(), d.sd.to_numpy()
    imp = d.total_game_importance.to_numpy()
    line = (d.market_base.abs() if market == 'spread' else d.market_base).to_numpy()
    for wlo, whi, wlabel in WEEK_WINDOWS:
        in_week = (week >= wlo) & (week <= whi)
        if in_week.sum() < GATE['min_picks']:
            continue
        sd_q = np.quantile(sd[in_week], [0., .25, .5, .75, 1.])
        imp_q = np.nanquantile(imp[in_week], [0., .5, .75, 1.])
        for elo in EDGES:
            in_edge = in_week & (edge >= elo)
            for qlo, qhi, sdlabel in SD_BANDS:
                lo = np.interp(qlo, [0, .25, .5, .75, 1.], sd_q)
                hi = np.interp(qhi, [0, .25, .5, .75, 1.], sd_q)
                in_sd = in_edge & (sd >= lo) & (sd <= hi)
                for ilo, ihi, ilabel in IMP_BANDS:
                    if ilabel == 'any imp':
                        in_imp = in_sd
                    else:
                        a = np.interp(ilo, [0, .5, .75, 1.], imp_q)
                        b = np.interp(ihi, [0, .5, .75, 1.], imp_q)
                        in_imp = in_sd & (imp >= a) & (imp <= b)
                    for llo, lhi, llabel in LINE_BANDS[market]:
                        mask = in_imp & (line >= llo) & (line <= lhi)
                        if mask.sum() < GATE['min_picks']:
                            continue
                        labels.append(f'{wlabel:10} e>={elo:g} {sdlabel:8} {ilabel:9} {llabel}')
                        masks.append(mask)
    return labels, np.array(masks, dtype=np.float32)


def survivors(masks, wins, playable, season):
    """Which rules clear the whole gate. `wins` may be a games x draws matrix
    of shuffled outcomes, in which case the answer is per draw."""
    single = wins.ndim == 1
    w = wins[:, None] if single else wins
    p = playable.astype(np.float32)
    n = masks @ p
    rate = np.divide(masks @ w, n[:, None], out=np.zeros((len(masks), w.shape[1]), np.float32),
                     where=n[:, None] > 0)
    ok = (n[:, None] >= GATE['min_picks']) & (rate >= GATE['min_rate'])
    for spans, floor in [(ERAS, GATE['min_era']), (HALVES, GATE['min_half'])]:
        for lo, hi in spans:
            span = ((season >= lo) & (season <= hi)).astype(np.float32) * p
            n_s = masks @ span
            r = np.divide(masks @ (w * span[:, None]), n_s[:, None],
                          out=np.zeros((len(masks), w.shape[1]), np.float32), where=n_s[:, None] > 0)
            ok &= (n_s[:, None] < GATE['min_era_n']) | (r >= floor)
    return (ok[:, 0], rate[:, 0], n) if single else (ok, rate, n)


def chance(masks, wins, playable, season, draws=200, seed=99):
    """The same gate, run against coin-flip picks: how many rules survive by
    luck alone. Flipping the side keeps every sample size and every condition
    intact and destroys only the model's skill."""
    rng = np.random.default_rng(seed)
    flip = rng.random((len(playable), draws)) < .5
    shuffled = np.where(flip, (playable & ~wins.astype(bool))[:, None], wins[:, None]).astype(np.float32)
    ok, _, _ = survivors(masks, shuffled, playable, season)
    return ok.sum(axis=0)


def units(frame):
    """Flat-stake profit at -110 where the real price is missing."""
    played = frame[~frame.push]
    wins = int(played.win.sum())
    return len(played), (wins / len(played) if len(played) else float('nan')), wins * (100 / 110) - (len(played) - wins)


def report(folder, market, draws=200, top=15):
    d = load(folder, market)
    labels, masks = rules(d, market)
    wins = (d.win & ~d.push).to_numpy().astype(np.float32)
    playable, season = (~d.push).to_numpy(), d.season.to_numpy()
    ok, rate, counts = survivors(masks, wins, playable, season)
    null = chance(masks, wins, playable, season, draws=draws)
    seasons = d.season.nunique()
    print(f'=== {Path(folder).parent.parent.name} · {market} · {len(d)} graded games, {seasons} seasons ===')
    print(f'  {len(labels)} rules searched · {int(ok.sum())} cleared the gate · '
          f'chance clears {null.mean():.1f} (95th pct {np.quantile(null, .95):.0f}) · '
          f'p = {(null >= ok.sum()).mean():.3f}')
    if not ok.any():
        return
    kept = sorted(((labels[i], rate[i], counts[i]) for i in np.where(ok)[0]), key=lambda r: -r[1])
    for label, r, n in kept[:top]:
        u = n * (r * (100 / 110) - (1 - r))
        print(f'    {100*r:5.1f}%  n={int(n):4d}  {u/seasons:+5.2f}u/yr  {label}')
    if len(kept) > top:
        print(f'    ... and {len(kept) - top} more')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', help='A backtest run folder, e.g. data/bt/model_2.0/2010-2025/<hash>')
    parser.add_argument('--market', choices=['spread', 'total', 'both'], default='both')
    parser.add_argument('--draws', type=int, default=200, help='Coin-flip replicas for the chance baseline')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args(argv)
    for market in (['spread', 'total'] if args.market == 'both' else [args.market]):
        report(args.folder, market, args.draws, args.top)
        print()


if __name__ == '__main__':
    sys.exit(main())
