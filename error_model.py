#!/usr/bin/env python3
"""Predict how badly the model will miss, then ask whether the games it
expects to get right are the ones it picks well.

The landscape search chopped the space into a few hundred thousand
threshold cells and ranked them, which pays a heavy multiple-comparison
price: 294,000 cells produce 67% hit rates out of shuffled outcomes. This
asks the same question with one model instead of a grid.

    |prediction - margin|  ~  ensemble SD + leverage + week + line + ...

Fit that on past seasons, score the current one, and every game gets a
continuous "expected miss". Bucket by it and read the hit rate off each
bucket. There is no cell to cherry-pick: the deciles are fixed by the
score, and the hit rate is a readout the fit never saw.

Everything is walk-forward -- a season's expected miss comes only from
seasons before it -- so the ranking is one the packet could actually have
produced on the day.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Known before kickoff. Realized margin, score and error are not features
# -- that would be scoring the model on its own answers.
FEATURES = ['sd', 'leverage', 'week', 'abs_line', 'abs_edge',
            'total_line', 'wind_mph', 'feels_like_f', 'abs_rest', 'indoor']


def frame(folder, market='spread'):
    files = sorted(Path(folder).glob('*wk*.parquet'))
    if not files:
        raise SystemExit(f'no scored weeks in {folder}')
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if market == 'total':
        d = d.dropna(subset=['total_prediction', 'points', 'total_line']).copy()
        d['abs_error'] = (d.total_prediction - d.points).abs()
        d['edge'] = d.total_prediction - d.total_line
        d['covered'] = d.points - d.total_line
        d['sd'] = np.sqrt(d.total_variance.clip(lower=0))
    else:
        d = d.dropna(subset=['prediction', 'margin', 'spread_line']).copy()
        d['abs_error'] = (d.prediction - d.margin).abs()
        d['edge'] = d.prediction + d.spread_line
        d['covered'] = d.margin + d.spread_line
        d['sd'] = np.sqrt(d.variance.clip(lower=0))
    d['abs_edge'] = d.edge.abs()
    d['abs_line'] = d.spread_line.abs()
    d['leverage'] = d.total_game_importance
    d['abs_rest'] = (d.away_rest - d.home_rest).abs() if 'away_rest' in d else 0.
    d['indoor'] = d.weather_indoor if 'weather_indoor' in d else 0.
    for column in FEATURES:
        if column not in d:
            d[column] = 0.
        d[column] = pd.to_numeric(d[column], errors='coerce')
    return d[d.covered != 0].reset_index(drop=True)


def walk_forward(data, min_seasons=4):
    """Expected miss for every game, fitted only on earlier seasons.

    Gradient boosting rather than a line: the relationships are not
    expected to be linear (a 3-point ensemble SD on a pick'em is a
    different thing from one on a 14-point favourite) and the interactions
    are the point. Shallow and heavily regularised, because the target is
    noisy and the sample is a few thousand games."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    seasons = sorted(data.season.unique())
    expected = pd.Series(np.nan, index=data.index)
    for season in seasons[min_seasons:]:
        past = data[data.season < season]
        now = data.season == season
        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=.05,
            min_samples_leaf=60, l2_regularization=1., random_state=1337)
        model.fit(past[FEATURES], past.abs_error)
        expected[now] = model.predict(data.loc[now, FEATURES])
    return expected


def report(data, expected, buckets=10):
    scored = data[expected.notna()].copy()
    scored['expected_miss'] = expected[expected.notna()]
    scored['won'] = np.sign(scored.edge) == np.sign(scored.covered)
    scored['bucket'] = pd.qcut(scored.expected_miss, buckets, labels=False, duplicates='drop')
    print(f'{len(scored)} games scored out of sample, {scored.season.min()}-{scored.season.max()}')
    print(f'overall: MAE {scored.abs_error.mean():.3f}, hit rate {scored.won.mean():.1%}\n')
    print(f"{'bucket':>8} {'expected':>9} {'actual':>8} {'hit':>8} {'n':>6} {'u/season':>9}")
    print('-' * 56)
    seasons = scored.season.nunique()
    for b, group in scored.groupby('bucket'):
        units = group.won.sum() * (100 / 110) - (~group.won).sum()
        print(f'{int(b) + 1:>8} {group.expected_miss.mean():>9.2f} {group.abs_error.mean():>8.2f} '
              f'{group.won.mean():>8.1%} {len(group):>6} {units / seasons:>9.2f}')
    best = scored[scored.bucket == 0]
    print(f'\n  corr(expected miss, actual miss) = {scored.expected_miss.corr(scored.abs_error):+.3f}')
    print(f'  corr(expected miss, won)         = {scored.expected_miss.corr(scored.won.astype(float)):+.3f}')
    return scored


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder')
    p.add_argument('--market', choices=['spread', 'total'], default='spread')
    p.add_argument('--buckets', type=int, default=10)
    args = p.parse_args(argv)
    data = frame(args.folder, args.market)
    return report(data, walk_forward(data), args.buckets)


if __name__ == '__main__':
    main()
