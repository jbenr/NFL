#!/usr/bin/env python3
"""alpha_juicer: find pick setups that hold up, by trying every combination.

Every graded pick carries things worth slicing on: how far the model is from
the market (the differential), how much the ensemble disagrees with itself
(SD), how much the game matters to the playoff picture (leverage, per team and
combined), when it was played, and -- on the sided-spread architecture, which
asks each side for the margin separately -- how far apart the two sides landed
(the gap). This searches bands of all of them, and every subset, so
"differential and leverage with no SD condition" is a candidate exactly like
"all of them" is -- then reports what survives.

The point is not to find a 65% cell. With a few thousand cells, coin flips
produce one. The point is to find a cell that is still there when the model
changes, so every candidate is scored on ALL the completed backtests at once and
ranked by its WORST model, not its best. A setup that reads 68% on the run it
was found on and 52% on the run next to it is a fit to that run's noise, and
ranking by the worst model refuses to reward it.

The gauntlet, in the order that kills candidates fastest:

  volume    at least --min-n graded picks on the primary run (default 100)
  every     model in the comparison set clears --floor (default 54%), counting
            only models with enough picks to judge -- this is the consistency
            requirement, and it is what most candidates die on
  eras      every four-season era with enough games clears --era-floor (50%)
  halves    both halves of the record clear --half-floor (52%)
  forward   walk-forward: the rule is bet in a season only if its record BEFORE
            that season already qualified, so the reported forward number is
            what you could actually have collected

and above the results sits the null: outcomes are reshuffled (keeping every
filter, sample size and push), the whole search is rerun, and the report says
how many rules cleared the same gates on randomized data. If chance produces 60
survivors and the search found 40, the search found nothing.

    python alpha_juicer.py --run model_2.0/weighted --market spread
    python alpha_juicer.py --run model_2.0/weighted --market total --weeks 1-12
    python alpha_juicer.py --run model_2.2 --market spread --rank primary --emit
    python alpha_juicer.py --list
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BT = Path('data/bt')
ERAS = [(2010, 2013), (2014, 2017), (2018, 2021), (2022, 2025)]
BANDS = [('S', .600), ('A', .575), ('B', .540)]


# ----------------------------------------------------------------- the runs
def parts(folder):
    """A run's prediction files, in either layout: the two-sided backtest
    writes one parquet per week, the shared/joint one writes one per market."""
    weeks = sorted(Path(folder).glob('*wk*.parquet'))
    return weeks if weeks else sorted(Path(folder).glob('*_spread.parquet')) + \
                               sorted(Path(folder).glob('*_total.parquet'))


def runs():
    """Every completed backtest under data/bt, newest first: label -> folder."""
    found = {}
    for config in BT.glob('*/*/*/config.json'):
        folder = config.parent
        files = parts(folder)
        if not files:
            continue
        spec = json.loads(config.read_text())
        # config['model'] is the architecture on a shared/joint run and the
        # version id ('model-2.0') on a two-sided one, so normalize it rather
        # than trusting the field: the architecture is what decides which runs
        # a rule has to replicate on.
        # Newer runs record it outright; older ones put the architecture in
        # 'model' for shared/joint and the version id for everything else.
        architecture = spec.get('architecture') or (
            spec['model'] if spec.get('model') in ['shared', 'joint'] else 'two-sided')
        family = spec.get('model_version', folder.parents[1].name)
        if architecture != 'two-sided':
            family += f'-{architecture}'
        label = f"{family}/{spec.get('calculation', '?')}"
        # Weeks, for the "is this run finished" check: a per-market layout has
        # them inside one file rather than one file each.
        count = len(files) if files[0].name.count('wk') else int(
            pd.read_parquet(files[0], columns=['season', 'week']).drop_duplicates().shape[0])
        found[f"{label}@{spec.get('start_season')}-{spec.get('season')}"] = dict(
            folder=folder, weeks=count, config=spec, architecture=architecture,
            written=max(f.stat().st_mtime for f in files))
    return dict(sorted(found.items(), key=lambda kv: -kv[1]['written']))


def resolve(name):
    """'model_2.2', 'model_2.0/steep' or a folder path -> folder."""
    if Path(name).is_dir():
        return Path(name)
    matches = [v['folder'] for k, v in runs().items() if k.startswith(name) or name in k]
    if not matches:
        raise SystemExit(f'No backtest matches {name!r}. Known runs:\n  ' + '\n  '.join(runs()))
    return matches[0]


def load(folder, market):
    """Graded picks for one market, settled by the same backtester code the
    packet uses, with the four sliceable quantities attached."""
    from backtester import market_panel, settle
    files = parts(folder)
    per_market = [f for f in files if f.name.endswith(f'_{market}.parquet')]
    if per_market:
        # Shared/joint layout: already one graded market panel per market.
        data = pd.concat([pd.read_parquet(f) for f in per_market], ignore_index=True)
        if 'market' not in data:
            data['market'] = market
    else:
        games = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        data = market_panel(games, market)
        if market == 'total':
            data['prediction'], data['variance'] = data.total_prediction, data.total_variance
    data['edge'] = data.prediction - data.market_base
    data = settle(data, 0.)
    data['sd'] = np.sqrt(data.variance.clip(lower=0))
    data['abs_edge'] = data.edge.abs()
    # Leverage: the combined number the packet already carries, the two teams'
    # own numbers, the gap between them, and how much the SIDE being picked
    # has riding on it -- a favourite with nothing to play for is the usual
    # story for a December upset, and only this last one can see that.
    data['leverage'] = data.total_game_importance
    data['leverage_gap'] = (data.away_importance - data.home_importance).abs()
    picked_away = data.edge > 0 if market == 'spread' else pd.Series(False, index=data.index)
    data['picked_leverage'] = np.where(picked_away, data.away_importance, data.home_importance)
    data['opponent_leverage'] = np.where(picked_away, data.home_importance, data.away_importance)
    data['line_magnitude'] = data.market_base.abs() if market == 'spread' else data.market_base
    # sided-spread asks each side for the margin and averages the two; the
    # gap between them is that architecture's own disagreement signal, which
    # the ensemble SD cannot see (it measures spread across members, not
    # across sides). NaN on every other architecture, which collapses the
    # dimension to 'any gap' and leaves those searches unchanged.
    data['gap'] = data.side_gap.abs() if 'side_gap' in data else np.nan
    return data.dropna(subset=['residual']).reset_index(drop=True)


def record(mask, data):
    """(n, hit rate, units at -110), pushes excluded from the rate."""
    played = mask & (~data.push).to_numpy()
    n = int(played.sum())
    wins = int((played & data.win.to_numpy()).sum())
    return n, (wins / n if n else float('nan')), wins * (100 / 110) - (n - wins)


def band(rate):
    for name, floor in BANDS:
        if rate >= floor:
            return name
    return None


# ---------------------------------------------------------------- the space
SD_QUANTILES = [.2, .25, .33, .5, .67, .75]
EDGE_QUANTILES = [.4, .5, .6, .7, .8, .9, .95]


def cut_points(values, quantiles, step):
    """Absolute cut points taken from the sample itself, rounded to `step` so
    a rule reads like a rule rather than a fitted decimal."""
    picks = sorted({round(float(values.quantile(q)) / step) * step for q in quantiles})
    return [p for p in picks if np.isfinite(p)]


def dimensions(market, weeks=None, sample=None):
    """The four sliceable quantities, each as bands plus an 'any' option, so
    every subset of conditions is searched -- differential alone, differential
    and leverage with no SD cut, SD and leverage with no differential, and so
    on.

    `sample` is the run being searched, and the bands come from it, because a
    fixed ladder is blind on a model it was not written for: the two-sided
    runs carry ensemble SDs around 4.2, the shared run around 1.65, so cuts of
    3.5-5.0 select every game it has and the whole dimension does nothing.

    The two scales are handled differently on purpose. Ensemble SD is internal
    to a model and means nothing across models, so its bands are PERCENTILES
    of whichever run is being measured -- "the tightest quarter of this
    model's own predictions" is the idea that transfers, and each run resolves
    it against its own distribution. The differential is market-relative: 3
    points off the number is 3 points off the number whoever predicted it, so
    its cuts are absolute, just chosen from this sample so they actually split
    it. Leverage is already a 0-1 scale shared by every run, so it is fixed.

    Labels carry both readings (`SD<=p25 (1.58)`) -- the percentile is what
    was searched, the absolute number is what the packet would have to apply
    to a single game.
    """
    windows = {'any week': (1, 22), 'wk1-3': (1, 3), 'wk1-6': (1, 6), 'wk4-8': (4, 8), 'wk1-12': (1, 12),
               'wk5-12': (5, 12), 'wk5-8': (5, 8), 'wk9-12': (9, 12), 'wk13-14': (13, 14),
               'wk15-18': (15, 18), 'playoffs': (19, 22), 'wk5+': (5, 22), 'wk9+': (9, 22),
               'wk13+': (13, 22)}
    week = {name: (lambda d, lo=lo, hi=hi: (d.week >= lo) & (d.week <= hi)) for name, (lo, hi) in windows.items()}
    week['wk13-14+playoffs'] = lambda d: d.week.between(13, 14) | (d.week >= 19)
    if weeks:
        lo, hi = weeks
        week = {f'wk{lo}-{hi}': lambda d, lo=lo, hi=hi: (d.week >= lo) & (d.week <= hi)}

    edge = {'any edge': lambda d: pd.Series(True, index=d.index)}
    cuts = (cut_points(sample.abs_edge, EDGE_QUANTILES, .5) if sample is not None
            else ([2, 3, 4, 5, 6, 7, 8] if market == 'spread' else [3, 4, 5, 6, 7, 8]))
    for cut in cuts:
        edge[f'edge>={cut:g}'] = lambda d, c=cut: d.abs_edge >= c
    for lo, hi in zip(cuts, cuts[1:]):
        edge[f'edge {lo:g}-{hi:g}'] = lambda d, lo=lo, hi=hi: (d.abs_edge >= lo) & (d.abs_edge < hi)

    sd = {'any SD': lambda d: pd.Series(True, index=d.index)}
    for q in SD_QUANTILES:
        here = f' ({sample.sd.quantile(q):.2f})' if sample is not None else ''
        sd[f'SD<=p{int(q*100)}{here}'] = lambda d, q=q: d.sd <= d.sd.quantile(q)
        sd[f'SD>p{int(q*100)}{here}'] = lambda d, q=q: d.sd > d.sd.quantile(q)
    band_here = (f' ({sample.sd.quantile(.25):.2f}-{sample.sd.quantile(.75):.2f})'
                 if sample is not None else '')
    sd[f'SD p25-p75{band_here}'] = lambda d: d.sd.between(d.sd.quantile(.25), d.sd.quantile(.75))

    lev = {'any leverage': lambda d: pd.Series(True, index=d.index)}
    for cut in [.3, .5, .7, .85]:
        lev[f'leverage>={cut:g}'] = lambda d, c=cut: d.leverage >= c
    lev['leverage<=0.3'] = lambda d: d.leverage <= .3
    lev['leverage 0.3-0.7'] = lambda d: d.leverage.between(.3, .7)
    # The averaged number hides the cases that matter most: a desperate
    # team against an eliminated one averages to middling. These read the
    # two sides separately.
    for cut in [.3, .5]:
        lev[f'gap>={cut:g} (mismatched)'] = lambda d, c=cut: d.leverage_gap >= c
    lev['gap<=0.2 (evenly matched)'] = lambda d: d.leverage_gap <= .2
    if market == 'spread':
        for cut in [.5, .7]:
            lev[f'picked side cares>={cut:g}'] = lambda d, c=cut: d.picked_leverage >= c
        lev['picked side cares<=0.3'] = lambda d: d.picked_leverage <= .3
        lev['picked side cares more'] = lambda d: d.picked_leverage > d.opponent_leverage + .2
        lev['picked side cares less'] = lambda d: d.picked_leverage + .2 < d.opponent_leverage
        lev['opponent has nothing on it'] = lambda d: d.opponent_leverage <= .3
        lev['opponent is desperate (>=0.7)'] = lambda d: d.opponent_leverage >= .7
        lev['both teams care (>=0.5)'] = lambda d: np.minimum(d.away_importance, d.home_importance) >= .5
        lev['neither team cares (<=0.3)'] = lambda d: np.maximum(d.away_importance, d.home_importance) <= .3
        lev['one cares, one does not'] = lambda d: (np.maximum(d.away_importance, d.home_importance) >= .6) & (
            np.minimum(d.away_importance, d.home_importance) <= .3)
    # The two sides' disagreement, where the architecture has one. Banded by
    # percentile like the ensemble SD, and for the same reason: it is
    # internal to a model and its absolute scale means nothing across them.
    gap = {'any gap': lambda d: pd.Series(True, index=d.index)}
    # `gap` is attached by load(); a sample handed straight to dimensions()
    # (tests, ad-hoc slices) will not carry it, and its absence just means
    # the dimension collapses to 'any gap'.
    if sample is not None and 'gap' in sample and sample.gap.notna().any():
        for q in (.25, .5):
            here = f' ({sample.gap.quantile(q):.2f})'
            gap[f'gap<=p{int(q*100)}{here}'] = lambda d, q=q: d.gap <= d.gap.quantile(q)
        for q in (.5, .75):
            here = f' ({sample.gap.quantile(q):.2f})'
            gap[f'gap>p{int(q*100)}{here}'] = lambda d, q=q: d.gap > d.gap.quantile(q)
    return dict(week=week, edge=edge, sd=sd, leverage=lev, gap=gap)


def combos(market, weeks=None, sample=None):
    """Every (week, edge, SD, leverage, gap) option tuple."""
    space = dimensions(market, weeks, sample)
    names = list(space)
    for picks in itertools.product(*(space[n].items() for n in names)):
        yield tuple((n, label) for n, (label, _) in zip(names, picks))


def precompute(frames, market, weeks=None, sample=None):
    """Every option's mask on every run, once. A twenty-thousand-combination
    search re-evaluates the same dozen predicates otherwise, and the whole
    point of the tool is that it can afford to be exhaustive."""
    space = dimensions(market, weeks, sample)
    # Each run resolves the predicates against ITS OWN frame, which is what
    # makes a percentile band mean the same thing on a model with a different
    # SD scale while an absolute differential stays absolute.
    return {run: {dim: {label: test(frame).to_numpy() for label, test in options.items()}
                  for dim, options in space.items()}
            for run, frame in frames.items()}


def build(combo, run_masks):
    """A candidate's mask, from precomputed option masks for one run."""
    mask = None
    for dim, label in combo:
        piece = run_masks[dim][label]
        mask = piece.copy() if mask is None else (mask & piece)
    return mask


def label_of(combo):
    return ' · '.join(label for _, label in combo if not label.startswith('any'))  or 'everything'


# ----------------------------------------------------------------- the null
def noise_floor(data, masks, args, draws=400, seed=1337, rule_chunk=2500):
    """What this exact search returns on randomized outcomes: the best rate any
    rule reaches, and how many rules clear the volume/rate/era/halves gates.
    The cross-model gate is left out -- the runs share these games, so a
    shuffle cannot be carried to them honestly -- which means real survivors
    are held to a stricter standard than this null is."""
    playable = (~data.push).to_numpy()
    wins = (data.win & ~data.push).to_numpy().astype(np.float32)
    slices = [data.season.between(lo, hi).to_numpy() for lo, hi in ERAS]
    slices += [(data.season <= 2017).to_numpy(), (data.season >= 2018).to_numpy()]
    rng = np.random.default_rng(seed)
    flips = rng.random((len(data), draws)) < .5
    shuffled = np.where(flips, (playable & ~data.win.to_numpy())[:, None], wins[:, None]).astype(np.float32)
    best = np.zeros(draws)
    survivors = np.zeros(draws)
    for start in range(0, len(masks), rule_chunk):
        block = np.array([m & playable for m in masks[start:start + rule_chunk]], dtype=np.float32)
        counts = block.sum(axis=1)
        block, counts = block[counts >= args.min_n], counts[counts >= args.min_n]
        if not len(block):
            continue
        rates = (block @ shuffled) / counts[:, None]
        passes = rates >= args.floor
        for i, keep in enumerate(slices):
            sliced = block * keep[None, :]
            sub_counts = sliced.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                sub_rates = (sliced @ shuffled) / sub_counts[:, None]
            floor = args.era_floor if i < len(ERAS) else args.half_floor
            thin = sub_counts < (args.era_min_n if i < len(ERAS) else 30)
            passes &= (sub_rates >= floor) | thin[:, None] | ~np.isfinite(sub_rates)
        best = np.maximum(best, rates.max(axis=0))
        survivors += passes.sum(axis=0)
    return best, survivors


# -------------------------------------------------------------- the scoring
def walk_forward(mask, data, floor, min_history=50):
    bet = np.zeros(len(data), bool)
    for season in sorted(data.season.unique()):
        past = mask & (data.season < season).to_numpy()
        n, rate, _ = record(past, data)
        if n >= min_history and rate >= floor:
            bet |= mask & (data.season == season).to_numpy()
    return record(bet, data)


def evaluate(combo, frames, cache, primary, market, args):
    mask = build(combo, cache[primary])
    data = frames[primary]
    n, rate, units = record(mask, data)
    if n < args.min_n or not np.isfinite(rate) or rate < args.floor:
        return None
    across = {primary: (n, rate)}
    for name, frame in frames.items():
        if name == primary:
            continue
        across[name] = record(build(combo, cache[name]), frame)[:2]
    judged = [r for count, r in across.values() if count >= args.cross_min_n]
    worst = min(judged) if judged else float('nan')
    if args.rank == 'worst' and (not judged or worst < args.floor):
        return None
    eras = [record(mask & data.season.between(lo, hi).to_numpy(), data) for lo, hi in ERAS]
    measured = [r for count, r, _ in eras if count >= args.era_min_n]
    if measured and min(measured) < args.era_floor:
        return None
    halves = [record(mask & (data.season <= 2017).to_numpy(), data),
              record(mask & (data.season >= 2018).to_numpy(), data)]
    judged_halves = [h[1] for h in halves if h[0] >= 30]
    if judged_halves and min(judged_halves) < args.half_floor:
        return None
    return dict(label=label_of(combo), combo=combo, mask=mask, n=n, rate=rate, units=units,
                worst=worst, across=across, eras=eras, halves=halves, tier=band(worst if args.rank == 'worst' else rate),
                forward=walk_forward(mask, data, args.floor))


def fold(results, limit=.7):
    """Keep one candidate per family: a grid makes dozens of near-identical
    slices of any real effect, and listing them all inflates one finding."""
    keep = []
    for candidate in results:
        if not any((candidate['mask'] & other['mask']).sum() /
                   max(1, min(candidate['mask'].sum(), other['mask'].sum())) >= limit for other in keep):
            keep.append(candidate)
    return keep


def report(results, data, args, market, primary, searched, best_draws, survivor_draws):
    print(f'\n=== alpha_juicer · {primary} · {market} · {searched} combinations ===')
    print(f'    {len(data)} graded games, {data.season.min()}-{data.season.max()}, '
          f'ranked by {"worst model" if args.rank == "worst" else "this model"}')
    print('    SD bands are percentiles, re-derived on each model (the number in brackets is this run\'s '
          'absolute cut); differential cuts are absolute points everywhere')
    print(f'    null: on randomized outcomes the best cell reaches {100 * np.median(best_draws):.1f}% '
          f'(95th pct {100 * np.quantile(best_draws, .95):.1f}%), and '
          f'{np.median(survivor_draws):.0f} cells clear the volume/rate/era/halves gates '
          f'(95th pct {np.quantile(survivor_draws, .95):.0f})')
    print(f'    gates: n>={args.min_n}, every model>={100*args.floor:.0f}%, era>={100*args.era_floor:.0f}%, '
          f'halves>={100*args.half_floor:.0f}%')
    # A gate over a slice with no games in it passes for free. On a partial or
    # short run that quietly disarms the era and halves checks, which are the
    # two doing most of the work -- so say so rather than printing survivors
    # that only had to clear one hurdle.
    live = sum(1 for lo, hi in ERAS if int(data.season.between(lo, hi).sum()) >= args.era_min_n)
    halves = [int((data.season <= 2017).sum()), int((data.season >= 2018).sum())]
    if live < len(ERAS) or min(halves) < 30:
        print(f'    WARNING: this run covers {data.season.nunique()} seasons -- {live} of {len(ERAS)} eras and '
              f'{sum(1 for h in halves if h >= 30)} of 2 halves have enough games. The era and halves gates '
              'pass automatically on the empty ones, so everything below is provisional.')
    if not results:
        print('\n    Nothing survived. In this space, on these models, there is no setup that is '
              'simultaneously profitable, era-stable and model-stable.')
        return []
    keep = fold(sorted(results, key=lambda r: -(r['worst'] if args.rank == 'worst' else r['rate'])))
    print(f'\n    {len(results)} combinations passed every gate; {len(keep)} families after folding overlaps.')
    if len(results) <= np.quantile(survivor_draws, .95):
        print('    That is inside what randomized outcomes produce -- treat these as leads, not findings.')
    print()
    for candidate in keep[:args.top]:
        eras = ' / '.join(f'{100*r:.0f}' if count >= args.era_min_n else '–' for count, r, _ in candidate['eras'])
        halves = ' / '.join(f'{100*h[1]:.0f}' for h in candidate['halves'])
        fn, frate, funits = candidate['forward']
        print(f'  [{candidate["tier"] or "–"}] {candidate["label"]}')
        print(f'       this model {100*candidate["rate"]:.1f}% of {candidate["n"]}, {candidate["units"]:+.1f}u '
              f'({candidate["units"]/16:+.2f}u a season)  ·  worst model {100*candidate["worst"]:.1f}%')
        print(f'       eras {eras}  ·  halves {halves}  ·  ' +
              (f'walk-forward {100*frate:.1f}% of {fn}' if fn else 'walk-forward never qualified'))
        print('       ' + '  '.join(f'{name.split("@")[0]} {100*rate:.1f}% ({count})'
                                     for name, (count, rate) in candidate['across'].items()))
    return keep


def emit(keep, market):
    print(f'\n# --- {market} candidates for weekly_packet.PICK_BUCKETS ---')
    for candidate in keep:
        eras = ' / '.join(f'{100*r:.0f}' if count else '–' for count, r, _ in candidate['eras'])
        across = ', '.join(f'{name.split("@")[0]} {100*rate:.1f}%'
                           for name, (count, rate) in list(candidate['across'].items())[1:])
        print(f"        dict(rule={candidate['label']!r},\n"
              f"             test=lambda week, importance, sd, line, edge: ...,   # hand-write from the rule\n"
              f"             model=TIER_MODEL, rate={candidate['rate']:.3f}, n={candidate['n']}, "
              f"worth='{candidate['units']/16:+.1f}u a season on ~{candidate['n']/16:.0f} picks',\n"
              f"             eras='{eras}%',\n             siblings={across!r},\n             note='...'),")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', help='Primary backtest: model_2.2, model_2.0/steep, or a folder')
    parser.add_argument('--market', choices=['spread', 'total', 'both'], default='both')
    parser.add_argument('--weeks', help='Restrict the search to a window, e.g. 1-12')
    parser.add_argument('--rank', choices=['worst', 'primary'], default='worst',
                        help="'worst' (default) scores every candidate by its weakest model")
    parser.add_argument('--min-n', type=int, default=100)
    parser.add_argument('--cross-min-n', type=int, default=60, help='Picks a model needs before it gets a vote')
    parser.add_argument('--floor', type=float, default=.54)
    parser.add_argument('--era-floor', type=float, default=.50)
    parser.add_argument('--era-min-n', type=int, default=25)
    parser.add_argument('--half-floor', type=float, default=.52)
    parser.add_argument('--draws', type=int, default=200)
    parser.add_argument('--top', type=int, default=12)
    parser.add_argument('--siblings', choices=['family', 'all', 'none'], default='family',
                        help="Which runs a candidate must also hold up on. 'family' (default) uses only runs "
                             "of the same ARCHITECTURE -- a shared-model rule should replicate across shared "
                             "runs, not across a two-sided one that owns a different part of the season. "
                             "'all' demands every completed run, 'none' scores the primary run alone.")
    parser.add_argument('--only', help='Comma-separated run labels to compare against (overrides --siblings)')
    parser.add_argument('--emit', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        for label, info in runs().items():
            print(f"{label:34} {info['weeks']:4d} weeks  {info['folder']}")
        return
    if not args.run:
        parser.error('--run is required (or --list)')
    weeks = tuple(int(x) for x in args.weeks.split('-')) if args.weeks else None
    table = runs()
    folder = resolve(args.run)
    primary = next((k for k, v in table.items() if v['folder'] == folder), str(folder))
    family = table[primary]['architecture'] if primary in table else 'two-sided'
    wanted = {k: v['folder'] for k, v in table.items() if v['weeks'] >= 300}
    if args.siblings == 'none':
        wanted = {}
    elif args.siblings == 'family':
        wanted = {k: v for k, v in wanted.items() if table[k]['architecture'] == family}
    if args.only:
        wanted = {k: v['folder'] for k, v in table.items() if any(o in k for o in args.only.split(','))}
    wanted[primary] = folder
    others = [k for k in wanted if k != primary]
    print(f'  primary: {primary} ({family}); must also hold on: '
          + (', '.join(others) if others else 'nothing -- no sibling runs of this architecture, so a '
                                              'result here is unreplicated'), flush=True)
    for market in (['spread', 'total'] if args.market == 'both' else [args.market]):
        frames = {name: load(path, market) for name, path in wanted.items()}
        cache = precompute(frames, market, weeks, sample=frames[primary])
        space = list(combos(market, weeks, sample=frames[primary]))
        sd = frames[primary].sd
        print(f'  {market}: bands taken from this run -- SD p25/p50/p75 = '
              f'{sd.quantile(.25):.2f}/{sd.quantile(.5):.2f}/{sd.quantile(.75):.2f}, '
              f'differential cuts {cut_points(frames[primary].abs_edge, EDGE_QUANTILES, .5)}', flush=True)
        print(f'  {market}: {len(space)} combinations x {len(frames)} models...', flush=True)
        results, masks = [], []
        for combo in space:
            mask = build(combo, cache[primary])
            masks.append(mask)
            if mask.sum() < args.min_n:
                continue
            candidate = evaluate(combo, frames, cache, primary, market, args)
            if candidate is not None:
                results.append(candidate)
        best_draws, survivor_draws = noise_floor(frames[primary], masks, args, draws=args.draws)
        keep = report(results, frames[primary], args, market, primary, len(space), best_draws, survivor_draws)
        if args.emit and keep:
            emit(keep, market)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
