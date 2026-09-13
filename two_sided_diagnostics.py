"""Hit rate (raw and cutoff-optimized) for a two-sided backtester run
(backtester.py --model two-sided), for spread and total separately.

Reads whatever per-week parquet files already exist in the run's output
directory -- works fine on a run still in progress, not just a finished one.
Read-only; never touches the model or its saved predictions.

Usage:
    python two_sided_diagnostics.py data/bt/two_sided/2025/<fingerprint>

"raw" = every single pick the model made (diff_cutoff=0, no SD filter) --
same number backtester.py's own summary.json reports as "all_picks".
"optimized" = split the games in half by week; find the best diff/SD cutoff
on the first half (calibration), report how that exact cutoff performs on
the second half (validation) -- never picks a cutoff and grades it on the
same games. With one season's worth of games this split is thin (a genuine
limitation, not hidden), so treat "optimized" as a rough signal, not a
validated policy -- the codebase's usual multi-season walk-forward
(optimus_prime.py, backtester.py's own evaluate()) is what real cutoff
validation looks like.
"""
import argparse
from pathlib import Path

import pandas as pd

from backtester import cutoff_grid, market_panel, policy_sd_cutoff, roi_interval, score


def load_predictions(output_dir):
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob('*_wk*.parquet'))
    if not files:
        raise ValueError(f'No per-week prediction files found in {output_dir}')
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), len(files)


def diagnose(output_dir, min_bets=20, calibration_frac=0.5):
    data, n_weeks = load_predictions(output_dir)
    print(f'{len(data)} games loaded from {n_weeks} weekly files in {output_dir}')
    report = {}
    for market in ['spread', 'total']:
        panel = market_panel(data, market)
        if market == 'total':
            panel['prediction'], panel['variance'] = panel.total_prediction, panel.total_variance
        panel['edge'] = panel.prediction - panel.market_base
        raw = score(panel)

        weeks = sorted(panel.week_id.unique())
        split = weeks[max(1, int(len(weeks) * calibration_frac))]
        calibration, validation = panel[panel.week_id < split], panel[panel.week_id >= split]
        grid = cutoff_grid(calibration, min_bets)
        eligible = grid[grid.eligible].sort_values(['pnl_units', 'n'], ascending=False)
        if eligible.empty:
            optimized = dict(status='INSUFFICIENT_CALIBRATION_BETS')
        else:
            chosen = eligible.iloc[0]
            diff, sd = float(chosen.diff_cutoff), policy_sd_cutoff(chosen)
            optimized = dict(diff_cutoff=diff, sd_cutoff=sd,
                             calibration=score(calibration, diff, sd),
                             validation=score(validation, diff, sd),
                             roi_95=roi_interval(validation, diff, sd))
        report[market] = dict(raw_all_picks=raw, optimized=optimized)

        print(f'\n{market.upper()} ({len(panel)} games, {len(calibration)} calibration / {len(validation)} validation)')
        print(f'  raw, every pick:      n={raw["n"]:3d}  win_rate={_fmt(raw["win_rate"])}  pnl={raw["pnl_units"]:+.2f}u')
        if 'validation' in optimized:
            c, v = optimized['calibration'], optimized['validation']
            print(f'  optimized cutoff:     diff>={optimized["diff_cutoff"]}  sd<={_fmt(optimized["sd_cutoff"])}')
            print(f'    calibration:        n={c["n"]:3d}  win_rate={_fmt(c["win_rate"])}  pnl={c["pnl_units"]:+.2f}u')
            print(f'    validation:         n={v["n"]:3d}  win_rate={_fmt(v["win_rate"])}  pnl={v["pnl_units"]:+.2f}u  '
                  f'roi_95={_fmt_interval(optimized["roi_95"])}')
        else:
            print(f'  optimized:            {optimized["status"]}')
    return report


def _fmt(value):
    return f'{value:.3f}' if isinstance(value, (int, float)) else str(value)


def _fmt_interval(interval):
    lo, hi = interval
    return 'n/a' if lo is None else f'[{lo:+.3f}, {hi:+.3f}]'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dir', help='A two-sided run directory, e.g. data/bt/two_sided/2025/<fingerprint>')
    ap.add_argument('--min-bets', type=int, default=20, help='Minimum calibration bets for a cutoff to be eligible')
    ap.add_argument('--calibration-frac', type=float, default=0.5,
                    help='Fraction of weeks (by week_id) used for calibration; the rest is validation')
    args = ap.parse_args()
    diagnose(args.output_dir, args.min_bets, args.calibration_frac)
