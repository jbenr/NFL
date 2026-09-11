"""Which of YOUR features actually carry edge against the spread?

Tests only the 32 features that already exist in data_crunchski_2.py / comp_stats()
-- the ones model_shredski.py trains on. No external features, no candidate
catalog, nothing pulled from anywhere else. If it isn't in PRODUCTION_FEATURES
below (copied verbatim from model_shredski.py's own feature list), this script
doesn't know about it and never touches it.

(An earlier version of this file also tested a second, separate set of research
features from feature_scan_data.py -- EPA/CPOE/standings-based edges that don't
exist anywhere in your production pipeline. That was answering a question you'd
already moved past. Dropped entirely; not imported, not referenced.)

Method, for each of your 32 features F:
  - full        : ATS win rate with all 32 features.
  - drop_F      : ATS win rate with F removed, the other 31 kept.
                  If this is WORSE than full, F is pulling real weight.
  - F_alone     : ATS win rate with just F (+ home_field_adv, the control).
                  Does F carry standalone signal on its own?
  drop_F and full are compared as a PAIRED difference on the identical games
  (same weekly folds, same test rows) -- matching feature_scan.py's own
  paired_improvement() approach -- because paired comparison has much more
  power than eyeballing two overlapping independent confidence intervals.

Metric is ATS win rate, not MAE: MAE improvements don't reliably translate into
more covers, which is the thing that pays. Bar for real edge is stated
explicitly -- 52.4%, the standard -110 breakeven -- not just ">50%".

Walk-forward is WEEKLY (refit on every strictly-prior game), not season blocks,
for far more bootstrap resampling units than a 3-season-fold design allows.

A fast Ridge stands in for the real 100-model TF ensemble so refitting 32 times
per week runs in seconds, not hours. Treat any verdict here as a screen to
confirm with the real back_test() in main.py, not as proof.

Usage:
    python edge_scan.py
    python edge_scan.py --lookback-weeks 150 --min-train-weeks 40
    NFL_WORKERS=4 python edge_scan.py --lookback-weeks 150   # cap prep_test_train's workers

Runtime note: the feature panel is built via prep_test_train, whose cost grows
worse than linearly with --lookback-weeks. 60 weeks runs in under a minute on
8 cores; 150+ takes several minutes.

Output: data/edge_scan/<timestamp>/report.html, results.csv, manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_crunchski_2 as dc

BREAKEVEN = 0.524  # standard -110 vig breakeven; the actual bar for "edge"

# Copied verbatim from model_shredski.py's modelo(). This IS your feature set --
# if you add/rename a feature there, mirror it here or this script silently
# stops covering it.
PRODUCTION_FEATURES = [
    "away_off_run_ypp", "away_def_run_ypp", "away_off_pass_ypp", "away_def_pass_ypp",
    "away_off_pass_completion_%", "away_def_pass_completion_%",
    "away_off_series_success_%", "away_def_series_success_%",
    "away_off_first_down_pp", "away_def_first_down_pp",
    "away_off_third_down_%", "away_def_third_down_%",
    "away_off_fourth_down_%", "away_def_fourth_down_%",
    "away_off_turnovers_pp", "away_def_turnovers_pp",
    "away_off_penalties_pp", "away_def_penalties_pp",
    "away_off_qb_elo", "away_def_qb_elo",
    "away_off_explosive_run_%", "away_def_explosive_run_%",
    "away_off_explosive_pass_%", "away_def_explosive_pass_%",
    "away_off_stuff_%", "away_def_stuff_%",
    "away_off_sack_%", "away_def_sack_%",
    "away_off_qb_hit_%", "away_def_qb_hit_%",
    "away_rest_adv", "home_field_adv",
    "away_game_importance",  # matchup_importance(), wired into additional_features() this session
]

REG_ALPHAS = [1.0, 5.0, 20.0, 50.0, 100.0, 200.0]
TOPK_SIZES = [4, 8, 16, 32]


def _bootstrap_ci(values: np.ndarray, week_id: np.ndarray, reps: int = 2000, seed: int = 1337):
    """Block-bootstrap the mean of `values` by week."""
    df = pd.DataFrame({"v": values, "week_id": week_id})
    weekly = df.groupby("week_id")["v"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(weekly), size=(reps, len(weekly)))
    means = weekly["sum"].to_numpy()[idx].sum(axis=1) / np.maximum(weekly["count"].to_numpy()[idx].sum(axis=1), 1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _make_model(alpha: float = 20.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def weekly_hits(panel: pd.DataFrame, features: list[str], min_train_weeks: int, alpha: float = 20.0):
    """Refit weekly on strictly earlier games; predict this week's ATS side.
    Returns per-game hit (bool), week_id, and game key columns, aligned so two
    calls with different `features` can be compared row-for-row (paired)."""
    weeks = panel[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    weeks["week_id"] = range(len(weeks))
    p = panel.merge(weeks, on=["season", "week"], how="left")

    rows = []
    for _, row in weeks.iterrows():
        wid = row["week_id"]
        if wid < min_train_weeks:
            continue
        train = p[p["week_id"] < wid]
        test = p[p["week_id"] == wid]
        if train.empty or test.empty:
            continue
        model = _make_model(alpha).fit(train[features], train["market_residual"])
        pred = model.predict(test[features])
        actual = test["market_residual"].to_numpy()
        push = actual == 0
        hit = (np.sign(pred) == np.sign(actual))
        out = test[["season", "week", "away_team", "home_team", "week_id"]].copy()
        out["hit"] = hit
        out["push"] = push
        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "week", "away_team", "home_team", "week_id", "hit", "push"])


def summarize(hits: pd.DataFrame) -> dict:
    h = hits[~hits["push"]]
    if h.empty:
        return dict(n=0, win_rate=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    win_rate = float(h["hit"].mean())
    ci_low, ci_high = _bootstrap_ci(h["hit"].to_numpy(), h["week_id"].to_numpy())
    return dict(n=int(len(h)), win_rate=win_rate, ci_low=ci_low, ci_high=ci_high)


def paired_gain(reference: pd.DataFrame, trial: pd.DataFrame) -> dict:
    """Positive means trial beats reference on the identical games. Used to
    compare drop_F against full without the noise of two overlapping but
    independently-bootstrapped intervals."""
    keys = ["season", "week", "away_team", "home_team"]
    m = reference[keys + ["hit", "push", "week_id"]].merge(
        trial[keys + ["hit"]], on=keys, suffixes=("_ref", "_trial"), validate="one_to_one")
    m = m[~m["push"]]
    if m.empty:
        return dict(n=0, gain=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    gain = (m["hit_trial"].astype(int) - m["hit_ref"].astype(int)).to_numpy()
    ci_low, ci_high = _bootstrap_ci(gain, m["week_id"].to_numpy())
    return dict(n=int(len(m)), gain=float(gain.mean()), ci_low=ci_low, ci_high=ci_high)


def build_panel(root: str, asof_season: int, asof_week: int, lookback_weeks: int) -> pd.DataFrame:
    print(f"Building feature panel (lookback={lookback_weeks} weeks) from your own "
          f"data_crunchski_2 pipeline...", flush=True)
    prod = dc.prep_test_train(asof_season, asof_week, lookback_weeks)
    prod = dc.additional_features(prod)

    sched = pd.read_parquet(f"{root}/data/sched.parquet")[
        ["season", "week", "away_team", "home_team", "spread_line"]]
    panel = prod.merge(sched, on=["season", "week", "away_team", "home_team"],
                       how="left", validate="one_to_one")
    panel["margin"] = panel["away_score"] - panel["home_score"]
    # nflverse spread_line is the market's expected HOME margin; verified this
    # session against the-odds-api's own sign convention (they match).
    panel["market_residual"] = panel["margin"] + panel["spread_line"]

    panel = panel.dropna(subset=PRODUCTION_FEATURES + ["market_residual"])
    print(f"Panel: {len(panel)} games, {panel['season'].nunique()} seasons, "
          f"{panel.groupby(['season','week']).ngroups} weeks after dropna.")
    return panel


def verdict_drop(gain_low: float, gain_high: float) -> str:
    """Sign convention: gain = hit_drop - hit_full. Negative-and-confident means
    dropping the feature hurts -> the feature is significant. Positive-and-
    confident means dropping it HELPS -> the feature is actively hurting the
    model (an overfitting/noise candidate, not a keeper)."""
    if gain_high < 0:
        return "significant (drop hurts)"
    if gain_low > 0:
        return "hurts model (drop helps)"
    return ""


def verdict_alone(win_rate: float, ci_low: float) -> str:
    if ci_low > BREAKEVEN:
        return "EDGE alone"
    if win_rate > BREAKEVEN:
        return "maybe alone"
    return ""


def run_scan(panel: pd.DataFrame, min_train_weeks: int) -> pd.DataFrame:
    print(f"\nFitting full model ({len(PRODUCTION_FEATURES)} features)...", flush=True)
    full_hits = weekly_hits(panel, PRODUCTION_FEATURES, min_train_weeks)
    full_summary = summarize(full_hits)
    print(f"  full  n={full_summary['n']:4d}  win_rate={full_summary['win_rate']:.3f}  "
          f"CI=[{full_summary['ci_low']:.3f}, {full_summary['ci_high']:.3f}]", flush=True)

    rows = [dict(feature="(all 32)", variant="full", **full_summary,
                 drop_gain=0.0, drop_ci_low=0.0, drop_ci_high=0.0, drop_verdict="",
                 alone_verdict="")]

    for i, feat in enumerate(PRODUCTION_FEATURES, 1):
        remaining = [f for f in PRODUCTION_FEATURES if f != feat]
        drop_hits = weekly_hits(panel, remaining, min_train_weeks)
        drop_summary = summarize(drop_hits)
        pg = paired_gain(full_hits, drop_hits)  # gain = hit_drop - hit_full

        alone_feats = [feat] if feat == "home_field_adv" else [feat, "home_field_adv"]
        alone_summary = summarize(weekly_hits(panel, alone_feats, min_train_weeks))

        rows.append(dict(
            feature=feat, variant=f"drop_{feat}", **drop_summary,
            drop_gain=pg["gain"], drop_ci_low=pg["ci_low"], drop_ci_high=pg["ci_high"],
            drop_verdict=verdict_drop(pg["ci_low"], pg["ci_high"]),
            alone_win_rate=alone_summary["win_rate"], alone_ci_low=alone_summary["ci_low"],
            alone_ci_high=alone_summary["ci_high"],
            alone_verdict=verdict_alone(alone_summary["win_rate"], alone_summary["ci_low"]),
        ))
        if i % 8 == 0 or i == len(PRODUCTION_FEATURES):
            print(f"  ...{i}/{len(PRODUCTION_FEATURES)} features tested", flush=True)

    return pd.DataFrame(rows)


def run_overfit_diagnostics(panel: pd.DataFrame, min_train_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is the full 32-feature set well-regularized, or would fewer features /
    heavier regularization do better? A feature failing its own drop-test can
    mean either 'this feature is noise' or 'the whole set is overfit and no
    individual feature will look significant until that's fixed' -- this
    tells you which."""
    print("\nRegularization sweep (all 32 features, varying alpha)...", flush=True)
    reg_rows = []
    for alpha in REG_ALPHAS:
        s = summarize(weekly_hits(panel, PRODUCTION_FEATURES, min_train_weeks, alpha=alpha))
        s["alpha"] = alpha
        reg_rows.append(s)
        print(f"  alpha={alpha:6.1f}  win_rate={s['win_rate']:.3f}  "
              f"CI=[{s['ci_low']:.3f}, {s['ci_high']:.3f}]", flush=True)

    print("\nFeature-count sweep (top-K by |correlation| with market_residual; "
          "whole-panel ranking is a quick screen, not a leak-free selection -- "
          "the walk-forward refit below still only uses strictly prior weeks)...", flush=True)
    corr = panel[PRODUCTION_FEATURES].corrwith(panel["market_residual"]).abs().sort_values(ascending=False)
    topk_rows = []
    for k in TOPK_SIZES:
        feats = corr.head(k).index.tolist()
        s = summarize(weekly_hits(panel, feats, min_train_weeks))
        s["k"] = k
        s["top_features"] = ", ".join(feats[:5]) + (", ..." if k > 5 else "")
        topk_rows.append(s)
        print(f"  top-{k:2d}  win_rate={s['win_rate']:.3f}  CI=[{s['ci_low']:.3f}, {s['ci_high']:.3f}]", flush=True)

    return pd.DataFrame(reg_rows), pd.DataFrame(topk_rows)


def write_bar_chart(out_dir: Path, panel: pd.DataFrame, results: pd.DataFrame,
                    args: argparse.Namespace) -> None:
    """One chart. Plain matplotlib -- not model_shredski's chart helper, which
    imports TensorFlow at module load; this tool's entire point is staying fast
    and TF-free.

    Bar value = -drop_gain, i.e. how much the hit rate FALLS if you remove the
    feature. Positive bar = feature is pulling weight, keep it. Negative bar =
    the model does better without it, prune candidate. Green/red/gray marks
    which ones clear the paired-CI bar for "confident," not just "positive.\""""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = results[results["variant"] == "full"].iloc[0]
    feats = results[results["variant"] != "full"].copy()
    feats["value"] = -feats["drop_gain"]
    feats["err_low"] = -feats["drop_ci_high"]
    feats["err_high"] = -feats["drop_ci_low"]
    feats = feats.sort_values("value")

    colors = feats["drop_verdict"].map({
        "significant (drop hurts)": "#2a9d3f",
        "hurts model (drop helps)": "#d64545",
    }).fillna("#999999")

    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(feats))))
    y = np.arange(len(feats))
    ax.barh(y, feats["value"], xerr=[feats["value"] - feats["err_low"], feats["err_high"] - feats["value"]],
           color=colors, capsize=2, height=0.7)
    ax.set_yticks(y, feats["feature"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Win-rate contribution (paired drop-test; + = keep, − = prune candidate)")
    ax.set_title(
        f"Feature significance -- your 32 production features only\n"
        f"Full model: {full['win_rate']:.1%} ATS  (n={full['n']}, {panel['season'].min()}-"
        f"{panel['season'].max()})  |  breakeven: {BREAKEVEN:.1%}  |  green=significant, red=hurts model",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--asof-season", type=int, default=2025)
    ap.add_argument("--asof-week", type=int, default=22)
    ap.add_argument("--lookback-weeks", type=int, default=60)
    ap.add_argument("--min-train-weeks", type=int, default=20)
    ap.add_argument("--diagnostics", action="store_true",
                    help="also run the regularization/top-K overfitting sweeps (slower, off by default)")
    args = ap.parse_args()

    os.chdir(args.root)
    panel = build_panel(".", args.asof_season, args.asof_week, args.lookback_weeks)

    results = run_scan(panel, args.min_train_weeks)

    print()
    feats = results[results["variant"] != "full"].sort_values("drop_gain")
    for _, r in feats.iterrows():
        print(f"  {r['feature']:32s} drop_gain={r['drop_gain']:+.4f}  "
              f"CI=[{r['drop_ci_low']:+.4f}, {r['drop_ci_high']:+.4f}]  {r['drop_verdict']}")

    out_dir = Path("data/edge_scan") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    write_bar_chart(out_dir, panel, results, args)

    if args.diagnostics:
        reg, topk = run_overfit_diagnostics(panel, args.min_train_weeks)
        reg.to_csv(out_dir / "regularization.csv", index=False)
        topk.to_csv(out_dir / "topk.csv", index=False)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(vars(args) | {"n_games": len(panel), "breakeven": BREAKEVEN,
                                "n_features": len(PRODUCTION_FEATURES)}, f, indent=2)

    print(f"\nSaved -> {out_dir}/feature_importance.png")


if __name__ == "__main__":
    main()
