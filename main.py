# Go Commies
import numpy as np
import pandas as pd
from datetime import datetime

import utils
from tabulate import tabulate, tabulate_formats
import os
import glob
import data_crunchski
import data_crunchski_2
import data_pullson
from pathlib import Path

# model_shredski and feature_select both import TensorFlow at module level.
# prep_test_train's worker processes re-import this file, so importing them up
# here made every worker load TF (~5s per prep_test_train call) for code it
# never runs. Imported inside the functions that actually need them instead.

import requests
pd.set_option('display.max_columns', None)


def download_team_logos(teams, logo_dir='data/logos'):
    """
    Given a list/array of team abbreviations, download the NFL team logos
    from ESPN if they are not already present in logo_dir.
    """
    if not os.path.exists(logo_dir):
        os.makedirs(logo_dir)
    for team in teams:
        # Construct the local file path.
        logo_path = os.path.join(logo_dir, f'{team}.png')
        if not os.path.exists(logo_path):
            # Construct the ESPN URL for the team logo.
            if team == "WAS":
                url = "https://cdn.freebiesupply.com/images/thumbs/2x/washington-redskins-logo.png"
            else:
                url = f'https://a.espncdn.com/i/teamlogos/nfl/500/{team}.png'
            response = requests.get(url)
            if response.status_code == 200:
                with open(logo_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download logo for {team} from {url}")


def validated_highlights(result, policy, model_spec, asof):
    """Fail closed: a different ensemble or a future validation cannot qualify."""
    empty = pd.Series(False, index=result.index)
    try:
        supported = (bool(model_spec) and policy['model_spec'] == model_spec
                     and policy['status'] == 'PAPER QUALIFIED'
                     and policy['evaluation'] == 'weekly_walk_forward'
                     and tuple(policy['validated_through']) < tuple(asof)
                     and policy['calibration']['n'] >= 60
                     and policy['calibration']['pnl_units'] > 0
                     and policy['validation']['n'] >= 60
                     and policy['validation']['pnl_units'] > 0
                     and policy['validation_roi_95'][0] > 0)
        if not supported:
            return empty
        cutoff = policy['sd_cutoff']
        edge = float(policy['diff_cutoff'])
        if not np.isfinite(edge) or edge < 0 or (cutoff is not None and (not np.isfinite(cutoff) or cutoff < 0)):
            return empty
        selected = result['diff'].abs().ge(edge) & result['diff'].ne(0)
        selected &= np.isfinite(result['diff']) & np.isfinite(result['sd']) & result['sd'].ge(0)
        return selected if cutoff is None else selected & result['sd'].le(cutoff)
    except (KeyError, TypeError, ValueError, IndexError):
        return empty


def h_to_the_tml(pred, season, week, lookback, tag):
    qb = pd.read_parquet(f'data/qb/qb_{season}_{week}_{lookback}.parquet')

    sched = pd.read_parquet('data/sched.parquet')
    sched = sched[(sched.season == season) & (sched.week == week)]

    # pull_odds() only ever returns UPCOMING games, and the merges below join on
    # team names alone -- no season/week. For a week that has already been played
    # that silently pairs some future rematch's line with this week's prediction
    # (e.g. a Dec 2026 BUF@DEN line against a Jan 2026 divisional prediction).
    # So for a finished week, use the stored closing line instead: the same
    # spread_line back_test() grades against. Sign convention matches -- the
    # odds-api away-side point and nflverse spread_line are both the expected
    # home margin. Column order here must match pull_odds() because the 'var'
    # rename below is positional.
    if len(sched) and sched['result'].notna().all():
        lines = pd.DataFrame({
            'date': pd.to_datetime(sched['gameday']).dt.date,
            'away_team': sched['away_team'],
            'spread': sched['spread_line'],
            'home_team': sched['home_team'],
            'total': sched['total_line'],
        }).reset_index(drop=True)
        print(f"  lines     stored closing lines ({season} wk{week} already played)")
    else:
        lines = data_pullson.pull_odds()

    sched.loc[:, 'away_qb'] = sched['away_qb_name'].apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")
    sched.loc[:, 'home_qb'] = sched['home_qb_name'].apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")
    sched['away_qb'] = sched['away_qb'].apply(utils.strip_suffix)
    sched['home_qb'] = sched['home_qb'].apply(utils.strip_suffix)

    sched = pd.merge(sched, qb[['name', 'weighted_qb_elo']], left_on='away_qb', right_on='name', how='left').rename(
        columns={'weighted_qb_elo': 'away_qb_elo'}
    ).drop(columns='name')
    sched = pd.merge(sched, qb[['name', 'weighted_qb_elo']], left_on='home_qb', right_on='name', how='left').rename(
        columns={'weighted_qb_elo': 'home_qb_elo'}
    ).drop(columns='name')

    sched = sched[[
        'away_team', 'home_team', 'gameday', 'gametime', 'away_qb_elo', 'home_qb_elo', 'away_qb', 'home_qb'
    ]]
    pred.columns = pred.columns.get_level_values(0)

    result = pd.merge(lines, pred, on=['away_team', 'home_team'], how='left')
    result = pd.merge(result, sched, on=['away_team', 'home_team'], how='left')

    def rename_col_by_index(dataframe, index_mapping):
        dataframe.columns = [index_mapping.get(i, col) for i, col in enumerate(dataframe.columns)]
        return dataframe

    # Renaming columns using the function.
    new_column_mapping = {6: 'var'}
    result = rename_col_by_index(result, new_column_mapping)

    result['diff'] = result['prediction'] + result['spread']
    result['diff_abs'] = abs(result['prediction'] + result['spread'])

    def pick_func(x):
        if x['diff'] < 0:
            return x['home_team']
        elif x['diff'] > 0:
            return x['away_team']
        else:
            return None

    result['pick'] = result.apply(pick_func, axis=1)
    result['sd'] = np.sqrt(result['var'].where(result['var'] >= 0))
    import json
    try:
        policy = json.loads(Path('data/optimize_picks/current/neural_summary.json').read_text()).get('spread', {})
    except (OSError, ValueError):
        policy = {}
    selected = validated_highlights(result, policy, pred.attrs.get('model_spec'), (season, week))
    picks = set(result.loc[selected, 'pick'])
    print(f'  highlights {len(picks)} picks supported by matching-model walk-forward validation')

    # Adding logos
    all_teams = pd.concat([result['away_team'], result['home_team']]).unique()
    download_team_logos(all_teams, logo_dir='data/logos')

    result['away_logo'] = result['away_team'].apply(lambda team: f'../../logos/{team}.png')
    result['home_logo'] = result['home_team'].apply(lambda team: f'../../logos/{team}.png')

    result = result[['gameday', 'gametime',
                     'away_qb', 'away_qb_elo', 'away_logo', 'away_team',
                     'spread', 'prediction',
                     'home_team', 'home_logo', 'home_qb', 'home_qb_elo',
                     'diff_abs', 'sd', 'pick']]

    result['gameday'] = pd.to_datetime(result['gameday'])
    # gametime is always 'HH:MM'; naming the format skips the dateutil fallback
    result['gametime'] = pd.to_datetime(result['gametime'], format='%H:%M').dt.time

    result = result.sort_values(by=['gameday', 'gametime', 'away_team'])
    result = result.dropna(subset=['prediction'])
    result['prediction'] = result['prediction'] * -1
    result = result.round(1).rename(columns={'diff_abs': 'diff'})

    if not os.path.exists('data/results'):
        os.makedirs('data/results')
    result.to_csv(f'{tag}/results_{season}_{week}_{lookback}.csv')

    def style_fonts_and_borders(val):
        return 'font-size: 16px; font-family: Arial; border: 2px solid gray'

    def format_bold(x):
        return 'font-weight: bold'

    def set_precision(val, precision):
        return f'{val:.{precision}f}'

    def style_spread(val, precision):
        if val > 0:
            return f'+{val:.{precision}f}'
        else:
            return f'{val:.{precision}f}'

    def highlight_picks(x):
        return 'background-color: #ffe590' if x in picks else ''

    result = result.reset_index(drop=True)

    # Create the HTML representation with styling.
    # Note the format functions for the logo columns:
    # we wrap the local file path in an <img> tag.
    html = (result.style
            .background_gradient(subset=['diff'], cmap='Greens')
            .background_gradient(subset=['sd'], cmap='Reds')
            .map(style_fonts_and_borders)
            .format({
        'away_qb_elo': lambda x: set_precision(x, precision=1),
        'home_qb_elo': lambda x: set_precision(x, precision=1),
        'gameday': lambda x: x.strftime('%a %m/%d'),
        'gametime': lambda x: x.strftime("%I:%M %p").lstrip('0'),
        'spread': lambda x: style_spread(x, precision=1),
        'prediction': lambda x: style_spread(x, precision=1),
        'diff': lambda x: set_precision(x, precision=1),
        'sd': lambda x: set_precision(x, precision=1),
        # Wrap the logo file path in an <img> tag.
        'away_logo': lambda x: f'<img src="{x}" alt="Away Logo" height="25">' if pd.notnull(x) else '',
        'home_logo': lambda x: f'<img src="{x}" alt="Home Logo" height="25">' if pd.notnull(x) else ''
    })
            .map(highlight_picks, subset=['away_team', 'home_team', 'pick'])
            )
    # IMPORTANT: disable escaping so that the <img> tags render as images.
    html.to_html(f'{tag}/html_{season}_{week}_{lookback}.html', escape=False)

    print(tabulate(result, headers='keys'))

def pull_bt(hist_reach: int, lookback: int):
    import model_shredski  # local: keeps TF out of prep_test_train's workers
    sched = pd.read_parquet('data/sched.parquet')
    sched.dropna(subset=['result'], inplace=True)
    sw_pairs = (
        sched[sched['game_type'] == 'REG']
        .groupby(['season', 'week'])
        .size()
        .index
        .tolist()
    )
    sw_pairs.reverse()  # now: [(latest_season, latest_week), ... older...]
    if hist_reach and hist_reach > 0:
        run_list = sw_pairs[:hist_reach]
    else:
        run_list = sw_pairs  # run everything
    if not run_list:
        print("No (season, week) pairs to run.")
        return
    out_dir = Path(f"data/bt/{lookback}")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(run_list)
    print(f"Backtest starting | lookback={lookback} | total weeks={total}")
    ok = 0
    fail = 0
    for idx, (season, week) in enumerate(run_list, start=1):
        print(f"[{idx:>3}/{total}] S{season} W{week} …", end=" ")
        try:
            dat_path = Path(f"data/stats/dat_{season}_{week}_{lookback}.parquet")
            try:
                # If you want to re-use cached data when present, uncomment these two lines:
                if dat_path.exists():
                    df = pd.read_parquet(dat_path)
                else:
                    df = data_crunchski_2.prep_test_train(season, week, lookback)
                    df.to_parquet(dat_path)
            except Exception as e:
                fail += 1
                print(f"[DATA FAIL] {e}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                tag = f"data/results/{season}_{week}_{lookback}"
                pred = model_shredski.modelo(df, season, week, tag, bt=True).round(1)
                if hasattr(pred.columns, "nlevels") and pred.columns.nlevels > 1:
                    pred.columns = pred.columns.get_level_values(0)
                csv_path = out_dir / f"bt_{season}_{week}_{lookback}.csv"
                pred.to_csv(csv_path, index=False)
                if 'prediction' in pred.columns:
                    mean_spread = float(pred['prediction'].mean())
                    print(f"[OK] → {csv_path.name} | mean spread {mean_spread:+.2f}")
                else:
                    print(f"[OK] → {csv_path.name}")
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[MODEL FAIL] {e}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] Unexpected error: {e}")

    print(f"Backtest done | OK={ok} FAIL={fail} | lookback={lookback}")

def back_test(bt: int,
              *,
              odds_line: int = -110,
              seed: int = 42,
              bootstrap_n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        _HAS_SK = True
    except Exception:
        _HAS_SK = False

    def _american_to_decimal(american: int) -> float:
        if american > 0:
            return american / 100.0
        return 100.0 / abs(american)

    def _kelly_fraction(p: float, american: int) -> float:
        b = _american_to_decimal(american)
        q = 1.0 - p
        f = (b * p - q) / b
        return max(0.0, f)

    def _bootstrap_ci(successes: np.ndarray, n: np.ndarray, reps: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
        if n.sum() == 0:
            return (np.nan, np.nan)
        idx = rng.integers(0, len(successes), size=(reps, len(successes)))
        wins = successes[idx]
        den  = n[idx]
        wr = np.divide(wins.sum(axis=1), np.maximum(den.sum(axis=1), 1), where=True)
        lo, hi = np.nanquantile(wr, [alpha/2, 1 - alpha/2])
        return float(lo), float(hi)

    def _bucket_win_table(df: pd.DataFrame, col: str, bins, labels=None, name: str = "") -> pd.DataFrame:
        x = df.copy()
        x["bucket"] = pd.cut(x[col], bins=bins, labels=labels, include_lowest=True, right=True)
        g = x.groupby("bucket", observed=True)["dinner"].agg(wins="sum", games="count")
        g["win_rate"] = g["wins"] / g["games"].replace(0, np.nan)

        rows = []
        for bkt, sub in x.groupby("bucket", observed=True):
            s = sub["dinner"].astype(int).values
            n = np.ones_like(s)
            lo, hi = _bootstrap_ci(s, n, reps=bootstrap_n)
            rows.append((bkt, lo, hi))
        cis = pd.DataFrame(rows, columns=["bucket", "lo", "hi"]).set_index("bucket")

        out = g.join(cis, how="left").reset_index()
        out.columns = [name or "bucket", "wins", "games", "win_rate", "ci_low", "ci_high"]
        return out

    def _spearman_monotonicity(df: pd.DataFrame, bucket_col: str = "bucket", rate_col: str = "win_rate") -> float:
        from scipy.stats import spearmanr
        y = df[rate_col].astype(float).values
        ord_idx = np.arange(len(df))
        rho, pval = spearmanr(ord_idx, y, nan_policy="omit")
        return float(rho), float(pval)

    files = []
    root = f"data/bt/{bt}"
    for fname in os.listdir(root):
        parts = fname.split("_")
        try:
            if len(parts) >= 4 and parts[3][:2] == str(bt):
                tmp = pd.read_csv(f"{root}/{fname}")
                tmp["week"] = int(parts[2])
                tmp["season"] = int(parts[1])
                files.append(tmp)
        except Exception as e:
            print(f"[WARN] {fname}: {e}")
    if not files:
        raise RuntimeError(f"No CSVs found in {root} for bt={bt}")

    bt_df = pd.concat(files, ignore_index=True)
    sched = pd.read_parquet("data/sched.parquet")
    bt_df = pd.merge(sched, bt_df, how="left",
                     on=["week", "season", "away_team", "home_team"])
    bt_df = bt_df.dropna(subset=["prediction", "result"]).copy()

    bt_df["prediction"] = bt_df["prediction"] * -1
    bt_df["abs_pred"] = bt_df["prediction"].abs()
    bt_df["diff"] = bt_df["spread_line"] - bt_df["prediction"]
    bt_df["diff_abs"] = bt_df["diff"].abs()

    def _pick(x):
        if x["diff"] < 0:
            return x["home_team"]
        elif x["diff"] > 0:
            return x["away_team"]
        else:
            return None

    def _winner(x):
        if x["result"] < x["spread_line"]:
            return x["away_team"]
        elif x["result"] > x["spread_line"]:
            return x["home_team"]
        else:
            return None

    bt_df["pick"] = bt_df.apply(_pick, axis=1)
    bt_df["winner"] = bt_df.apply(_winner, axis=1)
    bt_df["dinner"] = (bt_df["pick"] == bt_df["winner"]).astype(int)

    def _sign(z):
        return 0 if z == 0 else (1 if z > 0 else -1)
    bt_df["switcherooni"] = (
        bt_df["prediction"].apply(_sign) != bt_df["spread_line"].apply(_sign)
    ).astype(int)

    bt_df["abs_spread"] = bt_df["spread_line"].abs()
    bt_df = bt_df[bt_df["season"] == 2025].copy()

    out_agg = f"data/bt/{bt}/bt_{bt}_agg.csv"
    try:
        bt_df.to_csv(out_agg, index=False)
        print(f"[INFO] wrote {out_agg}")
    except Exception as e:
        print(f"[WARN] could not write {out_agg}: {e}")

    eval_df = bt_df.dropna(subset=["winner"]).copy()
    n_games = len(eval_df)
    if n_games == 0:
        print("[INFO] No evaluable games (no winners).")
        return bt_df

    print(f"\nTotal evaluable games: {n_games}")
    print(f"Overall hit rate: {eval_df['dinner'].mean():.3f}")

    edge_tbl = _bucket_win_table(
        eval_df,
        col="diff_abs",
        bins=[0, 1, 2, 3, 5, 10_000],
        labels=["<1", "1-2", "2-3", "3-5", "5+"],
        name="edge(|model-vegas|)"
    )
    print("\nWin rate by |model - vegas| edge (bootstrapped 95% CI):")
    print(edge_tbl.to_string(index=False))
    try:
        rho, p = _spearman_monotonicity(edge_tbl)
        print(f"Monotonicity (Spearman rho): {rho:.3f}, p={p:.3f}")
    except Exception:
        pass

    eval_df["var_bucket"] = pd.qcut(eval_df["variance"], 4, labels=["low","med-low","med-high","high"])
    var_tbl = _bucket_win_table(
        eval_df.rename(columns={"var_bucket": "bucket"}),
        col="variance",
        bins=list(np.quantile(eval_df["variance"], [0, .25, .5, .75, 1.0])),
        labels=None,
        name="variance_bin"
    )
    var_tbl["variance_bin"] = ["low","med-low","med-high","high"]
    print("\nWin rate by prediction variance (quartiles, bootstrapped 95% CI):")
    print(var_tbl.to_string(index=False))

    had_model = False
    if _HAS_SK:
        try:
            X = eval_df[["diff_abs", "variance", "switcherooni"]].copy()
            X["diff_switch"] = X["diff_abs"] * X["switcherooni"]

            def _zs(s):
                v = s.std(ddof=0)
                return (s - s.mean()) / (v if v > 0 else 1)
            X["diff_abs_z"] = _zs(X["diff_abs"])
            X["variance_z"] = _zs(np.sqrt(X["variance"].clip(lower=0)))
            X["diff_switch_z"] = _zs(X["diff_switch"])
            Xm = X[["diff_abs_z", "variance_z", "switcherooni", "diff_switch_z"]]

            y = eval_df["dinner"].astype(int)

            lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            lr.fit(Xm, y)
            eval_df["prob_raw"] = lr.predict_proba(Xm)[:, 1]

            iso = IsotonicRegression(out_of_bounds="clip")
            eval_df["prob_win"] = iso.fit_transform(eval_df["prob_raw"], y)

            coef = pd.Series(lr.coef_[0], index=Xm.columns).to_frame("coef")
            print("\nLogit coefficients (z-scored features; higher ⇒ more confident):")
            print(coef.to_string())

            eval_df["p_bucket"] = pd.cut(eval_df["prob_win"], bins=[0,.5,.6,.7,.8,.9,1.0],
                                         labels=["≤0.5","0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9",">0.9"],
                                         include_lowest=True, right=True)
            calib_tbl = (
                eval_df.groupby("p_bucket", observed=True)["dinner"]
                .agg(avg_win="mean", games="count")
                .reset_index()
            )
            print("\nCalibration by calibrated probability bins:")
            print(calib_tbl.to_string(index=False))

            had_model = True

            eval_df["conf_score"] = (
                lr.coef_[0][Xm.columns.get_loc("diff_abs_z")] * Xm["diff_abs_z"] +
                lr.coef_[0][Xm.columns.get_loc("variance_z")] * Xm["variance_z"] +
                lr.coef_[0][Xm.columns.get_loc("switcherooni")] * Xm["switcherooni"] +
                lr.coef_[0][Xm.columns.get_loc("diff_switch_z")] * Xm["diff_switch_z"]
            )
            eval_df["conf_bucket"] = pd.qcut(eval_df["conf_score"], 5, labels=["low","med-low","med","med-high","high"])
            conf_tbl = (
                eval_df.groupby("conf_bucket", observed=True)["dinner"]
                .agg(wins="sum", games="count", win_rate="mean")
                .reset_index()
            )
            print("\nWin rate by learned confidence (quintiles):")
            print(conf_tbl.to_string(index=False))

        except Exception as e:
            print(f"\n[WARN] Modeling/calibration skipped: {e}")

    else:
        print("\n[INFO] sklearn not available; skipping logistic + isotonic calibration.")

    switch_tbl = (
        eval_df.groupby("switcherooni", observed=True)["dinner"]
        .agg(wins="sum", games="count", win_rate="mean")
        .reset_index()
        .rename(columns={"switcherooni": "switch"})
    )
    print("\nSwitcherooni win rate (0=same side, 1=disagree):")
    print(switch_tbl.to_string(index=False))

    if had_model:
        p_use = eval_df["prob_win"].clip(0.01, 0.99)
    else:
        p_use = pd.Series(eval_df["dinner"].mean(), index=eval_df.index)

    kelly_frac = p_use.apply(lambda p: _kelly_fraction(p, odds_line)).clip(0, 0.05)  # cap at 5%
    picks = eval_df[[
        "season","week","away_team","home_team","spread_line","prediction","variance",
        "diff","diff_abs","switcherooni","pick","winner","dinner"
    ]].copy()
    picks["prob_win"] = p_use.values
    picks["kelly_f"]  = kelly_frac.values
    picks = picks.sort_values(["prob_win","diff_abs"], ascending=[False, False])

    out_picks = f"data/bt/{bt}/bt_{bt}_picks.csv"
    try:
        picks.to_csv(out_picks, index=False)
        print(f"\n[INFO] wrote ranked picks to {out_picks}")
    except Exception as e:
        print(f"[WARN] could not write picks: {e}")

    keep_cols = ["season","week","away_team","home_team","diff_abs","variance","switcherooni",
                 "dinner","pick","winner"]
    if "prob_win" in eval_df.columns:
        keep_cols += ["prob_win"]
    if "conf_score" in eval_df.columns:
        keep_cols += ["conf_score"]
    bt_df = bt_df.merge(eval_df[keep_cols], how="left",
                        on=["season","week","away_team","home_team"])
    return bt_df


def run_fs(data: pd.DataFrame, season: int, week: int):
    from feature_select import run_feature_selection, FSConfig  # local: TF import
    df = data.copy()
    df["result"] = df["away_score"] - df["home_score"]

    features = [
        "away_off_run_ypp","away_def_run_ypp",
        "away_off_pass_ypp","away_def_pass_ypp",
        "away_off_pass_completion_%","away_def_pass_completion_%",
        "away_off_series_success_%","away_def_series_success_%",
        "away_off_first_down_pp","away_def_first_down_pp",
        "away_off_third_down_%","away_def_third_down_%",
        "away_off_fourth_down_%","away_def_fourth_down_%",
        "away_off_turnovers_pp","away_def_turnovers_pp",
        "away_off_penalties_pp","away_def_penalties_pp",
        "away_off_qb_elo","away_def_qb_elo",
        "away_off_explosive_run_%","away_def_explosive_run_%",
        "away_off_explosive_pass_%","away_def_explosive_pass_%",
        "away_off_stuff_%","away_def_stuff_%",
        "away_off_sack_%","away_def_sack_%",
        "away_off_qb_hit_%","away_def_qb_hit_%",
        "away_rest","home_field_adv"
    ]

    train = (
        df[~((df.season == season) & (df.week == week))]
        .dropna(subset=features + ["result"])
        .copy()
    )

    cfg = FSConfig(
        cv_mode="walk",                # use ("season","week") order
        n_splits=5,
        order_cols=("season", "week"),
        repeats_perm=5,
        random_state=1337,
        corr_threshold=0.95,
        ablate_top_k=10,
        ablation_patience=1,
        epochs=120,
        batch_size=64,
        epoch_log_every=5,             # see progress every 5 epochs
        verbose_fit=0,                 # we print via EpochLogger
        use_mixed_precision=False,
        use_xla=False,
        tag=f"{season}_wk{week}",
    )

    out = run_feature_selection(train, features, target="result", cfg=cfg)
    print("\n[FS] Suggested keep list:\n", out["keep_list"])
    print("[FS] Plots:",
          "\n  Perm:", out["plots"]["perm_plot"],
          "\n  MI  :", out["plots"]["mi_plot"],
          "\n  Abl :", out["ablation"]["plot"])
    return out


def run(season, week, lookback, bt=False):
    import model_shredski  # local: keeps TF out of prep_test_train's workers
    from optimize_picks import PRODUCTION_FEATURES, neural_spec
    tag = f"data/results/{season}_{week}_{lookback}"

    df = data_crunchski_2.prep_test_train(season, week, lookback, history_weeks=lookback)

    df = data_crunchski_2.additional_features(df)
    # Match the packet/research row order so a fixed seed means the same fit.
    df = df.sort_values(['season', 'week', 'away_team', 'home_team']).reset_index(drop=True)

    if not os.path.exists(f'{tag}'): os.makedirs(tag, exist_ok=True)
    df.to_parquet(f'{tag}/dat_{season}_{week}_{lookback}.parquet')

    features = [f for f in PRODUCTION_FEATURES if f != 'away_game_importance']
    pred = model_shredski.modelo(df, season, week, tag, bt=bt, random_state=1337,
                                features=features, round_predictions=False)
    pred.attrs['model_spec'] = neural_spec(features, lookback, 'mean')
    pred.attrs['model_spec']['train_weeks'] = lookback
    return pred


if __name__ == '__main__':
    import argparse
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'research':
        from optimize_picks import main
        main(sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description='Existing weekly model, with optional research packets.')
        parser.add_argument('--season', type=int, default=2026)
        parser.add_argument('--week', type=int, default=1)
        parser.add_argument('--lookback', type=int, default=20)
        parser.add_argument('--packet', action='store_true', help='Spread and total packet with cached neural explanations')
        parser.add_argument('--refresh', action='store_true', help='Explicitly refresh the requested season only')
        args = parser.parse_args()
        if args.refresh:
            data_pullson.pull_sched([args.season])
            data_pullson.pull_pbp([args.season])
        if args.packet:
            from weekly_packet import neural_packet
            pred = neural_packet(args.season, args.week, args.lookback)
        else:
            pred = run(args.season, args.week, args.lookback, bt=True)
        if pred is not None:
            utils.pdf(pred.assign(sd=np.sqrt(pred.variance)).drop(columns='variance').round(1))
            h_to_the_tml(pred, args.season, args.week, args.lookback,
                         tag=f'data/results/{args.season}_{args.week}_{args.lookback}')
