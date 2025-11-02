import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split


# -------- helpers (clean, with team colors + swapped logos + relabels) --------

import os
from pathlib import Path
from textwrap import shorten
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def _pretty_feat(name: str) -> str:
    """Human-friendly feature label fallback."""
    return name.replace("_", " ").title()


def _add_logo(ax, logo_path: str | Path, xy=(0.01, 0.99), zoom=0.12, align="left"):
    """Place a logo *inside the axes* so it never goes off-page."""
    try:
        p = Path(logo_path)
        if not p.exists():
            return
        img = plt.imread(p)
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            xy,
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0, 1) if align == "left" else (1, 1),
            zorder=10,
            pad=0.0,
        )
        ax.add_artist(ab)
    except Exception:
        pass  # fail silently on any logo issues


def _load_team_colors(json_path: str | Path = "data/logos/team_colors.json") -> dict:
    try:
        with Path(json_path).open("r") as f:
            return json.load(f)
    except Exception:
        return {}  # fallback empty


def save_feature_importance_hbar(
    mean_imp: pd.Series,
    std_imp: pd.Series,
    title: str,
    path: str | Path,
    annotate: bool = True,
    logo_left: str | Path | None = None,
    logo_right: str | Path | None = None,
    left_logo_zoom: float = 0.12,
    right_logo_zoom: float = 0.12,
    label_override: list[str] | None = None,
    bar_colors: list[str] | None = None,
    fixed_order: list[str] | None = None,     # NEW: lock bar order
):
    """
    Clean horizontal bar chart for feature importance.
    If fixed_order is provided, bars follow that order (no value sort).
    """
    # ---- order & prep ----
    if fixed_order is not None:
        # keep only features present and in the given order
        order = [f for f in fixed_order if f in mean_imp.index]
    else:
        order = mean_imp.sort_values(ascending=False).index

    mean_imp = mean_imp.loc[order]
    std_imp = std_imp.reindex(order).fillna(0.0)
    std_vals = np.nan_to_num(std_imp.values, nan=0.0)

    # Labels
    if label_override is not None:
        # label_override is a mapping-by-index; reindex to our 'order'
        label_map = pd.Series(label_override, index=mean_imp.index)
        labels = list(label_map.loc[order].values)
    else:
        labels = [ _pretty_feat(x) for x in mean_imp.index ]

    n = len(mean_imp)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No features to display", ha="center", va="center", fontsize=12)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return

    # ---- layout ----
    fig_h = max(3.2, 0.42 * n + 1.6)
    fig_w = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)

    y = np.arange(n)
    ax.barh(
        y,
        mean_imp.values,
        xerr=std_vals,
        capsize=3,
        height=0.58,
        linewidth=0,
        color=bar_colors if bar_colors is not None else None,
    )
    ax.invert_yaxis()

    # y tick labels (truncate if too long)
    yticklabels = [shorten(lbl, width=38, placeholder="…") for lbl in labels]
    ax.set_yticks(y, yticklabels, fontsize=10)

    # x range/padding
    x_min = float(np.nanmin(mean_imp.values - std_vals))
    x_max = float(np.nanmax(mean_imp.values + std_vals))
    if not np.isfinite(x_min): x_min = 0.0
    if not np.isfinite(x_max): x_max = 1.0
    rng = (x_max - x_min) if x_max > x_min else 1.0
    ax.set_xlim(x_min - 0.08 * rng, x_max + 0.12 * rng)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Feature Importance (Δ loss; higher = more important)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(title, fontsize=18, weight="bold", pad=10)

    # ---- value annotations ----
    if annotate:
        xr0, xr1 = ax.get_xlim()
        xrng = xr1 - xr0
        inside_thresh = 0.10 * xrng
        for i, val in enumerate(mean_imp.values):
            if not np.isfinite(val):
                continue
            txt = f"{val:+.3f}"
            if abs(val) > inside_thresh:
                txt_x = val - np.sign(val) * 0.01 * xrng
                ax.text(
                    txt_x, y[i], txt,
                    va="center",
                    ha="right" if val > 0 else "left",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                txt_x = val + 0.012 * xrng * (1 if val >= 0 else -1)
                ax.text(
                    txt_x, y[i], txt,
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=9,
                )

    # ---- logos ----
    if logo_left:
        _add_logo(ax, logo_left, xy=(0.01, 1.02), zoom=left_logo_zoom, align="left")
    if logo_right:
        _add_logo(ax, logo_right, xy=(0.99, 1.02), zoom=right_logo_zoom, align="right")

    longest = max(len(lbl) for lbl in yticklabels)
    left = min(0.40, 0.18 + 0.007 * longest)
    fig.subplots_adjust(left=left, right=0.97, top=0.90, bottom=0.08)

    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---------- adapter (keeps your old call sites unchanged) ----------

def _resolve_team_logo(team: str) -> str | None:
    """Best-effort logo finder; edit paths/patterns to your repo."""
    candidates = [
        f"data/logos/{team}.png",
        f"../logos/{team}.png",
        f"../../logos/{team}.png"
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _mk_edge_labels(index: pd.Index, away_team: str, home_team: str) -> list[str]:
    """
    Build readable labels:
      - 'away_off_*'  -> '{AWAY} Off ...'
      - 'away_def_*'  -> '{HOME} Def ...'
    """
    out = []
    for feat in index:
        if feat.startswith("away_off_"):
            suffix = feat.replace("away_off_", "")
            label = f"{away_team} " + suffix.replace("_", " ").title()
        elif feat.startswith("away_def_"):
            suffix = feat.replace("away_def_", "")
            label = f"{home_team} " + suffix.replace("_", " ").title()
        else:
            label = _pretty_feat(feat)
        out.append(label)
    return out


def save_matchup_edges_hbar(
    edge_series: pd.Series,
    away_team: str,
    home_team: str,
    path: str,
    feature_order: list[str] | None = None,
    add_logos: bool = True,
    annotate: bool = True,
    pred_value=-2.5
):
    """
    Backwards-compatible wrapper around save_feature_importance_hbar.
    - Swapped logos: HOME on left, AWAY on right
    - Team-colored bars: away color for positive edges, home color for negative
    - y-labels: use team names instead of 'Away Off/Def'
    - Bars ordered by the feature-importance index when provided
    """
    s = edge_series
    if feature_order is not None:
        present = [f for f in feature_order if f in s.index]
        s = s.reindex(present)

    std = pd.Series(0.0, index=s.index)  # no uncertainty for edges

    # Team colors
    team_colors = _load_team_colors()
    away_color = team_colors.get(away_team, "#2b83ba")
    home_color = team_colors.get(home_team, "#d7191c")
    bar_colors = [away_color if v >= 0 else home_color for v in s.values]

    # Labels (team-specific renames)
    labels = _mk_edge_labels(s.index, away_team=away_team, home_team=home_team)

    subtitle = ""
    if pred_value is not None and np.isfinite(pred_value):
        subtitle = f"\nPredicted spread: {pred_value:+.1f}"

    title = f"{away_team} @ {home_team} — Feature Edges (+ Away, − Home){subtitle}"

    # SWAP logos: HOME on left, AWAY on right
    logo_left = _resolve_team_logo(home_team) if add_logos else None
    logo_right = _resolve_team_logo(away_team) if add_logos else None

    save_feature_importance_hbar(
        mean_imp=s,
        std_imp=std,
        title=title,
        path=path,
        annotate=annotate,
        logo_left=logo_left,
        logo_right=logo_right,
        label_override=labels,
        bar_colors=bar_colors,
        fixed_order=list(s.index),  # <— lock order to feature-importance index
    )


def integrated_gradients(model, x_batch: tf.Tensor, baseline: tf.Tensor, m_steps: int = 64):
    """
    Integrated Gradients for a batch.
    Returns:
      attributions: (n, d)
      base_pred:    (n,)
      f_x:          (n,)
    """
    x = tf.cast(x_batch, tf.float32)            # (n, d)
    n = tf.shape(x)[0]
    d = tf.shape(x)[1]

    baseline = tf.cast(baseline, tf.float32)
    if baseline.shape.rank == 2 and baseline.shape[0] == 1:
        baseline_t = tf.repeat(baseline, repeats=n, axis=0)   # (n, d)
    elif baseline.shape.rank == 2 and baseline.shape[0] == n:
        baseline_t = baseline
    else:
        raise ValueError(f"baseline must be shape (1, d) or (n, d); got {baseline.shape}")

    alphas = tf.linspace(0.0, 1.0, m_steps + 1)          # (m+1,)
    alphas = tf.reshape(alphas, (-1, 1, 1))               # (m+1, 1, 1)

    # Build interpolation path: (m+1, n, d)
    path = baseline_t[None, :, :] + alphas * (x[None, :, :] - baseline_t[None, :, :])

    # Flatten path into batch: ((m+1)*n, d)
    path_flat = tf.reshape(path, ((m_steps + 1) * n, d))

    with tf.GradientTape() as tape:
        tape.watch(path_flat)
        preds_flat = model(path_flat, training=False)     # ((m+1)*n, 1)
    grads_flat = tape.gradient(preds_flat, path_flat)     # ((m+1)*n, d)

    # Reshape grads back to (m+1, n, d) and average across the path
    grads = tf.reshape(grads_flat, (m_steps + 1, n, d))
    avg_grads = tf.reduce_mean(grads, axis=0)             # (n, d)

    attributions = (x - baseline_t) * avg_grads           # (n, d)

    base_pred = tf.squeeze(model(baseline_t, training=False), axis=-1).numpy()  # (n,)
    f_x = tf.squeeze(model(x,           training=False), axis=-1).numpy()       # (n,)
    return attributions.numpy(), base_pred, f_x




# -------- model --------

def modelo(data, season, week, tag):
    dat = data.copy()
    dat.loc[:, 'result'] = dat['away_score'] - dat['home_score']

    target = 'result'
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
        "away_off_explosive_run_%", "away_def_explosive_run_%",
        "away_off_explosive_pass_%", "away_def_explosive_pass_%",
        "away_off_stuff_%", "away_def_stuff_%",
        "away_off_sack_%", "away_def_sack_%",
        "away_off_qb_hit_%", "away_def_qb_hit_%"
    ]

    preds = dat[(dat.season == season) & (dat.week == week)].copy()
    train = dat[~((dat.season == season) & (dat.week == week))].copy()
    X, Y = train[features], train[target]

    # custom loss
    def sign_penalty(y_true, y_pred):
        penalty = 1.3
        loss = tf.where(tf.less(y_true * y_pred, 0),
                        penalty * tf.square(y_true - y_pred),
                        tf.square(y_true - y_pred))
        return tf.reduce_mean(loss, axis=-1)
    keras.losses.sign_penalty = sign_penalty

    # model builder
    def create_model():
        m = Sequential()
        m.add(Dropout(0.1))
        m.add(Dense(X.shape[1], input_dim=X.shape[1], activation='elu'))
        m.add(Dense((X.shape[1] + 1) // 2, activation='elu'))
        m.add(Dense((X.shape[1] + 1) // 3, activation='elu'))
        m.add(Dense(1, activation='linear'))
        return m

    # permutation importance
    def permutation_importance(model, X_val, y_val, loss_fn, random_state=42):
        rng = np.random.default_rng(random_state)
        base_pred = model.predict(X_val, verbose=0).reshape(-1)
        base_loss = float(tf.keras.backend.get_value(
            loss_fn(tf.constant(y_val.values, dtype=tf.float32),
                    tf.constant(base_pred, dtype=tf.float32))
        ))
        importances = {}
        X_val_arr = X_val.to_numpy(copy=True)
        for j, col in enumerate(X_val.columns):
            X_perm = X_val_arr.copy()
            rng.shuffle(X_perm[:, j])
            y_perm = model.predict(X_perm, verbose=0).reshape(-1)
            loss = float(tf.keras.backend.get_value(
                loss_fn(tf.constant(y_val.values, dtype=tf.float32),
                        tf.constant(y_perm, dtype=tf.float32))
            ))
            importances[col] = loss - base_loss
        return pd.Series(importances)

    # run
    tf.keras.backend.clear_session()
    X_tr, X_val, y_tr, y_val = train_test_split(X, Y, test_size=0.2, random_state=1337)
    feat_means = X.mean()

    all_imps = []
    all_preds = []
    edges_runs = []                      # collect per-run matchup edges
    X_pred = preds[features].copy()      # fixed prediction design for all runs
    iterations = 100

    for i in range(iterations):
        model = create_model()
        opt = keras.optimizers.Adam(amsgrad=True)
        model.compile(optimizer=opt, loss=sign_penalty)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
        model.fit(X, Y, epochs=100, callbacks=[reduce_lr])

        train_preds = model.predict(X)
        test_preds = model.predict(preds[features])

        r2 = sklearn.metrics.r2_score(Y, train_preds)
        mae = sklearn.metrics.mean_absolute_error(Y, train_preds)
        mse = sklearn.metrics.mean_squared_error(Y, train_preds)
        print(f'\nR2: {r2}\nMAE:{mae}\nMSE:{mse}\n')

        if r2 > 0:
            run_df = preds[['away_team', 'home_team']].copy()
            run_df['prediction'] = np.asarray(test_preds).reshape(-1).astype(float)
            all_preds.append(run_df)

            baseline_vec = tf.constant(feat_means.values[None, :], dtype=tf.float32)  # dataset mean as baseline
            # baseline_vec = tf.zeros((1, X_pred.shape[1]), dtype=tf.float32)
            ig_attr, base_vals, preds_vals = integrated_gradients(
                model,
                tf.constant(X_pred.values, dtype=tf.float32),
                baseline=baseline_vec,
                m_steps=64,
            )

            edges_runs.append(ig_attr)

        imp = permutation_importance(model, X_val, y_val, sign_penalty, random_state=42 + i)
        all_imps.append(imp)

    # feature importance: avg + std; save chart
    imp_df = pd.concat(all_imps, axis=1) if all_imps else pd.DataFrame(index=features)
    avg_imp = imp_df.mean(axis=1).fillna(0.0)
    std_imp = imp_df.std(axis=1).fillna(0.0)

    if all_preds:
        stacked_preds = pd.concat(all_preds, ignore_index=True)
        avg_pred_spread = float(stacked_preds['prediction'].mean())
    else:
        avg_pred_spread = float('nan')

    os.makedirs(tag, exist_ok=True)
    fi_title = (
            f"Aggregated Feature Importance — {season} Week {week} "
            f"(Lookback: {tag.split('_')[-1] if '_' in tag else '—'})\n"
            "Error bars show ±1 std across model runs"
            + (f"   |   Mean predicted spread: {avg_pred_spread:+.1f}" if np.isfinite(avg_pred_spread) else "")
    )
    save_feature_importance_hbar(
        mean_imp=avg_imp,
        std_imp=std_imp,
        title=fi_title,
        path=os.path.join(tag, "feature_importance.png"),
    )

    # aggregate predictions across runs -> mean + variance
    if not all_preds:
        results = preds[['away_team', 'home_team']].copy()
        results['prediction'] = np.nan
        results['variance'] = np.nan
    else:
        stacked = pd.concat(all_preds, ignore_index=True)
        agg = (stacked
               .groupby(['away_team', 'home_team'])['prediction']
               .agg(prediction='mean', variance='var')
               .reset_index())
        agg['variance'] = agg['variance'].fillna(0.0)
        results = preds.merge(agg, on=['away_team', 'home_team'], how='left')

    results['prediction'] = results['prediction'].round(1)

    # holistic edges: average edges across valid runs
    try:
        if len(edges_runs) > 0:
            edges_stack = np.stack(edges_runs, axis=0)     # (runs, games, features)
            avg_edges = edges_stack.mean(axis=0)           # (games, features)
            feature_order = list(avg_imp.sort_values(ascending=False).index)
            for i in range(len(preds)):
                aw = preds.iloc[i]['away_team']
                hm = preds.iloc[i]['home_team']
                ser = pd.Series(avg_edges[i, :], index=features)
                pred_val = float(results.iloc[i]['prediction']) if 'prediction' in results.columns else None
                save_matchup_edges_hbar(
                    edge_series=ser,
                    away_team=aw,
                    home_team=hm,
                    path=os.path.join(tag, f"edge_{aw}_@_{hm}.png"),
                    feature_order=feature_order,
                    add_logos=True,
                    pred_value=pred_val
                )
    except Exception:
        pass

    return results[['away_team', 'home_team', 'prediction', 'variance']]
