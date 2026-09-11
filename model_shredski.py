# These MUST be set before tensorflow is imported -- TF reads them once, at
# import time. They used to sit below the import, which is why the CUDA /
# oneDNN / TF-TRT banners kept printing.
#   0 = all, 1 = no INFO, 2 = no INFO+WARNING, 3 = no INFO+WARNING+ERROR
# 3 is needed because the "Unable to register cuFFT/cuDNN/cuBLAS factory"
# lines are logged at ERROR level. They are harmless duplicate-registration
# notices, not real failures. Drop to "2" if you ever need to see TF errors.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pathlib import Path
from textwrap import shorten
import json

import sklearn
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed, cpu_count, parallel_config
from modelo_workers import initialize_worker, train_iteration

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import initializers
import tensorflow as tf
tf.get_logger().setLevel('ERROR')                  # silence python-side tf logging
tf.keras.mixed_precision.set_global_policy('float32')

import sys
import contextlib
from tqdm import tqdm


@contextlib.contextmanager
def _quiet_stderr():
    """Silence stderr at the file-descriptor level.

    The CUDA/XLA chatter ("could not open file to read NUMA node", "XLA service
    initialized", "Compiled cluster using XLA!") comes from C++ absl logging
    that writes to stderr *before* absl::InitializeLog() runs. No env var or
    python logger setting can filter it -- only an fd-level redirect.
    """
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


_TF_WARMED = False


def _warm_up_tf():
    """Trigger TF's one-time GPU/XLA init with stderr muted, so the banners are
    swallowed once here instead of appearing mid-run. Everything after this runs
    with stderr fully live, so real errors are never hidden."""
    global _TF_WARMED
    if _TF_WARMED:
        return
    _TF_WARMED = True
    try:
        with _quiet_stderr():
            w = Sequential([Input(shape=(1,)), Dense(1)])
            w.compile(optimizer='adam', loss='mse')
            z = np.zeros((2, 1), dtype='float32')
            w.fit(z, np.zeros(2, dtype='float32'), epochs=1, verbose=0)
            w.predict(z, verbose=0)
            del w
            tf.keras.backend.clear_session()
    except Exception:
        pass  # warm-up is cosmetic only; never let it break a run


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
    avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2, axis=0)

    attributions = (x - baseline_t) * avg_grads           # (n, d)

    base_pred = tf.squeeze(model(baseline_t, training=False), axis=-1).numpy()  # (n,)
    f_x = tf.squeeze(model(x,           training=False), axis=-1).numpy()       # (n,)
    return attributions.numpy(), base_pred, f_x


# -------- model --------

def sign_penalty(y_true, y_pred):
    loss = tf.where(tf.less(y_true * y_pred, 0),
                    1.3 * tf.square(y_true - y_pred),
                    tf.square(y_true - y_pred))
    return tf.reduce_mean(loss, axis=-1)


def create_model(n_features: int):
    return Sequential([
        Input(shape=(n_features,)),
        Dropout(0.10),
        Dense(n_features, activation="elu", kernel_initializer=initializers.HeNormal()),
        Dense((n_features + 1) // 2, activation="elu", kernel_initializer=initializers.HeNormal()),
        Dense(max(1, (n_features + 1) // 3), activation="elu", kernel_initializer=initializers.HeNormal()),
        Dense(1, activation="linear"),
    ])


def permutation_importance(model, X_val, y_val, loss_fn=sign_penalty, random_state=42):
    """Evaluate all feature shuffles together, preserving the old RNG order."""
    values = np.asarray(X_val, dtype=np.float32)
    rng = np.random.default_rng(random_state)
    batches = [values]
    for j in range(values.shape[1]):
        shuffled = values.copy()
        rng.shuffle(shuffled[:, j])
        batches.append(shuffled)
    predictions = np.asarray(model(np.concatenate(batches), training=False)).reshape(
        len(batches), len(values))
    target = tf.constant(np.asarray(y_val), dtype=tf.float32)
    losses = [float(loss_fn(target, tf.constant(p))) for p in predictions]
    return np.asarray(losses[1:]) - losses[0]


def squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


def _train_iteration(i, X, Y, X_pred, X_val, y_val, baseline, bt, epochs, device, random_state, market='spread'):
    """One independent model; return only small arrays, never TensorFlow state."""
    with tf.device('/CPU:0' if device == 'cpu' else '/GPU:0'):
        tf.keras.backend.clear_session()
        if random_state is not None:
            keras.utils.set_random_seed(random_state + i)
        model = create_model(X.shape[1])
        try:
            loss_fn = sign_penalty if market == 'spread' else squared_error
            model.compile(optimizer=keras.optimizers.Adam(amsgrad=True), loss=loss_fn)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
            model.fit(X, Y, epochs=epochs, callbacks=[reduce_lr], verbose=0)
            test_preds = np.asarray(model(X_pred, training=False)).reshape(-1)
            if not np.isfinite(test_preds).all():
                raise ValueError(f'Model iteration {i + 1} produced non-finite predictions')
            imp = edges = metrics = base_pred = None
            if not bt:
                train_preds = np.asarray(model(X, training=False)).reshape(-1)
                metrics = (
                    sklearn.metrics.r2_score(Y, train_preds),
                    sklearn.metrics.mean_absolute_error(Y, train_preds),
                    sklearn.metrics.mean_squared_error(Y, train_preds),
                )
                edges, base_pred, _ = integrated_gradients(
                    model, tf.constant(X_pred), tf.constant(baseline), m_steps=64)
                imp = permutation_importance(model, X_val, y_val, loss_fn=loss_fn, random_state=42 + i)
                if not np.isfinite(edges).all() or not np.isfinite(imp).all():
                    raise ValueError(f'Model iteration {i + 1} produced non-finite explanations')
            return test_preds, imp, edges, metrics, base_pred
        finally:
            del model
            tf.keras.backend.clear_session()


def modelo(data, season, week, tag, bt: bool = False, *, n_jobs=None,
           iterations=100, epochs=100, device='cpu', random_state=None,
           features=None, market='spread', round_predictions=True):
    """Train the ensemble on up to 8 CPU workers by default.

    Override worker count with n_jobs or NFL_MODEL_JOBS; n_jobs=1 runs locally.
    device='gpu' runs sequentially. Ensemble size, epochs, architecture and loss
    retain their existing defaults. random_state optionally seeds each model.
    """
    if iterations < 1 or epochs < 1:
        raise ValueError('iterations and epochs must be positive')
    if device not in ('cpu', 'gpu'):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if market not in ('spread', 'total'):
        raise ValueError("market must be 'spread' or 'total'")
    if n_jobs is None:
        n_jobs = 1 if device == 'gpu' else int(os.environ.get('NFL_MODEL_JOBS', min(8, cpu_count())))
    if n_jobs < 1:
        raise ValueError('n_jobs must be positive')
    if device == 'gpu' and n_jobs != 1:
        raise ValueError("device='gpu' requires n_jobs=1")
    n_jobs = min(n_jobs, iterations)
    tag = str(tag)
    dat = data.copy()
    dat.loc[:, 'result'] = (dat['away_score'] - dat['home_score'] if market == 'spread'
                            else dat['away_score'] + dat['home_score'])

    target = 'result'
    features = features or ([c for c in dat if c.startswith('total_') and c != 'total_line'] if market == 'total' else [
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
        "away_off_qb_hit_%", "away_def_qb_hit_%",
        "away_rest_adv", "home_field_adv"
    ])

    preds = dat[(dat.season == season) & (dat.week == week)].copy()
    train = dat[(dat.season < season) | ((dat.season == season) & (dat.week < week))].copy()
    X, Y = train[features], train[target]
    if preds.empty:
        raise ValueError(f'No prediction rows for {season} week {week}')
    if len(train) < 2:
        raise ValueError('At least two training rows are required')
    if market == 'total':
        # A rookie QB can have no prior Elo. Fit fill values on training only.
        fill = X.replace([np.inf, -np.inf], np.nan).median().fillna(0)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(fill)
        preds[features] = preds[features].replace([np.inf, -np.inf], np.nan).fillna(fill)
    # Fail before starting workers: an unscored training game poisons all weights.
    for name, values in [('training features', X), ('training targets', Y),
                         ('prediction features', preds[features])]:
        if not np.isfinite(np.asarray(values, dtype=np.float32)).all():
            raise ValueError(f'Non-finite {name}; use data prepared for {season} week {week} '
                             'with completed training games and finite features')
    baseline = X.mean().to_numpy(dtype=np.float32)[None, :]
    X = X.to_numpy(dtype=np.float32)
    Y = Y.to_numpy(dtype=np.float32)
    X_val = y_val = None
    if not bt:
        _, X_val, _, y_val = train_test_split(X, Y, test_size=0.2, random_state=1337)

    all_imps = [] if not bt else None
    all_preds = []
    edges_runs = [] if not bt else None
    X_pred = preds[features].to_numpy(dtype=np.float32)
    target_offset = float(Y.mean()) if market == 'total' else 0.0
    if market == 'total':
        # Raw scoring levels need train-only scaling; center points for stable fitting.
        center, scale = X.mean(axis=0), X.std(axis=0)
        scale[scale < 1e-6] = 1
        X, X_pred = (X - center) / scale, (X_pred - center) / scale
        baseline = (baseline - center) / scale
        Y = Y - target_offset
        if not bt:
            X_val, y_val = (X_val - center) / scale, y_val - target_offset
    baseline_runs = []

    # Each process owns its TF runtime; never fork an initialized runtime or
    # share Keras models across threads. Limit native thread pools per worker.
    with parallel_config(backend='loky', inner_max_num_threads=1):
        with Parallel(n_jobs=n_jobs, return_as='generator', batch_size=1,
                      initializer=initialize_worker) as pool:
            if n_jobs == 1:
                if device == 'gpu':
                    _warm_up_tf()
                runs = (_train_iteration(i, X, Y, X_pred, X_val, y_val, baseline,
                                         bt, epochs, device, random_state, market)
                        for i in range(iterations))
            else:
                runs = pool(delayed(train_iteration)(
                    i, X, Y, X_pred, X_val, y_val, baseline, bt, epochs, device, random_state, market)
                    for i in range(iterations))
            with tqdm(runs, total=iterations,
                      desc=f'Training ensemble ({season} wk{week}, {n_jobs} {device} workers)') as progress:
                for test_preds, imp, edges, metrics, base_pred in progress:
                    run_df = preds[['away_team', 'home_team']].copy()
                    run_df['prediction'] = test_preds.astype(float) + target_offset
                    all_preds.append(run_df)
                    if not bt:
                        all_imps.append(pd.Series(imp, index=features))
                        edges_runs.append(edges)
                        baseline_runs.append(base_pred + target_offset)
                        progress.set_postfix(r2=f'{metrics[0]:.3f}', mae=f'{metrics[1]:.3f}',
                                             mse=f'{metrics[2]:.3f}', refresh=False)

    # Global Feature Importance chart (only when bt=False)
    if not bt:
        imp_df = pd.concat(all_imps, axis=1) if all_imps else pd.DataFrame(index=features)
        avg_imp = imp_df.mean(axis=1).fillna(0.0)
        std_imp = imp_df.std(axis=1).fillna(0.0)

        # Mean predicted spread across runs/games
        if all_preds:
            stacked_preds = pd.concat(all_preds, ignore_index=True)
            avg_pred_spread = float(stacked_preds['prediction'].mean())
        else:
            avg_pred_spread = float('nan')

        os.makedirs(tag, exist_ok=True)
        fi_title = (
            f"Training-sample importance (not out-of-sample evidence) — {season} Week {week} "
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

    # Aggregate predictions across runs -> mean + variance
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

    if not bt:
        details = results[['away_team', 'home_team', 'prediction', 'variance']].copy()
        details['baseline'] = np.mean(baseline_runs, axis=0)
        for j, feature in enumerate(features):
            details['attr_' + feature] = np.mean(edges_runs, axis=0)[:, j]
        details['integration_residual'] = details.prediction - details.baseline - details.filter(like='attr_').sum(axis=1)
        details.to_csv(os.path.join(tag, 'explanations.csv'), index=False)
        pd.DataFrame({'feature': features, 'importance': avg_imp.reindex(features).values,
                      'std': std_imp.reindex(features).values}).to_csv(os.path.join(tag, 'importance.csv'), index=False)
    if round_predictions:
        results['prediction'] = results['prediction'].round(1)

    # Edge charts per game (only when bt=False)
    if not bt:
        try:
            if edges_runs:
                edges_stack = np.stack(edges_runs, axis=0)     # (runs, games, features)
                avg_edges = edges_stack.mean(axis=0)           # (games, features)

                # Order features by global importance if we computed it; else use original order
                if 'avg_imp' in locals():
                    feature_order = list(avg_imp.sort_values(ascending=False).index)
                else:
                    feature_order = features

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


if __name__ == "__main__":
    None
