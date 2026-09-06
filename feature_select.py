# feature_select.py
from __future__ import annotations

import os
import time
import math
import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class FSConfig:
    # CV
    cv_mode: str = "walk"     # "walk" uses (season, week) order when available, else KFold
    n_splits: int = 5

    # Permutation importance
    repeats_perm: int = 3
    random_state: int = 42

    # Ablation
    ablate_top_k: int = 5      # how many top features to attempt removing step-by-step
    ablation_patience: int = 1 # stop after this many consecutive degradations

    # Correlation pruning
    corr_threshold: float = 0.95

    # Training
    epochs: int = 60
    batch_size: int = 256
    epoch_log_every: int = 5
    verbose_fit: int = 0  # let our custom logger print instead

    # Runtime
    use_gpu: bool = True
    use_mixed_precision: bool = False
    use_xla: bool = False

    # Misc
    tag: str = "feature_selection"
    order_cols: Tuple[str, str] = ("season", "week")  # for walk-forward

# ────────────────────────────────────────────────────────────────────────────────
# Utilities & logging
# ────────────────────────────────────────────────────────────────────────────────

class EpochLogger(keras.callbacks.Callback):
    def __init__(self, every: int = 5):
        super().__init__()
        self.every = max(1, int(every))
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every == 0 or (epoch + 1) == self.params.get("epochs", 0):
            l = logs or {}
            msg = f"[FS][epoch {epoch+1}] loss={l.get('loss'):.4f}"
            if "val_loss" in l:
                msg += f"  val_loss={l['val_loss']:.4f}"
            print(msg, flush=True)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _std_scale(train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    Xtr = sc.fit_transform(train)
    Xva = sc.transform(val)
    return Xtr, Xva

# ────────────────────────────────────────────────────────────────────────────────
# CV splitters
# ────────────────────────────────────────────────────────────────────────────────

def _iter_splits(df: pd.DataFrame, cfg: FSConfig):
    """Yield (train_index, val_index) for each fold."""
    if cfg.cv_mode == "walk" and all(col in df.columns for col in cfg.order_cols):
        # sort by (season, week) (or whatever order_cols are)
        order = df[list(cfg.order_cols)].apply(tuple, axis=1)
        idx_sorted = np.argsort(order.values)
        n = len(df)
        fold_size = n // cfg.n_splits
        for k in range(cfg.n_splits):
            end = (k + 1) * fold_size if k < cfg.n_splits - 1 else n
            val_idx = idx_sorted[k * fold_size : end]
            tr_mask = np.ones(n, dtype=bool); tr_mask[val_idx] = False
            tr_idx = np.nonzero(tr_mask)[0]
            yield tr_idx, val_idx
    else:
        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        for tr, va in kf.split(df):
            yield tr, va

# ────────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────────

def _build_model(n_features: int) -> keras.Model:
    """Smaller, fast network for feature selection runs."""
    m = Sequential([
        Input(shape=(n_features,)),
        Dense(min(64, max(16, n_features // 2)), activation="relu"),
        Dense(min(32, max(8, n_features // 4)), activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-3), loss="mse")
    return m

def _train_and_score(model, X_tr, y_tr, X_va, y_va, cfg: FSConfig) -> Tuple[float, float]:
    callbacks = [
        EpochLogger(cfg.epoch_log_every),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=0
        ),
    ]
    model.fit(
        X_tr, y_tr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=0.1,
        verbose=cfg.verbose_fit,
        callbacks=callbacks
    )
    pred = model.predict(X_va, verbose=0).reshape(-1)
    mae = mean_absolute_error(y_va, pred)
    r2 = r2_score(y_va, pred)
    return mae, r2

# ────────────────────────────────────────────────────────────────────────────────
# Importance: permutation (batched), MI; correlation prune; ablation
# ────────────────────────────────────────────────────────────────────────────────

def _perm_importance_for_fold(model,
                              X_val: pd.DataFrame,
                              y_val: pd.Series,
                              base_mae: float,
                              repeats: int,
                              rng: np.random.Generator,
                              desc: str,
                              tqdm_mininterval: float = 0.2) -> pd.DataFrame:
    """
    Fast permutation importance:
    For each feature j:
      - Build 'repeats' permuted copies vertically (shape: repeats*n x d)
      - Single model.predict on the stack
      - ΔMAE = mean(MAE_perm) - base_mae
    """
    cols = list(X_val.columns)
    Xv = X_val.to_numpy(copy=True)
    yv = y_val.to_numpy()
    out = []
    n, d = Xv.shape

    for j in tqdm(range(d), desc=desc, total=d, mininterval=tqdm_mininterval):
        stack = np.repeat(Xv[None, :, :], repeats, axis=0)   # (repeats, n, d)
        for r in range(repeats):
            rng.shuffle(stack[r, :, j])
        stack = stack.reshape(repeats * n, d)
        preds = model.predict(stack, verbose=0).reshape(repeats, n)
        maes = np.abs(preds - yv[None, :]).mean(axis=1)
        d_mae = float(maes.mean() - base_mae)
        out.append({"feature": cols[j], "dMAE": d_mae})
    return pd.DataFrame(out)

def _mutual_info(df: pd.DataFrame, features: List[str], y: pd.Series) -> pd.DataFrame:
    X = df[features].to_numpy()
    mi = mutual_info_regression(X, y.to_numpy(), random_state=0)
    return pd.DataFrame({"feature": features, "mi": mi})

def _corr_prune(df: pd.DataFrame, features: List[str], thr: float) -> List[str]:
    if len(features) <= 1:
        return features
    corr = df[features].corr().abs()
    keep = []
    dropped = set()
    for f in corr.columns:
        if f in dropped:
            continue
        keep.append(f)
        to_drop = corr.index[(corr[f] > thr) & (corr.index != f)].tolist()
        for g in to_drop:
            dropped.add(g)
    return keep

def _ablation_loop(model_builder,
                   base_feats: List[str],
                   df_tr: pd.DataFrame,
                   df_va: pd.DataFrame,
                   target: str,
                   base_mae: float,
                   cfg: FSConfig) -> Tuple[List[set], List[Tuple[float, float]]]:
    """
    Sequentially remove the 'worst' features from candidate set (top-K by perm dMAE),
    stopping when validation MAE degrades for 'ablation_patience' consecutive steps.
    Returns all kept-sets along the path and their (MAE, R2).
    """
    kept_sets: List[set] = []
    scores: List[Tuple[float, float]] = []
    patience = 0

    # Work on a copy of candidate list
    candidates = list(base_feats)

    # Precompute a single permutation ranking on validation to decide removal order
    print("[FS] Precomputing permutation ranking for ablation…")
    mb = model_builder(len(candidates))
    Xtr, ytr = df_tr[candidates], df_tr[target]
    Xva, yva = df_va[candidates], df_va[target]
    # scale
    Xtr_s, Xva_s = _std_scale(Xtr.values, Xva.values)
    mb_mae, _ = _train_and_score(mb, Xtr_s, ytr.values, Xva_s, yva.values, cfg)
    base_mae_for_perm = mb_mae  # use its own baseline on that fold
    rng = np.random.default_rng(cfg.random_state + 1337)
    perm = _perm_importance_for_fold(mb, Xva, yva, base_mae_for_perm, repeats=2, rng=rng, desc="[FS] Ablation priming")
    perm_rank = perm.sort_values("dMAE", ascending=False)["feature"].tolist()

    remove_order = [f for f in perm_rank if f in candidates][:cfg.ablate_top_k]

    current = set(candidates)
    for step, f in enumerate(tqdm(remove_order, desc="[FS] Ablating")):
        trial = list(current - {f})
        mb2 = model_builder(len(trial))
        Xtr, ytr = df_tr[trial], df_tr[target]
        Xva, yva = df_va[trial], df_va[target]
        Xtr_s, Xva_s = _std_scale(Xtr.values, Xva.values)
        mae, r2 = _train_and_score(mb2, Xtr_s, ytr.values, Xva_s, yva.values, cfg)

        kept_sets.append(set(trial))
        scores.append((mae, r2))
        print(f"[FS][ablate step {step+1}] removed={f}, val_MAE={mae:.4f} (base={base_mae:.4f}), val_R2={r2:.4f}")

        if mae <= base_mae:
            # improvement or equal -> accept removal
            base_mae = mae
            current = set(trial)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.ablation_patience:
                print("[FS] Early stop ablation (no improvement).")
                break

    return kept_sets, scores

# ────────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────────

def _plot_perm(perm_df: pd.DataFrame, out_path: str) -> None:
    if perm_df.empty:
        return
    df = perm_df.copy().sort_values("perm_mean_dMAE", ascending=False)
    plt.figure(figsize=(10, max(3.0, 0.35 * len(df))))
    plt.barh(df["feature"], df["perm_mean_dMAE"], xerr=df["perm_std_dMAE"], capsize=3)
    plt.gca().invert_yaxis()
    plt.xlabel("ΔMAE upon permutation (higher ⇒ more important)")
    plt.title("Permutation Importance (CV mean ± 1 std)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_mi(mi_df: pd.DataFrame, out_path: str) -> None:
    if mi_df.empty:
        return
    df = mi_df.copy().sort_values("mi", ascending=False)
    plt.figure(figsize=(10, max(3.0, 0.35 * len(df))))
    plt.barh(df["feature"], df["mi"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mutual Information with target")
    plt.title("Mutual Information")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_ablation(scores: List[Tuple[float, float]], out_path: str) -> None:
    if not scores:
        return
    mae = [s[0] for s in scores]
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(mae) + 1), mae, marker="o")
    plt.xlabel("Ablation step")
    plt.ylabel("Validation MAE")
    plt.title("Ablation Path")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# Core runner
# ────────────────────────────────────────────────────────────────────────────────

def _cv_baseline_and_importance(df: pd.DataFrame,
                                features: List[str],
                                target: str,
                                model_builder,
                                cfg: FSConfig):
    rng = np.random.default_rng(cfg.random_state)

    fold_mae, fold_r2 = [], []
    perm_all: List[pd.DataFrame] = []

    print(f"[FS] Starting CV ({cfg.cv_mode}, splits={cfg.n_splits}) on {len(df)} rows, {len(features)} features")
    for fold, (tr, va) in enumerate(_iter_splits(df, cfg), start=1):
        print(f"[FS] Fold {fold}/{cfg.n_splits}")
        df_tr, df_va = df.iloc[tr], df.iloc[va]
        Xtr, ytr = df_tr[features], df_tr[target]
        Xva, yva = df_va[features], df_va[target]

        # scale
        Xtr_s, Xva_s = _std_scale(Xtr.values, Xva.values)

        model = model_builder(len(features))
        mae, r2 = _train_and_score(model, Xtr_s, ytr.values, Xva_s, yva.values, cfg)
        fold_mae.append(mae); fold_r2.append(r2)
        print(f"[FS]   fold MAE={mae:.4f}  R2={r2:.4f}")

        # baseline MAE on *unshuffled* val set (same as mae above)
        base_mae = mae
        # permutation importance (batched)
        perm_f = _perm_importance_for_fold(model, Xva, yva, base_mae,
                                           repeats=cfg.repeats_perm,
                                           rng=rng,
                                           desc=f"[FS] Perm (fold {fold})")
        perm_all.append(perm_f)

    base_mae = float(np.mean(fold_mae))
    base_r2 = float(np.mean(fold_r2))
    print(f"[FS] CV complete. Baseline: MAE={base_mae:.4f}  R2={base_r2:.4f}")

    # aggregate permutation across folds
    perm_df = pd.concat(perm_all, ignore_index=True)
    perm_summary = (perm_df
                    .groupby("feature")["dMAE"]
                    .agg(perm_mean_dMAE="mean", perm_std_dMAE="std", n_folds="count")
                    .reset_index())
    return perm_summary, base_mae, base_r2

def run_feature_selection(df: pd.DataFrame,
                          features: Sequence[str],
                          target: str,
                          cfg: Optional[FSConfig] = None) -> Dict:
    """
    Full FS pipeline with guaranteed keep_list in return payload.
    """
    t0 = time.time()
    cfg = cfg or FSConfig()
    np.random.seed(cfg.random_state)
    tf.random.set_seed(cfg.random_state)

    if cfg.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if cfg.use_xla:
        tf.config.optimizer.set_jit(True)

    outdir = os.path.join("data", "feature_selection")
    _ensure_dir(outdir)

    features = list(features)
    # basic clean
    df = df.dropna(subset=list(set(features + [target]))).copy()

    # 1) CV baseline + permutation importance
    perm_summary, base_mae, base_r2 = _cv_baseline_and_importance(
        df, features, target, _build_model, cfg
    )

    # 2) Mutual information (fast screen)
    print("[FS] Computing mutual information…")
    mi_summary = _mutual_info(df, features, df[target])

    # 3) Correlation pruning (keep reps under threshold)
    survivors = _corr_prune(df, features, cfg.corr_threshold)
    print(f"[FS] Baseline MAE={base_mae:.4f}  R2={base_r2:.4f}  |  kept {len(survivors)} features after corr prune")

    # 4) Walk-forward last split for ablation (or fall back to last KFold)
    #    Find a single (train, val) split representative (last chronological)
    splits = list(_iter_splits(df, cfg))
    tr_idx, va_idx = splits[-1]
    df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]

    # pick top-K by permutation (on CV) among survivors
    perm_surv = perm_summary[perm_summary["feature"].isin(survivors)]
    top_for_ablation = perm_surv.sort_values("perm_mean_dMAE", ascending=False)["feature"].tolist()
    top_for_ablation = top_for_ablation[:min(cfg.ablate_top_k, len(top_for_ablation))]

    ablation_sets: List[set] = []
    ablation_scores: List[Tuple[float, float]] = []
    ablation_plot_path = None
    if len(top_for_ablation) >= 2:
        ablation_sets, ablation_scores = _ablation_loop(
            _build_model, survivors, df_tr, df_va, target, base_mae, cfg
        )
        # plot ablation path
        ablation_plot_path = os.path.join(outdir, f"ablation_{cfg.tag}.png")
        _plot_ablation(ablation_scores, ablation_plot_path)
        print(f"[FS] Saved ablation plot → {ablation_plot_path}")
    else:
        print("[FS] Skipping ablation (not enough candidates after prune).")

    # 5) Plots for perm & MI
    perm_plot_path = os.path.join(outdir, f"perm_{cfg.tag}.png")
    mi_plot_path = os.path.join(outdir, f"mi_{cfg.tag}.png")
    _plot_perm(perm_summary, perm_plot_path)
    _plot_mi(mi_summary, mi_plot_path)

    # ── Synthesize suggested keep list (ALWAYS return keep_list) ────────────────
    # keep_by_perm: top-10 by ΔMAE (higher is more important)
    keep_by_perm: List[str] = []
    if not perm_summary.empty:
        k = min(10, len(perm_summary))
        keep_by_perm = (
            perm_summary.sort_values("perm_mean_dMAE", ascending=False)
                        .head(k)["feature"].tolist()
        )

    # keep_by_mi: top-10 by MI
    keep_by_mi: List[str] = []
    if not mi_summary.empty:
        k = min(10, len(mi_summary))
        keep_by_mi = mi_summary.sort_values("mi", ascending=False).head(k)["feature"].tolist()

    # keep_by_ablation: last non-hurting set if we ran ablation
    keep_by_ablation: List[str] = []
    if ablation_sets:
        keep_by_ablation = sorted(list(ablation_sets[-1]))

    survivors_set = set(survivors) if survivors is not None else set(features)
    union_set = (set(keep_by_perm) | set(keep_by_mi) | set(keep_by_ablation)) & survivors_set
    if not union_set:
        if keep_by_perm:
            union_set = set(keep_by_perm) & survivors_set
        elif keep_by_mi:
            union_set = set(keep_by_mi) & survivors_set
        else:
            union_set = survivors_set

    ordered_sources = [
        keep_by_perm,
        keep_by_mi,
        keep_by_ablation,
        [f for f in survivors_set if f not in set(keep_by_perm) | set(keep_by_mi) | set(keep_by_ablation)],
    ]
    seen = set()
    keep_list: List[str] = []
    for bucket in ordered_sources:
        for f in bucket:
            if f in union_set and f not in seen:
                keep_list.append(f); seen.add(f)

    # logs
    t1 = time.time()
    print(f"[FS] Suggested keep list (n={len(keep_list)}): {', '.join(keep_list[:20])}{' ...' if len(keep_list)>20 else ''}")
    print(f"[FS] Done in {t1 - t0:.1f}s")

    # payload
    out = {
        "baseline": {"mae": base_mae, "r2": base_r2},
        "perm_summary": perm_summary,
        "mi_summary": mi_summary,
        "ablation": {
            "sets": ablation_sets,
            "scores": ablation_scores,
            "plot": ablation_plot_path,
        },
        "survivors": list(survivors_set),
        "keep_by_perm": keep_by_perm,
        "keep_by_mi": keep_by_mi,
        "keep_by_ablation": keep_by_ablation,
        "keep_list": keep_list,
        "plots": {"perm_plot": perm_plot_path, "mi_plot": mi_plot_path},
        "timing_sec": float(t1 - t0),
        "config": dataclasses.asdict(cfg),
    }
    return out

if __name__ == "__main__":
    None