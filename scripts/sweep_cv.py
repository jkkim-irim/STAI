"""K-fold cross-validation sweep with finer grid around best regions.

Builds on `scripts/sweep.py`:
  - Replaces single train/val split with stratified K-fold (default 5-fold)
  - Each config trains K times, scored by **mean ± std** of val macro-F1
  - Lower variance across folds → less risk of val-set overfitting
  - Finer grid focused on top performers from the initial sweep

Run:
    python scripts/sweep_cv.py --data datasets/ev_battery_qc_train.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import LabelEncoder, StandardScaler, load_csv  # noqa: E402
from src.svm import (  # noqa: E402
    KernelSVM,
    LinearHardMarginSVM,
    MultiClassOvR,
    SoftMarginSVM,
)


def _print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns list of (train_idx, val_idx) per fold, preserving per-class ratio."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    chunks_per_class: list[list[np.ndarray]] = []
    for c in classes:
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        chunks_per_class.append(np.array_split(ci, k))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_i in range(k):
        val_idx = np.concatenate([chunks_per_class[c][fold_i] for c in range(len(classes))])
        train_parts = []
        for c in range(len(classes)):
            for j in range(k):
                if j != fold_i:
                    train_parts.append(chunks_per_class[c][j])
        train_idx = np.concatenate(train_parts)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = n // len(classes)
    parts = []
    for c in classes:
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        parts.append(ci[: min(per, len(ci))])
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def cv_evaluate(
    label: str,
    factory_with_weight,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    train_cap: int,
    seed: int,
) -> dict:
    f1s, accs, fits = [], [], []
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        X_tr_full, y_tr_full = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        if train_cap and train_cap < len(X_tr_full):
            X_tr, y_tr = stratified_subsample(X_tr_full, y_tr_full, train_cap, seed + fold_i)
        else:
            X_tr, y_tr = X_tr_full, y_tr_full

        model = factory_with_weight()
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fits.append(time.time() - t0)
        pred_va = model.predict(X_va)
        accs.append(float((pred_va == y_va).mean()))
        f1s.append(macro_f1(y_va, pred_va, n_classes))

    f1_arr = np.array(f1s)
    acc_arr = np.array(accs)
    total_fit = float(sum(fits))
    _print(
        f"  {label:38s}  cv_f1={f1_arr.mean():.4f}±{f1_arr.std():.4f}  "
        f"cv_acc={acc_arr.mean():.4f}±{acc_arr.std():.4f}  "
        f"per-fold f1=[{', '.join(f'{x:.3f}' for x in f1s)}]  "
        f"fit={total_fit:.1f}s"
    )
    return {
        "label": label,
        "cv_f1_mean": float(f1_arr.mean()),
        "cv_f1_std": float(f1_arr.std()),
        "cv_acc_mean": float(acc_arr.mean()),
        "cv_acc_std": float(acc_arr.std()),
        "fit_seconds_total": total_fit,
        "per_fold_f1": f1s,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--train-cap", type=int, default=4000,
                   help="cap stratified subsample per fold's training (0=full)")
    args = p.parse_args()

    X_raw, y_str, _ = load_csv(args.data)
    enc = LabelEncoder().fit(y_str)
    y = enc.transform(y_str)
    sc = StandardScaler().fit(X_raw)
    X = sc.transform(X_raw)
    K = len(enc.classes_)

    folds = stratified_kfold(y, args.folds, args.seed)
    _print(f"K-fold CV: k={args.folds}, total n={len(y)}, per fold val~{len(folds[0][1])}")
    _print(f"  classes ({K}): {enc.classes_.tolist()}")
    _print(f"  train cap per fold: {args.train_cap or 'full'}")
    _print()

    # ---------------- baseline variants ----------------
    _print("[baselines]")
    cfgs_baseline = [
        ("linear-hard no-weight",
         lambda: MultiClassOvR(lambda: LinearHardMarginSVM(), class_weight=None)),
        ("linear-hard balanced",
         lambda: MultiClassOvR(lambda: LinearHardMarginSVM(), class_weight="balanced")),
        ("soft C=100 balanced",
         lambda: MultiClassOvR(lambda: SoftMarginSVM(C=100.0), class_weight="balanced")),
    ]

    # ---------------- finer poly grid (top region from prior sweep) ----------------
    _print()
    _print("[poly fine grid: degree x C]")
    cfgs_poly = []
    for d in [3, 4, 5]:
        for C in [3, 5, 10, 20, 50]:
            label = f"poly d={d} C={C}"
            cfgs_poly.append((
                label,
                lambda C=C, d=d: MultiClassOvR(
                    lambda C=C, d=d: KernelSVM(
                        C=C, kernel="poly", degree=d, gamma="scale", coef0=1.0
                    ),
                    class_weight="balanced",
                ),
            ))

    # ---------------- finer RBF grid (around C=100, gamma=0.05 / scale) ----------------
    _print()
    _print("[rbf fine grid: C x gamma]")
    cfgs_rbf = []
    for C in [10, 30, 100, 300]:
        for gamma in [0.02, 0.05, 0.1, "scale"]:
            label = f"rbf C={C} gamma={gamma}"
            cfgs_rbf.append((
                label,
                lambda C=C, g=gamma: MultiClassOvR(
                    lambda C=C, g=g: KernelSVM(C=C, kernel="rbf", gamma=g),
                    class_weight="balanced",
                ),
            ))

    results: list[dict] = []
    for group_name, group in [
        ("baselines", cfgs_baseline),
        ("poly fine", cfgs_poly),
        ("rbf fine", cfgs_rbf),
    ]:
        _print()
        _print(f"--- {group_name} ({len(group)} configs) ---")
        for label, factory in group:
            r = cv_evaluate(label, factory, X, y, folds, K, args.train_cap, args.seed)
            results.append(r)

    _print()
    _print("=== top 10 by mean cv macro-F1 (sorted) ===")
    results.sort(key=lambda r: r["cv_f1_mean"], reverse=True)
    for r in results[:10]:
        _print(
            f"  {r['label']:38s}  "
            f"cv_f1={r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.4f}  "
            f"cv_acc={r['cv_acc_mean']:.4f}±{r['cv_acc_std']:.4f}"
        )

    _print()
    _print("=== best per family ===")
    for prefix in ["linear-hard", "soft", "poly", "rbf"]:
        best = max(
            (r for r in results if r["label"].startswith(prefix)),
            key=lambda r: r["cv_f1_mean"],
            default=None,
        )
        if best is not None:
            _print(
                f"  {prefix:12s} -> {best['label']:30s}  "
                f"cv_f1={best['cv_f1_mean']:.4f}±{best['cv_f1_std']:.4f}"
            )


if __name__ == "__main__":
    main()
