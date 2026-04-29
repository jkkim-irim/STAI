"""Hyperparameter sweep on the STAI dataset.

Strategy:
  1. Fixed train/val split (seed=42) on full data
  2. Train subsample (stratified, capped at --train-cap) for fast sweep
  3. Same val set for every config — fair comparison
  4. Score by val macro-F1 (better than accuracy on imbalanced data)

Run:
    python scripts/sweep.py --data datasets/ev_battery_qc_train.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import LabelEncoder, StandardScaler, load_csv, train_val_split  # noqa: E402
from src.svm import (  # noqa: E402
    KernelSVM,
    LinearHardMarginSVM,
    MultiClassOvR,
    SoftMarginSVM,
)


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


def evaluate(model, X_tr, y_tr, X_va, y_va, n_classes: int, label: str) -> dict:
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    train_acc = float((pred_tr == y_tr).mean())
    val_acc = float((pred_va == y_va).mean())
    train_f1 = macro_f1(y_tr, pred_tr, n_classes)
    val_f1 = macro_f1(y_va, pred_va, n_classes)
    print(
        f"  {label:48s} fit={fit_s:5.1f}s  "
        f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
        f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f}"
    )
    return {
        "label": label,
        "fit_seconds": fit_s,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_macro_f1": train_f1,
        "val_macro_f1": val_f1,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--train-cap", type=int, default=3000,
                   help="stratified subsample size for sweep training")
    args = p.parse_args()

    X_raw, y_str, _ = load_csv(args.data)
    enc = LabelEncoder().fit(y_str)
    y_int = enc.transform(y_str)
    sc = StandardScaler().fit(X_raw)
    Xs = sc.transform(X_raw)
    X_tr_full, y_tr_full, X_va, y_va = train_val_split(
        Xs, y_int, val_ratio=args.val_ratio, seed=args.seed
    )
    X_tr, y_tr = stratified_subsample(X_tr_full, y_tr_full, args.train_cap, args.seed)
    K = len(enc.classes_)

    print(f"sweep: train(sample)={len(X_tr)}, val={len(X_va)}, classes={K}")
    print(f"classes: {enc.classes_.tolist()}")
    print(f"val class counts: {np.unique(y_va, return_counts=True)[1].tolist()}")
    print()

    results: list[dict] = []

    print("[linear hard-margin]")
    m = MultiClassOvR(lambda: LinearHardMarginSVM(), class_weight="balanced")
    results.append(evaluate(m, X_tr, y_tr, X_va, y_va, K, "linear-hard balanced"))
    m = MultiClassOvR(lambda: LinearHardMarginSVM(), class_weight=None)
    results.append(evaluate(m, X_tr, y_tr, X_va, y_va, K, "linear-hard no-weight"))

    print("\n[soft-margin: C sweep]")
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m = MultiClassOvR(lambda C=C: SoftMarginSVM(C=C), class_weight="balanced")
        results.append(evaluate(m, X_tr, y_tr, X_va, y_va, K, f"soft C={C}"))

    print("\n[kernel RBF: C × gamma sweep]")
    for C in [0.1, 1.0, 10.0, 100.0]:
        for gamma in ["scale", "auto", 0.05, 0.5, 5.0]:
            label = f"rbf C={C} gamma={gamma}"
            m = MultiClassOvR(
                lambda C=C, g=gamma: KernelSVM(C=C, kernel="rbf", gamma=g),
                class_weight="balanced",
            )
            results.append(evaluate(m, X_tr, y_tr, X_va, y_va, K, label))

    print("\n[kernel poly: C × degree sweep]")
    for C in [1.0, 10.0]:
        for d in [2, 3, 4]:
            label = f"poly C={C} degree={d}"
            m = MultiClassOvR(
                lambda C=C, d=d: KernelSVM(
                    C=C, kernel="poly", gamma="scale", degree=d, coef0=1.0
                ),
                class_weight="balanced",
            )
            results.append(evaluate(m, X_tr, y_tr, X_va, y_va, K, label))

    print("\n=== top 10 by val macro-F1 ===")
    results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    for r in results[:10]:
        print(
            f"  {r['label']:48s} "
            f"val_f1={r['val_macro_f1']:.4f}  val_acc={r['val_acc']:.4f}  "
            f"fit={r['fit_seconds']:.1f}s"
        )

    print("\n=== best per variant (by val macro-F1) ===")
    for prefix in ["linear-hard", "soft", "rbf", "poly"]:
        best = max(
            (r for r in results if r["label"].startswith(prefix)),
            key=lambda r: r["val_macro_f1"],
            default=None,
        )
        if best:
            print(
                f"  {prefix:12s} -> {best['label']}  "
                f"val_f1={best['val_macro_f1']:.4f}  val_acc={best['val_acc']:.4f}"
            )


if __name__ == "__main__":
    main()
