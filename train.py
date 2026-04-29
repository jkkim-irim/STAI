"""Train an SVM on the STAI EV-battery dataset and save the model.

Usage:
    python train.py --data datasets/ev_battery_qc_train.csv \
        --variant {linear|soft|kernel} --out models/model.pkl
"""

from __future__ import annotations

import argparse
import datetime
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import (  # noqa: E402
    FEATURE_COLUMNS,
    LabelEncoder,
    StandardScaler,
    load_csv,
    train_val_split,
)
from src.svm import (  # noqa: E402
    KernelSVM,
    LinearHardMarginSVM,
    MultiClassOvR,
    SoftMarginSVM,
)


def print_flush(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="path to training CSV")
    p.add_argument(
        "--variant",
        choices=["linear", "soft", "kernel"],
        required=True,
        help="linear=hard-margin, soft=soft-margin, kernel=nonlinear",
    )
    p.add_argument("--out", required=True, help="path to output model file (.pkl)")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--kernel", choices=["rbf", "poly", "linear"], default="rbf",
                   help="kernel type (only used when variant=kernel)")
    p.add_argument("--gamma", default="scale", help="float, 'scale', or 'auto'")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--coef0", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="balance positive/negative within each binary OvR sub-problem",
    )
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="subsample training set to N rows (stratified); 0 = use all",
    )
    return p.parse_args()


def make_factory(variant: str, args: argparse.Namespace):
    if variant == "linear":
        return lambda: LinearHardMarginSVM(tol=args.tol)
    if variant == "soft":
        return lambda: SoftMarginSVM(C=args.C, tol=args.tol)
    if variant == "kernel":
        # parse gamma: float or "scale"/"auto"
        try:
            gamma_val: float | str = float(args.gamma)
        except ValueError:
            gamma_val = args.gamma
        return lambda: KernelSVM(
            C=args.C,
            kernel=args.kernel,
            gamma=gamma_val,
            degree=args.degree,
            coef0=args.coef0,
            tol=args.tol,
        )
    raise ValueError(variant)


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = n // len(classes)
    idx_parts: list[np.ndarray] = []
    for c in classes:
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        idx_parts.append(ci[: min(per_class, len(ci))])
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, prefix: str = ""
) -> None:
    print_flush(f"{prefix}per-class metrics:")
    for ci, c in enumerate(classes):
        mask = y_true == ci
        if mask.sum() == 0:
            continue
        recall = (y_pred[mask] == ci).mean()
        pred_mask = y_pred == ci
        precision = (
            (y_true[pred_mask] == ci).mean() if pred_mask.sum() > 0 else 0.0
        )
        print_flush(
            f"{prefix}  {c:30s}  n={int(mask.sum()):5d}  "
            f"recall={recall:.4f}  precision={precision:.4f}"
        )


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, prefix: str = ""
) -> None:
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    short = [c[:10] for c in classes]
    print_flush(f"{prefix}confusion (rows=true, cols=pred):")
    print_flush(f"{prefix}  {'':22s}" + "".join(f"{s:>11s}" for s in short))
    for i, c in enumerate(classes):
        print_flush(
            f"{prefix}  {c[:20]:22s}"
            + "".join(f"{cm[i, j]:11d}" for j in range(K))
        )


def main() -> None:
    args = parse_args()
    print_flush(f"[stai-train] variant={args.variant}")
    print_flush(f"             args={vars(args)}")

    X_raw, y_str, _ = load_csv(args.data)
    print_flush(f"  loaded {X_raw.shape[0]} rows × {X_raw.shape[1]} features")

    encoder = LabelEncoder().fit(y_str)
    y_int = encoder.transform(y_str)
    print_flush(f"  classes ({len(encoder.classes_)}): {encoder.classes_.tolist()}")

    scaler = StandardScaler().fit(X_raw)
    Xs = scaler.transform(X_raw)

    X_tr, y_tr, X_va, y_va = train_val_split(
        Xs, y_int, val_ratio=args.val_ratio, seed=args.seed
    )
    print_flush(f"  split: train={len(X_tr)}, val={len(X_va)}")

    if args.max_train_samples and args.max_train_samples < len(X_tr):
        X_tr, y_tr = stratified_subsample(X_tr, y_tr, args.max_train_samples, args.seed)
        print_flush(f"  subsampled train to {len(X_tr)} (stratified)")

    cw = args.class_weight if args.class_weight != "none" else None
    factory = make_factory(args.variant, args)
    ovr = MultiClassOvR(base_factory=factory, class_weight=cw)

    t0 = time.time()
    ovr.fit(X_tr, y_tr, verbose=True)
    fit_seconds = time.time() - t0
    print_flush(f"  fit done in {fit_seconds:.1f}s")

    pred_tr = ovr.predict(X_tr)
    pred_va = ovr.predict(X_va)
    train_acc = (pred_tr == y_tr).mean()
    val_acc = (pred_va == y_va).mean()

    print_flush()
    print_flush(f"  train acc: {train_acc:.4f}")
    print_flush(f"  val   acc: {val_acc:.4f}")
    print_flush()
    per_class_report(y_va, pred_va, encoder.classes_, prefix="  [val] ")
    print_flush()
    confusion_matrix(y_va, pred_va, encoder.classes_, prefix="  [val] ")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "variant": args.variant,
        "svm": ovr,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": list(FEATURE_COLUMNS),
        "meta": {
            "seed": args.seed,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "fit_seconds": float(fit_seconds),
            "cli_args": vars(args),
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(state, f)
    print_flush(f"\n  model saved to {out_path}")


if __name__ == "__main__":
    main()
