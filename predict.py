"""Predict labels for a CSV using a trained STAI SVM model.

Usage:
    python predict.py --model models/model.pkl \
        --in datasets/some_input.csv --out pred.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import FEATURE_COLUMNS, NORMAL_LABEL, TARGET_COLUMN  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="pickled model from train.py")
    p.add_argument("--in", dest="input", required=True, help="input CSV (must have 6 feature columns)")
    p.add_argument("--out", required=True, help="output CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.model, "rb") as f:
        state = pickle.load(f)
    print(
        f"[stai-predict] variant={state['variant']}  "
        f"trained={state['meta']['trained_at']}  "
        f"train_acc={state['meta']['train_acc']:.4f}  "
        f"val_acc={state['meta']['val_acc']:.4f}"
    )

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"input CSV missing required feature columns: {missing}")

    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    Xs = state["scaler"].transform(X)
    pred_int = state["svm"].predict(Xs)
    pred_labels = state["encoder"].inverse_transform(pred_int)

    out_df = df.copy()
    out_df["Defect_Type_Pred"] = pred_labels

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"  saved {len(out_df)} predictions to {out_path}")

    if TARGET_COLUMN in df.columns:
        true = df[TARGET_COLUMN].fillna(NORMAL_LABEL).astype(str).to_numpy()
        acc = (pred_labels == true).mean()
        print(f"  accuracy on input (with labels): {acc:.4f}")
        classes = state["encoder"].classes_
        cls2i = {c: i for i, c in enumerate(classes)}
        K = len(classes)
        cm = np.zeros((K, K), dtype=int)
        for t, p in zip(true, pred_labels):
            if t in cls2i and p in cls2i:
                cm[cls2i[t], cls2i[p]] += 1
        short = [c[:10] for c in classes]
        print("  confusion (rows=true, cols=pred):")
        print("  " + " " * 22 + "".join(f"{s:>11s}" for s in short))
        for i, c in enumerate(classes):
            print(f"  {c[:20]:22s}" + "".join(f"{cm[i, j]:11d}" for j in range(K)))


if __name__ == "__main__":
    main()
