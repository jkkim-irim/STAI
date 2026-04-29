"""Data pipeline for STAI EV battery defect classification.

Public API:
    FEATURE_COLUMNS, META_COLUMNS, TARGET_COLUMN
    load_csv(path) -> (X, y, raw_df)
    LabelEncoder, StandardScaler
    train_val_split(X, y, val_ratio, seed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "Ambient_Temp_C",
    "Anode_Overhang_mm",
    "Electrolyte_Volume_ml",
    "Internal_Resistance_mOhm",
    "Capacity_mAh",
    "Retention_50Cycle_Pct",
)
META_COLUMNS: tuple[str, ...] = (
    "Cell_ID",
    "Batch_ID",
    "Production_Line",
    "Shift",
    "Supplier",
)
TARGET_COLUMN: str = "Defect_Type"
NORMAL_LABEL: str = "None"


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Read CSV and return feature matrix, label vector, and raw DataFrame.

    `Defect_Type` NaN values are filled with the string "None" (no defect).
    The label column is optional — predict-time inputs may omit it; in that
    case `y` is returned as None.
    """
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input CSV missing required feature columns: {missing}")
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if TARGET_COLUMN in df.columns:
        y_series = df[TARGET_COLUMN].fillna(NORMAL_LABEL).astype(str)
        y = y_series.to_numpy()
    else:
        y = None
    return X, y, df


class LabelEncoder:
    """String labels -> contiguous integer indices."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "LabelEncoder":
        self.classes_ = np.array(sorted(np.unique(y).tolist()))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder not fitted")
        index = {c: i for i, c in enumerate(self.classes_)}
        unknown = set(np.unique(y)) - set(self.classes_)
        if unknown:
            raise ValueError(f"unseen labels at transform: {sorted(unknown)}")
        return np.array([index[v] for v in y], dtype=np.int64)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_int: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder not fitted")
        return self.classes_[np.asarray(y_int, dtype=np.int64)]

    def to_dict(self) -> dict:
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder not fitted")
        return {"classes": np.asarray(self.classes_)}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelEncoder":
        enc = cls()
        enc.classes_ = np.asarray(d["classes"])
        return enc


class StandardScaler:
    """Per-feature standardization (mean 0, std 1)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std == 0.0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler not fitted")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler not fitted")
        return {"mean": self.mean_, "std": self.std_}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        sc = cls()
        sc.mean_ = np.asarray(d["mean"], dtype=np.float64)
        sc.std_ = np.asarray(d["std"], dtype=np.float64)
        return sc


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split — preserves per-class ratios."""
    rng = np.random.default_rng(seed)
    classes, _ = np.unique(y, return_counts=True)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    for c in classes:
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])
    train_idx_arr = np.concatenate(train_idx)
    val_idx_arr = np.concatenate(val_idx)
    rng.shuffle(train_idx_arr)
    rng.shuffle(val_idx_arr)
    return X[train_idx_arr], y[train_idx_arr], X[val_idx_arr], y[val_idx_arr]


def class_weights_balanced(y: np.ndarray) -> dict:
    """Inverse-frequency class weights — useful for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    k = len(classes)
    return {c: n / (k * cnt) for c, cnt in zip(classes, counts)}
