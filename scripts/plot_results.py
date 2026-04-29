"""Generate report-ready PNG visualizations for the STAI SVM project.

Outputs go under figures/ in 4 subfolders:
  data_overview/      — input data context (class dist, PCA scatter)
  decision_boundary/  — 2D PCA decision regions for each OvO model
  performance/        — metrics (acc, mF1, per-class F1, confusion)
  support_vectors/    — model complexity (SV counts)

All plots are square (figsize=(8, 8)) and saved at 120 DPI.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    MultiClassOvO,
    SoftMarginSVM,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIG_ROOT = Path("/home/jkkim/STAI/figures")
SUBDIRS = ["data_overview", "decision_boundary", "performance", "support_vectors"]
for sd in SUBDIRS:
    (FIG_ROOT / sd).mkdir(parents=True, exist_ok=True)

# Square figures, consistent style
plt.rcParams.update({
    "figure.figsize": (8, 8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

CLASS_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]  # red, orange, blue, green
SAVE_KW = {"dpi": 120, "bbox_inches": "tight"}


# ---------------------------------------------------------------------------
# Load data + best models
# ---------------------------------------------------------------------------

print("Loading data...", flush=True)
X_raw, y_str, _ = load_csv("/home/jkkim/STAI/datasets/ev_battery_qc_train.csv")
encoder = LabelEncoder().fit(y_str)
y_int = encoder.transform(y_str)
classes = encoder.classes_
K = len(classes)
class_short = ["Critical", "High IR", "None", "Poor Ret"]

scaler = StandardScaler().fit(X_raw)
X_scaled = scaler.transform(X_raw)
X_tr, y_tr, X_va, y_va = train_val_split(X_scaled, y_int, val_ratio=0.2, seed=42)

# Simple PCA (eigendecomposition of covariance)
def simple_pca(X: np.ndarray, n_components: int = 2):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc.T)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    return Xc @ V[:, idx[:n_components]], w[idx], V[:, idx]

X_pca, eigvals, eigvecs = simple_pca(X_scaled)
explained = eigvals[:2] / eigvals.sum()
X_pca_tr, y_pca_tr, X_pca_va, y_pca_va = train_val_split(X_pca, y_int, val_ratio=0.2, seed=42)

# Load 4 final OvO models
MODELS_DIR = Path("/home/jkkim/STAI/models")
MODEL_NAMES = [
    "linear_hard_ovo",
    "soft_C100_ovo",
    "kernel_poly_d3_C50_ovo",
    "kernel_rbf_C300_g005_ovo",
]
MODEL_DISPLAY = {
    "linear_hard_ovo":          "Linear (hard, OvO)",
    "soft_C100_ovo":            "Soft C=100 (OvO)",
    "kernel_poly_d3_C50_ovo":   "Kernel poly d=3 C=50 (OvO)",
    "kernel_rbf_C300_g005_ovo": "Kernel RBF C=300 γ=0.05 (OvO)",
}
states = {n: pickle.load(open(MODELS_DIR / f"{n}.pkl", "rb")) for n in MODEL_NAMES}


# ---------------------------------------------------------------------------
# 1. Data overview
# ---------------------------------------------------------------------------

print("[1/4] data overview...", flush=True)

# 1a. Class distribution
fig, ax = plt.subplots(figsize=(8, 8))
counts = np.bincount(y_int, minlength=K)
bars = ax.bar(range(K), counts, color=CLASS_COLORS, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(K))
ax.set_xticklabels(class_short, rotation=0)
ax.set_ylabel("Number of samples")
ax.set_title("Class Distribution (n=13,565)")
for b, n in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, n, f"{n}\n({100 * n / sum(counts):.2f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, max(counts) * 1.15)
plt.savefig(FIG_ROOT / "data_overview/class_distribution.png", **SAVE_KW)
plt.close()

# 1b. PCA 2D scatter
fig, ax = plt.subplots(figsize=(8, 8))
for i, c in enumerate(class_short):
    mask = y_int == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLASS_COLORS[i], label=f"{c} (n={mask.sum()})",
               alpha=0.4, s=10, edgecolors="none")
ax.set_xlabel(f"PC 1 ({100 * explained[0]:.1f}% var)")
ax.set_ylabel(f"PC 2 ({100 * explained[1]:.1f}% var)")
ax.set_title("PCA 2D Projection of Standardized Features")
ax.legend(loc="upper right")
plt.savefig(FIG_ROOT / "data_overview/pca_2d_scatter.png", **SAVE_KW)
plt.close()

# 1c. Feature distribution by class
feats = list(FEATURE_COLUMNS)
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for k, f in enumerate(feats):
    ax = axes[k // 2, k % 2]
    for i in range(K):
        ax.hist(X_raw[y_int == i, k], bins=30, alpha=0.5,
                color=CLASS_COLORS[i], label=class_short[i] if k == 0 else None)
    ax.set_title(f.replace("_", " "), fontsize=9)
    ax.tick_params(labelsize=8)
fig.suptitle("Feature Distributions by Class", y=1.0, fontsize=13)
fig.legend(class_short, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(FIG_ROOT / "data_overview/feature_distributions.png", **SAVE_KW)
plt.close()


# ---------------------------------------------------------------------------
# 2. Decision boundaries in 2D PCA
# ---------------------------------------------------------------------------

print("[2/4] decision boundaries (training mini-SVMs in PCA space)...", flush=True)


def stratified_subsample(X, y, n=2500, seed=42):
    rng = np.random.default_rng(seed)
    classes_local = np.unique(y)
    per = n // len(classes_local)
    parts = []
    for c in classes_local:
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        parts.append(ci[: min(per, len(ci))])
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


X_pca_tr_sub, y_pca_tr_sub = stratified_subsample(X_pca_tr, y_pca_tr, n=2500, seed=42)

PCA_FACTORIES = {
    "linear_hard_ovo":          lambda: LinearHardMarginSVM(),
    "soft_C100_ovo":            lambda: SoftMarginSVM(C=100),
    "kernel_poly_d3_C50_ovo":   lambda: KernelSVM(C=50, kernel="poly", degree=3, gamma="scale", coef0=1.0),
    "kernel_rbf_C300_g005_ovo": lambda: KernelSVM(C=300, kernel="rbf", gamma=0.05),
}

pca_models = {}
for name, factory in PCA_FACTORIES.items():
    print(f"  fitting {name} on PCA 2D…", flush=True)
    m = MultiClassOvO(factory, class_weight="balanced")
    m.fit(X_pca_tr_sub, y_pca_tr_sub)
    pca_models[name] = m


def plot_decision_boundary(model, X, y, title, ax):
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).astype(int).reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap([f"{c}33" for c in CLASS_COLORS])
    ax.contourf(xx, yy, Z, levels=np.arange(K + 1) - 0.5, cmap=cmap_bg, alpha=1.0)
    ax.contour(xx, yy, Z, levels=np.arange(K + 1) - 0.5, colors="white", linewidths=1.5)

    for i in range(K):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=CLASS_COLORS[i], label=class_short[i],
                   s=14, edgecolors="black", linewidths=0.3, alpha=0.85)
    ax.set_xlabel(f"PC 1 ({100 * explained[0]:.1f}%)")
    ax.set_ylabel(f"PC 2 ({100 * explained[1]:.1f}%)")
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


for name, m in pca_models.items():
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_decision_boundary(m, X_pca_va, y_pca_va, MODEL_DISPLAY[name], ax)
    ax.legend(loc="upper right")
    plt.savefig(FIG_ROOT / f"decision_boundary/{name}.png", **SAVE_KW)
    plt.close()

# 2x2 combined plot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, (name, m) in zip(axes.flat, pca_models.items()):
    plot_decision_boundary(m, X_pca_va, y_pca_va, MODEL_DISPLAY[name], ax)
axes[0, 0].legend(loc="upper right")
fig.suptitle("Decision Boundaries in PCA 2D Space (val data overlaid)", y=1.0, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_ROOT / "decision_boundary/all_4_compare.png", dpi=120, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 3. Performance metrics
# ---------------------------------------------------------------------------

print("[3/4] performance metrics...", flush=True)


def macro_f1(yt, yp, K_):
    f1s = []
    for c in range(K_):
        tp = ((yp == c) & (yt == c)).sum()
        fp = ((yp == c) & (yt != c)).sum()
        fn = ((yp != c) & (yt == c)).sum()
        if tp == 0:
            f1s.append(0.0)
            continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    return f1s, float(np.mean(f1s))


# Compute val metrics for each model
val_results = {}
for name, state in states.items():
    Xs_va = state["scaler"].transform(X_raw)  # rebuild from raw
    _, _, X_v, y_v = train_val_split(Xs_va, y_int, val_ratio=0.2, seed=42)
    pred = state["svm"].predict(X_v)
    acc = float((pred == y_v).mean())
    f1_per, mf1 = macro_f1(y_v, pred, K)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_v, pred):
        cm[int(t), int(p)] += 1
    val_results[name] = {
        "acc": acc, "mF1": mf1, "f1_per": f1_per, "cm": cm,
        "n_sv": sum(len(est.alpha_) for est in state["svm"].estimators_),
    }
    print(f"  {name}: acc={acc:.4f} mF1={mf1:.4f} #SV={val_results[name]['n_sv']}",
          flush=True)

# 3a. Accuracy + macro-F1 grouped bar
fig, ax = plt.subplots(figsize=(8, 8))
x = np.arange(len(MODEL_NAMES))
w = 0.35
accs = [val_results[n]["acc"] for n in MODEL_NAMES]
mf1s = [val_results[n]["mF1"] for n in MODEL_NAMES]
b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#1f77b4", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, mf1s, w, label="Macro F1", color="#ff7f0e", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_DISPLAY[n].replace(" (OvO)", "")
                     .replace("Linear (hard,", "Linear")
                     for n in MODEL_NAMES],
                   rotation=15, ha="right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Validation Performance — 4 OvO Models")
ax.legend(loc="lower right")
for bs, vs in [(b1, accs), (b2, mf1s)]:
    for b, v in zip(bs, vs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
plt.savefig(FIG_ROOT / "performance/accuracy_macro_f1.png", **SAVE_KW)
plt.close()

# 3b. Per-class F1 heatmap
fig, ax = plt.subplots(figsize=(8, 8))
F1_matrix = np.array([val_results[n]["f1_per"] for n in MODEL_NAMES])
im = ax.imshow(F1_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(K))
ax.set_xticklabels(class_short)
ax.set_yticks(range(len(MODEL_NAMES)))
ax.set_yticklabels([MODEL_DISPLAY[n].replace(" (OvO)", "") for n in MODEL_NAMES],
                   fontsize=9)
ax.set_title("Per-Class F1 Score (val)")
for i in range(len(MODEL_NAMES)):
    for j in range(K):
        ax.text(j, i, f"{F1_matrix[i, j]:.3f}",
                ha="center", va="center",
                color="white" if F1_matrix[i, j] < 0.5 else "black",
                fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F1")
plt.savefig(FIG_ROOT / "performance/per_class_f1_heatmap.png", **SAVE_KW)
plt.close()

# 3c. Confusion matrix for best model (linear_hard_ovo)
best_name = max(val_results, key=lambda n: val_results[n]["acc"])
cm = val_results[best_name]["cm"]
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, cmap="Blues", aspect="auto")
ax.set_xticks(range(K)); ax.set_xticklabels(class_short)
ax.set_yticks(range(K)); ax.set_yticklabels(class_short)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {MODEL_DISPLAY[best_name]}\n"
             f"val_acc={val_results[best_name]['acc']:.4f}, mF1={val_results[best_name]['mF1']:.4f}")
for i in range(K):
    for j in range(K):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=11, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(FIG_ROOT / "performance/confusion_matrix_best.png", **SAVE_KW)
plt.close()

# 3d. Per-class recall comparison (grouped bar)
fig, ax = plt.subplots(figsize=(8, 8))
recall_matrix = np.zeros((len(MODEL_NAMES), K))
for i, n in enumerate(MODEL_NAMES):
    cm_n = val_results[n]["cm"]
    row_sums = cm_n.sum(axis=1)
    recall_matrix[i] = np.where(row_sums > 0, np.diag(cm_n) / row_sums, 0)

x = np.arange(K); width = 0.2
for i, n in enumerate(MODEL_NAMES):
    ax.bar(x + (i - 1.5) * width, recall_matrix[i], width,
           label=MODEL_DISPLAY[n].replace(" (OvO)", ""),
           edgecolor="black", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(class_short)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Recall")
ax.set_title("Per-Class Recall — 4 OvO Models (val)")
ax.legend(loc="lower right", fontsize=8)
plt.savefig(FIG_ROOT / "performance/per_class_recall.png", **SAVE_KW)
plt.close()


# ---------------------------------------------------------------------------
# 4. Support vector counts
# ---------------------------------------------------------------------------

print("[4/4] support vector counts...", flush=True)

fig, ax = plt.subplots(figsize=(8, 8))
sv_counts = [val_results[n]["n_sv"] for n in MODEL_NAMES]
labels = [MODEL_DISPLAY[n].replace(" (OvO)", "") for n in MODEL_NAMES]
bars = ax.bar(range(len(MODEL_NAMES)), sv_counts,
              color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
              edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(MODEL_NAMES)))
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Number of Support Vectors (sum across OvO binaries)")
ax.set_title("Model Complexity — Support Vector Counts")
for b, n in zip(bars, sv_counts):
    ax.text(b.get_x() + b.get_width()/2, n, str(n),
            ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, max(sv_counts) * 1.12)
plt.savefig(FIG_ROOT / "support_vectors/sv_count_per_model.png", **SAVE_KW)
plt.close()


print(f"\nDone. Saved figures to {FIG_ROOT}/")
print("Subdirectories:", *SUBDIRS)
