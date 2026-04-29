"""SVM implementations from scratch.

Three variants:
    LinearHardMarginSVM  — dual QP, no slack (separable case)
    SoftMarginSVM        — dual QP with 0 ≤ α ≤ C (linearly inseparable)
    KernelSVM            — dual QP with kernel matrix (rbf / poly / linear)

All three share `_BaseSVM` which solves the dual via cvxopt:
    min  ½ αᵀ P α − 1ᵀ α
    s.t. Σ α_i y_i = 0
         0 ≤ α_i ≤ C_i        (C_i = C × sample_weight_i)
where P_ij = y_i y_j K(x_i, x_j).

Only used external optimizer: cvxopt.solvers.qp (general QP, not SVM-specific).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# cvxopt prints a banner by default — silence + tighten tolerances
import cvxopt  # noqa: E402
import cvxopt.solvers  # noqa: E402

cvxopt.solvers.options["show_progress"] = False
cvxopt.solvers.options["abstol"] = 1e-8
cvxopt.solvers.options["reltol"] = 1e-8
cvxopt.solvers.options["feastol"] = 1e-8


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def _kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
) -> np.ndarray:
    """Compute K(X1, X2) — shape (n1, n2)."""
    if kernel == "linear":
        return X1 @ X2.T
    if kernel == "rbf":
        # ||x-y||² = ||x||² + ||y||² - 2 x·y
        sq1 = np.sum(X1 * X1, axis=1, keepdims=True)
        sq2 = np.sum(X2 * X2, axis=1, keepdims=True).T
        sqdist = np.maximum(sq1 + sq2 - 2.0 * (X1 @ X2.T), 0.0)
        return np.exp(-gamma * sqdist)
    if kernel == "poly":
        return (gamma * (X1 @ X2.T) + coef0) ** degree
    raise ValueError(f"unknown kernel: {kernel!r}")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseSVM:
    """Shared dual-QP SVM core. Subclasses pin specific kernel/C settings.

    Binary classification only — multi-class is handled by `MultiClassOvR`.
    Labels must be in {-1, +1}.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "rbf", "poly"] = "linear",
        gamma: float | str = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-5,
        ridge: float = 1e-8,
    ) -> None:
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.ridge = float(ridge)

        # fitted state
        self.alpha_: np.ndarray | None = None
        self.support_vectors_: np.ndarray | None = None
        self.support_y_: np.ndarray | None = None  # ±1 labels of SVs
        self.b_: float | None = None
        self.w_: np.ndarray | None = None  # only for linear kernel
        self._gamma_value: float | None = None  # resolved at fit

    # -- helpers --------------------------------------------------------

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        if self.gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError(f"invalid gamma: {self.gamma!r}")

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return _kernel_matrix(
            X1, X2, self.kernel, self._gamma_value or 1.0, self.degree, self.coef0
        )

    # -- training -------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "_BaseSVM":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.array_equal(np.unique(y), np.array([-1.0, 1.0])):
            raise ValueError("y must contain only {-1, +1}")
        n = X.shape[0]

        self._gamma_value = self._resolve_gamma(X)

        K = self._kernel(X, X)
        # P = (y y^T) ⊙ K  + ridge*I  for numerical PD
        P = (np.outer(y, y) * K).astype(np.float64)
        P += self.ridge * np.eye(n)
        q = -np.ones(n, dtype=np.float64)

        # box: 0 ≤ α_i ≤ C * w_i
        if sample_weight is None:
            upper = np.full(n, self.C, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if sample_weight.shape[0] != n:
                raise ValueError("sample_weight length mismatch")
            upper = self.C * sample_weight

        # G x ≤ h:  -I α ≤ 0  AND  I α ≤ upper
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.concatenate([np.zeros(n), upper])
        # equality A x = b:  yᵀ α = 0
        A = y.reshape(1, -1)
        b_eq = np.zeros(1, dtype=np.float64)

        sol = cvxopt.solvers.qp(
            cvxopt.matrix(P),
            cvxopt.matrix(q),
            cvxopt.matrix(G),
            cvxopt.matrix(h),
            cvxopt.matrix(A),
            cvxopt.matrix(b_eq),
        )
        if sol["status"] not in ("optimal", "unknown"):
            raise RuntimeError(f"cvxopt QP failed: status={sol['status']}")
        alpha = np.array(sol["x"]).reshape(-1)
        alpha = np.clip(alpha, 0.0, None)

        # Support vectors: α > tol
        sv_mask = alpha > self.tol
        if not np.any(sv_mask):
            raise RuntimeError("no support vectors found — try increasing C or tol")
        self.alpha_ = alpha[sv_mask]
        self.support_vectors_ = X[sv_mask]
        self.support_y_ = y[sv_mask]

        # Bias b: average over margin SVs (0 < α < C*w_i)
        margin_mask = (alpha > self.tol) & (alpha < upper - self.tol)
        if np.any(margin_mask):
            X_m = X[margin_mask]
            y_m = y[margin_mask]
            K_m = self._kernel(X_m, self.support_vectors_)
            b_vals = y_m - (K_m * (self.alpha_ * self.support_y_)).sum(axis=1)
            self.b_ = float(np.mean(b_vals))
        else:
            # fall back to any SV
            K_all = self._kernel(self.support_vectors_, self.support_vectors_)
            b_vals = self.support_y_ - (K_all * (self.alpha_ * self.support_y_)).sum(axis=1)
            self.b_ = float(np.mean(b_vals))

        # cache w for linear kernel
        if self.kernel == "linear":
            self.w_ = (self.alpha_ * self.support_y_) @ self.support_vectors_

        return self

    # -- inference ------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.support_vectors_ is None or self.b_ is None:
            raise RuntimeError("model not fitted")
        X = np.asarray(X, dtype=np.float64)
        if self.kernel == "linear" and self.w_ is not None:
            return X @ self.w_ + self.b_
        K = self._kernel(X, self.support_vectors_)
        return (K * (self.alpha_ * self.support_y_)).sum(axis=1) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    # -- serialization --------------------------------------------------

    def to_dict(self) -> dict:
        if self.support_vectors_ is None:
            raise RuntimeError("model not fitted")
        return {
            "C": np.float64(self.C),
            "kernel": np.array(self.kernel),
            "gamma_value": np.float64(self._gamma_value or 1.0),
            "degree": np.int64(self.degree),
            "coef0": np.float64(self.coef0),
            "tol": np.float64(self.tol),
            "ridge": np.float64(self.ridge),
            "alpha": self.alpha_,
            "support_vectors": self.support_vectors_,
            "support_y": self.support_y_,
            "b": np.float64(self.b_),
            "w": self.w_ if self.w_ is not None else np.array([]),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_BaseSVM":
        m = cls(
            C=float(d["C"]),
            kernel=str(d["kernel"]),
            gamma=float(d["gamma_value"]),
            degree=int(d["degree"]),
            coef0=float(d["coef0"]),
            tol=float(d["tol"]),
            ridge=float(d["ridge"]),
        )
        m._gamma_value = float(d["gamma_value"])
        m.alpha_ = np.asarray(d["alpha"], dtype=np.float64)
        m.support_vectors_ = np.asarray(d["support_vectors"], dtype=np.float64)
        m.support_y_ = np.asarray(d["support_y"], dtype=np.float64)
        m.b_ = float(d["b"])
        w_arr = np.asarray(d["w"], dtype=np.float64)
        m.w_ = w_arr if w_arr.size > 0 else None
        return m


# ---------------------------------------------------------------------------
# Public variants
# ---------------------------------------------------------------------------

class LinearHardMarginSVM(_BaseSVM):
    """Hard-margin linear SVM. Use only when data is (nearly) linearly separable."""

    def __init__(self, tol: float = 1e-5, ridge: float = 1e-8) -> None:
        # large C ≡ no slack; not infinite to keep QP numerically stable
        super().__init__(C=1e8, kernel="linear", tol=tol, ridge=ridge)


class SoftMarginSVM(_BaseSVM):
    """Soft-margin linear SVM. Box constraint 0 ≤ α ≤ C tolerates outliers/overlap."""

    def __init__(self, C: float = 1.0, tol: float = 1e-5, ridge: float = 1e-8) -> None:
        super().__init__(C=C, kernel="linear", tol=tol, ridge=ridge)


class KernelSVM(_BaseSVM):
    """Nonlinear SVM via kernel trick. Default RBF."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["rbf", "poly", "linear"] = "rbf",
        gamma: float | str = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-5,
        ridge: float = 1e-8,
    ) -> None:
        super().__init__(
            C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,
            tol=tol, ridge=ridge,
        )


# ---------------------------------------------------------------------------
# Multi-class via One-vs-Rest
# ---------------------------------------------------------------------------

class MultiClassOvR:
    """One-vs-Rest wrapper. Trains one binary SVM per class.

    `class_weight='balanced'` rebalances within each binary problem so that
    the (small) positive class isn't drowned out by the negative class —
    important for the heavily imbalanced STAI dataset.
    """

    def __init__(
        self,
        base_factory,
        class_weight: Literal[None, "balanced"] = "balanced",
    ) -> None:
        self.base_factory = base_factory
        self.class_weight = class_weight
        self.classes_: np.ndarray | None = None
        self.estimators_: list[_BaseSVM] = []

    def __getstate__(self) -> dict:
        # base_factory is typically a lambda — strip it so pickle works
        state = self.__dict__.copy()
        state["base_factory"] = None
        return state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiClassOvR":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        n = len(y)
        for c in self.classes_:
            y_bin = np.where(y == c, 1.0, -1.0)
            if self.class_weight == "balanced":
                n_pos = int((y_bin == 1).sum())
                n_neg = n - n_pos
                w = np.where(y_bin == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
            else:
                w = None
            est = self.base_factory()
            est.fit(X, y_bin, sample_weight=w)
            self.estimators_.append(est)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        scores = np.column_stack([est.decision_function(X) for est in self.estimators_])
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("model not fitted")
        idx = np.argmax(self.decision_function(X), axis=1)
        return self.classes_[idx]

    def to_dict(self) -> dict:
        return {
            "classes": np.asarray(self.classes_),
            "n_estimators": np.int64(len(self.estimators_)),
            **{f"est_{i}": est.to_dict() for i, est in enumerate(self.estimators_)},
        }

    @classmethod
    def from_dict(cls, d: dict, base_cls: type[_BaseSVM]) -> "MultiClassOvR":
        m = cls(base_factory=base_cls)
        m.classes_ = np.asarray(d["classes"])
        n = int(d["n_estimators"])
        m.estimators_ = [base_cls.from_dict(d[f"est_{i}"]) for i in range(n)]
        return m
