from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scorecard_segment_eval.schema import Gates


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class WoEEncoder:
    """Train-only quantile/category WoE. Unseen levels map to 0."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.edges_: dict[str, np.ndarray | None] = {}
        self.woe_: dict[str, dict[str, float]] = {}
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        yv = y.to_numpy(dtype=float)
        n_bad = float(yv.sum())
        n_good = float(len(yv) - n_bad)
        eps = 0.5
        self.columns_ = list(X.columns)
        for col in X.columns:
            s = X[col]
            numeric = pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > self.n_bins
            if numeric:
                qs = np.linspace(0.0, 1.0, self.n_bins + 1)
                edges = np.unique(np.quantile(pd.to_numeric(s, errors="coerce").dropna(), qs))
                if len(edges) < 3:
                    numeric = False
                else:
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    self.edges_[col] = edges
                    keys = pd.cut(pd.to_numeric(s, errors="coerce"), bins=edges, include_lowest=True).astype("string").fillna("__na__")
            if not numeric:
                self.edges_[col] = None
                keys = s.astype("string").fillna("__na__")
            frame = pd.DataFrame({"y": yv, "bin": keys.to_numpy()})
            g = frame.groupby("bin", dropna=False)
            bad = g["y"].sum()
            tot = g.size()
            good = tot - bad
            woe = np.log(((good + eps) / max(n_good, 1.0)) / ((bad + eps) / max(n_bad, 1.0)))
            self.woe_[col] = woe.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.columns_:
            s = X[col]
            edges = self.edges_[col]
            if edges is not None:
                keys = pd.cut(pd.to_numeric(s, errors="coerce"), bins=edges, include_lowest=True).astype("string").fillna("__na__")
            else:
                keys = s.astype("string").fillna("__na__")
            mapping = self.woe_[col]
            cols.append(keys.map(lambda k: mapping.get(k, 0.0)).to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.zeros((len(X), 0))


def fit_segment_lr(X_woe: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(X_woe, y)
    return model


def refit_same_predictors(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    pred_cols: list[str],
    target_col: str,
    gates: Gates,
) -> np.ndarray:
    enc = WoEEncoder(n_bins=gates.n_woe_bins)
    enc.fit(train[pred_cols], train[target_col])
    x_tr = enc.transform(train[pred_cols])
    x_ho = enc.transform(holdout[pred_cols])
    model = fit_segment_lr(x_tr, train[target_col].to_numpy(dtype=int))
    return model.predict_proba(x_ho)[:, 1]


def recalibrate_pd(p_train: np.ndarray, y_train: np.ndarray, p_holdout: np.ndarray) -> np.ndarray:
    """Two-parameter logistic rescale: logit(p*) = a + b * logit(p)."""
    z_tr = _logit(np.asarray(p_train, dtype=float)).reshape(-1, 1)
    y = np.asarray(y_train, dtype=int)
    if y.min() == y.max():
        return np.clip(np.asarray(p_holdout, dtype=float), 1e-6, 1.0 - 1e-6)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(z_tr, y)
    z_ho = _logit(np.asarray(p_holdout, dtype=float)).reshape(-1, 1)
    return model.predict_proba(z_ho)[:, 1]


def apply_recalibrate_fn(p_train, y_train, p_holdout) -> np.ndarray:
    return recalibrate_pd(p_train, y_train, p_holdout)
