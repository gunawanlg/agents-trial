import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from typing import Any, Dict, Tuple


def _as_arrays(y, p):
    # type: (Any, Any) -> Tuple[np.ndarray, np.ndarray]
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return y, p


def auc(y, p):
    y, p = _as_arrays(y, p)
    if len(y) == 0 or y.min() == y.max() or p.min() == p.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def gini(y, p):
    a = auc(y, p)
    if np.isnan(a):
        return float("nan")
    return 2.0 * a - 1.0


def ks_stat(y, p):
    y, p = _as_arrays(y, p)
    n_bad = y.sum()
    n_good = len(y) - n_bad
    if n_bad == 0 or n_good == 0:
        return float("nan")
    order = np.argsort(p)
    y_sorted = y[order]
    cdf_bad = np.cumsum(y_sorted) / n_bad
    cdf_good = np.cumsum(1.0 - y_sorted) / n_good
    return float(np.max(np.abs(cdf_bad - cdf_good)))


def brier(y, p):
    y, p = _as_arrays(y, p)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def logloss(y, p):
    y, p = _as_arrays(y, p)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(log_loss(y, p, labels=[0.0, 1.0]))


def observed_expected(y, p):
    # type: (Any, Any) -> Dict[str, float]
    y, p = _as_arrays(y, p)
    if len(y) == 0:
        return {"n": 0.0, "defaults": 0.0, "obs_rate": float("nan"), "mean_pd": float("nan"), "oe": float("nan")}
    mean_pd = float(p.mean())
    obs_rate = float(y.mean())
    oe = obs_rate / mean_pd if mean_pd > 0 else float("nan")
    return {
        "n": float(len(y)),
        "defaults": float(y.sum()),
        "obs_rate": obs_rate,
        "mean_pd": mean_pd,
        "oe": oe,
    }


def ece(y, p, n_bins=10):
    y, p = _as_arrays(y, p)
    if len(y) < n_bins:
        n_bins = max(int(len(y)), 1)
    try:
        ranks = pd.qcut(p, q=n_bins, duplicates="drop")
    except ValueError:
        return float("nan")
    df = pd.DataFrame({"y": y, "p": p, "bin": ranks})
    try:
        grouped = df.groupby("bin", observed=True)
    except TypeError:
        grouped = df.groupby("bin")
    if grouped.ngroups == 0:
        return float("nan")
    abs_err = (grouped["p"].mean() - grouped["y"].mean()).abs()
    weights = grouped.size() / len(df)
    return float((abs_err * weights).sum())


def hosmer_lemeshow(y, p, n_bins=10):
    # type: (Any, Any, int) -> Dict[str, float]
    y, p = _as_arrays(y, p)
    out = {"hl_stat": float("nan"), "hl_df": float("nan")}
    if len(y) < n_bins:
        return out
    try:
        ranks = pd.qcut(p, q=n_bins, duplicates="drop")
    except ValueError:
        return out
    df = pd.DataFrame({"y": y, "p": p, "bin": ranks})
    try:
        g = df.groupby("bin", observed=True)
    except TypeError:
        g = df.groupby("bin")
    n = g.size().to_numpy(dtype=float) if hasattr(g.size(), "to_numpy") else np.asarray(g.size(), dtype=float)
    obs = g["y"].sum().to_numpy(dtype=float) if hasattr(g["y"].sum(), "to_numpy") else np.asarray(g["y"].sum(), dtype=float)
    exp = g["p"].sum().to_numpy(dtype=float) if hasattr(g["p"].sum(), "to_numpy") else np.asarray(g["p"].sum(), dtype=float)
    var = np.clip(exp * (1.0 - exp / np.clip(n, 1.0, None)), 1e-12, None)
    stat = float(np.sum((obs - exp) ** 2 / var))
    out["hl_stat"] = stat
    out["hl_df"] = float(max(len(n) - 2, 1))
    return out


def performance_bundle(y, p, n_ece_bins=10):
    # type: (Any, Any, int) -> Dict[str, float]
    oe = observed_expected(y, p)
    hl = hosmer_lemeshow(y, p, n_bins=n_ece_bins)
    bundle = dict(oe)
    bundle["gini"] = gini(y, p)
    bundle["auc"] = auc(y, p)
    bundle["ks"] = ks_stat(y, p)
    bundle["brier"] = brier(y, p)
    bundle["logloss"] = logloss(y, p)
    bundle["ece"] = ece(y, p, n_bins=n_ece_bins)
    bundle.update(hl)
    return bundle
