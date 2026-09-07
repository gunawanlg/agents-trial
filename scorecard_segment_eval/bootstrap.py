import numpy as np

from scorecard_segment_eval.metrics import gini


def bootstrap_gini_ci(
    y,
    p,
    n_bootstrap=200,
    seed=42,
    z=1.64,
):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    point = gini(y, p)
    if n < 10 or np.isnan(point):
        return {"gini": point, "gini_se": float("nan"), "gini_ci_low": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        draws[i] = gini(y[idx], p[idx])
    draws = draws[np.isfinite(draws)]
    if len(draws) < 10:
        return {"gini": point, "gini_se": float("nan"), "gini_ci_low": float("nan")}
    se = float(np.std(draws, ddof=1))
    return {"gini": point, "gini_se": se, "gini_ci_low": point - z * se}


def bootstrap_delta_gini(
    y,
    p_refit,
    p_pooled,
    n_bootstrap=200,
    seed=42,
    z=1.64,
):
    y = np.asarray(y, dtype=float)
    p_refit = np.asarray(p_refit, dtype=float)
    p_pooled = np.asarray(p_pooled, dtype=float)
    n = len(y)
    g_r = gini(y, p_refit)
    g_p = gini(y, p_pooled)
    delta = g_r - g_p if np.isfinite(g_r) and np.isfinite(g_p) else float("nan")
    if n < 10 or np.isnan(delta):
        return {
            "gini_refit": g_r,
            "gini_pooled": g_p,
            "delta_gini": delta,
            "delta_gini_se": float("nan"),
            "delta_gini_ci_low": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        draws[i] = gini(y[idx], p_refit[idx]) - gini(y[idx], p_pooled[idx])
    draws = draws[np.isfinite(draws)]
    se = float(np.std(draws, ddof=1)) if len(draws) >= 10 else float("nan")
    ci_low = delta - z * se if np.isfinite(se) else float("nan")
    return {
        "gini_refit": g_r,
        "gini_pooled": g_p,
        "delta_gini": delta,
        "delta_gini_se": se,
        "delta_gini_ci_low": ci_low,
    }
