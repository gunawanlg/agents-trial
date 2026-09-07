"""Bootstrap confidence intervals for Gini and Gini differences.

Every replicate draws from its own generator, seeded from ``(seed, replicate
index)`` only.  Replicate ``i`` therefore uses the same resample no matter which
worker executes it or in what order, which is what makes serial and parallel
runs bit-identical.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from scorecard_segment_eval.metrics import gini
from scorecard_segment_eval.parallel import map_jobs, resolve_n_jobs, spawn_seeds


def _chunks(n_items, n_chunks):
    # type: (int, int) -> List[Sequence[int]]
    if n_items <= 0:
        return []
    return [list(part) for part in np.array_split(np.arange(n_items), max(n_chunks, 1)) if len(part)]


def _replicate_indices(seed, n_rows):
    # type: (int, int) -> np.ndarray
    return np.random.default_rng(seed).integers(0, n_rows, size=n_rows)


def _run_replicates(statistic, seeds, n_rows, n_jobs=None, cap=8):
    # type: (Any, Sequence[int], int, Optional[int], int) -> np.ndarray
    """Evaluate ``statistic(indices)`` for every replicate, in order."""
    n = len(seeds)
    if n == 0:
        return np.empty(0, dtype=float)
    workers = resolve_n_jobs(n_jobs, cap=cap, n_items=n)
    parts = _chunks(n, workers)

    def _one(part):
        # type: (Sequence[int]) -> np.ndarray
        out = np.empty(len(part), dtype=float)
        for j, replicate in enumerate(part):
            idx = _replicate_indices(seeds[replicate], n_rows)
            out[j] = statistic(idx)
        return out

    results = map_jobs(_one, parts, n_jobs=workers, cap=cap)
    return np.concatenate(results) if results else np.empty(0, dtype=float)


def bootstrap_gini_ci(y, p, n_bootstrap=200, seed=42, z=1.64, n_jobs=None, cap=8):
    # type: (Any, Any, int, int, float, Optional[int], int) -> Dict[str, float]
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    point = gini(y, p)
    if n < 10 or np.isnan(point):
        return {"gini": point, "gini_se": float("nan"), "gini_ci_low": float("nan")}
    seeds = spawn_seeds(seed, n_bootstrap)

    def _stat(idx):
        # type: (np.ndarray) -> float
        return gini(y[idx], p[idx])

    draws = _run_replicates(_stat, seeds, n, n_jobs=n_jobs, cap=cap)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 10:
        return {"gini": point, "gini_se": float("nan"), "gini_ci_low": float("nan")}
    se = float(np.std(draws, ddof=1))
    return {"gini": point, "gini_se": se, "gini_ci_low": point - z * se}


def bootstrap_delta_gini(y, p_refit, p_pooled, n_bootstrap=200, seed=42, z=1.64, n_jobs=None, cap=8):
    # type: (Any, Any, Any, int, int, float, Optional[int], int) -> Dict[str, float]
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
    seeds = spawn_seeds(seed, n_bootstrap)

    def _stat(idx):
        # type: (np.ndarray) -> float
        return gini(y[idx], p_refit[idx]) - gini(y[idx], p_pooled[idx])

    draws = _run_replicates(_stat, seeds, n, n_jobs=n_jobs, cap=cap)
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
