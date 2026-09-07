"""Discrimination, calibration and approval-rate metrics."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from scorecard_segment_eval.schema import Gates

#: Keys returned by :func:`matched_ar_comparison`, always present.
MATCHED_AR_KEYS = (
    "ar_reference_cutoff",
    "ar_segment",
    "ar_reference",
    "ar_gap",
    "ar_gap_triggered",
    "matched_ar",
    "matched_ar_anchor",
    "matched_ar_threshold_segment",
    "matched_ar_threshold_reference",
    "matched_ar_n",
    "matched_ar_events",
    "gini_at_matched_ar",
    "gini_reference_at_matched_ar",
    "gini_at_matched_ar_gap",
    "gini_at_matched_ar_ratio",
)

#: A Gini ratio is only reported when the anchored reference Gini is at least
#: this large in absolute value; below that the denominator is noise and the
#: ratio explodes.  The absolute gap stays available in every case.
MIN_RATIO_DENOMINATOR = 0.05


def _as_arrays(y, p):
    # type: (Any, Any) -> Tuple[np.ndarray, np.ndarray]
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return y, p


def auc(y, p):
    # type: (Any, Any) -> float
    y, p = _as_arrays(y, p)
    if len(y) == 0 or y.min() == y.max() or p.min() == p.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def gini(y, p):
    # type: (Any, Any) -> float
    a = auc(y, p)
    if np.isnan(a):
        return float("nan")
    return 2.0 * a - 1.0


def ks_stat(y, p):
    # type: (Any, Any) -> float
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
    # type: (Any, Any) -> float
    y, p = _as_arrays(y, p)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def logloss(y, p):
    # type: (Any, Any) -> float
    y, p = _as_arrays(y, p)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(log_loss(y, p, labels=[0.0, 1.0]))


def observed_expected(y, p):
    # type: (Any, Any) -> Dict[str, float]
    y, p = _as_arrays(y, p)
    if len(y) == 0:
        return {
            "n": 0.0,
            "defaults": 0.0,
            "obs_rate": float("nan"),
            "mean_pd": float("nan"),
            "oe": float("nan"),
        }
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
    # type: (Any, Any, int) -> float
    y, p = _as_arrays(y, p)
    if len(y) < n_bins:
        n_bins = max(int(len(y)), 1)
    try:
        ranks = pd.qcut(p, q=n_bins, duplicates="drop")
    except ValueError:
        return float("nan")
    df = pd.DataFrame({"y": y, "p": p, "bin": ranks})
    grouped = df.groupby("bin", observed=True)
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
    g = df.groupby("bin", observed=True)
    n = g.size().to_numpy(dtype=float)
    obs = g["y"].sum().to_numpy(dtype=float)
    exp = g["p"].sum().to_numpy(dtype=float)
    var = np.clip(exp * (1.0 - exp / np.clip(n, 1.0, None)), 1e-12, None)
    stat = float(np.sum((obs - exp) ** 2 / var))
    out["hl_stat"] = stat
    out["hl_df"] = float(max(len(n) - 2, 1))
    return out


def performance_bundle(y, p, n_ece_bins=10):
    # type: (Any, Any, int) -> Dict[str, float]
    oe = observed_expected(y, p)
    hl = hosmer_lemeshow(y, p, n_bins=n_ece_bins)
    bundle = {}  # type: Dict[str, float]
    bundle.update(oe)
    bundle.update(
        {
            "gini": gini(y, p),
            "auc": auc(y, p),
            "ks": ks_stat(y, p),
            "brier": brier(y, p),
            "logloss": logloss(y, p),
            "ece": ece(y, p, n_bins=n_ece_bins),
        }
    )
    bundle.update(hl)
    return bundle


# --------------------------------------------------------------------------
# Approval rate / matched-AR Gini (section A2)
# --------------------------------------------------------------------------
def score_cutoff_for_ar(p, approval_rate):
    # type: (Any, float) -> float
    """Score cutoff that approves ``approval_rate`` of the population.

    The score is a PD, so *lower is better*: an application is approved when
    ``score <= cutoff``.
    """
    arr = np.asarray(p, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    rate = float(np.clip(approval_rate, 0.0, 1.0))
    return float(np.quantile(arr, rate))


def approval_rate(p, cutoff):
    # type: (Any, float) -> float
    """Share of the population approved at ``cutoff`` (``score <= cutoff``)."""
    arr = np.asarray(p, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0 or not np.isfinite(cutoff):
        return float("nan")
    return float(np.mean(valid <= cutoff))


def _empty_matched_ar():
    # type: () -> Dict[str, Any]
    return {
        "ar_reference_cutoff": float("nan"),
        "ar_segment": float("nan"),
        "ar_reference": float("nan"),
        "ar_gap": float("nan"),
        "ar_gap_triggered": False,
        "matched_ar": float("nan"),
        "matched_ar_anchor": None,
        "matched_ar_threshold_segment": float("nan"),
        "matched_ar_threshold_reference": float("nan"),
        "matched_ar_n": float("nan"),
        "matched_ar_events": float("nan"),
        "gini_at_matched_ar": float("nan"),
        "gini_reference_at_matched_ar": float("nan"),
        "gini_at_matched_ar_gap": float("nan"),
        "gini_at_matched_ar_ratio": float("nan"),
    }


def matched_ar_comparison(y_segment, p_segment, y_reference, p_reference, gates=None):
    # type: (Any, Any, Any, Any, Optional[Gates]) -> Dict[str, Any]
    """Gini at a comparable approval rate for a segment vs. the portfolio.

    Why this exists: a segment whose score distribution sits far from the
    portfolio's is effectively judged on a different slice of the risk
    spectrum.  At the portfolio cutoff its approval rate (AR) can be, say, 55%
    against the portfolio's 85%, and the two Ginis are then not like-for-like.

    Procedure:

    1. Take the portfolio cutoff that approves ``Gates.reference_approval_rate``
       of the reference population and read off the AR each group would get.
    2. If ``|AR_segment - AR_reference| >= Gates.ar_gap_trigger`` the comparison
       is flagged and the *lower-AR side* becomes the anchor -- the tighter of
       the two operating points, which both groups can actually reach.
    3. Re-derive a group-specific threshold that hits that matched AR inside
       each group, keep only the approved rows, and recompute Gini there.

    Every key in :data:`MATCHED_AR_KEYS` is always present.  The Gini columns
    stay null when the gap does not trigger or when the anchored sub-population
    is too small (``Gates.matched_ar_min_n`` / ``matched_ar_min_events``).
    """
    gates = gates or Gates()
    out = _empty_matched_ar()
    y_seg = np.asarray(y_segment, dtype=float)
    p_seg = np.asarray(p_segment, dtype=float)
    y_ref = np.asarray(y_reference, dtype=float)
    p_ref = np.asarray(p_reference, dtype=float)
    if len(y_seg) == 0 or len(y_ref) == 0 or len(y_seg) != len(p_seg) or len(y_ref) != len(p_ref):
        return out

    cutoff = score_cutoff_for_ar(p_ref, gates.reference_approval_rate)
    ar_seg = approval_rate(p_seg, cutoff)
    ar_ref = approval_rate(p_ref, cutoff)
    out["ar_reference_cutoff"] = cutoff
    out["ar_segment"] = ar_seg
    out["ar_reference"] = ar_ref
    if not (np.isfinite(ar_seg) and np.isfinite(ar_ref)):
        return out
    gap = float(ar_seg - ar_ref)
    out["ar_gap"] = gap
    triggered = bool(abs(gap) >= float(gates.ar_gap_trigger))
    out["ar_gap_triggered"] = triggered
    if not triggered:
        return out

    matched = float(min(ar_seg, ar_ref))
    out["matched_ar"] = matched
    out["matched_ar_anchor"] = "segment" if ar_seg <= ar_ref else "reference"
    if matched <= 0.0:
        return out
    thr_seg = score_cutoff_for_ar(p_seg, matched)
    thr_ref = score_cutoff_for_ar(p_ref, matched)
    out["matched_ar_threshold_segment"] = thr_seg
    out["matched_ar_threshold_reference"] = thr_ref
    mask_seg = np.isfinite(p_seg) & (p_seg <= thr_seg)
    mask_ref = np.isfinite(p_ref) & (p_ref <= thr_ref)
    n_seg = int(mask_seg.sum())
    events_seg = float(y_seg[mask_seg].sum()) if n_seg else 0.0
    out["matched_ar_n"] = float(n_seg)
    out["matched_ar_events"] = events_seg
    if n_seg < int(gates.matched_ar_min_n) or events_seg < float(gates.matched_ar_min_events):
        return out
    g_seg = gini(y_seg[mask_seg], p_seg[mask_seg])
    g_ref = gini(y_ref[mask_ref], p_ref[mask_ref])
    out["gini_at_matched_ar"] = g_seg
    out["gini_reference_at_matched_ar"] = g_ref
    if np.isfinite(g_seg) and np.isfinite(g_ref):
        out["gini_at_matched_ar_gap"] = float(g_seg - g_ref)
        if abs(g_ref) >= MIN_RATIO_DENOMINATOR:
            out["gini_at_matched_ar_ratio"] = float(g_seg / g_ref)
    return out
