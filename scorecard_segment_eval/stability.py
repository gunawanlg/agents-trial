"""Predictor stability over vintages.

A refit that only *fits better* is not enough: the new coefficients have to be
trustworthy going forward.  For every candidate predictor this module measures,
across scoring vintages,

* **distribution drift** -- PSI of the WoE-bin distribution of each vintage
  against the reference (all-vintages) distribution, on the grouping's own
  bins;
* **discriminatory power** -- univariate Gini and IV per vintage.  A vintage
  only counts as weak when its Gini falls below the floor by *more than
  sampling noise*, using the Hanley-McNeil standard error of the AUC.  Without
  that correction the minimum over a dozen small monthly vintages is always far
  below the pooled value and every predictor looks unstable;
* **sign consistency** -- the share of vintages whose univariate slope of the
  target on the predictor's WoE has the same sign as the reference slope.

The three are folded into a ``stability_score`` in ``[0, 1]`` and a boolean
``stable`` flag, both driven by the ``stability_*`` fields on
:class:`~scorecard_segment_eval.schema.Gates`.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scorecard_segment_eval.binning import BinningModel
from scorecard_segment_eval.characteristics import psi_from_labels
from scorecard_segment_eval.metrics import gini
from scorecard_segment_eval.parallel import map_jobs
from scorecard_segment_eval.schema import Gates

STABILITY_COLUMNS = (
    "feature",
    "n_vintages",
    "assessed",
    "psi_max",
    "psi_mean",
    "gini_reference",
    "gini_min",
    "gini_median",
    "gini_ratio_min",
    "gini_ratio_median",
    "weak_vintage_share",
    "iv_reference",
    "iv_min",
    "iv_cv",
    "sign_consistency",
    "stability_score",
    "stable",
    "stability_flags",
)


def gini_standard_error(y, score):
    # type: (np.ndarray, np.ndarray) -> float
    """Hanley-McNeil standard error of the Gini for one sample.

    Closed form, so it costs nothing next to a bootstrap and is available even
    for the small per-vintage slices where it matters most.
    """
    y_arr = np.asarray(y, dtype=float)
    n_pos = float((y_arr == 1).sum())
    n_neg = float((y_arr == 0).sum())
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    g = gini(y_arr, score)
    if not np.isfinite(g):
        return float("nan")
    a = max(0.5 * (g + 1.0), 1.0 - 0.5 * (g + 1.0))
    a = float(np.clip(a, 0.5, 1.0 - 1e-9))
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (
        a * (1.0 - a)
        + (n_pos - 1.0) * (q1 - a * a)
        + (n_neg - 1.0) * (q2 - a * a)
    ) / (n_pos * n_neg)
    if var <= 0:
        return float("nan")
    return float(2.0 * np.sqrt(var))


def vintage_labels(dates, freq="M"):
    # type: (Any, str) -> pd.Series
    """Period labels (default calendar month) for a date-like series."""
    parsed = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    periods = parsed.dt.to_period(freq)
    return periods.astype("object").where(periods.notna(), None)


def _slope_sign(x, y):
    # type: (np.ndarray, np.ndarray) -> float
    """Sign of the least-squares slope of ``y`` on ``x``.

    Used instead of a logistic fit: for a binary target the LS slope has the
    same sign as the logistic coefficient, and it cannot fail to converge.
    """
    if len(x) < 5:
        return 0.0
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return 0.0
    cov = float(np.cov(np.vstack([x, y]))[0, 1])
    if cov > 0:
        return 1.0
    if cov < 0:
        return -1.0
    return 0.0


def _empty_stability_row(feature, flags):
    # type: (str, List[str]) -> Dict[str, Any]
    return {
        "feature": feature,
        "n_vintages": 0,
        "assessed": False,
        "psi_max": float("nan"),
        "psi_mean": float("nan"),
        "gini_reference": float("nan"),
        "gini_min": float("nan"),
        "gini_median": float("nan"),
        "gini_ratio_min": float("nan"),
        "gini_ratio_median": float("nan"),
        "weak_vintage_share": float("nan"),
        "iv_reference": float("nan"),
        "iv_min": float("nan"),
        "iv_cv": float("nan"),
        "sign_consistency": float("nan"),
        "stability_score": float("nan"),
        "stable": True,
        "stability_flags": ",".join(flags),
    }


def _score_from_parts(psi_max, weak_vintage_share, sign_consistency, gates):
    # type: (float, float, float, Gates) -> float
    """Composite in ``[0, 1]``: drift, persistence of power, and sign agreement."""
    parts = []
    if np.isfinite(psi_max) and gates.stability_psi_max > 0:
        parts.append(float(np.clip(1.0 - psi_max / gates.stability_psi_max, 0.0, 1.0)))
    if np.isfinite(weak_vintage_share) and gates.stability_max_weak_vintage_share > 0:
        parts.append(
            float(
                np.clip(
                    1.0 - weak_vintage_share / gates.stability_max_weak_vintage_share, 0.0, 1.0
                )
            )
        )
    if np.isfinite(sign_consistency):
        parts.append(float(np.clip(sign_consistency, 0.0, 1.0)))
    if not parts:
        return float("nan")
    return float(np.mean(parts))


def predictor_stability(
    df,
    pred_cols,
    target_col,
    date_col,
    gates=None,
    grouping=None,
    freq="M",
    min_rows_per_vintage=30,
    n_jobs=None,
):
    # type: (pd.DataFrame, Sequence[str], str, Optional[str], Optional[Gates], Optional[BinningModel], str, int, Optional[int]) -> pd.DataFrame
    """One stability row per predictor.  Parallel over predictors."""
    gates = gates or Gates()
    features = [c for c in pred_cols if c in df.columns]
    if not features or target_col not in df.columns:
        return pd.DataFrame(columns=list(STABILITY_COLUMNS))

    frame = df.reset_index(drop=True)
    y_all = frame[target_col].to_numpy(dtype=float)
    if date_col is None or date_col not in frame.columns:
        rows = [_empty_stability_row(f, ["no_date_column"]) for f in features]
        return pd.DataFrame(rows)[list(STABILITY_COLUMNS)]

    periods = vintage_labels(frame[date_col], freq=freq)
    counts = periods.value_counts(dropna=True)
    kept = sorted([p for p in counts.index if counts[p] >= min_rows_per_vintage], key=str)
    if grouping is None:
        if date_col is not None and date_col in frame.columns:
            fit_frame = frame[list(features) + [c for c in [date_col] if c not in features]]
            grouping = BinningModel.fit(
                fit_frame, frame[target_col], gates=gates, n_jobs=1, date_col=date_col
            )
        else:
            grouping = BinningModel.fit(frame[features], frame[target_col], gates=gates, n_jobs=1)

    def _one(feature):
        # type: (str) -> Dict[str, Any]
        spec = grouping.specs.get(feature)
        if spec is None:
            return _empty_stability_row(feature, ["no_grouping"])
        labels_all = spec.assign(frame[feature])
        woe_all = spec.transform_woe(frame[feature])
        order = spec.bin_order()
        ref_gini = gini(y_all, -woe_all)
        ref_iv = float(spec.iv)
        ref_sign = _slope_sign(woe_all, y_all)
        if len(kept) < max(int(gates.stability_min_vintages), 2):
            row = _empty_stability_row(feature, ["insufficient_vintages"])
            row["n_vintages"] = len(kept)
            row["gini_reference"] = ref_gini
            row["iv_reference"] = ref_iv
            return row
        if not np.isfinite(ref_gini) or abs(ref_gini) < float(gates.stability_min_reference_gini):
            # A predictor with no univariate signal has no stability to speak
            # of; judging its sign or power over vintages would be judging
            # noise, so it is reported but never allowed to veto a refit.
            row = _empty_stability_row(feature, ["no_reference_signal"])
            row["n_vintages"] = len(kept)
            row["gini_reference"] = ref_gini
            row["iv_reference"] = ref_iv
            return row

        psis = []  # type: List[float]
        ginis = []  # type: List[float]
        ivs = []  # type: List[float]
        signs = []  # type: List[float]
        weak = []  # type: List[bool]
        floor = float(gates.stability_gini_ratio_min) * abs(ref_gini)
        for period in kept:
            mask = (periods == period).to_numpy()
            if mask.sum() < min_rows_per_vintage:
                continue
            y_v = y_all[mask]
            labels_v = labels_all[mask]
            woe_v = woe_all[mask]
            psis.append(psi_from_labels(labels_v, labels_all, order=order)["psi"])
            g_v = gini(y_v, -woe_v)
            ginis.append(g_v)
            ivs.append(_iv_from_labels(labels_v, y_v, order))
            signs.append(_slope_sign(woe_v, y_v))
            se_v = gini_standard_error(y_v, -woe_v)
            if not np.isfinite(g_v):
                weak.append(False)
            else:
                bound = abs(g_v) + (gates.delta_gini_z * se_v if np.isfinite(se_v) else 0.0)
                weak.append(bool(bound < floor))

        psi_arr = np.asarray(psis, dtype=float)
        gini_arr = np.asarray(ginis, dtype=float)
        iv_arr = np.asarray(ivs, dtype=float)
        sign_arr = np.asarray(signs, dtype=float)
        finite_gini = gini_arr[np.isfinite(gini_arr)]
        finite_iv = iv_arr[np.isfinite(iv_arr)]
        gini_min = float(np.min(np.abs(finite_gini))) if len(finite_gini) else float("nan")
        gini_median = float(np.median(finite_gini)) if len(finite_gini) else float("nan")
        ratio_min = (
            float(gini_min / abs(ref_gini)) if np.isfinite(gini_min) else float("nan")
        )
        ratio_median = (
            float(abs(gini_median) / abs(ref_gini)) if np.isfinite(gini_median) else float("nan")
        )
        weak_share = float(np.mean(weak)) if len(weak) else float("nan")
        iv_cv = (
            float(np.std(finite_iv, ddof=1) / np.mean(finite_iv))
            if len(finite_iv) >= 2 and np.mean(finite_iv) > 1e-12
            else float("nan")
        )
        if ref_sign == 0 or not len(sign_arr):
            sign_consistency = float("nan")
        else:
            sign_consistency = float(np.mean(sign_arr == ref_sign))

        psi_max = float(np.nanmax(psi_arr)) if len(psi_arr) else float("nan")
        psi_mean = float(np.nanmean(psi_arr)) if len(psi_arr) else float("nan")
        score = _score_from_parts(psi_max, weak_share, sign_consistency, gates)

        flags = []  # type: List[str]
        if np.isfinite(psi_max) and psi_max > gates.stability_psi_max:
            flags.append("distribution_drift")
        if np.isfinite(weak_share) and weak_share > gates.stability_max_weak_vintage_share:
            flags.append("power_loss_in_vintage")
        if np.isfinite(ratio_median) and ratio_median < gates.stability_gini_ratio_min:
            flags.append("power_loss_median")
        if np.isfinite(sign_consistency) and sign_consistency < gates.stability_sign_consistency_min:
            flags.append("sign_flip")
        if np.isfinite(score) and score < gates.stability_score_min:
            flags.append("low_stability_score")
        overlap_pairs = grouping.overlapping_event_rate_pairs(
            frame,
            y_all,
            date_col,
            feature,
            freq=freq,
            min_rows=min_rows_per_vintage,
            z=float(getattr(gates, "delta_gini_z", 1.64) or 1.64),
        )
        if overlap_pairs:
            flags.append("overlapping_event_rate_bounds")
        return {
            "feature": feature,
            "n_vintages": len(psis),
            "assessed": True,
            "psi_max": psi_max,
            "psi_mean": psi_mean,
            "gini_reference": ref_gini,
            "gini_min": gini_min,
            "gini_median": gini_median,
            "gini_ratio_min": ratio_min,
            "gini_ratio_median": ratio_median,
            "weak_vintage_share": weak_share,
            "iv_reference": ref_iv,
            "iv_min": float(np.min(finite_iv)) if len(finite_iv) else float("nan"),
            "iv_cv": iv_cv,
            "sign_consistency": sign_consistency,
            "stability_score": score,
            "stable": not flags,
            "stability_flags": ",".join(flags),
        }

    jobs = n_jobs if n_jobs is not None else gates.n_jobs
    rows = map_jobs(_one, features, n_jobs=jobs, cap=gates.max_workers_cap)
    return pd.DataFrame(rows)[list(STABILITY_COLUMNS)]


def _iv_from_labels(labels, y, order=None):
    # type: (Sequence[Any], np.ndarray, Optional[Sequence[str]]) -> float
    from scorecard_segment_eval.binning import woe_table

    table = woe_table(labels, y)
    if table.empty:
        return float("nan")
    return float(table["iv_part"].sum())


def stability_summary(table, gates=None):
    # type: (pd.DataFrame, Optional[Gates]) -> Dict[str, Any]
    """Collapse a stability table into a refit gate decision.

    A refit is accepted on stability grounds only when no *assessed* predictor
    is flagged and the weakest assessed score clears
    ``Gates.stability_score_min``.  When nothing could be assessed the gate
    passes but says so, so it never silently blocks an otherwise sound refit.
    """
    gates = gates or Gates()
    out = {
        "stability_assessed": False,
        "stability_score": float("nan"),
        "stability_pass": True,
        "stability_n_predictors": 0,
        "stability_n_unstable": 0,
        "unstable_predictors": "",
        "stability_reason": "not_assessed",
    }
    if table is None or len(table) == 0:
        return out
    out["stability_n_predictors"] = int(len(table))
    assessed = table.loc[table["assessed"].fillna(False).astype(bool)] if "assessed" in table else table
    if len(assessed) == 0:
        out["stability_reason"] = "insufficient_vintages"
        return out
    scores = pd.to_numeric(assessed["stability_score"], errors="coerce").dropna()
    weakest = float(scores.min()) if len(scores) else float("nan")
    unstable = assessed.loc[~assessed["stable"].fillna(True).astype(bool)]
    out["stability_assessed"] = True
    out["stability_score"] = weakest
    out["stability_n_unstable"] = int(len(unstable))
    out["unstable_predictors"] = ",".join(str(f) for f in unstable["feature"].tolist())
    score_ok = (not np.isfinite(weakest)) or weakest >= gates.stability_score_min
    passed = bool(len(unstable) == 0 and score_ok)
    out["stability_pass"] = passed
    if passed:
        out["stability_reason"] = "stable"
    elif len(unstable):
        flags = sorted(set(",".join(unstable["stability_flags"].fillna("").tolist()).split(",")) - set([""]))
        out["stability_reason"] = ";".join(flags) if flags else "unstable"
    else:
        out["stability_reason"] = "low_stability_score"
    return out
