"""Per-characteristic diagnostics: WoE shape, IV and type-aware PSI.

PSI branches on the variable type (section A3 of the upgrade brief):

* If the model consumes the characteristic as a **categorical / WoE-binned**
  variable, PSI is computed on the *existing* bins taken from the grouping
  definition (``grouping.json`` / a fitted
  :class:`~scorecard_segment_eval.binning.BinningModel`).  No new cut points
  are invented, so the reported shift is the shift the model actually sees.
* If the characteristic is **numerical** and ungrouped, bin edges are derived
  from deciles computed at the **portfolio (reference) level** and then applied
  unchanged to the local/segment level, so both distributions are measured on
  one fixed ruler.

Both paths return the same set of keys and record which method was used, plus
explicit accounting for missing values and out-of-reference-range values.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scorecard_segment_eval.binning import (
    MISSING_LABEL,
    OTHER_LABEL,
    BinningModel,
    BinSpec,
    assign_numeric_bins,
    edge_labels,
    information_value,
    woe_table,
)
from scorecard_segment_eval.metrics import gini
from scorecard_segment_eval.parallel import map_jobs
from scorecard_segment_eval.schema import Gates, ScorecardColumns

PSI_EPSILON = 1e-6

METHOD_GROUPING = "grouping_bins"
METHOD_CATEGORICAL = "categorical_levels"
METHOD_NUMERIC_DECILES = "numeric_portfolio_deciles"
METHOD_UNAVAILABLE = "unavailable"

#: Keys every PSI computation returns, so the shape is stable across branches.
PSI_KEYS = (
    "psi",
    "psi_method",
    "psi_n_bins",
    "psi_worst_bin",
    "psi_worst_bin_contrib",
    "psi_missing_share_segment",
    "psi_missing_share_reference",
    "psi_out_of_range_share_segment",
    "psi_empty_bins",
)


def _empty_psi(method=METHOD_UNAVAILABLE):
    # type: (str) -> Dict[str, Any]
    return {
        "psi": float("nan"),
        "psi_method": method,
        "psi_n_bins": 0,
        "psi_worst_bin": None,
        "psi_worst_bin_contrib": float("nan"),
        "psi_missing_share_segment": float("nan"),
        "psi_missing_share_reference": float("nan"),
        "psi_out_of_range_share_segment": float("nan"),
        "psi_empty_bins": 0,
    }


def portfolio_decile_edges(reference_values, n_bins=10):
    # type: (Any, int) -> Optional[np.ndarray]
    """Fixed bin edges from portfolio-level quantiles of ``reference_values``.

    The outer edges are infinite so the ruler still accepts local values that
    lie outside the portfolio's observed range.
    """
    values = pd.to_numeric(pd.Series(reference_values), errors="coerce").dropna()
    if values.nunique() < 3:
        return None
    qs = np.linspace(0.0, 1.0, max(int(n_bins), 2) + 1)
    edges = np.unique(np.quantile(values.to_numpy(dtype=float), qs)).astype(float)
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi_from_labels(labels_segment, labels_reference, order=None, epsilon=PSI_EPSILON):
    # type: (Sequence[Any], Sequence[Any], Optional[Sequence[str]], float) -> Dict[str, Any]
    """PSI between two already-binned samples over a fixed bin order.

    Empty bins are floored at ``epsilon`` rather than dropped, so a bin that
    exists in the portfolio but is absent locally still contributes.
    """
    seg = pd.Series(list(labels_segment), dtype="object")
    ref = pd.Series(list(labels_reference), dtype="object")
    if len(seg) == 0 or len(ref) == 0:
        return _empty_psi()
    seg_counts = seg.value_counts()
    ref_counts = ref.value_counts()
    if order is None:
        bins = list(ref_counts.index)
        for label in seg_counts.index:
            if label not in bins:
                bins.append(label)
    else:
        bins = list(order)
        for label in list(ref_counts.index) + list(seg_counts.index):
            if label not in bins:
                bins.append(label)
    seg_share = np.asarray([float(seg_counts.get(b, 0.0)) for b in bins], dtype=float)
    ref_share = np.asarray([float(ref_counts.get(b, 0.0)) for b in bins], dtype=float)
    empty_bins = int(((seg_share == 0) | (ref_share == 0)).sum())
    seg_share = np.clip(seg_share / max(seg_share.sum(), 1.0), epsilon, None)
    ref_share = np.clip(ref_share / max(ref_share.sum(), 1.0), epsilon, None)
    contrib = (seg_share - ref_share) * np.log(seg_share / ref_share)
    worst = int(np.argmax(contrib)) if len(contrib) else -1
    out = _empty_psi()
    out.update(
        {
            "psi": float(np.nansum(contrib)),
            "psi_n_bins": int(len(bins)),
            "psi_worst_bin": str(bins[worst]) if worst >= 0 else None,
            "psi_worst_bin_contrib": float(contrib[worst]) if worst >= 0 else float("nan"),
            "psi_missing_share_segment": float((seg == MISSING_LABEL).mean()),
            "psi_missing_share_reference": float((ref == MISSING_LABEL).mean()),
            "psi_empty_bins": empty_bins,
        }
    )
    return out


def psi_for_feature(
    segment_values,
    reference_values,
    spec=None,
    n_bins=10,
    treat_as=None,
    epsilon=PSI_EPSILON,
):
    # type: (Any, Any, Optional[BinSpec], int, Optional[str], float) -> Dict[str, Any]
    """Type-aware PSI for one characteristic.

    ``spec`` is a grouping definition for the feature; when present the
    existing bins are reused.  ``treat_as`` (``"categorical"`` / ``"numeric"``)
    forces a branch, otherwise the dtype decides.
    """
    seg = pd.Series(segment_values).reset_index(drop=True)
    ref = pd.Series(reference_values).reset_index(drop=True)
    if len(seg) == 0 or len(ref) == 0:
        return _empty_psi()

    if spec is not None:
        labels_seg = spec.assign(seg)
        labels_ref = spec.assign(ref)
        out = psi_from_labels(labels_seg, labels_ref, order=spec.bin_order(), epsilon=epsilon)
        out["psi_method"] = METHOD_GROUPING
        out["psi_out_of_range_share_segment"] = _out_of_range_share(seg, ref, spec)
        return out

    if treat_as is None:
        numeric_like = pd.api.types.is_numeric_dtype(ref) and ref.nunique(dropna=True) > 2
        treat_as = "numeric" if numeric_like else "categorical"

    if treat_as == "categorical":
        labels_seg = _levels(seg)
        labels_ref = _levels(ref)
        order = list(pd.Series(labels_ref).value_counts().index)
        out = psi_from_labels(labels_seg, labels_ref, order=order, epsilon=epsilon)
        out["psi_method"] = METHOD_CATEGORICAL
        unseen = ~pd.Series(labels_seg).isin(set(order))
        out["psi_out_of_range_share_segment"] = float(unseen.mean()) if len(unseen) else float("nan")
        return out

    edges = portfolio_decile_edges(ref, n_bins=n_bins)
    if edges is None:
        labels_seg = _levels(seg)
        labels_ref = _levels(ref)
        out = psi_from_labels(labels_seg, labels_ref, epsilon=epsilon)
        out["psi_method"] = METHOD_CATEGORICAL
        out["psi_out_of_range_share_segment"] = float("nan")
        return out
    labels = edge_labels(edges)
    labels_seg = assign_numeric_bins(seg, edges, labels)
    labels_ref = assign_numeric_bins(ref, edges, labels)
    out = psi_from_labels(labels_seg, labels_ref, order=labels + [MISSING_LABEL], epsilon=epsilon)
    out["psi_method"] = METHOD_NUMERIC_DECILES
    out["psi_out_of_range_share_segment"] = _out_of_range_share(seg, ref, None)
    return out


def _levels(series):
    # type: (Any) -> np.ndarray
    s = pd.Series(series).reset_index(drop=True)
    filled = s.astype("object").where(s.notna(), MISSING_LABEL)
    return filled.map(lambda v: MISSING_LABEL if v is None else str(v)).to_numpy(dtype=object)


def _out_of_range_share(segment_values, reference_values, spec):
    # type: (Any, Any, Optional[BinSpec]) -> float
    """Share of local values outside the reference's observed numeric range."""
    if spec is not None and spec.kind != "numeric":
        levels = set()
        for members in (spec.groups or {}).values():
            levels.update(members)
        seg_levels = pd.Series(_levels(segment_values))
        known = levels | set([MISSING_LABEL])
        return float((~seg_levels.isin(known)).mean()) if len(seg_levels) else float("nan")
    seg = pd.to_numeric(pd.Series(segment_values), errors="coerce")
    ref = pd.to_numeric(pd.Series(reference_values), errors="coerce").dropna()
    if ref.empty or seg.dropna().empty:
        return float("nan")
    lo = float(ref.min())
    hi = float(ref.max())
    valid = seg.dropna()
    return float(((valid < lo) | (valid > hi)).mean())


def psi_table(
    df_segment,
    df_reference,
    features,
    grouping=None,
    gates=None,
    categorical_features=(),
    n_jobs=None,
):
    # type: (pd.DataFrame, pd.DataFrame, Sequence[str], Optional[BinningModel], Optional[Gates], Sequence[str], Optional[int]) -> pd.DataFrame
    """PSI for many characteristics, one row each, parallel over features."""
    gates = gates or Gates()
    forced = set(categorical_features or ())
    usable = [f for f in features if f in df_segment.columns and f in df_reference.columns]

    def _one(feature):
        # type: (str) -> Dict[str, Any]
        spec = None
        if grouping is not None:
            spec = grouping.specs.get(feature)
        row = psi_for_feature(
            df_segment[feature],
            df_reference[feature],
            spec=spec,
            n_bins=gates.n_woe_bins,
            treat_as="categorical" if feature in forced else None,
        )
        row["feature"] = feature
        return row

    jobs = n_jobs if n_jobs is not None else gates.n_jobs
    rows = map_jobs(_one, usable, n_jobs=jobs, cap=gates.max_workers_cap)
    if not rows:
        return pd.DataFrame(columns=["feature"] + list(PSI_KEYS))
    return pd.DataFrame(rows)[["feature"] + list(PSI_KEYS)]


# --------------------------------------------------------------------------
# WoE shape diagnostics
# --------------------------------------------------------------------------
def _bin_labels_for_diagnostics(series_ref, series_seg, spec, n_bins):
    # type: (Any, Any, Optional[BinSpec], int) -> Any
    """Return ``(labels_ref, labels_seg, method, numeric)`` on one fixed ruler."""
    if spec is not None:
        return spec.assign(series_ref), spec.assign(series_seg), METHOD_GROUPING, spec.kind == "numeric"
    numeric_like = pd.api.types.is_numeric_dtype(series_ref) and pd.Series(series_ref).nunique(dropna=True) > 2
    if not numeric_like:
        return _levels(series_ref), _levels(series_seg), METHOD_CATEGORICAL, False
    edges = portfolio_decile_edges(series_ref, n_bins=n_bins)
    if edges is None:
        return _levels(series_ref), _levels(series_seg), METHOD_CATEGORICAL, False
    labels = edge_labels(edges)
    return (
        assign_numeric_bins(series_ref, edges, labels),
        assign_numeric_bins(series_seg, edges, labels),
        METHOD_NUMERIC_DECILES,
        True,
    )


def feature_diagnostics(
    df_seg,
    df_all,
    cols,
    gates,
    grouping=None,
    n_jobs=None,
):
    # type: (pd.DataFrame, pd.DataFrame, ScorecardColumns, Gates, Optional[BinningModel], Optional[int]) -> pd.DataFrame
    """IV, WoE-shape and type-aware PSI for every model characteristic."""
    target = cols.col_target
    if target is None or target not in df_seg.columns:
        return pd.DataFrame()
    y_seg = df_seg[target].to_numpy(dtype=float)
    y_all = df_all[target].to_numpy(dtype=float)
    woe_declared = set(cols.cols_pred_woe or ())
    features = list(dict.fromkeys(list(cols.cols_pred or ()) + list(cols.cols_pred_woe or ())))
    usable = [f for f in features if f in df_seg.columns and f in df_all.columns]

    def _one(feature):
        # type: (str) -> Dict[str, Any]
        spec = grouping.specs.get(feature) if grouping is not None else None
        s_all = df_all[feature]
        s_seg = df_seg[feature]
        b_all, b_seg, method, numeric = _bin_labels_for_diagnostics(s_all, s_seg, spec, gates.n_woe_bins)
        woe_all = woe_table(b_all, y_all)["woe"]
        woe_seg = woe_table(b_seg, y_seg)["woe"]
        aligned = pd.concat([woe_all.rename("all"), woe_seg.rename("seg")], axis=1).dropna()
        if len(aligned) >= 3:
            corr = float(spearmanr(aligned["all"], aligned["seg"]).correlation)
        else:
            corr = float("nan")
        if numeric:
            scores = pd.to_numeric(s_seg, errors="coerce")
            filled = scores.fillna(scores.median() if scores.notna().any() else 0.0)
            uni_g = gini(y_seg, filled.to_numpy(dtype=float))
        else:
            uni_g = float("nan")
        row = {
            "feature": feature,
            "iv_segment": information_value(b_seg, y_seg),
            "iv_overall": information_value(b_all, y_all),
            "woe_spearman": corr,
            "univariate_gini_segment": uni_g,
            "rank_reversal": bool(np.isfinite(corr) and corr < gates.woe_spearman_reversal),
            "grouping_used": spec is not None,
        }
        row.update(
            psi_for_feature(
                s_seg,
                s_all,
                spec=spec,
                n_bins=gates.n_woe_bins,
                treat_as="categorical" if (spec is None and feature in woe_declared and not numeric) else None,
            )
        )
        return row

    jobs = n_jobs if n_jobs is not None else gates.n_jobs
    rows = map_jobs(_one, usable, n_jobs=jobs, cap=gates.max_workers_cap)
    return pd.DataFrame(rows)


def shape_divergence(char_table, gates):
    # type: (pd.DataFrame, Gates) -> bool
    if char_table.empty or "psi" not in char_table.columns:
        return False
    psi_hit = bool((char_table["psi"].fillna(0.0) >= gates.psi_woe_material).any())
    reverse = bool(char_table["rank_reversal"].fillna(False).any())
    return psi_hit or reverse
