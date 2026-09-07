"""Characteristic diagnostics: grouping/WoE-bin PSI vs portfolio-decile PSI for numeric."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scorecard_segment_eval.compat import as_str_keys, is_numeric_series, safe_groupby
from scorecard_segment_eval.grouping import apply_grouping, load_grouping
from scorecard_segment_eval.metrics import gini
from scorecard_segment_eval.schema import Gates, ScorecardColumns


def _quantile_edges(s, n_bins):
    # type: (pd.Series, int) -> Optional[np.ndarray]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.nunique() < 3:
        return None
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    try:
        edges = np.unique(np.quantile(s, qs))
    except Exception:
        edges = np.unique(np.percentile(s, qs * 100.0))
    if len(edges) < 3:
        return None
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _woe_by_bin(y, bins):
    eps = 0.5
    n_bad = float(y.sum())
    n_good = float(len(y) - n_bad)
    frame = pd.DataFrame({"y": y, "bin": bins})
    g = safe_groupby(frame, "bin", dropna=False)
    bad = g["y"].sum()
    tot = g.size()
    good = tot - bad
    woe = np.log(((good + eps) / max(n_good, 1.0)) / ((bad + eps) / max(n_bad, 1.0)))
    return woe


def _iv(y, bins):
    eps = 0.5
    n_bad = float(y.sum()) + eps * 2
    n_good = float(len(y) - y.sum()) + eps * 2
    frame = pd.DataFrame({"y": y, "bin": bins})
    g = safe_groupby(frame, "bin", dropna=False)
    bad = (g["y"].sum() + eps) / n_bad
    good = (g.size() - g["y"].sum() + eps) / n_good
    woe = np.log(good / bad)
    return float(((good - bad) * woe).sum())


def _psi(share_seg, share_all):
    aligned = pd.concat([share_seg.rename("s"), share_all.rename("a")], axis=1).fillna(1e-6)
    aligned = aligned.clip(lower=1e-6)
    return float(((aligned["s"] - aligned["a"]) * np.log(aligned["s"] / aligned["a"])).sum())


def looks_like_woe_feature(name):
    n = str(name).lower()
    return n.endswith("_woe") or n.endswith("_bin") or "woe" in n.split("_")


def _feature_bins(s_all, s_seg, feat, cols, gates, grouping):
    """Return (bins_all, bins_seg, method).

    Categorical / WoE / grouping.json → use those bins.
    Numeric raw features → portfolio-level deciles, then apply to the segment.
    """
    grouping = grouping or {}
    if feat in grouping:
        spec = grouping[feat]
        return apply_grouping(s_all, spec), apply_grouping(s_seg, spec), "grouping"

    woe_cols = list(cols.cols_pred_woe or [])
    named_woe = feat in woe_cols or looks_like_woe_feature(feat)
    nuniq = int(s_all.nunique(dropna=True))
    # Discrete WoE/category bins: non-numeric, or numeric with few levels.
    # Continuous WoE *values* (many unique floats) still use portfolio deciles.
    discrete = (not is_numeric_series(s_all)) or (named_woe and nuniq <= max(30, int(getattr(gates, "n_psi_deciles", 10) or 10) * 3))
    if discrete:
        return as_str_keys(s_all), as_str_keys(s_seg), "categorical"

    n_decile = int(getattr(gates, "n_psi_deciles", 10) or 10)
    edges = _quantile_edges(s_all, n_decile)
    if edges is None:
        return as_str_keys(s_all), as_str_keys(s_seg), "categorical"
    b_all = as_str_keys(pd.cut(pd.to_numeric(s_all, errors="coerce"), bins=edges, include_lowest=True))
    b_seg = as_str_keys(pd.cut(pd.to_numeric(s_seg, errors="coerce"), bins=edges, include_lowest=True))
    return b_all, b_seg, "decile"


def feature_diagnostics(df_seg, df_all, cols, gates, grouping=None):
    # type: (pd.DataFrame, pd.DataFrame, ScorecardColumns, Gates, Optional[Dict[str, Any]]) -> pd.DataFrame
    y_seg = df_seg[cols.col_target].to_numpy(dtype=float)
    y_all = df_all[cols.col_target].to_numpy(dtype=float)
    features = list(dict.fromkeys(list(cols.cols_pred) + list(cols.cols_pred_woe) + list(cols.cols_pred_used)))
    grouping = grouping if grouping is not None else load_grouping(
        getattr(cols, "grouping_path", None) or getattr(gates, "grouping_path", None)
    )
    rows = []  # type: List[dict]
    for feat in features:
        if feat not in df_seg.columns or feat not in df_all.columns:
            continue
        s_all = df_all[feat]
        s_seg = df_seg[feat]
        b_all, b_seg, method = _feature_bins(s_all, s_seg, feat, cols, gates, grouping)
        b_all_np = b_all.to_numpy() if hasattr(b_all, "to_numpy") else np.asarray(b_all)
        b_seg_np = b_seg.to_numpy() if hasattr(b_seg, "to_numpy") else np.asarray(b_seg)

        woe_all = _woe_by_bin(y_all, b_all_np)
        woe_seg = _woe_by_bin(y_seg, b_seg_np)
        aligned = pd.concat([woe_all.rename("all"), woe_seg.rename("seg")], axis=1).dropna()
        if len(aligned) >= 3:
            corr = float(spearmanr(aligned["all"], aligned["seg"]).correlation)
        else:
            corr = float("nan")
        share_all = pd.Series(b_all_np).value_counts(normalize=True)
        share_seg = pd.Series(b_seg_np).value_counts(normalize=True)
        numeric = is_numeric_series(s_seg) and method == "decile"
        if numeric:
            uni_g = gini(y_seg, pd.to_numeric(s_seg, errors="coerce").fillna(s_seg.rank(method="average")))
        else:
            uni_g = gini(y_seg, pd.Series(b_seg_np).astype("category").cat.codes.astype(float))
        rows.append(
            {
                "feature": feat,
                "iv_segment": _iv(y_seg, b_seg_np),
                "iv_overall": _iv(y_all, b_all_np),
                "psi": _psi(share_seg, share_all),
                "woe_spearman": corr,
                "univariate_gini_segment": uni_g,
                "rank_reversal": bool(np.isfinite(corr) and corr < gates.woe_spearman_reversal),
                "psi_method": method,
            }
        )
    return pd.DataFrame(rows)


def shape_divergence(char_table, gates):
    # type: (pd.DataFrame, Gates) -> bool
    if char_table.empty:
        return False
    psi_hit = bool((char_table["psi"] >= gates.psi_woe_material).any())
    reverse = bool(char_table["rank_reversal"].fillna(False).any())
    return psi_hit or reverse
