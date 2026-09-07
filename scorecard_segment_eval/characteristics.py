import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scorecard_segment_eval.metrics import gini
from scorecard_segment_eval.schema import Gates, ScorecardColumns


def _quantile_edges(s, n_bins):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.nunique() < 3:
        return None
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(s, qs))
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _woe_by_bin(y, bins):
    eps = 0.5
    n_bad = float(y.sum())
    n_good = float(len(y) - n_bad)
    frame = pd.DataFrame({"y": y, "bin": bins})
    g = frame.groupby("bin", dropna=False)
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
    g = frame.groupby("bin", dropna=False)
    bad = (g["y"].sum() + eps) / n_bad
    good = (g.size() - g["y"].sum() + eps) / n_good
    woe = np.log(good / bad)
    return float(((good - bad) * woe).sum())


def _psi(share_seg, share_all):
    aligned = pd.concat([share_seg.rename("s"), share_all.rename("a")], axis=1).fillna(1e-6)
    aligned = aligned.clip(lower=1e-6)
    return float(((aligned["s"] - aligned["a"]) * np.log(aligned["s"] / aligned["a"])).sum())


def _configured_bins(series, grouping):
    """Apply grouping.json-style category maps or numeric cut points."""
    if not grouping:
        return series.astype(str).fillna("__na__")
    if isinstance(grouping, list):
        edges = np.unique(np.r_[-np.inf, grouping, np.inf])
        return pd.cut(pd.to_numeric(series, errors="coerce"), bins=edges, include_lowest=True).astype(str)
    if isinstance(grouping, dict):
        edges = grouping.get("edges")
        if edges is not None:
            clean = [float(value) for value in edges if np.isfinite(float(value))]
            edges = np.unique(np.r_[-np.inf, clean, np.inf])
            return pd.cut(pd.to_numeric(series, errors="coerce"), bins=edges, include_lowest=True).astype(str)
        mapping = grouping.get("mapping", grouping.get("categories", grouping))
        return series.astype(str).fillna("__na__").map(lambda value: mapping.get(value, "__other__"))
    return series.astype(str).fillna("__na__")


def feature_diagnostics(df_seg, df_all, cols, gates):
    y_seg = df_seg[cols.col_target].to_numpy(dtype=float)
    y_all = df_all[cols.col_target].to_numpy(dtype=float)
    features = list(dict.fromkeys(list(cols.cols_pred) + list(cols.cols_pred_woe)))
    rows = []
    for feat in features:
        if feat not in df_seg.columns or feat not in df_all.columns:
            continue
        s_all = df_all[feat]
        s_seg = df_seg[feat]
        numeric = pd.api.types.is_numeric_dtype(s_all) and feat not in cols.cols_pred_woe
        grouping = cols.grouping.get(feat) if getattr(cols, "grouping", None) else None
        if grouping or feat in cols.cols_pred_woe or not numeric:
            b_all = _configured_bins(s_all, grouping).to_numpy()
            b_seg = _configured_bins(s_seg, grouping).to_numpy()
            binning_source = "grouping" if grouping else ("woe_values" if feat in cols.cols_pred_woe else "categories")
        elif numeric:
            edges = _quantile_edges(s_all, gates.n_psi_bins)
            if edges is None:
                b_all = s_all.astype(str).fillna("__na__").to_numpy()
                b_seg = s_seg.astype(str).fillna("__na__").to_numpy()
                binning_source = "values"
            else:
                b_all = pd.cut(pd.to_numeric(s_all, errors="coerce"), bins=edges, include_lowest=True).astype(str)
                b_seg = pd.cut(pd.to_numeric(s_seg, errors="coerce"), bins=edges, include_lowest=True).astype(str)
                b_all = b_all.to_numpy()
                b_seg = b_seg.to_numpy()
                binning_source = "portfolio_deciles"

        woe_all = _woe_by_bin(y_all, b_all)
        woe_seg = _woe_by_bin(y_seg, b_seg)
        aligned = pd.concat([woe_all.rename("all"), woe_seg.rename("seg")], axis=1).dropna()
        if len(aligned) >= 3:
            corr = float(spearmanr(aligned["all"], aligned["seg"]).correlation)
        else:
            corr = float("nan")
        share_all = pd.Series(b_all).value_counts(normalize=True)
        share_seg = pd.Series(b_seg).value_counts(normalize=True)
        uni_g = gini(y_seg, pd.to_numeric(s_seg, errors="coerce").fillna(s_seg.rank(method="average"))) if numeric else float("nan")
        rows.append(
            {
                "feature": feat,
                "iv_segment": _iv(y_seg, b_seg),
                "iv_overall": _iv(y_all, b_all),
                "psi": _psi(share_seg, share_all),
                "psi_binning": binning_source,
                "woe_spearman": corr,
                "univariate_gini_segment": uni_g,
                "rank_reversal": bool(np.isfinite(corr) and corr < gates.woe_spearman_reversal),
            }
        )
    return pd.DataFrame(rows)


def shape_divergence(char_table, gates):
    if char_table.empty:
        return False
    psi_hit = bool((char_table["psi"] >= gates.psi_woe_material).any())
    reverse = bool(char_table["rank_reversal"].fillna(False).any())
    return psi_hit or reverse
