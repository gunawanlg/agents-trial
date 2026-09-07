"""Same-predictor refit: optimized WoE, optional XGBoost submodel, vintage-stable selection."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from scorecard_segment_eval.compat import as_str_keys, is_numeric_series, safe_groupby, to_numpy, warn_user
from scorecard_segment_eval.metrics import gini
from scorecard_segment_eval.parallel import run_parallel
from scorecard_segment_eval.schema import Gates


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _quantile_edges(s, n_bins):
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


def _tree_edges(x, y, max_leaf=8, min_samples=50, max_depth=4, random_state=0):
    """Supervised bin edges from a shallow decision tree (opt-binning fallback)."""
    xv = pd.to_numeric(pd.Series(x), errors="coerce")
    yv = np.asarray(y, dtype=float)
    mask = xv.notna().to_numpy() if hasattr(xv.notna(), "to_numpy") else np.asarray(xv.notna())
    if mask.sum() < max(int(min_samples) * 2, 30):
        return None
    y_fit = yv[mask]
    if np.unique(y_fit).size < 2:
        return None
    x_fit = xv.to_numpy()[mask].reshape(-1, 1)
    leaf = max(2, int(max_leaf))
    min_leaf = max(5, int(min_samples))
    tree = DecisionTreeClassifier(
        max_leaf_nodes=leaf,
        min_samples_leaf=min_leaf,
        max_depth=int(max_depth),
        random_state=int(random_state),
    )
    try:
        tree.fit(x_fit, y_fit.astype(int))
    except Exception:
        return None
    thresholds = [t for t in tree.tree_.threshold if t != -2]
    if not thresholds:
        return None
    edges = np.unique(np.asarray(thresholds, dtype=float))
    return np.concatenate(([-np.inf], edges, [np.inf]))


def _optbinning_edges(x, y, max_bins=10):
    try:
        from optbinning import OptimalBinning
    except Exception:
        return None
    xv = pd.to_numeric(pd.Series(x), errors="coerce")
    yv = np.asarray(y, dtype=float)
    mask = xv.notna().to_numpy() if hasattr(xv.notna(), "to_numpy") else np.asarray(xv.notna())
    if mask.sum() < 40 or np.unique(yv[mask]).size < 2:
        return None
    try:
        optb = OptimalBinning(name="x", dtype="numerical", max_n_bins=int(max_bins), solver="cp")
        optb.fit(xv.to_numpy()[mask], yv[mask].astype(int))
        splits = getattr(optb, "splits", None)
        if splits is None or len(splits) == 0:
            return None
        return np.concatenate(([-np.inf], np.asarray(splits, dtype=float), [np.inf]))
    except Exception:
        return None


def _resolve_edges(s, y, n_bins, strategy, max_leaf, min_bin, max_depth, random_state=0):
    strategy = (strategy or "tree").lower()
    edges = None
    if strategy == "optbinning":
        edges = _optbinning_edges(s, y, max_bins=n_bins)
        if edges is None:
            edges = _tree_edges(s, y, max_leaf=max_leaf, min_samples=min_bin, max_depth=max_depth, random_state=random_state)
    elif strategy == "tree":
        edges = _tree_edges(s, y, max_leaf=max_leaf, min_samples=min_bin, max_depth=max_depth, random_state=random_state)
    if edges is None:
        edges = _quantile_edges(s, n_bins)
    return edges


class WoEEncoder(object):
    """Train-only WoE. Numeric bins from a tree / opt-binning / quantiles; unseen -> 0."""

    def __init__(
        self,
        n_bins=10,
        strategy="tree",
        tree_max_leaf=8,
        tree_min_bin=50,
        tree_max_depth=4,
        n_jobs=1,
        random_state=0,
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.tree_max_leaf = tree_max_leaf
        self.tree_min_bin = tree_min_bin
        self.tree_max_depth = tree_max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.edges_ = {}  # type: Dict[str, Optional[np.ndarray]]
        self.woe_ = {}  # type: Dict[str, Dict[str, float]]
        self.columns_ = []  # type: List[str]
        self.bin_method_ = {}  # type: Dict[str, str]

    def _fit_column(self, payload):
        col, s, yv, n_bad, n_good, eps = payload
        numeric = is_numeric_series(s) and s.nunique(dropna=True) > min(3, self.n_bins)
        method = "categorical"
        edges = None
        if numeric:
            edges = _resolve_edges(
                s,
                yv,
                n_bins=self.n_bins,
                strategy=self.strategy,
                max_leaf=self.tree_max_leaf,
                min_bin=self.tree_min_bin,
                max_depth=self.tree_max_depth,
                random_state=self.random_state,
            )
            if edges is None or len(edges) < 3:
                numeric = False
                edges = None
            else:
                method = self.strategy if self.strategy in ("tree", "optbinning", "quantile") else "tree"
                if self.strategy == "optbinning" and method:
                    method = "optbinning" if _optbinning_edges is not None else "tree"
        if numeric and edges is not None:
            keys = as_str_keys(pd.cut(pd.to_numeric(s, errors="coerce"), bins=edges, include_lowest=True))
        else:
            edges = None
            method = "categorical"
            keys = as_str_keys(s)
        frame = pd.DataFrame({"y": yv, "bin": to_numpy(keys)})
        g = safe_groupby(frame, "bin", dropna=False)
        bad = g["y"].sum()
        tot = g.size()
        good = tot - bad
        woe = np.log(((good + eps) / max(n_good, 1.0)) / ((bad + eps) / max(n_bad, 1.0)))
        return col, edges, woe.to_dict(), method

    def fit(self, X, y):
        # type: (pd.DataFrame, pd.Series) -> "WoEEncoder"
        yv = to_numpy(y, dtype=float)
        n_bad = float(yv.sum())
        n_good = float(len(yv) - n_bad)
        eps = 0.5
        self.columns_ = list(X.columns)
        payloads = [(col, X[col], yv, n_bad, n_good, eps) for col in X.columns]
        fitted = run_parallel(self._fit_column, payloads, n_jobs=self.n_jobs, prefer="threads")
        for col, edges, mapping, method in fitted:
            self.edges_[col] = edges
            self.woe_[col] = mapping
            self.bin_method_[col] = method
        return self

    def transform(self, X):
        # type: (pd.DataFrame) -> np.ndarray
        cols = []
        for col in self.columns_:
            s = X[col]
            edges = self.edges_[col]
            if edges is not None:
                keys = as_str_keys(pd.cut(pd.to_numeric(s, errors="coerce"), bins=edges, include_lowest=True))
            else:
                keys = as_str_keys(s)
            mapping = self.woe_[col]
            mapped = keys.map(lambda k: mapping.get(k, 0.0))
            cols.append(to_numpy(mapped, dtype=float))
        if cols:
            return np.column_stack(cols)
        return np.zeros((len(X), 0))


def fit_segment_lr(X_woe, y, C=1e6):
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    model.fit(X_woe, y)
    return model


def _fit_xgb_submodel(X, y, max_depth=3, n_estimators=80, learning_rate=0.08, random_state=0):
    try:
        from xgboost import XGBClassifier
    except Exception:
        raise ImportError(
            "submodel=True requires xgboost. Install xgboost or set submodel=False to use logistic regression."
        )
    depth = min(int(max_depth), 3)
    params = dict(
        max_depth=depth,
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
        n_jobs=1,
        verbosity=0,
    )
    model = None
    last_err = None
    for extra in (dict(eval_metric="logloss", objective="binary:logistic"), dict(objective="binary:logistic"), dict()):
        try:
            model = XGBClassifier(**params)
            if extra:
                model.set_params(**extra)
            model.fit(X, y)
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            model = None
    if model is None:
        raise RuntimeError("XGBoost submodel failed to fit: " + str(last_err))
    return model


def _predict_proba_positive(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(X))
    return _sigmoid(np.asarray(model.predict(X), dtype=float))


def predictor_vintage_stability(
    df,
    pred_cols,
    target_col,
    date_col,
    n_bins=10,
    min_vintages=3,
    psi_max=0.25,
    gini_cv_max=0.50,
):
    """Per-predictor stability across monthly vintages (PSI first-vs-last, Gini CV, sign flips)."""
    rows = []  # type: List[Dict[str, Any]]
    if date_col is None or date_col not in df.columns or df.empty:
        for col in pred_cols:
            rows.append(
                {
                    "feature": col,
                    "n_vintages": 0,
                    "psi_first_last": float("nan"),
                    "gini_cv": float("nan"),
                    "sign_flips": 0,
                    "stable": True,
                    "reason": "no_date",
                }
            )
        return pd.DataFrame(rows)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    months = dates.dt.to_period("M").astype(str)
    y = to_numpy(df[target_col], dtype=float)
    vintages = [v for v in sorted(pd.unique(months)) if v == v and str(v) != "NaT"]
    if len(vintages) < 2:
        for col in pred_cols:
            rows.append(
                {
                    "feature": col,
                    "n_vintages": len(vintages),
                    "psi_first_last": float("nan"),
                    "gini_cv": float("nan"),
                    "sign_flips": 0,
                    "stable": True,
                    "reason": "too_few_vintages",
                }
            )
        return pd.DataFrame(rows)

    first, last = vintages[0], vintages[-1]
    for col in pred_cols:
        s = df[col]
        ginis = []
        signs = []
        for v in vintages:
            part_mask = months == v
            if int(np.sum(part_mask)) < 30:
                continue
            y_v = y[part_mask.to_numpy() if hasattr(part_mask, "to_numpy") else np.asarray(part_mask)]
            if np.unique(y_v).size < 2:
                continue
            x_v = pd.to_numeric(s[part_mask], errors="coerce")
            if is_numeric_series(s):
                score = x_v.fillna(x_v.median() if x_v.notna().any() else 0.0)
            else:
                score = s[part_mask].astype("category").cat.codes.astype(float)
            g = gini(y_v, to_numpy(score, dtype=float))
            ginis.append(g)
            corr = np.corrcoef(to_numpy(score, dtype=float), y_v)
            signs.append(np.sign(corr[0, 1]) if corr.shape == (2, 2) and np.isfinite(corr[0, 1]) else 0.0)

        psi_val = float("nan")
        if is_numeric_series(s):
            edges = _quantile_edges(s, n_bins)
            if edges is not None:
                b_first = as_str_keys(pd.cut(pd.to_numeric(s[months == first], errors="coerce"), bins=edges, include_lowest=True))
                b_last = as_str_keys(pd.cut(pd.to_numeric(s[months == last], errors="coerce"), bins=edges, include_lowest=True))
                share_f = pd.Series(to_numpy(b_first)).value_counts(normalize=True)
                share_l = pd.Series(to_numpy(b_last)).value_counts(normalize=True)
                aligned = pd.concat([share_l.rename("s"), share_f.rename("a")], axis=1).fillna(1e-6).clip(lower=1e-6)
                psi_val = float(((aligned["s"] - aligned["a"]) * np.log(aligned["s"] / aligned["a"])).sum())
        else:
            share_f = as_str_keys(s[months == first]).value_counts(normalize=True)
            share_l = as_str_keys(s[months == last]).value_counts(normalize=True)
            aligned = pd.concat([share_l.rename("s"), share_f.rename("a")], axis=1).fillna(1e-6).clip(lower=1e-6)
            psi_val = float(((aligned["s"] - aligned["a"]) * np.log(aligned["s"] / aligned["a"])).sum())

        ginis_f = [g for g in ginis if np.isfinite(g)]
        if ginis_f:
            mean_g = float(np.mean(np.abs(ginis_f)))
            gini_cv = float(np.std(ginis_f, ddof=1) / mean_g) if len(ginis_f) >= 2 and mean_g > 1e-9 else 0.0
        else:
            gini_cv = float("nan")
        sign_vals = [sg for sg in signs if sg != 0]
        flips = 0
        for i in range(1, len(sign_vals)):
            if sign_vals[i] != sign_vals[i - 1]:
                flips += 1

        reasons = []
        if np.isfinite(psi_val) and psi_val >= psi_max:
            reasons.append("psi")
        if np.isfinite(gini_cv) and gini_cv >= gini_cv_max and len(ginis_f) >= min_vintages:
            reasons.append("gini_cv")
        if flips > 0 and len(sign_vals) >= min_vintages:
            reasons.append("sign_flip")
        stable = len(reasons) == 0
        rows.append(
            {
                "feature": col,
                "n_vintages": len(ginis_f),
                "psi_first_last": psi_val,
                "gini_cv": gini_cv,
                "sign_flips": flips,
                "stable": stable,
                "reason": ",".join(reasons) if reasons else "ok",
            }
        )
    return pd.DataFrame(rows)


class RefitResult(object):
    def __init__(self, p_holdout, p_holdout_stable=None, dropped=None, stability=None, model_kind="lr", used_pred=None):
        self.p_holdout = p_holdout
        self.p_holdout_stable = p_holdout_stable if p_holdout_stable is not None else p_holdout
        self.dropped = list(dropped or [])
        self.stability = stability if stability is not None else pd.DataFrame()
        self.model_kind = model_kind
        self.used_pred = list(used_pred or [])


def _encoder_from_gates(gates):
    gates = gates or Gates()
    return WoEEncoder(
        n_bins=gates.n_woe_bins,
        strategy=getattr(gates, "woe_strategy", "tree"),
        tree_max_leaf=getattr(gates, "woe_tree_max_leaf", 8),
        tree_min_bin=getattr(gates, "woe_tree_min_bin", 50),
        tree_max_depth=getattr(gates, "woe_tree_max_depth", 4),
        n_jobs=getattr(gates, "n_jobs", 1),
        random_state=getattr(gates, "bootstrap_seed", 0),
    )


def _fit_predict_model(x_tr, y_tr, x_ho, submodel, gates):
    y_tr = np.asarray(y_tr, dtype=int)
    if submodel:
        model = _fit_xgb_submodel(
            x_tr,
            y_tr,
            max_depth=getattr(gates, "xgb_max_depth", 3),
            n_estimators=getattr(gates, "xgb_n_estimators", 80),
            learning_rate=getattr(gates, "xgb_learning_rate", 0.08),
            random_state=getattr(gates, "bootstrap_seed", 0),
        )
        kind = "xgboost"
        C = None
    else:
        kind = "lr"
        C = 1e6
        model = fit_segment_lr(x_tr, y_tr, C=C)
    return _predict_proba_positive(model, x_ho), kind


def refit_segment_model(
    train,
    holdout,
    pred_cols,
    target_col,
    gates=None,
    date_col=None,
    submodel=None,
    **kwargs
):
    """Fit a segment model on train, score holdout.

    ``submodel=True`` uses XGBoost with max_depth<=3 (at most 3-feature interactions).
    Unstable predictors (vintage PSI / Gini CV / sign flip) are dropped for the
    stability-constrained score; the performance fit still uses all predictors.
    """
    gates = gates or Gates()
    if submodel is None:
        submodel = bool(kwargs.get("submodel", getattr(gates, "submodel", False)))
    pred_cols = [c for c in list(pred_cols) if c in train.columns]
    if not pred_cols:
        raise ValueError("refit_segment_model requires at least one predictor column")

    date_col = date_col or kwargs.get("date_col")
    stability = predictor_vintage_stability(
        train,
        pred_cols,
        target_col,
        date_col,
        n_bins=gates.n_woe_bins,
        min_vintages=getattr(gates, "stability_min_vintages", 3),
        psi_max=getattr(gates, "stability_psi_max", 0.25),
        gini_cv_max=getattr(gates, "stability_gini_cv_max", 0.50),
    )
    stable_cols = pred_cols
    dropped = []  # type: List[str]
    if not stability.empty and "stable" in stability.columns:
        flagged = stability.loc[~stability["stable"].astype(bool), "feature"].tolist()
        dropped = [c for c in flagged if c in pred_cols]
        kept = [c for c in pred_cols if c not in dropped]
        if kept:
            stable_cols = kept
        else:
            dropped = []
            stable_cols = pred_cols

    enc_all = _encoder_from_gates(gates)
    enc_all.fit(train[pred_cols], train[target_col])
    x_tr = enc_all.transform(train[pred_cols])
    x_ho = enc_all.transform(holdout[pred_cols])
    p_perf, kind = _fit_predict_model(
        x_tr, train[target_col], x_ho, submodel=submodel, gates=gates
    )

    if stable_cols == pred_cols:
        p_stable = p_perf
    else:
        enc_st = _encoder_from_gates(gates)
        enc_st.fit(train[stable_cols], train[target_col])
        x_tr_s = enc_st.transform(train[stable_cols])
        x_ho_s = enc_st.transform(holdout[stable_cols])
        # Slightly stronger shrinkage when we already dropped noisy vintages
        if not submodel:
            y_tr = to_numpy(train[target_col], dtype=int)
            model = fit_segment_lr(x_tr_s, y_tr, C=10.0)
            p_stable = _predict_proba_positive(model, x_ho_s)
        else:
            p_stable, _k = _fit_predict_model(
                x_tr_s, train[target_col], x_ho_s, submodel=True, gates=gates
            )

    return RefitResult(
        p_holdout=np.asarray(p_perf, dtype=float),
        p_holdout_stable=np.asarray(p_stable, dtype=float),
        dropped=dropped,
        stability=stability,
        model_kind=kind,
        used_pred=pred_cols,
    )


def refit_same_predictors(
    train,
    holdout,
    pred_cols,
    target_col,
    gates,
    submodel=False,
    date_col=None,
    **kwargs
):
    """Backward-compatible helper: returns holdout PD from the performance refit."""
    if "submodel" in kwargs and submodel is False:
        submodel = kwargs.get("submodel", False)
    result = refit_segment_model(
        train,
        holdout,
        pred_cols,
        target_col,
        gates=gates,
        date_col=date_col,
        submodel=submodel,
        **kwargs
    )
    return result.p_holdout


def recalibrate_pd(p_train, y_train, p_holdout):
    """Two-parameter logistic rescale: logit(p*) = a + b * logit(p)."""
    z_tr = _logit(np.asarray(p_train, dtype=float)).reshape(-1, 1)
    y = np.asarray(y_train, dtype=int)
    if y.min() == y.max():
        return np.clip(np.asarray(p_holdout, dtype=float), 1e-6, 1.0 - 1e-6)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(z_tr, y)
    z_ho = _logit(np.asarray(p_holdout, dtype=float)).reshape(-1, 1)
    return model.predict_proba(z_ho)[:, 1]


def apply_recalibrate_fn(p_train, y_train, p_holdout):
    return recalibrate_pd(p_train, y_train, p_holdout)
