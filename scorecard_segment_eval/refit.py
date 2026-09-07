import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional, Tuple

from scorecard_segment_eval.schema import Gates


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class WoEEncoder:
    """Target-aware train-only WoE grouping. Unseen levels map to 0."""

    def __init__(self, n_bins=10, min_bin_fraction=0.03):
        self.n_bins = n_bins
        self.min_bin_fraction = min_bin_fraction
        self.edges_ = {}
        self.category_groups_ = {}
        self.woe_ = {}
        self.columns_ = []

    def _numeric_edges(self, values, y):
        valid = values.notna()
        if valid.sum() < 4 or values[valid].nunique() < 3:
            return None
        tree = DecisionTreeClassifier(
            criterion="entropy",
            max_leaf_nodes=self.n_bins,
            min_samples_leaf=max(2, int(valid.sum() * self.min_bin_fraction)),
            random_state=17,
        )
        tree.fit(values[valid].to_numpy().reshape(-1, 1), y[valid])
        thresholds = tree.tree_.threshold
        thresholds = np.sort(thresholds[thresholds != -2])
        if not len(thresholds):
            return None
        return np.r_[-np.inf, thresholds, np.inf]

    def _categorical_groups(self, values, y):
        frame = pd.DataFrame({"value": values.astype(str).fillna("__na__"), "y": y})
        rates = frame.groupby("value")["y"].mean().sort_values()
        if len(rates) <= self.n_bins:
            return dict((value, value) for value in rates.index)
        ranks = dict((value, rank) for rank, value in enumerate(rates.index))
        ranked = frame["value"].map(ranks).to_numpy().reshape(-1, 1)
        tree = DecisionTreeClassifier(
            criterion="entropy",
            max_leaf_nodes=self.n_bins,
            min_samples_leaf=max(2, int(len(frame) * self.min_bin_fraction)),
            random_state=17,
        )
        tree.fit(ranked, y)
        leaves = tree.apply(np.arange(len(rates)).reshape(-1, 1))
        return dict((value, "group_%s" % leaf) for value, leaf in zip(rates.index, leaves))

    def fit(self, X, y):
        y = pd.Series(y, index=X.index)
        yv = y.to_numpy(dtype=float)
        n_bad = float(yv.sum())
        n_good = float(len(yv) - n_bad)
        eps = 0.5
        self.columns_ = list(X.columns)
        for col in X.columns:
            s = X[col]
            numeric = pd.api.types.is_numeric_dtype(s)
            if numeric:
                numeric_s = pd.to_numeric(s, errors="coerce")
                edges = self._numeric_edges(numeric_s, y)
                if edges is None:
                    numeric = False
                else:
                    self.edges_[col] = edges
                    self.category_groups_[col] = None
                    keys = pd.cut(numeric_s, bins=edges, include_lowest=True).astype(str)
            if not numeric:
                self.edges_[col] = None
                groups = self._categorical_groups(s, y)
                self.category_groups_[col] = groups
                keys = s.astype(str).fillna("__na__").map(lambda value: groups.get(value, "__other__"))
            frame = pd.DataFrame({"y": yv, "bin": keys.to_numpy()})
            g = frame.groupby("bin", dropna=False)
            bad = g["y"].sum()
            tot = g.size()
            good = tot - bad
            woe = np.log(((good + eps) / max(n_good, 1.0)) / ((bad + eps) / max(n_bad, 1.0)))
            self.woe_[col] = woe.to_dict()
        return self

    def transform(self, X):
        cols = []
        for col in self.columns_:
            s = X[col]
            edges = self.edges_[col]
            if edges is not None:
                keys = pd.cut(pd.to_numeric(s, errors="coerce"), bins=edges, include_lowest=True).astype(str)
            else:
                groups = self.category_groups_[col]
                keys = s.astype(str).fillna("__na__").map(lambda value: groups.get(value, "__other__"))
            mapping = self.woe_[col]
            cols.append(keys.map(lambda k: mapping.get(k, 0.0)).to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.zeros((len(X), 0))


def fit_segment_lr(X_woe, y):
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(X_woe, y)
    return model


def _predictor_stability(encoded, dates, n_bins=10):
    """Worst vintage-to-portfolio PSI for each encoded predictor."""
    periods = pd.to_datetime(dates, errors="coerce").dt.to_period("M").reset_index(drop=True)
    output = []
    for idx in range(encoded.shape[1]):
        values = pd.Series(encoded[:, idx])
        edges = np.unique(np.percentile(values, np.linspace(0, 100, min(n_bins, 10) + 1)))
        if len(edges) < 3:
            output.append(0.0)
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        all_share = pd.cut(values, edges).value_counts(normalize=True)
        worst = 0.0
        for period in periods.dropna().unique():
            local = pd.cut(values[(periods == period).to_numpy()], edges).value_counts(normalize=True)
            aligned = pd.concat([local.rename("local"), all_share.rename("all")], axis=1).fillna(1e-6).clip(lower=1e-6)
            worst = max(worst, float(((aligned.local - aligned["all"]) * np.log(aligned.local / aligned["all"])).sum()))
        output.append(worst)
    return np.asarray(output)


def refit_same_predictors(train, holdout, pred_cols, target_col, gates, **kwargs):
    """Refit with optimized WoE and a performance/stability model objective.

    ``submodel=True`` uses optional XGBoost; its max depth is always capped at 3.
    ``return_details=True`` returns predictions and model diagnostics.
    """
    enc = WoEEncoder(n_bins=gates.n_woe_bins)
    enc.fit(train[pred_cols], train[target_col])
    x_tr = enc.transform(train[pred_cols])
    x_ho = enc.transform(holdout[pred_cols])
    y_tr = train[target_col].to_numpy(dtype=int)
    stability = _predictor_stability(x_tr, train[kwargs.get("date_col")]) if kwargs.get("date_col") in train else np.zeros(len(pred_cols))
    stable_idx = np.where(stability <= kwargs.get("max_woe_psi", gates.max_woe_psi))[0]
    if not len(stable_idx):
        stable_idx = np.arange(len(pred_cols))

    submodel = bool(kwargs.get("submodel", False))
    XGBClassifier = None
    if submodel:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("submodel=True requires the optional 'xgboost' package")

    all_idx = np.arange(len(pred_cols))
    candidates = [all_idx]
    if not np.array_equal(all_idx, stable_idx):
        candidates.append(stable_idx)
    candidate_results = []
    for candidate_idx in candidates:
        if submodel:
            model = XGBClassifier(
                max_depth=min(int(kwargs.get("max_depth", 3)), 3),
                n_estimators=int(kwargs.get("n_estimators", 100)),
                learning_rate=float(kwargs.get("learning_rate", 0.05)),
                random_state=int(kwargs.get("random_state", 17)),
                eval_metric="logloss",
            )
            model.fit(x_tr[:, candidate_idx], y_tr)
        else:
            model = fit_segment_lr(x_tr[:, candidate_idx], y_tr)
        candidate_predictions = model.predict_proba(x_ho[:, candidate_idx])[:, 1]
        candidate_gini = float("nan")
        if target_col in holdout and holdout[target_col].nunique() > 1:
            candidate_gini = 2.0 * roc_auc_score(holdout[target_col], candidate_predictions) - 1.0
        mean_psi = float(stability[candidate_idx].mean())
        objective = candidate_gini - gates.refit_stability_penalty * mean_psi
        candidate_results.append((objective, candidate_gini, mean_psi, candidate_idx, candidate_predictions))
    chosen = max(candidate_results, key=lambda item: item[0] if np.isfinite(item[0]) else -np.inf)
    objective, holdout_gini, mean_psi, stable_idx, predictions = chosen
    details = {
        "predictor_stability": dict(zip(pred_cols, stability)),
        "selected_predictors": [pred_cols[i] for i in stable_idx],
        "mean_selected_psi": mean_psi,
        "holdout_gini": holdout_gini,
        "selection_objective": objective,
        "candidate_objectives": [item[0] for item in candidate_results],
        "submodel": submodel,
        "grouping": dict(
            (
                col,
                {
                    "edges": (
                        [float(value) for value in enc.edges_[col] if np.isfinite(value)]
                        if enc.edges_[col] is not None
                        else None
                    ),
                    "categories": enc.category_groups_[col],
                },
            )
            for col in pred_cols
        ),
    }
    return (predictions, details) if kwargs.get("return_details", False) else predictions


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
