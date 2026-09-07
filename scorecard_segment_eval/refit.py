"""Same-predictor refit, optional XGBoost sub-model, and recalibration.

Three upgrades over the original module (section A1 of the brief):

1. **Optimal WoE binning.**  :class:`WoEEncoder` now delegates to
   :mod:`scorecard_segment_eval.binning`, which prefers :mod:`optbinning` when
   installed and otherwise uses a constrained decision-tree grouping.  The
   fitted definition is exposed via :meth:`WoEEncoder.to_grouping` so it can be
   written to ``grouping.json``.
2. **Optional sub-model pillar.**  ``submodel=True`` fits an XGBoost model with
   ``max_depth`` hard-capped at 3 (at most 3-way interactions).  Without
   :mod:`xgboost` installed it warns and falls back to the logistic path
   instead of raising.
3. **Stability-aware acceptance.**  :func:`refit_with_diagnostics` returns
   per-predictor vintage stability alongside the holdout predictions, so the
   caller can require *both* a performance gain and stability.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scorecard_segment_eval.binning import BinningModel
from scorecard_segment_eval.schema import Gates
from scorecard_segment_eval.stability import predictor_stability, stability_summary

#: Absolute ceiling on sub-model depth; callers may lower it, never raise it.
MAX_SUBMODEL_DEPTH = 3

#: Sub-model defaults; every one of these is overridable by the caller except
#: ``max_depth``, which is clamped.
DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": MAX_SUBMODEL_DEPTH,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "binary:logistic",
    "n_jobs": 1,
    "verbosity": 0,
}


def _logit(p):
    # type: (np.ndarray) -> np.ndarray
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    # type: (np.ndarray) -> np.ndarray
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def xgboost_available():
    # type: () -> bool
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


class WoEEncoder(object):
    """Optimal-binning WoE encoder.

    Backwards compatible with the previous equal-frequency implementation:
    ``fit`` / ``transform`` keep their signatures and ``edges_``, ``woe_`` and
    ``columns_`` are still populated.  ``n_bins`` remains accepted and is used
    as the maximum bin count when no explicit gates are supplied.
    """

    def __init__(self, n_bins=10, gates=None, method=None, n_jobs=None):
        # type: (int, Optional[Gates], Optional[str], Optional[int]) -> None
        self.n_bins = n_bins
        if gates is None:
            gates = Gates(n_woe_bins=n_bins, binning_max_bins=max(int(n_bins), 2))
        if method is not None:
            gates = _with_binning_method(gates, method)
        self.gates = gates
        self.n_jobs = n_jobs
        self.model_ = None  # type: Optional[BinningModel]
        self.edges_ = {}  # type: Dict[str, Optional[np.ndarray]]
        self.woe_ = {}  # type: Dict[str, Dict[str, float]]
        self.columns_ = []  # type: List[str]

    def fit(self, X, y):
        # type: (pd.DataFrame, Any) -> "WoEEncoder"
        self.model_ = BinningModel.fit(X, y, gates=self.gates, n_jobs=self.n_jobs)
        self.columns_ = list(self.model_.columns)
        self.edges_ = {}
        self.woe_ = {}
        for col in self.columns_:
            spec = self.model_.specs[col]
            self.edges_[col] = None if spec.edges is None else np.asarray(spec.edges, dtype=float)
            self.woe_[col] = dict(spec.woe)
        return self

    def transform(self, X):
        # type: (pd.DataFrame) -> np.ndarray
        if self.model_ is None:
            raise ValueError("WoEEncoder.transform called before fit")
        return self.model_.transform(X)

    def bin_frame(self, X):
        # type: (pd.DataFrame) -> pd.DataFrame
        if self.model_ is None:
            raise ValueError("WoEEncoder.bin_frame called before fit")
        return self.model_.bin_frame(X)

    def to_grouping(self):
        # type: () -> BinningModel
        """The fitted grouping definition (serialisable to ``grouping.json``)."""
        if self.model_ is None:
            raise ValueError("WoEEncoder.to_grouping called before fit")
        return self.model_

    @classmethod
    def from_grouping(cls, model):
        # type: (BinningModel) -> "WoEEncoder"
        encoder = cls(n_bins=model.gates.binning_max_bins, gates=model.gates)
        encoder.model_ = model
        encoder.columns_ = list(model.columns)
        for col in encoder.columns_:
            spec = model.specs[col]
            encoder.edges_[col] = None if spec.edges is None else np.asarray(spec.edges, dtype=float)
            encoder.woe_[col] = dict(spec.woe)
        return encoder


def _with_binning_method(gates, method):
    # type: (Gates, str) -> Gates
    payload = dict(gates.__dict__)
    payload["binning_method"] = method
    return Gates(**payload)


def fit_segment_lr(X_woe, y):
    # type: (np.ndarray, np.ndarray) -> LogisticRegression
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(X_woe, y)
    return model


def submodel_params(gates=None, overrides=None):
    # type: (Optional[Gates], Optional[Dict[str, Any]]) -> Dict[str, Any]
    """Sub-model hyper-parameters with ``max_depth`` clamped to <= 3."""
    gates = gates or Gates()
    overrides = dict(overrides or {})
    params = dict(DEFAULT_XGB_PARAMS)
    params.update(overrides)
    cap = max(min(int(gates.submodel_max_depth), MAX_SUBMODEL_DEPTH), 1)
    requested = int(params.get("max_depth", cap))
    if requested > cap and "max_depth" in overrides:
        warnings.warn(
            "sub-model max_depth=%d exceeds the %d-way interaction cap; clamping to %d"
            % (requested, cap, cap)
        )
    params["max_depth"] = min(requested, cap)
    params.setdefault("random_state", int(gates.bootstrap_seed))
    return params


def fit_submodel(X_woe, y, gates=None, xgb_params=None):
    # type: (np.ndarray, np.ndarray, Optional[Gates], Optional[Dict[str, Any]]) -> Any
    """Fit the XGBoost pillar sub-model, or ``None`` when unavailable."""
    if not xgboost_available():
        warnings.warn(
            "submodel=True requested but the optional 'xgboost' package is not installed; "
            "falling back to the logistic refit. Install with: pip install "
            "'scorecard-segment-eval[submodel]'"
        )
        return None
    from xgboost import XGBClassifier

    params = submodel_params(gates, xgb_params)
    model = XGBClassifier(**params)
    model.fit(np.asarray(X_woe, dtype=float), np.asarray(y, dtype=int))
    return model


@dataclass
class RefitResult:
    """Everything a caller needs to accept or reject a refit."""

    p_holdout: np.ndarray
    method: str
    grouping: Optional[BinningModel] = None
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability_summary: Dict[str, Any] = field(default_factory=dict)
    coefficients: Dict[str, float] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    def stability_pass(self):
        # type: () -> bool
        return bool(self.stability_summary.get("stability_pass", True))

    def stability_score(self):
        # type: () -> float
        return float(self.stability_summary.get("stability_score", float("nan")))


def refit_with_diagnostics(
    train,
    holdout,
    pred_cols,
    target_col,
    gates=None,
    submodel=False,
    xgb_params=None,
    date_col=None,
    grouping=None,
    n_jobs=None,
):
    # type: (pd.DataFrame, pd.DataFrame, Sequence[str], str, Optional[Gates], bool, Optional[Dict[str, Any]], Optional[str], Optional[BinningModel], Optional[int]) -> RefitResult
    """Refit on ``train``, score ``holdout``, and diagnose predictor stability.

    The grouping is fitted on the training rows only (or reused when supplied),
    so the holdout stays untouched.  Stability is measured on the training rows
    as well, because acceptance has to be decidable before the refit is used.
    """
    gates = gates or Gates()
    pred_cols = [c for c in pred_cols if c in train.columns and c in holdout.columns]
    messages = []  # type: List[str]
    if not pred_cols:
        return RefitResult(
            p_holdout=np.full(len(holdout), float("nan")),
            method="none",
            messages=["no_usable_predictors"],
        )

    y_train = train[target_col].to_numpy(dtype=int)
    if grouping is None:
        encoder = WoEEncoder(n_bins=gates.n_woe_bins, gates=gates, n_jobs=n_jobs)
        encoder.fit(train[pred_cols], train[target_col])
        grouping = encoder.to_grouping()
    x_train = grouping.transform(train[pred_cols])
    x_holdout = grouping.transform(holdout[pred_cols])

    method = "logistic_woe"
    coefficients = {}  # type: Dict[str, float]
    p_holdout = None  # type: Optional[np.ndarray]
    if submodel:
        booster = fit_submodel(x_train, y_train, gates=gates, xgb_params=xgb_params)
        if booster is None:
            messages.append("xgboost_unavailable_fallback_logistic")
        else:
            method = "xgboost_submodel"
            p_holdout = np.asarray(booster.predict_proba(x_holdout)[:, 1], dtype=float)
            importances = getattr(booster, "feature_importances_", None)
            if importances is not None:
                coefficients = dict(
                    (col, float(val)) for col, val in zip(grouping.columns, list(importances))
                )
    if p_holdout is None:
        model = fit_segment_lr(x_train, y_train)
        p_holdout = np.asarray(model.predict_proba(x_holdout)[:, 1], dtype=float)
        coefficients = dict((col, float(val)) for col, val in zip(grouping.columns, model.coef_[0]))
        coefficients["__intercept__"] = float(model.intercept_[0])

    stability = predictor_stability(
        train,
        pred_cols,
        target_col,
        date_col,
        gates=gates,
        grouping=grouping,
        n_jobs=n_jobs,
    )
    summary = stability_summary(stability, gates=gates)
    return RefitResult(
        p_holdout=p_holdout,
        method=method,
        grouping=grouping,
        stability=stability,
        stability_summary=summary,
        coefficients=coefficients,
        messages=messages,
    )


def refit_same_predictors(
    train,
    holdout,
    pred_cols,
    target_col,
    gates,
    submodel=False,
    xgb_params=None,
    date_col=None,
    grouping=None,
    n_jobs=None,
):
    # type: (pd.DataFrame, pd.DataFrame, Sequence[str], str, Gates, bool, Optional[Dict[str, Any]], Optional[str], Optional[BinningModel], Optional[int]) -> np.ndarray
    """Holdout PDs from a same-predictor refit (unchanged return contract)."""
    result = refit_with_diagnostics(
        train,
        holdout,
        pred_cols,
        target_col,
        gates=gates,
        submodel=submodel,
        xgb_params=xgb_params,
        date_col=date_col,
        grouping=grouping,
        n_jobs=n_jobs,
    )
    return result.p_holdout


def recalibrate_pd(p_train, y_train, p_holdout):
    # type: (Any, Any, Any) -> np.ndarray
    """Two-parameter logistic rescale: ``logit(p*) = a + b * logit(p)``."""
    z_tr = _logit(np.asarray(p_train, dtype=float)).reshape(-1, 1)
    y = np.asarray(y_train, dtype=int)
    if y.min() == y.max():
        return np.clip(np.asarray(p_holdout, dtype=float), 1e-6, 1.0 - 1e-6)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(z_tr, y)
    z_ho = _logit(np.asarray(p_holdout, dtype=float)).reshape(-1, 1)
    return model.predict_proba(z_ho)[:, 1]


def apply_recalibrate_fn(p_train, y_train, p_holdout):
    # type: (Any, Any, Any) -> np.ndarray
    return recalibrate_pd(p_train, y_train, p_holdout)
