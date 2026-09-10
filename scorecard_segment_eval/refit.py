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

Refit and recalibration both return a :class:`FittedModelArtifact` that holds
the grouping (the WoE transformation) and the fitted estimator (logistic
regression or XGBoost) so a later run can reproduce the scores.  A refit also
compares the segment grouping against the original / portfolio grouping
(``grouping.json`` or the grouping implied by ``cols_pred_woe``) and writes
per-bin notes for any material difference.
"""

import json
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scorecard_segment_eval.binning import (
    MISSING_LABEL,
    BinSpec,
    BinningModel,
    compare_groupings,
    grouping_from_woe_columns,
)
from scorecard_segment_eval.schema import Gates, map_pred_to_woe
from scorecard_segment_eval.stability import predictor_stability, stability_summary

ARTIFACT_VERSION = 1
MODEL_FILENAME = "model.pkl"
GROUPING_FILENAME = "grouping.json"
META_FILENAME = "meta.json"
COMPARISON_FILENAME = "grouping_comparison.json"
SCORECARD_FILENAME = "scorecard.sql"

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


def _safe_path_token(value):
    # type: (Any) -> str
    text = "none" if value is None else str(value)
    text = text.strip() or "unnamed"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:80]


def _ensure_dir(path):
    # type: (str) -> str
    directory = os.path.abspath(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


@dataclass
class FittedModelArtifact:
    """Grouping + estimator bundle that can score a later sample.

    ``kind`` is ``"refit"`` (new WoE bins + LR/XGBoost) or ``"recalibrate"``
    (two-parameter logistic rescale of an existing PD).  :meth:`save` writes
    a directory containing ``grouping.json``, ``model.pkl``, ``meta.json``
    and, for logistic refits, a production-shaped ``scorecard.sql``.
    """

    kind: str
    method: str
    grouping: Optional[BinningModel] = None
    model: Any = None
    feature_names: List[str] = field(default_factory=list)
    pred_woe_map: Dict[str, str] = field(default_factory=dict)
    coefficients: Dict[str, float] = field(default_factory=dict)
    grouping_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    segment_col: Optional[str] = None
    segment_value: Optional[str] = None
    score_col: Optional[str] = None
    messages: List[str] = field(default_factory=list)

    def predict_proba(self, X, score=None):
        # type: (Any, Any) -> np.ndarray
        """Reproduce scores.  Recalibration consumes a PD; refit consumes raw X."""
        if self.model is None:
            raise ValueError("FittedModelArtifact has no model to score with")
        if self.kind == "recalibrate":
            if score is None:
                if isinstance(X, pd.DataFrame) and self.score_col and self.score_col in X.columns:
                    score = X[self.score_col]
                else:
                    score = X
            z = _logit(np.asarray(score, dtype=float)).reshape(-1, 1)
            return np.asarray(self.model.predict_proba(z)[:, 1], dtype=float)
        if self.grouping is None:
            raise ValueError("refit artifact is missing its WoE grouping")
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.feature_names if c in X.columns] or self.feature_names
            matrix = self.grouping.transform(X[cols] if cols else X)
        else:
            matrix = np.asarray(X, dtype=float)
            if self.grouping is not None and matrix.ndim == 2 and matrix.shape[1] != len(self.grouping.columns):
                matrix = self.grouping.transform(pd.DataFrame(matrix, columns=list(self.feature_names)))
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)

    def to_meta(self):
        # type: () -> Dict[str, Any]
        comparison = []
        if self.grouping_comparison is not None and not self.grouping_comparison.empty:
            comparison = json.loads(self.grouping_comparison.to_json(orient="records"))
        return {
            "version": ARTIFACT_VERSION,
            "kind": self.kind,
            "method": self.method,
            "feature_names": list(self.feature_names),
            "pred_woe_map": dict(self.pred_woe_map),
            "coefficients": dict(self.coefficients),
            "segment_col": self.segment_col,
            "segment_value": None if self.segment_value is None else str(self.segment_value),
            "score_col": self.score_col,
            "messages": list(self.messages),
            "has_grouping": self.grouping is not None,
            "has_model": self.model is not None,
            "grouping_comparison": comparison,
        }

    def _should_write_scorecard_sql(self):
        # type: () -> bool
        if self.kind == "recalibrate":
            return False
        if self.method == "xgboost_submodel":
            return False
        if self.grouping is None or not self.grouping.columns:
            return False
        return True

    def to_scorecard_sql(self, table="_SOURCETABLENAME_", score_alias="SCORE"):
        # type: (str, str) -> str
        """Render this artefact as production scorecard SQL (CASE WHEN + sigmoid)."""
        from scorecard_segment_eval.sql_model import render_scorecard_sql

        if not self._should_write_scorecard_sql():
            raise ValueError(
                "cannot render scorecard SQL for kind=%s method=%s" % (self.kind, self.method)
            )
        coefficients = dict(self.coefficients or {})
        if "__intercept__" not in coefficients and self.model is not None:
            intercept = getattr(self.model, "intercept_", None)
            if intercept is not None:
                coefficients["__intercept__"] = float(np.asarray(intercept).reshape(-1)[0])
            coef = getattr(self.model, "coef_", None)
            if coef is not None and self.grouping is not None:
                for name, val in zip(self.grouping.columns, np.asarray(coef).reshape(-1)):
                    coefficients.setdefault(name, float(val))
        return render_scorecard_sql(
            self.grouping,
            coefficients,
            table=table,
            score_alias=score_alias,
            pred_woe_map=self.pred_woe_map,
        )

    def save(self, directory):
        # type: (str) -> str
        """Write the grouping, the estimator and metadata into ``directory``."""
        directory = _ensure_dir(directory)
        meta = self.to_meta()
        with open(os.path.join(directory, META_FILENAME), "w") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        if self.grouping is not None:
            self.grouping.save(os.path.join(directory, GROUPING_FILENAME))
        if self.model is not None:
            with open(os.path.join(directory, MODEL_FILENAME), "wb") as handle:
                pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.grouping_comparison is not None and not self.grouping_comparison.empty:
            records = json.loads(self.grouping_comparison.to_json(orient="records"))
            with open(os.path.join(directory, COMPARISON_FILENAME), "w") as handle:
                json.dump(records, handle, indent=2)
        if self._should_write_scorecard_sql():
            with open(os.path.join(directory, SCORECARD_FILENAME), "w") as handle:
                handle.write(self.to_scorecard_sql())
        return directory

    @classmethod
    def load(cls, directory):
        # type: (str) -> "FittedModelArtifact"
        directory = os.path.abspath(directory)
        with open(os.path.join(directory, META_FILENAME), "r") as handle:
            meta = json.load(handle)
        grouping_path = os.path.join(directory, GROUPING_FILENAME)
        grouping = BinningModel.load(grouping_path) if os.path.isfile(grouping_path) else None
        model_path = os.path.join(directory, MODEL_FILENAME)
        model = None
        if os.path.isfile(model_path):
            with open(model_path, "rb") as handle:
                model = pickle.load(handle)
        comparison = pd.DataFrame()
        comparison_path = os.path.join(directory, COMPARISON_FILENAME)
        if os.path.isfile(comparison_path):
            with open(comparison_path, "r") as handle:
                comparison = pd.DataFrame(json.load(handle))
        elif meta.get("grouping_comparison"):
            comparison = pd.DataFrame(meta["grouping_comparison"])
        return cls(
            kind=str(meta.get("kind", "refit")),
            method=str(meta.get("method", "unknown")),
            grouping=grouping,
            model=model,
            feature_names=list(meta.get("feature_names") or []),
            pred_woe_map=dict(meta.get("pred_woe_map") or {}),
            coefficients=dict(meta.get("coefficients") or {}),
            grouping_comparison=comparison,
            segment_col=meta.get("segment_col"),
            segment_value=meta.get("segment_value"),
            score_col=meta.get("score_col"),
            messages=list(meta.get("messages") or []),
        )


def save_fitted_artifact(artifact, directory):
    # type: (FittedModelArtifact, str) -> str
    """Write one :class:`FittedModelArtifact` directory."""
    return artifact.save(directory)


def load_fitted_artifact(directory):
    # type: (str) -> FittedModelArtifact
    """Read a directory written by :func:`save_fitted_artifact`."""
    return FittedModelArtifact.load(directory)


def save_fitted_artifacts(artifacts, directory):
    # type: (Sequence[FittedModelArtifact], str) -> List[str]
    """Write every artifact under ``directory/{segment}/{value}/{kind}``."""
    directory = _ensure_dir(directory)
    written = []  # type: List[str]
    for artifact in artifacts or ():
        sub = os.path.join(
            directory,
            _safe_path_token(artifact.segment_col or "segment"),
            _safe_path_token(artifact.segment_value),
            _safe_path_token(artifact.kind),
        )
        written.append(artifact.save(sub))
    return written


@dataclass
class RefitResult:
    """Everything a caller needs to accept or reject a refit."""

    p_holdout: np.ndarray
    method: str
    grouping: Optional[BinningModel] = None
    model: Any = None
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability_summary: Dict[str, Any] = field(default_factory=dict)
    coefficients: Dict[str, float] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    grouping_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    pred_woe_map: Dict[str, str] = field(default_factory=dict)
    artifact: Optional[FittedModelArtifact] = None

    def stability_pass(self):
        # type: () -> bool
        return bool(self.stability_summary.get("stability_pass", True))

    def stability_score(self):
        # type: () -> float
        return float(self.stability_summary.get("stability_score", float("nan")))

    def save(self, directory):
        # type: (str) -> str
        artifact = self.artifact or self.to_artifact()
        return artifact.save(directory)

    def to_artifact(self, segment_col=None, segment_value=None):
        # type: (Optional[str], Optional[str]) -> FittedModelArtifact
        if self.artifact is not None:
            if segment_col is not None:
                self.artifact.segment_col = segment_col
            if segment_value is not None:
                self.artifact.segment_value = None if segment_value is None else str(segment_value)
            return self.artifact
        return FittedModelArtifact(
            kind="refit",
            method=self.method,
            grouping=self.grouping,
            model=self.model,
            feature_names=list(self.grouping.columns) if self.grouping is not None else [],
            pred_woe_map=dict(self.pred_woe_map),
            coefficients=dict(self.coefficients),
            grouping_comparison=self.grouping_comparison,
            segment_col=segment_col,
            segment_value=None if segment_value is None else str(segment_value),
            messages=list(self.messages),
        )


def _resolve_portfolio_grouping(
    portfolio_grouping,
    pred_woe_map,
    portfolio_frame,
    pred_cols,
    target_col,
    gates,
    n_jobs,
):
    # type: (Optional[BinningModel], Dict[str, str], Optional[pd.DataFrame], Sequence[str], str, Gates, Optional[int]) -> Tuple[Optional[BinningModel], List[str]]
    """Pick the original / portfolio grouping to compare a segment refit against."""
    messages = []  # type: List[str]
    if portfolio_grouping is not None:
        return portfolio_grouping, messages
    if pred_woe_map and portfolio_frame is not None:
        reconstructed = grouping_from_woe_columns(portfolio_frame, pred_woe_map, gates=gates)
        if reconstructed is not None:
            messages.append("portfolio_grouping_from_woe_columns")
            return reconstructed, messages
        messages.append("cols_pred_woe_not_discrete_grouping")
    if portfolio_frame is not None and pred_cols and target_col in getattr(portfolio_frame, "columns", []):
        usable = [c for c in pred_cols if c in portfolio_frame.columns]
        if usable:
            fitted = BinningModel.fit(
                portfolio_frame[usable], portfolio_frame[target_col], gates=gates, n_jobs=n_jobs
            )
            messages.append("portfolio_grouping_fitted_on_reference_frame")
            return fitted, messages
    return None, messages


def _logit_spec_for_refit(feature, portfolio_grouping):
    # type: (str, Optional[BinningModel]) -> BinSpec
    """Keep a predictor in logit form instead of re-binning it as WoE."""
    port = None
    if portfolio_grouping is not None:
        port = portfolio_grouping.specs.get(feature)
    if port is not None and (port.kind == "logit" or port.transform == "logit"):
        spec = BinSpec.from_dict(port.to_dict())
        if "refit_keeps_logit_form" not in spec.notes:
            spec.notes.append("refit_keeps_logit_form")
        return spec
    return BinSpec(
        feature=feature,
        kind="logit",
        method="logit",
        transform="logit",
        notes=["refit_keeps_logit_form"],
        labels=["logit"],
        woe={"logit": 0.0, MISSING_LABEL: 0.0},
    )


def _build_refit_grouping(
    train,
    pred_cols,
    target_col,
    gates,
    n_jobs,
    logit_cols,
    portfolio_grouping,
    messages,
):
    # type: (pd.DataFrame, Sequence[str], str, Gates, Optional[int], Optional[Sequence[str]], Optional[BinningModel], List[str]) -> BinningModel
    """Fit WoE bins, but keep SQL VAL / LIN predictors as ``log(p/(1-p))``."""
    logit_keep = [c for c in pred_cols if c in set(logit_cols or [])]
    if not logit_keep:
        encoder = WoEEncoder(n_bins=gates.n_woe_bins, gates=gates, n_jobs=n_jobs)
        encoder.fit(train[list(pred_cols)], train[target_col])
        return encoder.to_grouping()

    messages.append("refit_keeps_logit_form:" + ",".join(logit_keep))
    specs = {}  # type: Dict[str, BinSpec]
    woe_cols = [c for c in pred_cols if c not in set(logit_keep)]
    if woe_cols:
        fitted = BinningModel.fit(
            train[woe_cols], train[target_col], gates=gates, n_jobs=n_jobs
        )
        for col in woe_cols:
            specs[col] = fitted.specs[col]
    for col in logit_keep:
        specs[col] = _logit_spec_for_refit(col, portfolio_grouping)
    model = BinningModel(specs=specs, gates=gates)
    model.columns = list(pred_cols)
    return model


def _logistic_refit_method(grouping):
    # type: (Optional[BinningModel]) -> str
    """Name the logistic refit by whether predictors stayed WoE, logit, or mixed."""
    if grouping is None:
        return "logistic_woe"
    kinds = []
    for col in grouping.columns:
        spec = grouping.specs.get(col)
        if spec is None:
            continue
        if spec.kind == "logit" or spec.transform == "logit":
            kinds.append("logit")
        else:
            kinds.append("woe")
    if not kinds or all(k == "woe" for k in kinds):
        return "logistic_woe"
    if all(k == "logit" for k in kinds):
        return "logistic_logit"
    return "logistic_mixed"


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
    portfolio_grouping=None,
    pred_woe_map=None,
    portfolio_frame=None,
    cols_pred_woe=None,
    segment_col=None,
    segment_value=None,
    logit_cols=None,
):
    # type: (pd.DataFrame, pd.DataFrame, Sequence[str], str, Optional[Gates], bool, Optional[Dict[str, Any]], Optional[str], Optional[BinningModel], Optional[int], Optional[BinningModel], Optional[Dict[str, str]], Optional[pd.DataFrame], Optional[Sequence[str]], Optional[str], Any, Optional[Sequence[str]]) -> RefitResult
    """Refit on ``train``, score ``holdout``, and diagnose predictor stability.

    The grouping is fitted on the training rows only (or reused when supplied),
    so the holdout stays untouched.  Stability is measured on the training rows
    as well, because acceptance has to be decidable before the refit is used.

    When ``cols_pred_woe`` or ``pred_woe_map`` is supplied, the segment
    grouping is compared to the original / portfolio grouping (``portfolio_grouping``,
    else reconstructed from the WoE columns, else fitted on ``portfolio_frame``)
    and per-bin notes are attached to the saved grouping.

    Predictors listed in ``logit_cols`` (typically SQL ``_VAL`` / ``_LIN``
    columns) stay in logit form ``log(p/(1-p))`` instead of being re-binned as
    WoE.  A supplied ``grouping`` (including one parsed from scorecard SQL) is
    the starting definition: per-bin stats are refreshed on the training rows
    and compared against the portfolio / original SQL grouping.  Adjacent
    bins whose vintage event-rate bounds overlap are merged.  When the
    supplied object is also the portfolio grouping it is cloned first so the
    SQL definition is not mutated.
    """
    gates = gates or Gates()
    pred_cols = [c for c in pred_cols if c in train.columns and c in holdout.columns]
    messages = []  # type: List[str]
    mapping = dict(pred_woe_map or {})
    if not mapping:
        mapping = map_pred_to_woe(pred_cols, cols_pred_woe)
    if not pred_cols:
        empty = RefitResult(
            p_holdout=np.full(len(holdout), float("nan")),
            method="none",
            messages=["no_usable_predictors"],
            pred_woe_map=mapping,
        )
        empty.artifact = empty.to_artifact(segment_col=segment_col, segment_value=segment_value)
        return empty

    y_train = train[target_col].to_numpy(dtype=int)
    if grouping is None:
        grouping = _build_refit_grouping(
            train,
            pred_cols,
            target_col,
            gates,
            n_jobs,
            logit_cols,
            portfolio_grouping,
            messages,
        )
    else:
        if portfolio_grouping is None:
            portfolio_grouping = grouping
            messages.append("portfolio_grouping_from_supplied_grouping")
        if grouping is portfolio_grouping:
            grouping = grouping.clone()
            messages.append("segment_grouping_cloned_from_supplied")
        grouping.refresh_stats(train, train[target_col])
        messages.append("grouping_stats_refreshed_on_train")
    if date_col is not None and date_col in train.columns:
        before_bins = dict(
            (col, len(grouping.specs[col].labels))
            for col in grouping.columns
            if col in grouping.specs
        )
        grouping.merge_overlapping_event_rate_bounds(
            train, train[target_col], date_col, gates=gates
        )
        for col, n_before in before_bins.items():
            n_after = len(grouping.specs[col].labels)
            if n_after < n_before:
                messages.append(
                    "merged_overlapping_event_rate_bounds:%s:%d->%d" % (col, n_before, n_after)
                )
    x_train = grouping.transform(train[pred_cols])
    x_holdout = grouping.transform(holdout[pred_cols])

    method = "logistic_woe"
    coefficients = {}  # type: Dict[str, float]
    p_holdout = None  # type: Optional[np.ndarray]
    model = None  # type: Any
    if submodel:
        booster = fit_submodel(x_train, y_train, gates=gates, xgb_params=xgb_params)
        if booster is None:
            messages.append("xgboost_unavailable_fallback_logistic")
        else:
            method = "xgboost_submodel"
            model = booster
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
        method = _logistic_refit_method(grouping)

    reference = portfolio_frame
    resolved_portfolio, port_messages = _resolve_portfolio_grouping(
        portfolio_grouping,
        mapping,
        reference,
        pred_cols,
        target_col,
        gates,
        n_jobs,
    )
    messages.extend(port_messages)
    comparison = compare_groupings(
        grouping, resolved_portfolio, pred_woe_map=mapping, gates=gates
    )

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
    result = RefitResult(
        p_holdout=p_holdout,
        method=method,
        grouping=grouping,
        model=model,
        stability=stability,
        stability_summary=summary,
        coefficients=coefficients,
        messages=messages,
        grouping_comparison=comparison,
        pred_woe_map=mapping,
    )
    result.artifact = result.to_artifact(segment_col=segment_col, segment_value=segment_value)
    return result


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
    portfolio_grouping=None,
    pred_woe_map=None,
    portfolio_frame=None,
    cols_pred_woe=None,
    logit_cols=None,
):
    # type: (pd.DataFrame, pd.DataFrame, Sequence[str], str, Gates, bool, Optional[Dict[str, Any]], Optional[str], Optional[BinningModel], Optional[int], Optional[BinningModel], Optional[Dict[str, str]], Optional[pd.DataFrame], Optional[Sequence[str]], Optional[Sequence[str]]) -> np.ndarray
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
        portfolio_grouping=portfolio_grouping,
        pred_woe_map=pred_woe_map,
        portfolio_frame=portfolio_frame,
        cols_pred_woe=cols_pred_woe,
        logit_cols=logit_cols,
    )
    return result.p_holdout


@dataclass
class RecalibrateResult:
    """Two-parameter PD rescale plus the fitted estimator (and optional grouping)."""

    p_holdout: np.ndarray
    model: Any = None
    intercept: float = float("nan")
    slope: float = float("nan")
    grouping: Optional[BinningModel] = None
    artifact: Optional[FittedModelArtifact] = None
    messages: List[str] = field(default_factory=list)

    def save(self, directory):
        # type: (str) -> str
        artifact = self.artifact or self.to_artifact()
        return artifact.save(directory)

    def to_artifact(self, segment_col=None, segment_value=None, score_col=None):
        # type: (Optional[str], Optional[str], Optional[str]) -> FittedModelArtifact
        if self.artifact is not None:
            if segment_col is not None:
                self.artifact.segment_col = segment_col
            if segment_value is not None:
                self.artifact.segment_value = None if segment_value is None else str(segment_value)
            if score_col is not None:
                self.artifact.score_col = score_col
            return self.artifact
        coefficients = {"__intercept__": float(self.intercept), "logit_pd": float(self.slope)}
        return FittedModelArtifact(
            kind="recalibrate",
            method="logistic_pd",
            grouping=self.grouping,
            model=self.model,
            feature_names=["logit_pd"],
            coefficients=coefficients,
            segment_col=segment_col,
            segment_value=None if segment_value is None else str(segment_value),
            score_col=score_col,
            messages=list(self.messages),
        )


def recalibrate_with_diagnostics(
    p_train,
    y_train,
    p_holdout,
    grouping=None,
    score_col=None,
    segment_col=None,
    segment_value=None,
):
    # type: (Any, Any, Any, Optional[BinningModel], Optional[str], Optional[str], Any) -> RecalibrateResult
    """Two-parameter logistic rescale, returning the fitted model for reuse."""
    z_tr = _logit(np.asarray(p_train, dtype=float)).reshape(-1, 1)
    y = np.asarray(y_train, dtype=int)
    p_ho = np.asarray(p_holdout, dtype=float)
    messages = []  # type: List[str]
    if y.min() == y.max():
        clipped = np.clip(p_ho, 1e-6, 1.0 - 1e-6)
        result = RecalibrateResult(
            p_holdout=clipped,
            grouping=grouping,
            messages=["one_class_train_passthrough"],
        )
        result.artifact = result.to_artifact(
            segment_col=segment_col, segment_value=segment_value, score_col=score_col
        )
        return result
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(z_tr, y)
    z_ho = _logit(p_ho).reshape(-1, 1)
    scored = np.asarray(model.predict_proba(z_ho)[:, 1], dtype=float)
    result = RecalibrateResult(
        p_holdout=scored,
        model=model,
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0][0]),
        grouping=grouping,
        messages=messages,
    )
    result.artifact = result.to_artifact(
        segment_col=segment_col, segment_value=segment_value, score_col=score_col
    )
    return result


def recalibrate_pd(p_train, y_train, p_holdout):
    # type: (Any, Any, Any) -> np.ndarray
    """Two-parameter logistic rescale: ``logit(p*) = a + b * logit(p)``."""
    return recalibrate_with_diagnostics(p_train, y_train, p_holdout).p_holdout


def apply_recalibrate_fn(p_train, y_train, p_holdout):
    # type: (Any, Any, Any) -> np.ndarray
    return recalibrate_pd(p_train, y_train, p_holdout)
