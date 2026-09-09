"""Smart data creation: infer the analysis metadata instead of demanding it.

The user supplies a table in a database plus the minimum that cannot be
guessed:

================  ======================================================
``col_id``        the credit case id, e.g. ``SKP_CREDIT_CASE``
``col_score``     the pooled model's PD / score column
``cols_pred_used``the predictors the pooled model actually consumes
================  ======================================================

``cols_pred_used`` may be omitted when ``model_sql`` / ``model_sql_path`` is
a production logistic scorecard (CASE WHEN WoE bins + ``LN(p/(1-p))`` VAL
columns + ``LINEAR_SCORE``).  The parser fills ``cols_pred``,
``cols_pred_woe``, ``cols_pred_used``, the grouping (with SQL null imputation)
and a sklearn logistic whose ``coef_`` / ``intercept_`` reproduce
``PD = 1/(1+exp(-B^T X))``.

Everything else is either **inferred from the database** (with a loud warning
every single time, so an inferred setting is never mistaken for a supplied
one) or **declared missing**, in which case the analyses that depend on it are
reported as blocked rather than silently skipped:

* inferable -- ``col_date``, ``cols_segment``, ``col_target`` / ``target_str``,
  ``col_obs`` / ``target_obs``;
* not inferable -- ``cols_pred`` (no refit without it) and ``cols_pred_woe``
  (no recalibration, and no grouping-based PSI when a ``grouping.json`` exists
  but ``cols_pred`` does not).

All lookups go through SQL held in ``scorecard_segment_eval/sql/*.sql`` and an
injectable executor (see :mod:`scorecard_segment_eval.dbio`), so nothing here
needs a real database to be tested.

Typical flow::

    meta = resolve_metadata(table="risk.base", col_id="SKP_CREDIT_CASE",
                            col_score="PD", cols_pred_used=[...],
                            executor=executor)
    print(render_metadata_summary(meta))
    confirmation = confirm_settings(meta, auto_confirm=True,
                                    settings_path="settings.json")
    df = build_analysis_frame(meta, executor, base=local_frame)
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from scorecard_segment_eval import config
from scorecard_segment_eval.dbio import render_sql
from scorecard_segment_eval.schema import ANALYSIS_KEYS, ScorecardColumns
from scorecard_segment_eval.sql_model import parse_scorecard_sql, parse_scorecard_sql_path

#: Environment variable that forces non-interactive confirmation.
AUTO_CONFIRM_ENV = "SCORECARD_EVAL_AUTO_CONFIRM"

SETTINGS_VERSION = 1


class MetadataInferenceWarning(UserWarning):
    """Raised whenever a setting is fetched or inferred rather than supplied."""


@dataclass
class CapabilityReport:
    """Which analyses the resolved metadata supports, and why not otherwise."""

    available: List[str] = field(default_factory=list)
    blocked: Dict[str, str] = field(default_factory=dict)

    def is_available(self, analysis):
        # type: (str) -> bool
        return analysis in self.available

    def to_frame(self):
        # type: () -> pd.DataFrame
        rows = [{"analysis": name, "status": "available", "reason": ""} for name in self.available]
        for name in sorted(self.blocked):
            rows.append({"analysis": name, "status": "blocked", "reason": self.blocked[name]})
        return pd.DataFrame(rows, columns=["analysis", "status", "reason"])

    def render_text(self):
        # type: () -> str
        lines = ["Available analyses:"]
        for name in self.available:
            lines.append("  [x] " + name)
        if not self.available:
            lines.append("  (none)")
        lines.append("Blocked analyses:")
        for name in sorted(self.blocked):
            lines.append("  [ ] " + name + " -- " + self.blocked[name])
        if not self.blocked:
            lines.append("  (none)")
        return "\n".join(lines)


@dataclass
class ResolvedMetadata:
    """Resolved column metadata plus the provenance of every field."""

    table: str
    columns: ScorecardColumns
    sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    target_str: Optional[str] = None
    target_obs: Optional[str] = None
    portfolio: Optional[str] = None
    grouping_path: Optional[str] = None
    catalogue: List[str] = field(default_factory=list)
    capabilities: CapabilityReport = field(default_factory=CapabilityReport)
    formula: Optional[str] = None
    coefficients: Dict[str, float] = field(default_factory=dict)
    model_sql_path: Optional[str] = None
    grouping: Any = None
    scorecard: Any = None

    def inferred_fields(self):
        # type: () -> List[str]
        return sorted(k for k, v in self.sources.items() if str(v).startswith("inferred"))

    def supplied_fields(self):
        # type: () -> List[str]
        return sorted(k for k, v in self.sources.items() if v == "supplied")

    def missing_fields(self):
        # type: () -> List[str]
        return sorted(k for k, v in self.sources.items() if v == "missing")

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "version": SETTINGS_VERSION,
            "table": self.table,
            "columns": self.columns.as_dict(),
            "sources": dict(self.sources),
            "warnings": list(self.warnings),
            "target_str": self.target_str,
            "target_obs": self.target_obs,
            "portfolio": self.portfolio,
            "grouping_path": self.grouping_path,
            "catalogue": list(self.catalogue),
            "formula": self.formula,
            "coefficients": dict(self.coefficients),
            "model_sql_path": self.model_sql_path,
        }

    @classmethod
    def from_dict(cls, payload):
        # type: (Dict[str, Any]) -> "ResolvedMetadata"
        meta = cls(
            table=payload.get("table", ""),
            columns=ScorecardColumns.from_dict(payload.get("columns") or {}),
            sources=dict(payload.get("sources") or {}),
            warnings=list(payload.get("warnings") or []),
            target_str=payload.get("target_str"),
            target_obs=payload.get("target_obs"),
            portfolio=payload.get("portfolio"),
            grouping_path=payload.get("grouping_path"),
            catalogue=list(payload.get("catalogue") or []),
            formula=payload.get("formula"),
            coefficients=dict(payload.get("coefficients") or {}),
            model_sql_path=payload.get("model_sql_path"),
        )
        if meta.model_sql_path and os.path.isfile(str(meta.model_sql_path)):
            try:
                parsed = parse_scorecard_sql_path(str(meta.model_sql_path))
                meta.scorecard = parsed
                meta.grouping = parsed.grouping
                if not meta.formula:
                    meta.formula = parsed.formula
                if not meta.coefficients:
                    meta.coefficients = dict(parsed.coefficients)
            except (OSError, ValueError):
                pass
        elif meta.grouping_path and os.path.isfile(str(meta.grouping_path)):
            try:
                from scorecard_segment_eval.binning import load_grouping

                meta.grouping = load_grouping(str(meta.grouping_path))
            except (OSError, ValueError, KeyError):
                pass
        meta.capabilities = resolve_capabilities(meta)
        return meta


# --------------------------------------------------------------------------
# Metadata resolution
# --------------------------------------------------------------------------
def _emit(meta, message, warn):
    # type: (ResolvedMetadata, str, bool) -> None
    meta.warnings.append(message)
    if warn:
        warnings.warn(message, MetadataInferenceWarning, stacklevel=3)


def _catalogue(executor, table):
    # type: (Optional[Callable[[str], pd.DataFrame]], str) -> Tuple[List[str], Dict[str, str]]
    if executor is None:
        return [], {}
    frame = executor(render_sql("describe_table", table=table))
    if frame is None or len(frame) == 0:
        return [], {}
    names = [str(v) for v in frame["column_name"].tolist()]
    types = {}
    if "data_type" in frame.columns:
        for name, dtype in zip(names, frame["data_type"].tolist()):
            types[name] = str(dtype)
    return names, types


def _first_present(candidates, catalogue):
    # type: (Sequence[str], Sequence[str]) -> Optional[str]
    lowered = dict((str(c).lower(), str(c)) for c in catalogue)
    for candidate in candidates:
        if str(candidate) in catalogue:
            return str(candidate)
        if str(candidate).lower() in lowered:
            return lowered[str(candidate).lower()]
    return None


def _infer_date_column(meta, catalogue, types, warn):
    # type: (ResolvedMetadata, List[str], Dict[str, str], bool) -> Optional[str]
    found = _first_present(config.DATE_COLUMN_CANDIDATES, catalogue)
    if found is not None:
        _emit(
            meta,
            "col_date was not supplied: inferred %r from the table catalogue "
            "(known date-column name)." % (found,),
            warn,
        )
        meta.sources["col_date"] = "inferred:catalogue_candidate"
        return found
    for name in catalogue:
        dtype = str(types.get(name, "")).lower()
        if any(token in dtype for token in config.DATE_TYPE_TOKENS):
            _emit(
                meta,
                "col_date was not supplied: inferred %r from the table catalogue "
                "(data type %r)." % (name, types.get(name)),
                warn,
            )
            meta.sources["col_date"] = "inferred:catalogue_datatype"
            return name
    _emit(
        meta,
        "col_date was not supplied and no date-like column could be found; "
        "vintage stability and the time holdout are unavailable.",
        warn,
    )
    meta.sources["col_date"] = "missing"
    return None


def _infer_segment_columns(meta, executor, catalogue, sample_limit, col_id, warn):
    # type: (ResolvedMetadata, Optional[Callable[[str], pd.DataFrame]], List[str], int, str, bool) -> List[str]
    found = []  # type: List[str]
    for candidate in config.SEGMENT_COLUMN_CANDIDATES:
        name = _first_present([candidate], catalogue)
        if name is None or name in found:
            continue
        if executor is not None:
            profile = executor(
                render_sql(
                    "column_profile",
                    table=meta.table,
                    column=name,
                    col_id=col_id,
                    sample_limit=sample_limit,
                )
            )
            if profile is None or len(profile) == 0:
                continue
            if len(profile) > config.MAX_SEGMENT_CARDINALITY:
                _emit(
                    meta,
                    "candidate segmentation column %r has %d distinct values "
                    "(> %d) and was skipped."
                    % (name, len(profile), config.MAX_SEGMENT_CARDINALITY),
                    warn,
                )
                continue
        found.append(name)
    if found:
        _emit(
            meta,
            "cols_segment was not supplied: inferred %s from the table catalogue."
            % (", ".join(repr(f) for f in found),),
            warn,
        )
        meta.sources["cols_segment"] = "inferred:catalogue_candidate"
        return found
    _emit(
        meta,
        "cols_segment was not supplied and no low-cardinality candidate column "
        "was found; only the portfolio-level view is available.",
        warn,
    )
    meta.sources["cols_segment"] = "missing"
    return []


def _dominant_portfolio(meta, executor, catalogue, sample_limit, col_id, warn):
    # type: (ResolvedMetadata, Optional[Callable[[str], pd.DataFrame]], List[str], int, str, bool) -> Optional[str]
    if executor is None:
        return None
    col_portfolio = _first_present(config.PORTFOLIO_COLUMN_CANDIDATES, catalogue)
    if col_portfolio is None:
        return None
    frame = executor(
        render_sql(
            "fetch_portfolio",
            table=meta.table,
            col_id=col_id,
            col_portfolio=col_portfolio,
            sample_limit=sample_limit,
        )
    )
    if frame is None or len(frame) == 0:
        return None
    top = frame.iloc[0]
    portfolio = None if pd.isna(top["portfolio"]) else str(top["portfolio"])
    if portfolio is not None:
        _emit(
            meta,
            "portfolio was not supplied: read %r from column %r (%s cases) to "
            "drive target detection."
            % (portfolio, col_portfolio, top.get("n_cases")),
            warn,
        )
    return portfolio


def _resolve_target(
    meta,
    executor,
    catalogue,
    col_target,
    target_str,
    col_obs,
    target_obs,
    portfolio,
    sample_limit,
    col_id,
    warn,
):
    # type: (...) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
    """Resolve ``(col_target, col_obs, target_str, target_obs)``.

    Precedence: an explicitly supplied target or observation flag wins; then
    each is derived from the other; then the dominant portfolio decides via
    :data:`scorecard_segment_eval.config.PORTFOLIO_TARGET_MAP`; finally the
    configured default pair is used.
    """
    target = col_target or target_str
    obs = col_obs or target_obs
    if col_target or target_str:
        meta.sources["col_target"] = "supplied"
    if col_obs or target_obs:
        meta.sources["col_obs"] = "supplied"

    if target and not obs:
        derived = config.obs_from_target(target)
        _emit(
            meta,
            "col_obs was not supplied: inferred %r from the supplied target %r."
            % (derived, target),
            warn,
        )
        obs = derived
        meta.sources["col_obs"] = "inferred:from_col_target"
    elif obs and not target:
        derived = config.target_from_obs(obs)
        _emit(
            meta,
            "col_target was not supplied: inferred %r from the supplied "
            "observation flag %r." % (derived, obs),
            warn,
        )
        target = derived
        meta.sources["col_target"] = "inferred:from_col_obs"
    elif not target and not obs:
        detected = config.target_for_portfolio(portfolio)
        if detected is not None:
            target, obs = detected
            _emit(
                meta,
                "col_target/col_obs were not supplied: portfolio %r implies "
                "target %r and observation flag %r." % (portfolio, target, obs),
                warn,
            )
            meta.sources["col_target"] = "inferred:portfolio=" + str(portfolio)
            meta.sources["col_obs"] = "inferred:portfolio=" + str(portfolio)
        else:
            target = config.DEFAULT_TARGET
            obs = config.DEFAULT_TARGET_OBS
            _emit(
                meta,
                "col_target/col_obs were not supplied and the portfolio did not "
                "resolve; defaulting to %r / %r." % (target, obs),
                warn,
            )
            meta.sources["col_target"] = "default"
            meta.sources["col_obs"] = "default"

    if catalogue:
        target = _validate_against_catalogue(meta, target, catalogue, "col_target", config.known_targets(), warn)
        obs = _validate_against_catalogue(meta, obs, catalogue, "col_obs", config.known_obs_flags(), warn)
    return target, obs, target, obs


def _validate_against_catalogue(meta, name, catalogue, field_name, fallbacks, warn):
    # type: (ResolvedMetadata, Optional[str], List[str], str, Sequence[str], bool) -> Optional[str]
    if name is None:
        return None
    resolved = _first_present([name], catalogue)
    if resolved is not None:
        return resolved
    alternative = _first_present(fallbacks, catalogue)
    if alternative is not None:
        _emit(
            meta,
            "%s resolved to %r which is not in the table; falling back to %r "
            "found in the catalogue." % (field_name, name, alternative),
            warn,
        )
        meta.sources[field_name] = "inferred:catalogue_fallback"
        return alternative
    _emit(
        meta,
        "%s resolved to %r but no such column exists in %s; analyses that need "
        "it are blocked." % (field_name, name, meta.table),
        warn,
    )
    meta.sources[field_name] = "missing"
    return None


def resolve_metadata(
    table,
    col_id,
    col_score,
    cols_pred_used=None,
    executor=None,
    col_date=None,
    cols_segment=None,
    col_target=None,
    target_str=None,
    col_obs=None,
    target_obs=None,
    cols_pred=None,
    cols_pred_woe=None,
    col_fantomas=None,
    portfolio=None,
    grouping_path=None,
    sample_limit=100000,
    warn=True,
    model_sql=None,
    model_sql_path=None,
):
    # type: (str, str, str, Optional[Sequence[str]], Optional[Callable[[str], pd.DataFrame]], Optional[str], Optional[Sequence[str]], Optional[str], Optional[str], Optional[str], Optional[str], Optional[Sequence[str]], Optional[Sequence[str]], Optional[str], Optional[str], Optional[str], int, bool, Optional[str], Optional[str]) -> ResolvedMetadata
    """Resolve the full analysis metadata from a minimal user specification.

    Raises ``ValueError`` when a mandatory input is absent.  Every field that
    ends up inferred or fetched emits a :class:`MetadataInferenceWarning` and
    is recorded in ``sources``.

    ``model_sql`` / ``model_sql_path`` is a production logistic scorecard in
    SQL form.  When supplied it fills ``cols_pred``, ``cols_pred_woe``,
    ``cols_pred_used``, the grouping (with SQL null imputation) and the
    pooled logistic coefficients, unless the caller already set those.
    """
    parsed = None
    sql_path = str(model_sql_path) if model_sql_path else None
    if sql_path:
        parsed = parse_scorecard_sql_path(sql_path)
    elif model_sql:
        parsed = parse_scorecard_sql(model_sql)

    if not table:
        raise ValueError("table is mandatory")
    if not col_id:
        raise ValueError("col_id is mandatory (the credit case id)")
    if not col_score:
        raise ValueError("col_score is mandatory")
    if not cols_pred_used and parsed is None:
        raise ValueError("cols_pred_used is mandatory (predictors the pooled model uses)")

    meta = ResolvedMetadata(table=str(table), columns=ScorecardColumns(col_id=str(col_id)))
    meta.sources["col_id"] = "supplied"
    meta.sources["col_score"] = "supplied"
    if cols_pred_used:
        meta.sources["cols_pred_used"] = "supplied"
    elif parsed is not None:
        meta.sources["cols_pred_used"] = "inferred:sql_model"
        cols_pred_used = list(parsed.cols_pred_used)
        _emit(
            meta,
            "cols_pred_used was not supplied: inferred %s from the scorecard SQL."
            % (list(cols_pred_used),),
            warn,
        )
    else:
        meta.sources["cols_pred_used"] = "missing"

    catalogue, types = _catalogue(executor, meta.table)
    meta.catalogue = list(catalogue)
    if executor is None:
        _emit(
            meta,
            "no SQL executor was supplied: nothing can be looked up from the "
            "database, so only the settings you passed will be used.",
            warn,
        )

    if col_date:
        meta.sources["col_date"] = "supplied"
        resolved_date = str(col_date)
    else:
        resolved_date = _infer_date_column(meta, catalogue, types, warn)

    if cols_segment:
        meta.sources["cols_segment"] = "supplied"
        resolved_segments = [str(c) for c in cols_segment]
    else:
        resolved_segments = _infer_segment_columns(
            meta, executor, catalogue, sample_limit, str(col_id), warn
        )

    if portfolio:
        meta.sources["portfolio"] = "supplied"
        resolved_portfolio = str(portfolio)
    elif not (col_target or target_str or col_obs or target_obs):
        resolved_portfolio = _dominant_portfolio(
            meta, executor, catalogue, sample_limit, str(col_id), warn
        )
        meta.sources["portfolio"] = (
            "inferred:catalogue_portfolio" if resolved_portfolio else "missing"
        )
    else:
        resolved_portfolio = None
    meta.portfolio = resolved_portfolio

    resolved_target, resolved_obs, target_name, obs_name = _resolve_target(
        meta,
        executor,
        catalogue,
        col_target,
        target_str,
        col_obs,
        target_obs,
        resolved_portfolio,
        sample_limit,
        str(col_id),
        warn,
    )
    meta.target_str = target_name
    meta.target_obs = obs_name

    if cols_pred:
        meta.sources["cols_pred"] = "supplied"
        resolved_pred = [str(c) for c in cols_pred]
    elif parsed is not None and parsed.cols_pred:
        resolved_pred = list(parsed.cols_pred)
        meta.sources["cols_pred"] = "inferred:sql_model"
        _emit(
            meta,
            "cols_pred was not supplied: inferred %s from the scorecard SQL."
            % (resolved_pred,),
            warn,
        )
    else:
        resolved_pred = []
        meta.sources["cols_pred"] = "missing"
        _emit(
            meta,
            "cols_pred was not supplied and cannot be inferred from the "
            "database: the same-predictor refit (and the XGBoost sub-model) "
            "cannot be performed.",
            warn,
        )

    if cols_pred_woe:
        meta.sources["cols_pred_woe"] = "supplied"
        resolved_woe = [str(c) for c in cols_pred_woe]
    elif parsed is not None and parsed.cols_pred_woe:
        resolved_woe = list(parsed.cols_pred_woe)
        meta.sources["cols_pred_woe"] = "inferred:sql_model"
        _emit(
            meta,
            "cols_pred_woe was not supplied: inferred %s from the scorecard SQL."
            % (resolved_woe,),
            warn,
        )
    else:
        resolved_woe = []
        meta.sources["cols_pred_woe"] = "missing"
        _emit(
            meta,
            "cols_pred_woe was not supplied and cannot be inferred from the "
            "database: recalibration on the WoE inputs cannot be performed.",
            warn,
        )

    pred_map = dict(parsed.pred_map) if parsed is not None else {}

    if grouping_path:
        meta.grouping_path = str(grouping_path)
        exists = os.path.isfile(str(grouping_path))
        if parsed is not None and parsed.grouping is not None and not exists:
            parsed.grouping.save(str(grouping_path))
            exists = True
            meta.sources["grouping_path"] = "inferred:sql_model"
            _emit(
                meta,
                "grouping_path %r did not exist: wrote the grouping parsed from "
                "the scorecard SQL (including null imputation)." % (str(grouping_path),),
                warn,
            )
        else:
            meta.sources["grouping_path"] = "supplied" if exists else "missing"
        if not exists:
            _emit(
                meta,
                "grouping_path %r does not exist: grouping-based PSI cannot be "
                "performed." % (str(grouping_path),),
                warn,
            )
        elif not resolved_pred:
            _emit(
                meta,
                "grouping_path %r exists but cols_pred was not supplied: the "
                "grouping cannot be applied, so grouping-based PSI and "
                "recalibration on grouped inputs cannot be performed."
                % (str(grouping_path),),
                warn,
            )
    elif parsed is not None and parsed.grouping is not None:
        meta.sources["grouping_path"] = "inferred:sql_model"
        meta.grouping = parsed.grouping
    else:
        meta.sources["grouping_path"] = "missing"

    if col_fantomas:
        meta.sources["col_fantomas"] = "supplied"

    if parsed is not None:
        meta.formula = parsed.formula
        meta.coefficients = dict(parsed.coefficients)
        meta.model_sql_path = sql_path
        meta.grouping = parsed.grouping
        meta.scorecard = parsed
        if parsed.grouping is not None and meta.grouping_path:
            # already saved above when the file was missing
            pass

    meta.columns = ScorecardColumns(
        col_id=str(col_id),
        col_date=resolved_date,
        col_obs=resolved_obs,
        col_target=resolved_target,
        col_score=str(col_score),
        cols_pred=resolved_pred,
        cols_segment=resolved_segments,
        col_fantomas=str(col_fantomas) if col_fantomas else None,
        cols_pred_woe=resolved_woe,
        cols_pred_used=[str(c) for c in (cols_pred_used or [])],
        pred_map=pred_map,
    )
    meta.capabilities = resolve_capabilities(meta)
    return meta


# --------------------------------------------------------------------------
# Capability resolution
# --------------------------------------------------------------------------
def resolve_capabilities(meta):
    # type: (ResolvedMetadata) -> CapabilityReport
    """Report exactly which analyses the resolved metadata supports.

    ``blocked`` maps an analysis key from
    :data:`scorecard_segment_eval.schema.ANALYSIS_KEYS` to the reason it cannot
    run, so the caller never has to guess why an output table came back empty.
    """
    cols = meta.columns
    available = []  # type: List[str]
    blocked = {}  # type: Dict[str, str]

    def _mark(name, ok, reason):
        # type: (str, bool, str) -> None
        if ok:
            available.append(name)
        else:
            blocked[name] = reason

    has_score = bool(cols.col_score)
    has_target = bool(cols.col_target)
    has_segments = bool(cols.cols_segment)
    has_date = bool(cols.col_date)
    has_pred = bool(cols.cols_pred)
    has_woe = bool(cols.cols_pred_woe)
    grouping_ok = bool(meta.grouping is not None) or (
        bool(meta.grouping_path)
        and meta.sources.get("grouping_path") in ("supplied", "inferred:sql_model")
    )

    _mark(
        "segment_performance",
        has_score and has_target and has_segments,
        "needs col_score, col_target and cols_segment; missing: "
        + _missing_list([("col_score", has_score), ("col_target", has_target), ("cols_segment", has_segments)]),
    )
    _mark(
        "vintage_stability",
        has_score and has_target and has_date,
        "needs col_date to group vintages; missing: "
        + _missing_list([("col_score", has_score), ("col_target", has_target), ("col_date", has_date)]),
    )
    _mark(
        "matched_ar_gini",
        has_score and has_target and has_segments,
        "needs col_score, col_target and cols_segment to compare approval rates; missing: "
        + _missing_list([("col_score", has_score), ("col_target", has_target), ("cols_segment", has_segments)]),
    )
    _mark(
        "psi_numeric",
        has_target and bool(cols.cols_pred_used or cols.cols_pred),
        "needs characteristics to profile (cols_pred_used or cols_pred) and col_target",
    )
    if grouping_ok and has_pred:
        _mark("psi_grouping", True, "")
    elif not grouping_ok:
        _mark("psi_grouping", False, "no usable grouping.json was supplied")
    else:
        _mark("psi_grouping", False, "grouping.json exists but cols_pred is missing, so it cannot be applied")
    _mark(
        "recalibration",
        has_woe and has_score and has_target,
        "needs cols_pred_woe (the WoE inputs the pooled model consumes); missing: "
        + _missing_list([("cols_pred_woe", has_woe), ("col_score", has_score), ("col_target", has_target)]),
    )
    _mark(
        "refit",
        has_pred and has_target and has_date,
        "needs cols_pred (not inferable) and col_date for the time holdout; missing: "
        + _missing_list([("cols_pred", has_pred), ("col_target", has_target), ("col_date", has_date)]),
    )
    submodel_ok = has_pred and has_target and has_date and _xgboost_available()
    _mark(
        "submodel",
        submodel_ok,
        "needs cols_pred plus the optional 'xgboost' extra"
        if (has_pred and has_target and has_date)
        else "needs cols_pred, col_target, col_date and the optional 'xgboost' extra",
    )
    _mark(
        "report",
        has_score and has_target and has_segments,
        "needs a segment evaluation to report on",
    )
    ordered = [key for key in ANALYSIS_KEYS if key in available]
    return CapabilityReport(available=ordered, blocked=blocked)


def _missing_list(pairs):
    # type: (Sequence[Tuple[str, bool]]) -> str
    missing = [name for name, ok in pairs if not ok]
    return ", ".join(missing) if missing else "none"


def _xgboost_available():
    # type: () -> bool
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


def _format_pred_woe_map(mapping, cols_pred):
    # type: (Dict[str, str], Sequence[str]) -> str
    if not cols_pred and not mapping:
        return ""
    parts = []
    for pred in list(cols_pred or ()):
        woe = mapping.get(pred)
        parts.append("%s -> %s" % (pred, woe if woe else "(none)"))
    return "; ".join(parts)
def render_metadata_summary(meta):
    # type: (ResolvedMetadata) -> str
    """Plain-text summary of the resolved metadata, provenance and warnings."""
    cols = meta.columns
    lines = ["Resolved analysis metadata", "=" * 26, "table: " + str(meta.table)]
    fields = [
        ("col_id", cols.col_id),
        ("col_score", cols.col_score),
        ("col_date", cols.col_date),
        ("col_target", cols.col_target),
        ("col_obs", cols.col_obs),
        ("col_fantomas", cols.col_fantomas),
        ("cols_segment", cols.cols_segment),
        ("cols_pred", cols.cols_pred),
        ("cols_pred_woe", cols.cols_pred_woe),
        ("pred_woe_map", _format_pred_woe_map(cols.pred_woe_map(), cols.cols_pred)),
        ("cols_pred_used", cols.cols_pred_used),
        ("grouping_path", meta.grouping_path),
        ("portfolio", meta.portfolio),
    ]
    for name, value in fields:
        source = meta.sources.get(name, "not_set")
        if name == "pred_woe_map":
            source = (
                "derived"
                if (cols.cols_pred_woe and cols.cols_pred) or cols.pred_map
                else source
            )
        shown = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append("  %-16s %-40s [%s]" % (name + ":", shown if shown else "-", source))
    if meta.formula:
        lines.append("")
        lines.append("Pooled model formula:")
        for line in str(meta.formula).splitlines():
            lines.append("  " + line)
    if meta.warnings:
        lines.append("")
        lines.append("Warnings (%d):" % (len(meta.warnings),))
        for message in meta.warnings:
            lines.append("  ! " + message)
    lines.append("")
    lines.append(meta.capabilities.render_text())
    return "\n".join(lines)


@dataclass
class ConfirmationResult:
    confirmed: bool
    settings_path: Optional[str] = None
    summary: str = ""
    auto: bool = False


def _default_prompt(message):
    # type: (str) -> str
    return input(message)


def _auto_confirm_requested():
    # type: () -> bool
    if str(os.environ.get(AUTO_CONFIRM_ENV, "")).strip().lower() in ("1", "true", "yes"):
        return True
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return True
    try:
        return not stdin.isatty()
    except Exception:
        return True


def _display(text):
    # type: (str) -> None
    """Print the summary, using IPython's renderer when we are in a notebook."""
    try:
        from IPython.display import display  # type: ignore
        from IPython.display import Markdown  # type: ignore
    except Exception:
        print(text)
        return
    try:
        display(Markdown("```\n" + text + "\n```"))
    except Exception:
        print(text)


def confirm_settings(
    meta,
    auto_confirm=None,
    prompt=None,
    settings_path=None,
    save=True,
    display_fn=None,
):
    # type: (ResolvedMetadata, Optional[bool], Optional[Callable[[str], str]], Optional[str], bool, Optional[Callable[[str], None]]) -> ConfirmationResult
    """Show the resolved metadata and capabilities, then ask the user to confirm.

    Notebook-friendly: the summary is rendered through ``IPython.display`` when
    available and the answer is read with :func:`input`.  Scripts and tests can
    bypass the prompt entirely with ``auto_confirm=True``, by setting
    ``SCORECARD_EVAL_AUTO_CONFIRM=1``, or simply by running without a TTY --
    all three auto-confirm and say so.

    On confirmation the settings are written to ``settings_path`` (the user is
    prompted for the file name when it is not given), and
    :func:`load_settings` reads them back losslessly on later runs.
    """
    summary = render_metadata_summary(meta)
    show = display_fn or _display
    show(summary)

    auto = bool(auto_confirm) if auto_confirm is not None else _auto_confirm_requested()
    asker = prompt or _default_prompt

    if auto:
        warnings.warn(
            "auto-confirming the resolved metadata without asking (non-interactive "
            "session or auto_confirm=True); review the summary above.",
            MetadataInferenceWarning,
            stacklevel=2,
        )
        confirmed = True
    else:
        answer = str(asker("Proceed with these settings? [y/N]: ")).strip().lower()
        confirmed = answer in ("y", "yes")
        if not confirmed:
            return ConfirmationResult(confirmed=False, summary=summary, auto=False)

    path = settings_path
    if save and not path:
        if auto:
            path = config.DEFAULT_SETTINGS_FILENAME
        else:
            answer = str(
                asker("Save settings as [%s]: " % (config.DEFAULT_SETTINGS_FILENAME,))
            ).strip()
            path = answer or config.DEFAULT_SETTINGS_FILENAME
    if save and path:
        save_settings(meta, path)
    return ConfirmationResult(
        confirmed=confirmed, settings_path=path if save else None, summary=summary, auto=auto
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_settings(meta, path):
    # type: (ResolvedMetadata, str) -> str
    """Persist resolved settings as JSON.  Round-trips losslessly."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(meta.to_dict(), handle, indent=2, sort_keys=True)
    return path


def load_settings(path):
    # type: (str) -> ResolvedMetadata
    """Load settings written by :func:`save_settings`, as a cache/reference."""
    with open(path, "r") as handle:
        payload = json.load(handle)
    version = payload.get("version")
    if version is not None and int(version) > SETTINGS_VERSION:
        warnings.warn(
            "settings file %r was written by a newer version (%s > %s); unknown "
            "fields are ignored." % (path, version, SETTINGS_VERSION),
            MetadataInferenceWarning,
        )
    return ResolvedMetadata.from_dict(payload)


# --------------------------------------------------------------------------
# Frame assembly
# --------------------------------------------------------------------------
def required_columns(meta):
    # type: (ResolvedMetadata) -> List[str]
    """Every column the resolved analyses need, de-duplicated and ordered."""
    cols = meta.columns
    wanted = [cols.col_id, cols.col_score, cols.col_date, cols.col_target, cols.col_obs, cols.col_fantomas]
    wanted.extend(cols.cols_segment or ())
    wanted.extend(cols.cols_pred or ())
    wanted.extend(cols.cols_pred_woe or ())
    wanted.extend(cols.cols_pred_used or ())
    out = []  # type: List[str]
    for name in wanted:
        if name and name not in out:
            out.append(str(name))
    return out


def _scorecard_fillable(meta, frame):
    # type: (ResolvedMetadata, Optional[pd.DataFrame]) -> List[str]
    parsed = meta.scorecard
    if parsed is None or frame is None:
        return []
    preds = list(parsed.cols_pred or [])
    if not preds or any(c not in frame.columns for c in preds):
        return []
    fillable = []  # type: List[str]
    if meta.columns.col_score:
        fillable.append(str(meta.columns.col_score))
    fillable.extend(list(parsed.cols_pred_woe or ()))
    fillable.extend(list(parsed.cols_pred_used or ()))
    out = []  # type: List[str]
    for name in fillable:
        if name and name not in out:
            out.append(name)
    return out


def _apply_scorecard_columns(meta, frame, warn):
    # type: (ResolvedMetadata, pd.DataFrame, bool) -> pd.DataFrame
    parsed = meta.scorecard
    if parsed is None:
        return frame
    preds = list(parsed.cols_pred or [])
    if not preds or any(c not in frame.columns for c in preds):
        return frame
    score_col = meta.columns.col_score or "SCORE"
    scored = parsed.transform_frame(frame, score_col=score_col)
    added = []  # type: List[str]
    for col in scored.columns:
        if col not in frame.columns:
            frame[col] = scored[col]
            added.append(col)
        elif col == score_col and frame[col].isna().all():
            frame[col] = scored[col]
            added.append(col)
    if added and warn:
        warnings.warn(
            "filled %s from the parsed scorecard SQL (PD = 1/(1+exp(-B^T X)))."
            % (", ".join(added),),
            MetadataInferenceWarning,
            stacklevel=3,
        )
    return frame


def build_analysis_frame(meta, executor, base=None, warn=True):
    # type: (ResolvedMetadata, Optional[Callable[[str], pd.DataFrame]], Optional[pd.DataFrame], bool) -> pd.DataFrame
    """Assemble the analysis frame, fetching only what ``base`` is missing.

    Fetched columns are merged on ``col_id``.  A warning names every column
    that came from the database rather than from the caller.

    When a production scorecard SQL model is attached to ``meta`` and the
    raw predictors are already in the frame, missing score / WoE / VAL
    columns are computed from that model instead of being fetched.
    """
    wanted = required_columns(meta)
    col_id = meta.columns.col_id
    if base is not None:
        frame = base.reset_index(drop=True).copy()
        missing = [c for c in wanted if c not in frame.columns]
    else:
        frame = None
        missing = list(wanted)
    fillable = _scorecard_fillable(meta, frame)
    if fillable:
        missing = [c for c in missing if c not in fillable]
    if not missing:
        if frame is None:
            return pd.DataFrame()
        return _apply_scorecard_columns(meta, frame, warn)
    if executor is None:
        raise ValueError(
            "columns %s are not in the supplied frame and no executor was given"
            % (", ".join(missing),)
        )
    to_fetch = [c for c in missing if c != col_id]
    fetched = executor(
        render_sql("fetch_columns_by_id", table=meta.table, col_id=col_id, columns=to_fetch)
    )
    if warn:
        warnings.warn(
            "fetched %s from %s using %s; these values were not supplied by you."
            % (", ".join(to_fetch), meta.table, col_id),
            MetadataInferenceWarning,
            stacklevel=2,
        )
    if frame is None:
        frame = fetched.reset_index(drop=True)
    else:
        frame = frame.merge(fetched, on=col_id, how="left")
    return _apply_scorecard_columns(meta, frame, warn)
