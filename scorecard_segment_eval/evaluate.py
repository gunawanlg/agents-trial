"""Segment-level evaluation: Q1 verdict, Q2 action and supporting diagnostics."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scorecard_segment_eval.binning import BinningModel
from scorecard_segment_eval.bootstrap import bootstrap_delta_gini, bootstrap_gini_ci
from scorecard_segment_eval.characteristics import feature_diagnostics, shape_divergence
from scorecard_segment_eval.decision import (
    ar_artifact_suspected,
    is_important,
    q1_verdict,
    q2_action,
    q2_portfolio_action,
)
from scorecard_segment_eval.metrics import (
    MATCHED_AR_KEYS,
    brier,
    gini,
    logloss,
    matched_ar_comparison,
    performance_bundle,
)
from scorecard_segment_eval.parallel import map_jobs, ordered_concat
from scorecard_segment_eval.population import fantomas_mask, observable_mask, time_holdout_mask
from scorecard_segment_eval.refit import (
    FittedModelArtifact,
    recalibrate_with_diagnostics,
    refit_with_diagnostics,
    save_fitted_artifacts,
)
from scorecard_segment_eval.schema import Gates, ScorecardColumns
from scorecard_segment_eval.stability import predictor_stability

OVERALL_SEGMENT_COL = "__overall__"
OVERALL_SEGMENT_VALUE = "ALL"

STABILITY_RESULT_KEYS = (
    "refit_method",
    "stability_assessed",
    "stability_score",
    "stability_pass",
    "stability_n_predictors",
    "stability_n_unstable",
    "unstable_predictors",
    "stability_reason",
)


@dataclass
class SegmentEvalResult:
    """Tables produced by :func:`evaluate_segments`."""

    segment_summary: pd.DataFrame
    decisions: pd.DataFrame
    refit_comparison: pd.DataFrame
    characteristics: pd.DataFrame
    vintage: pd.DataFrame
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    grouping_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    grouping_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    fitted_artifacts: List[FittedModelArtifact] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def save_artifacts(self, directory):
        # type: (str) -> List[str]
        """Persist every recalibrate / refit grouping and estimator."""
        return save_fitted_artifacts(self.fitted_artifacts, directory)


def _vintage_ratio(obs, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> Tuple[Optional[float], pd.DataFrame]
    if obs.empty or cols.col_date is None or cols.col_date not in obs.columns:
        return None, pd.DataFrame()
    dates = pd.to_datetime(obs[cols.col_date], errors="coerce")
    months = dates.dt.to_period("M")
    rows = []
    for period, part in obs.groupby(months, dropna=True):
        bundle = performance_bundle(part[cols.col_target], part[cols.col_score])
        rows.append(
            {
                "vintage": str(period),
                "n": bundle["n"],
                "defaults": bundle["defaults"],
                "gini": bundle["gini"],
                "oe": bundle["oe"],
            }
        )
    table = pd.DataFrame(rows)
    if table.empty or table["gini"].notna().sum() < 2:
        return None, table
    table = table.sort_values("vintage")
    mid = max(len(table) // 2, 1)
    early = table.iloc[:mid]["gini"].mean()
    latest = table.iloc[-1]["gini"]
    if not np.isfinite(early) or early == 0 or not np.isfinite(latest):
        return None, table
    return float(latest / early), table


def _slice_metrics(part, cols, gates, n_jobs=None):
    # type: (pd.DataFrame, ScorecardColumns, Gates, Optional[int]) -> Dict[str, float]
    bundle = performance_bundle(
        part[cols.col_target], part[cols.col_score], n_ece_bins=gates.n_ece_bins
    )
    ci = bootstrap_gini_ci(
        part[cols.col_target],
        part[cols.col_score],
        n_bootstrap=gates.n_bootstrap,
        seed=gates.bootstrap_seed,
        z=gates.delta_gini_z,
        n_jobs=n_jobs,
        cap=gates.max_workers_cap,
    )
    bundle["gini_se"] = ci["gini_se"]
    bundle["gini_ci_low"] = ci["gini_ci_low"]
    return bundle


def _empty_matched_ar_row():
    # type: () -> Dict[str, Any]
    row = dict((key, None) for key in MATCHED_AR_KEYS)
    row["ar_gap_triggered"] = False
    return row


def _empty_holdout_row():
    # type: () -> Dict[str, Any]
    row = {
        "n_holdout": None,
        "defaults_holdout": None,
        "gini_pooled": None,
        "gini_refit": None,
        "delta_gini": None,
        "delta_gini_se": None,
        "delta_gini_ci_low": None,
        "brier_pooled": None,
        "brier_refit": None,
        "logloss_pooled": None,
        "logloss_refit": None,
        "gini_recal": None,
        "brier_recal": None,
        "logloss_recal": None,
    }
    row["refit_method"] = None
    row["stability_assessed"] = False
    row["stability_score"] = None
    row["stability_pass"] = True
    row["stability_n_predictors"] = 0
    row["stability_n_unstable"] = 0
    row["unstable_predictors"] = ""
    row["stability_reason"] = "not_attempted"
    return row


def _run_holdout_models(
    obs_seg,
    cols,
    gates,
    do_refit,
    do_recal,
    grouping=None,
    submodel=False,
    xgb_params=None,
    n_jobs=None,
    obs_all=None,
    segment_col=None,
    segment_value=None,
):
    # type: (pd.DataFrame, ScorecardColumns, Gates, bool, bool, Optional[BinningModel], bool, Optional[Dict[str, Any]], Optional[int], Optional[pd.DataFrame], Optional[str], Any) -> Tuple[Dict[str, Any], pd.DataFrame, List[FittedModelArtifact], pd.DataFrame]
    out = _empty_holdout_row()
    stability_table = pd.DataFrame()
    artifacts = []  # type: List[FittedModelArtifact]
    grouping_notes = pd.DataFrame()
    if len(obs_seg) < 40 or cols.col_date is None or cols.col_date not in obs_seg.columns:
        return out, stability_table, artifacts, grouping_notes
    ho = time_holdout_mask(obs_seg[cols.col_date], gates.holdout_frac)
    train = obs_seg.loc[~ho]
    holdout = obs_seg.loc[ho]
    if train[cols.col_target].sum() < 5 or holdout[cols.col_target].sum() < 5:
        return out, stability_table, artifacts, grouping_notes
    y = holdout[cols.col_target].to_numpy(dtype=float)
    p_pooled = holdout[cols.col_score].to_numpy(dtype=float)
    out["n_holdout"] = float(len(holdout))
    out["defaults_holdout"] = float(y.sum())
    out["gini_pooled"] = gini(y, p_pooled)
    out["brier_pooled"] = brier(y, p_pooled)
    out["logloss_pooled"] = logloss(y, p_pooled)
    if do_recal:
        recal = recalibrate_with_diagnostics(
            train[cols.col_score].to_numpy(),
            train[cols.col_target].to_numpy(),
            p_pooled,
            grouping=grouping,
            score_col=cols.col_score,
            segment_col=segment_col,
            segment_value=segment_value,
        )
        p_re = recal.p_holdout
        out["gini_recal"] = gini(y, p_re)
        out["brier_recal"] = brier(y, p_re)
        out["logloss_recal"] = logloss(y, p_re)
        if recal.artifact is not None:
            artifacts.append(recal.artifact)
    if do_refit and cols.cols_pred:
        missing = [c for c in cols.cols_pred if c not in obs_seg.columns]
        if not missing:
            mapping = cols.pred_woe_map()
            refit = refit_with_diagnostics(
                train,
                holdout,
                list(cols.cols_pred),
                cols.col_target,
                gates=gates,
                submodel=submodel,
                xgb_params=xgb_params,
                date_col=cols.col_date,
                grouping=None,
                n_jobs=n_jobs,
                portfolio_grouping=grouping,
                pred_woe_map=mapping,
                portfolio_frame=obs_all if obs_all is not None else obs_seg,
                cols_pred_woe=list(cols.cols_pred_woe or []),
                segment_col=segment_col,
                segment_value=segment_value,
                logit_cols=cols.logit_pred_cols(grouping=grouping),
                pred_val_map=cols.pred_val_map(),
            )
            delta = bootstrap_delta_gini(
                y,
                refit.p_holdout,
                p_pooled,
                n_bootstrap=gates.n_bootstrap,
                seed=gates.bootstrap_seed,
                z=gates.delta_gini_z,
                n_jobs=n_jobs,
                cap=gates.max_workers_cap,
            )
            out.update(delta)
            out["brier_refit"] = brier(y, refit.p_holdout)
            out["logloss_refit"] = logloss(y, refit.p_holdout)
            out["refit_method"] = refit.method
            for key in (
                "stability_assessed",
                "stability_score",
                "stability_pass",
                "stability_n_predictors",
                "stability_n_unstable",
                "unstable_predictors",
                "stability_reason",
            ):
                if key in refit.stability_summary:
                    out[key] = refit.stability_summary[key]
            stability_table = refit.stability
            grouping_notes = refit.grouping_comparison
            if refit.artifact is not None:
                artifacts.append(refit.artifact)
    return out, stability_table, artifacts, grouping_notes


def _matched_ar_for_segment(obs_seg, obs_all, cols, gates):
    # type: (pd.DataFrame, pd.DataFrame, ScorecardColumns, Gates) -> Dict[str, Any]
    if obs_seg.empty or obs_all.empty:
        return _empty_matched_ar_row()
    return matched_ar_comparison(
        obs_seg[cols.col_target],
        obs_seg[cols.col_score],
        obs_all[cols.col_target],
        obs_all[cols.col_score],
        gates=gates,
    )


def _evaluate_one_segment(task):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Everything computed for a single ``(segment_col, segment_value)`` cell.

    Pure with respect to the inputs, so it can run on any worker.
    """
    seg_col = task["segment_col"]
    value = task["segment_value"]
    part = task["part"]
    obs_all = task["obs_all"]
    cols = task["cols"]  # type: ScorecardColumns
    gates = task["gates"]  # type: Gates
    overall_gini = task["overall_gini"]
    defaults_all = task["defaults_all"]
    n_all = task["n_all"]
    grouping = task["grouping"]
    submodel = task["submodel"]
    xgb_params = task["xgb_params"]
    n_jobs = task["inner_n_jobs"]

    fan = fantomas_mask(part, cols)
    obs = part.loc[observable_mask(part, cols)]
    obs_ex = obs.loc[~fantomas_mask(obs, cols)]
    n_ttd = len(part)
    volume_share = n_ttd / n_all if n_all else float("nan")
    defaults = float(obs[cols.col_target].sum()) if len(obs) else 0.0
    default_share = defaults / defaults_all if defaults_all else 0.0
    important = is_important(volume_share, default_share, gates)

    if len(obs):
        m_obs = _slice_metrics(obs, cols, gates, n_jobs=n_jobs)
    else:
        m_obs = performance_bundle([], [])
        m_obs.update({"gini_se": float("nan"), "gini_ci_low": float("nan")})
    matched = _matched_ar_for_segment(obs, obs_all, cols, gates)
    ratio, vint = _vintage_ratio(obs, cols)
    if not vint.empty:
        vint = vint.assign(segment_col=seg_col, segment_value=str(value))

    summary_rows = []  # type: List[Dict[str, Any]]
    base = {
        "segment_col": seg_col,
        "segment_value": str(value),
        "slice": "observable",
        "n_ttd": n_ttd,
        "volume_share": volume_share,
        "default_share": default_share,
        "fantomas_rate": float(fan.mean()) if n_ttd else float("nan"),
        "important": important,
    }
    row = dict(base)
    row.update(m_obs)
    row.update(matched)
    summary_rows.append(row)
    if len(obs_ex) and cols.col_fantomas:
        m_ex = _slice_metrics(obs_ex, cols, gates, n_jobs=n_jobs)
        row_ex = dict(base)
        row_ex["slice"] = "observable_ex_fantomas"
        row_ex["fantomas_rate"] = float(fan.mean())
        row_ex.update(m_ex)
        row_ex.update(_empty_matched_ar_row())
        summary_rows.append(row_ex)

    q1, failed = q1_verdict(
        n=m_obs.get("n", 0),
        defaults=m_obs.get("defaults", 0),
        gini_seg=m_obs.get("gini", float("nan")),
        gini_ci_low=m_obs.get("gini_ci_low", float("nan")),
        gini_overall=overall_gini,
        oe=m_obs.get("oe", float("nan")),
        ece_val=m_obs.get("ece", float("nan")),
        vintage_ratio=ratio,
        gates=gates,
    )

    char = (
        feature_diagnostics(obs, obs_all, cols, gates, grouping=grouping, n_jobs=n_jobs)
        if len(obs) and len(obs_all)
        else pd.DataFrame()
    )
    divergent = shape_divergence(char, gates) if not char.empty else False
    if not char.empty:
        char = char.assign(segment_col=seg_col, segment_value=str(value))

    do_recal = q1 == "WEAK" and "rank_order" not in failed and "calibration" in failed
    do_refit = (q1 == "WEAK" and "rank_order" in failed) or divergent
    hold, stability_table, artifacts, grouping_notes = _run_holdout_models(
        obs,
        cols,
        gates,
        do_refit=do_refit,
        do_recal=do_recal or do_refit,
        grouping=grouping,
        submodel=submodel,
        xgb_params=xgb_params,
        n_jobs=n_jobs,
        obs_all=obs_all,
        segment_col=seg_col,
        segment_value=value,
    )
    hold.update(
        {
            "segment_col": seg_col,
            "segment_value": str(value),
            "shape_divergent": divergent,
        }
    )
    if not grouping_notes.empty:
        grouping_notes = grouping_notes.assign(segment_col=seg_col, segment_value=str(value))
    if not stability_table.empty:
        stability_table = stability_table.assign(segment_col=seg_col, segment_value=str(value))

    action, reason = q2_action(
        q1=q1,
        failed_pillars=failed,
        important=important,
        shape_divergent=divergent,
        delta_gini=hold.get("delta_gini"),
        delta_gini_ci_low=hold.get("delta_gini_ci_low"),
        brier_refit=hold.get("brier_refit"),
        brier_pooled=hold.get("brier_pooled"),
        logloss_refit=hold.get("logloss_refit"),
        logloss_pooled=hold.get("logloss_pooled"),
        gates=gates,
        stability_pass_flag=bool(hold.get("stability_pass", True)),
        stability_reason=str(hold.get("stability_reason", "") or ""),
    )
    if do_recal:
        action, reason = "RECALIBRATE", "rank_order_ok_calibration_fail"

    decision = {
        "segment_col": seg_col,
        "segment_value": str(value),
        "q1_verdict": q1,
        "failed_pillars": ",".join(failed),
        "important": important,
        "shape_divergent": divergent,
        "q2_action": action,
        "q2_reason": reason,
        "volume_share": volume_share,
        "default_share": default_share,
        "gini": m_obs.get("gini"),
        "gini_ratio": (m_obs.get("gini") / overall_gini) if overall_gini else float("nan"),
        "oe": m_obs.get("oe"),
        "obs_rate": m_obs.get("obs_rate"),
        "mean_pd": m_obs.get("mean_pd"),
        "ece": m_obs.get("ece"),
        "vintage_gini_ratio": ratio,
        "ar_segment": matched.get("ar_segment"),
        "ar_reference": matched.get("ar_reference"),
        "ar_reference_cutoff": matched.get("ar_reference_cutoff"),
        "ar_gap": matched.get("ar_gap"),
        "ar_gap_triggered": matched.get("ar_gap_triggered"),
        "matched_ar": matched.get("matched_ar"),
        "matched_ar_anchor": matched.get("matched_ar_anchor"),
        "matched_ar_threshold_segment": matched.get("matched_ar_threshold_segment"),
        "matched_ar_threshold_reference": matched.get("matched_ar_threshold_reference"),
        "gini_at_matched_ar": matched.get("gini_at_matched_ar"),
        "gini_reference_at_matched_ar": matched.get("gini_reference_at_matched_ar"),
        "gini_at_matched_ar_gap": matched.get("gini_at_matched_ar_gap"),
        "gini_at_matched_ar_ratio": matched.get("gini_at_matched_ar_ratio"),
        "ar_artifact_suspected": ar_artifact_suspected(
            failed,
            matched.get("ar_gap_triggered"),
            matched.get("gini_at_matched_ar_ratio"),
            gates,
            gini_at_matched_ar=matched.get("gini_at_matched_ar"),
        ),
        "psi_method": _dominant_psi_method(char),
        "refit_method": hold.get("refit_method"),
        "stability_score": hold.get("stability_score"),
        "stability_pass": hold.get("stability_pass"),
        "unstable_predictors": hold.get("unstable_predictors"),
        "stability_reason": hold.get("stability_reason"),
    }
    return {
        "summary_rows": summary_rows,
        "decision": decision,
        "refit": hold,
        "characteristics": char,
        "vintage": vint,
        "stability": stability_table,
        "fitted_artifacts": artifacts,
        "grouping_comparison": grouping_notes,
    }


def _dominant_psi_method(char):
    # type: (pd.DataFrame) -> Optional[str]
    if char is None or char.empty or "psi_method" not in char.columns:
        return None
    counts = char["psi_method"].value_counts()
    if counts.empty:
        return None
    return str(counts.index[0])


def is_overall_segment(segment_col, segment_value=None):
    # type: (Any, Any) -> bool
    """True for the synthetic portfolio row ``__overall__`` / ``ALL``."""
    if str(segment_col) != OVERALL_SEGMENT_COL:
        return False
    if segment_value is None:
        return True
    return str(segment_value) == OVERALL_SEGMENT_VALUE


def _portfolio_grouping_summary(obs, cols, grouping):
    # type: (pd.DataFrame, ScorecardColumns, Optional[BinningModel]) -> pd.DataFrame
    """IV, bin count and univariate Gini of the production WoE grouping."""
    if grouping is None or obs is None or obs.empty:
        return pd.DataFrame()
    table = grouping.iv_table()
    if table is None or table.empty:
        return pd.DataFrame()
    y = None
    if cols.col_target is not None and cols.col_target in obs.columns:
        y = obs[cols.col_target].to_numpy(dtype=float)
    ginis = []
    for feature in table["feature"].tolist():
        spec = grouping.specs.get(feature) if grouping.specs else None
        if spec is None or feature not in obs.columns or y is None:
            ginis.append(float("nan"))
            continue
        woe = spec.transform_woe(obs[feature])
        is_logit = spec.kind == "logit" or spec.transform == "logit"
        score = woe if is_logit else -woe
        # Fold through a sigmoid so metrics.gini (which clips to a PD) keeps rank.
        pd_hat = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(score, dtype=float), -30, 30)))
        ginis.append(gini(y, pd_hat))
    out = table.copy()
    out["univariate_gini"] = ginis
    out["segment_col"] = OVERALL_SEGMENT_COL
    out["segment_value"] = OVERALL_SEGMENT_VALUE
    return out


def _evaluate_portfolio(
    df,
    obs_all,
    cols,
    gates,
    grouping,
    overall,
    n_all,
    defaults_all,
    n_jobs,
    submodel=False,
    xgb_params=None,
    whatif=False,
):
    # type: (pd.DataFrame, pd.DataFrame, ScorecardColumns, Gates, Optional[BinningModel], Dict[str, float], int, float, Optional[int], bool, Optional[Dict[str, Any]], bool) -> Dict[str, Any]
    """Q1 (and optional what-if refit) for the pooled book."""
    fan_all = fantomas_mask(df, cols)
    ratio, vint = _vintage_ratio(obs_all, cols)
    if not vint.empty:
        vint = vint.assign(segment_col=OVERALL_SEGMENT_COL, segment_value=OVERALL_SEGMENT_VALUE)
    grouping_summary = _portfolio_grouping_summary(obs_all, cols, grouping)

    q1, failed = q1_verdict(
        n=overall.get("n", 0),
        defaults=overall.get("defaults", 0),
        gini_seg=overall.get("gini", float("nan")),
        gini_ci_low=overall.get("gini_ci_low", float("nan")),
        gini_overall=overall.get("gini", float("nan")),
        oe=overall.get("oe", float("nan")),
        ece_val=overall.get("ece", float("nan")),
        vintage_ratio=ratio,
        gates=gates,
    )
    do_recal = q1 == "WEAK" and "rank_order" not in failed and "calibration" in failed
    do_refit = False
    if whatif:
        do_refit = True
        do_recal = True
    hold, refit_stability, artifacts, grouping_notes = _run_holdout_models(
        obs_all,
        cols,
        gates,
        do_refit=do_refit,
        do_recal=do_recal,
        grouping=grouping,
        submodel=submodel,
        xgb_params=xgb_params,
        n_jobs=n_jobs,
        obs_all=obs_all,
        segment_col=OVERALL_SEGMENT_COL,
        segment_value=OVERALL_SEGMENT_VALUE,
    )
    hold.update(
        {
            "segment_col": OVERALL_SEGMENT_COL,
            "segment_value": OVERALL_SEGMENT_VALUE,
            "shape_divergent": False,
            "whatif": bool(whatif),
        }
    )
    if not grouping_notes.empty:
        grouping_notes = grouping_notes.assign(
            segment_col=OVERALL_SEGMENT_COL, segment_value=OVERALL_SEGMENT_VALUE
        )
    production_stability = _overall_stability(df, obs_all, cols, gates, grouping, n_jobs)
    if not production_stability.empty:
        production_stability = production_stability.assign(stability_source="production")
    if not refit_stability.empty:
        refit_stability = refit_stability.assign(
            segment_col=OVERALL_SEGMENT_COL,
            segment_value=OVERALL_SEGMENT_VALUE,
            stability_source="refit",
        )
    stability_frames = []  # type: List[pd.DataFrame]
    if not production_stability.empty:
        stability_frames.append(production_stability)
    if not refit_stability.empty:
        stability_frames.append(refit_stability)
    stability_table = ordered_concat(stability_frames)

    action, reason = q2_portfolio_action(
        q1=q1,
        failed_pillars=failed,
        delta_gini=hold.get("delta_gini"),
        delta_gini_ci_low=hold.get("delta_gini_ci_low"),
        brier_refit=hold.get("brier_refit"),
        brier_pooled=hold.get("brier_pooled"),
        logloss_refit=hold.get("logloss_refit"),
        logloss_pooled=hold.get("logloss_pooled"),
        gates=gates,
        stability_pass_flag=bool(hold.get("stability_pass", True)),
        stability_reason=str(hold.get("stability_reason", "") or ""),
        whatif=bool(whatif),
    )
    overall_gini = overall.get("gini")
    gini_ratio = 1.0 if overall_gini and overall_gini == overall_gini and overall_gini != 0 else float("nan")
    summary = {
        "segment_col": OVERALL_SEGMENT_COL,
        "segment_value": OVERALL_SEGMENT_VALUE,
        "slice": "observable",
        "n_ttd": n_all,
        "volume_share": 1.0,
        "default_share": 1.0,
        "fantomas_rate": float(fan_all.mean()) if n_all else float("nan"),
        "important": True,
    }
    summary.update(overall)
    summary.update(_empty_matched_ar_row())
    decision = {
        "segment_col": OVERALL_SEGMENT_COL,
        "segment_value": OVERALL_SEGMENT_VALUE,
        "q1_verdict": q1,
        "failed_pillars": ",".join(failed),
        "important": True,
        "shape_divergent": False,
        "q2_action": action,
        "q2_reason": reason,
        "volume_share": 1.0,
        "default_share": 1.0 if defaults_all else 0.0,
        "gini": overall.get("gini"),
        "gini_ratio": gini_ratio,
        "oe": overall.get("oe"),
        "obs_rate": overall.get("obs_rate"),
        "mean_pd": overall.get("mean_pd"),
        "ece": overall.get("ece"),
        "vintage_gini_ratio": ratio,
        "ar_segment": None,
        "ar_reference": None,
        "ar_reference_cutoff": None,
        "ar_gap": None,
        "ar_gap_triggered": False,
        "matched_ar": None,
        "matched_ar_anchor": None,
        "matched_ar_threshold_segment": None,
        "matched_ar_threshold_reference": None,
        "gini_at_matched_ar": None,
        "gini_reference_at_matched_ar": None,
        "gini_at_matched_ar_gap": None,
        "gini_at_matched_ar_ratio": None,
        "ar_artifact_suspected": False,
        "psi_method": "grouping_bins" if grouping is not None else None,
        "refit_method": hold.get("refit_method"),
        "stability_score": hold.get("stability_score"),
        "stability_pass": hold.get("stability_pass"),
        "unstable_predictors": hold.get("unstable_predictors"),
        "stability_reason": hold.get("stability_reason"),
    }
    return {
        "summary_rows": [summary],
        "decision": decision,
        "refit": hold,
        "characteristics": pd.DataFrame(),
        "vintage": vint,
        "stability": stability_table,
        "fitted_artifacts": artifacts,
        "grouping_comparison": grouping_notes,
        "grouping_summary": grouping_summary,
    }


def _coerce_grouping(grouping):
    # type: (Any) -> Optional[BinningModel]
    """Accept a ``BinningModel`` or a parsed scorecard that carries one."""
    if grouping is None:
        return None
    if isinstance(grouping, BinningModel):
        return grouping
    inner = getattr(grouping, "grouping", None)
    if isinstance(inner, BinningModel):
        return inner
    return grouping


def evaluate_segments(
    df,
    cols,
    gates=None,
    grouping=None,
    n_jobs=None,
    submodel=False,
    xgb_params=None,
    portfolio_whatif=False,
    include_segments=True,
):
    # type: (pd.DataFrame, ScorecardColumns, Optional[Gates], Any, Optional[int], bool, Optional[Dict[str, Any]], bool, bool) -> SegmentEvalResult
    """Evaluate the pooled book and every segment value of every segmentation column.

    The pooled book is always scored as the synthetic segment ``__overall__`` /
    ``ALL``: Gini, calibration (O/E, ECE) and vintage stability, plus the
    production WoE grouping's IV / univariate Gini when ``grouping`` is
    supplied.  Pass ``portfolio_whatif=True`` to also run a same-predictor
    holdout refit on the whole book regardless of the Q1 verdict (see
    :func:`evaluate_portfolio_whatif`).

    ``grouping`` is a :class:`~scorecard_segment_eval.binning.BinningModel`, or
    a parsed SQL scorecard (``ScorecardSQLModel`` / ``ResolvedMetadata``) whose
    ``grouping`` attribute is used.  Pass the grouping parsed from production
    SQL so PSI and the refit comparison use those bins, including null
    imputation.

    ``n_jobs`` controls the per-segment fan-out (``None`` falls back to
    ``Gates.n_jobs``).  Results are identical for any worker count.
    ``include_segments=False`` skips the per-segment loop and keeps only the
    portfolio row.
    """
    grouping = _coerce_grouping(grouping)
    gates = gates or Gates()
    jobs = n_jobs if n_jobs is not None else gates.n_jobs
    obs_all = df.loc[observable_mask(df, cols)].copy()
    overall = _slice_metrics(obs_all, cols, gates, n_jobs=jobs)
    overall_gini = overall["gini"]
    n_all = len(df)
    defaults_all = float(obs_all[cols.col_target].sum()) if len(obs_all) else 0.0

    portfolio = _evaluate_portfolio(
        df,
        obs_all,
        cols,
        gates,
        grouping,
        overall,
        n_all,
        defaults_all,
        jobs,
        submodel=submodel,
        xgb_params=xgb_params,
        whatif=bool(portfolio_whatif),
    )

    tasks = []  # type: List[Dict[str, Any]]
    if include_segments:
        for seg_col in cols.cols_segment or ():
            if seg_col not in df.columns:
                continue
            for value, part in df.groupby(seg_col, dropna=False):
                tasks.append(
                    {
                        "segment_col": seg_col,
                        "segment_value": value,
                        "part": part,
                        "obs_all": obs_all,
                        "cols": cols,
                        "gates": gates,
                        "overall_gini": overall_gini,
                        "defaults_all": defaults_all,
                        "n_all": n_all,
                        "grouping": grouping,
                        "submodel": submodel,
                        "xgb_params": xgb_params,
                        # Inner loops stay serial: the outer fan-out already
                        # saturates the pool and parallel.map_jobs refuses to nest.
                        "inner_n_jobs": jobs,
                    }
                )

    outputs = map_jobs(_evaluate_one_segment, tasks, n_jobs=jobs, cap=gates.max_workers_cap)

    summary_rows = list(portfolio["summary_rows"])
    decision_rows = [portfolio["decision"]]
    refit_rows = [portfolio["refit"]]
    char_frames = []  # type: List[pd.DataFrame]
    vintage_frames = []  # type: List[pd.DataFrame]
    stability_frames = []  # type: List[pd.DataFrame]
    grouping_frames = []  # type: List[pd.DataFrame]
    artifacts = list(portfolio.get("fitted_artifacts") or [])
    if portfolio.get("vintage") is not None and not portfolio["vintage"].empty:
        vintage_frames.append(portfolio["vintage"])
    if portfolio.get("stability") is not None and not portfolio["stability"].empty:
        stability_frames.append(portfolio["stability"])
    grouping_notes = portfolio.get("grouping_comparison", pd.DataFrame())
    if grouping_notes is not None and not grouping_notes.empty:
        grouping_frames.append(grouping_notes)
    for out in outputs:
        summary_rows.extend(out["summary_rows"])
        decision_rows.append(out["decision"])
        refit_rows.append(out["refit"])
        char_frames.append(out["characteristics"])
        vintage_frames.append(out["vintage"])
        stability_frames.append(out["stability"])
        grouping_frames.append(out.get("grouping_comparison", pd.DataFrame()))
        artifacts.extend(out.get("fitted_artifacts") or [])

    meta = {
        "n_rows": int(len(df)),
        "n_observable": int(len(obs_all)),
        "overall_gini": overall_gini,
        "n_jobs": jobs,
        "submodel": bool(submodel),
        "grouping_supplied": grouping is not None,
        "pred_woe_map": cols.pred_woe_map(),
        "n_fitted_artifacts": len(artifacts),
        "portfolio_whatif": bool(portfolio_whatif),
        "include_segments": bool(include_segments),
        "gates": dict(gates.__dict__),
        "columns": cols.as_dict(),
    }
    return SegmentEvalResult(
        segment_summary=pd.DataFrame(summary_rows),
        decisions=pd.DataFrame(decision_rows),
        refit_comparison=pd.DataFrame(refit_rows),
        characteristics=ordered_concat(char_frames),
        vintage=ordered_concat(vintage_frames),
        stability=ordered_concat(stability_frames),
        grouping_comparison=ordered_concat(grouping_frames),
        grouping_summary=portfolio.get("grouping_summary", pd.DataFrame()),
        fitted_artifacts=artifacts,
        meta=meta,
    )


def evaluate_portfolio_whatif(
    df,
    cols,
    gates=None,
    grouping=None,
    n_jobs=None,
    submodel=False,
    xgb_params=None,
):
    # type: (pd.DataFrame, ScorecardColumns, Optional[Gates], Any, Optional[int], bool, Optional[Dict[str, Any]]) -> SegmentEvalResult
    """What-if: refit the pooled scorecard on the whole book.

    Evaluates portfolio-level Gini, calibration and vintage / WoE-grouping
    stability, then *always* runs a same-predictor holdout refit and a
    two-parameter PD recalibration, regardless of the Q1 verdict.  Per-segment
    cells are skipped; the result contains the ``__overall__`` / ``ALL`` row
    only.

    Use this when the question is "would re-estimating the same predictors on
    recent data beat the production PD?", not "is a dedicated segment model
    warranted?".
    """
    return evaluate_segments(
        df,
        cols,
        gates=gates,
        grouping=grouping,
        n_jobs=n_jobs,
        submodel=submodel,
        xgb_params=xgb_params,
        portfolio_whatif=True,
        include_segments=False,
    )


def _overall_stability(df, obs_all, cols, gates, grouping, n_jobs):
    # type: (pd.DataFrame, pd.DataFrame, ScorecardColumns, Gates, Optional[BinningModel], Optional[int]) -> pd.DataFrame
    """Portfolio-level predictor stability, so the report always has a baseline."""
    if not cols.cols_pred or cols.col_target is None or obs_all.empty:
        return pd.DataFrame()
    usable = [c for c in cols.cols_pred if c in obs_all.columns]
    if not usable:
        return pd.DataFrame()
    table = predictor_stability(
        obs_all,
        usable,
        cols.col_target,
        cols.col_date,
        gates=gates,
        grouping=grouping,
        n_jobs=n_jobs,
        reference_frame=obs_all,
        pred_val_map=cols.pred_val_map(),
    )
    if table.empty:
        return table
    return table.assign(segment_col="__overall__", segment_value="ALL")
