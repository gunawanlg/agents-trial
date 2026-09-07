from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scorecard_segment_eval.bootstrap import bootstrap_delta_gini, bootstrap_gini_ci
from scorecard_segment_eval.characteristics import feature_diagnostics, shape_divergence
from scorecard_segment_eval.compat import safe_groupby
from scorecard_segment_eval.decision import is_important, q1_verdict, q2_action
from scorecard_segment_eval.grouping import load_grouping
from scorecard_segment_eval.metrics import brier, gini, logloss, performance_bundle
from scorecard_segment_eval.parallel import run_parallel
from scorecard_segment_eval.population import approval_rate, fantomas_mask, observable_mask, time_holdout_mask
from scorecard_segment_eval.refit import recalibrate_pd, refit_segment_model
from scorecard_segment_eval.schema import Gates, ScorecardColumns


@dataclass
class SegmentEvalResult:
    segment_summary: pd.DataFrame
    decisions: pd.DataFrame
    refit_comparison: pd.DataFrame
    characteristics: pd.DataFrame
    vintage: pd.DataFrame


def _vintage_ratio(obs, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> Tuple[Optional[float], pd.DataFrame]
    if obs.empty or cols.col_date is None or cols.col_date not in obs.columns:
        return None, pd.DataFrame()
    dates = pd.to_datetime(obs[cols.col_date], errors="coerce")
    months = dates.dt.to_period("M")
    rows = []
    grouped = safe_groupby(obs, months, dropna=True)
    for period, part in grouped:
        bundle = performance_bundle(part[cols.col_target], part[cols.col_score])
        rows.append({"vintage": str(period), "n": bundle["n"], "defaults": bundle["defaults"], "gini": bundle["gini"], "oe": bundle["oe"]})
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


def _slice_metrics(part, cols, gates):
    # type: (pd.DataFrame, ScorecardColumns, Gates) -> Dict[str, float]
    bundle = performance_bundle(part[cols.col_target], part[cols.col_score], n_ece_bins=gates.n_ece_bins)
    ci = bootstrap_gini_ci(
        part[cols.col_target],
        part[cols.col_score],
        n_bootstrap=gates.n_bootstrap,
        seed=gates.bootstrap_seed,
        z=gates.delta_gini_z,
    )
    bundle["gini_se"] = ci["gini_se"]
    bundle["gini_ci_low"] = ci["gini_ci_low"]
    return bundle


def gini_at_approval_rate(part, cols, target_ar, higher_is_risk=True):
    """Gini on the approved slice after matching ``target_ar`` via a score cutoff.

    If ``higher_is_risk`` (PD), keep the best (lowest) scores until the keep-rate
    equals ``target_ar``. Observables among those kept are used for Gini.
    """
    if part is None or len(part) == 0 or target_ar is None or not np.isfinite(target_ar):
        return float("nan"), 0.0
    target_ar = float(np.clip(target_ar, 0.0, 1.0))
    n = len(part)
    n_keep = max(int(round(target_ar * n)), 1)
    n_keep = min(n_keep, n)
    scores = pd.to_numeric(part[cols.col_score], errors="coerce")
    if higher_is_risk:
        keep_idx = scores.nsmallest(n_keep).index
    else:
        keep_idx = scores.nlargest(n_keep).index
    sub = part.loc[keep_idx]
    obs = sub.loc[observable_mask(sub, cols)]
    if len(obs) < 20 or obs[cols.col_target].nunique() < 2:
        return float("nan"), float(len(obs))
    return gini(obs[cols.col_target], obs[cols.col_score]), float(len(obs))


def _run_holdout_models(obs_seg, cols, gates, do_refit, do_recal):
    # type: (pd.DataFrame, ScorecardColumns, Gates, bool, bool) -> Dict[str, Any]
    empty = {
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
        "gini_refit_stable": None,
        "delta_gini_stable": None,
        "refit_dropped": "",
        "refit_model": None,
    }
    if len(obs_seg) < 40:
        return empty
    if cols.col_date is None or cols.col_date not in obs_seg.columns:
        return empty
    ho = time_holdout_mask(obs_seg[cols.col_date], gates.holdout_frac)
    train = obs_seg.loc[~ho]
    holdout = obs_seg.loc[ho]
    if train[cols.col_target].sum() < 5 or holdout[cols.col_target].sum() < 5:
        return empty
    y = holdout[cols.col_target].to_numpy(dtype=float)
    p_pooled = holdout[cols.col_score].to_numpy(dtype=float)
    empty["n_holdout"] = float(len(holdout))
    empty["defaults_holdout"] = float(y.sum())
    empty["gini_pooled"] = gini(y, p_pooled)
    empty["brier_pooled"] = brier(y, p_pooled)
    empty["logloss_pooled"] = logloss(y, p_pooled)
    if do_recal:
        p_re = recalibrate_pd(train[cols.col_score].to_numpy(), train[cols.col_target].to_numpy(), p_pooled)
        empty["gini_recal"] = gini(y, p_re)
        empty["brier_recal"] = brier(y, p_re)
        empty["logloss_recal"] = logloss(y, p_re)
    if do_refit and cols.cols_pred:
        missing = [c for c in cols.cols_pred if c not in obs_seg.columns]
        if not missing:
            result = refit_segment_model(
                train,
                holdout,
                cols.cols_pred,
                cols.col_target,
                gates=gates,
                date_col=cols.col_date,
                submodel=bool(getattr(gates, "submodel", False)),
            )
            p_refit = result.p_holdout
            delta = bootstrap_delta_gini(
                y,
                p_refit,
                p_pooled,
                n_bootstrap=gates.n_bootstrap,
                seed=gates.bootstrap_seed,
                z=gates.delta_gini_z,
            )
            empty.update(delta)
            empty["brier_refit"] = brier(y, p_refit)
            empty["logloss_refit"] = logloss(y, p_refit)
            empty["gini_refit_stable"] = gini(y, result.p_holdout_stable)
            if np.isfinite(empty["gini_refit_stable"]) and np.isfinite(empty.get("gini_pooled") or float("nan")):
                empty["delta_gini_stable"] = empty["gini_refit_stable"] - empty["gini_pooled"]
            empty["refit_dropped"] = ",".join(result.dropped)
            empty["refit_model"] = result.model_kind
    return empty


def _evaluate_one_segment(payload):
    """Worker for one (segment_col, value) slice. Safe for thread pools."""
    (
        seg_col,
        value,
        part,
        cols,
        gates,
        overall_gini,
        n_all,
        defaults_all,
        obs_all,
        grouping,
        sibling_min_ar,
        overall_ar,
    ) = payload

    fan = fantomas_mask(part, cols)
    obs = part.loc[observable_mask(part, cols)]
    obs_ex = obs.loc[~fantomas_mask(obs, cols)]
    n_ttd = len(part)
    volume_share = n_ttd / n_all if n_all else float("nan")
    defaults = float(obs[cols.col_target].sum()) if len(obs) else 0.0
    default_share = defaults / defaults_all if defaults_all else 0.0
    important = is_important(volume_share, default_share, gates)
    ar = approval_rate(part, cols)
    ar_gap = abs(ar - overall_ar) if np.isfinite(ar) and np.isfinite(overall_ar) else float("nan")
    lower_ar = None
    gini_aligned = float("nan")
    n_aligned = float("nan")
    if np.isfinite(ar):
        candidates = [ar]
        if np.isfinite(overall_ar):
            candidates.append(overall_ar)
        if sibling_min_ar is not None and np.isfinite(sibling_min_ar):
            candidates.append(sibling_min_ar)
        lower_ar = min(candidates)
        gap_vs_lower = ar - lower_ar if np.isfinite(lower_ar) else 0.0
        material = (np.isfinite(ar_gap) and ar_gap >= gates.ar_gap_material) or (
            np.isfinite(gap_vs_lower) and gap_vs_lower >= gates.ar_gap_material
        )
        if material and lower_ar is not None:
            higher_is_risk = bool(getattr(cols, "score_higher_is_risk", True))
            gini_aligned, n_aligned = gini_at_approval_rate(part, cols, lower_ar, higher_is_risk=higher_is_risk)
        else:
            lower_ar = None

    m_obs = _slice_metrics(obs, cols, gates) if len(obs) else performance_bundle([], [])
    if len(obs) == 0:
        m_obs.update({"gini_se": float("nan"), "gini_ci_low": float("nan")})
    ratio, vint = _vintage_ratio(obs, cols)
    if not vint.empty:
        vint = vint.assign(segment_col=seg_col, segment_value=str(value))

    summary_rows = [
        {
            "segment_col": seg_col,
            "segment_value": str(value),
            "slice": "observable",
            "n_ttd": n_ttd,
            "volume_share": volume_share,
            "default_share": default_share,
            "fantomas_rate": float(fan.mean()) if n_ttd else float("nan"),
            "important": important,
            "approval_rate": ar,
            "ar_gap_vs_overall": ar_gap,
            "ar_aligned": lower_ar,
            "gini_ar_aligned": gini_aligned,
            "n_ar_aligned": n_aligned,
            **m_obs,
        }
    ]
    if len(obs_ex) and cols.col_fantomas:
        m_ex = _slice_metrics(obs_ex, cols, gates)
        summary_rows.append(
            {
                "segment_col": seg_col,
                "segment_value": str(value),
                "slice": "observable_ex_fantomas",
                "n_ttd": n_ttd,
                "volume_share": volume_share,
                "default_share": default_share,
                "fantomas_rate": float(fan.mean()),
                "important": important,
                "approval_rate": ar,
                "ar_gap_vs_overall": ar_gap,
                "ar_aligned": lower_ar,
                "gini_ar_aligned": gini_aligned,
                "n_ar_aligned": n_aligned,
                **m_ex,
            }
        )

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

    char = feature_diagnostics(obs, obs_all, cols, gates, grouping=grouping) if len(obs) and len(obs_all) else pd.DataFrame()
    divergent = shape_divergence(char, gates) if not char.empty else False
    if not char.empty:
        char = char.assign(segment_col=seg_col, segment_value=str(value))

    do_recal = q1 == "WEAK" and "rank_order" not in failed and "calibration" in failed
    do_refit = (q1 == "WEAK" and "rank_order" in failed) or divergent
    hold = _run_holdout_models(obs, cols, gates, do_refit=do_refit, do_recal=do_recal or do_refit)
    hold.update(
        {
            "segment_col": seg_col,
            "segment_value": str(value),
            "shape_divergent": divergent,
        }
    )

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
        "ece": m_obs.get("ece"),
        "vintage_gini_ratio": ratio,
        "approval_rate": ar,
        "ar_gap_vs_overall": ar_gap,
        "ar_aligned": lower_ar,
        "gini_ar_aligned": gini_aligned,
    }
    return {
        "summary_rows": summary_rows,
        "decision": decision,
        "hold": hold,
        "char": char,
        "vintage": vint,
    }


def evaluate_segments(df, cols, gates=None):
    # type: (pd.DataFrame, ScorecardColumns, Optional[Gates]) -> SegmentEvalResult
    gates = gates or Gates()
    grouping = load_grouping(getattr(cols, "grouping_path", None) or getattr(gates, "grouping_path", None))
    obs_all = df.loc[observable_mask(df, cols)].copy()
    fan_all = fantomas_mask(df, cols)
    overall = _slice_metrics(obs_all, cols, gates) if len(obs_all) else performance_bundle([], [])
    if len(obs_all) == 0:
        overall.update({"gini_se": float("nan"), "gini_ci_low": float("nan")})
    overall_gini = overall.get("gini", float("nan"))
    n_all = len(df)
    defaults_all = float(obs_all[cols.col_target].sum()) if len(obs_all) else 0.0
    overall_ar = approval_rate(df, cols)

    summary_rows = []  # type: List[dict]
    decision_rows = []  # type: List[dict]
    refit_rows = []  # type: List[dict]
    char_rows = []  # type: List[pd.DataFrame]
    vintage_rows = []  # type: List[pd.DataFrame]

    overall_row = {
        "segment_col": "__overall__",
        "segment_value": "ALL",
        "slice": "observable",
        "n_ttd": n_all,
        "volume_share": 1.0,
        "default_share": 1.0,
        "fantomas_rate": float(fan_all.mean()) if n_all else float("nan"),
        "important": True,
        "approval_rate": overall_ar,
        "ar_gap_vs_overall": 0.0,
        "ar_aligned": None,
        "gini_ar_aligned": float("nan"),
        "n_ar_aligned": float("nan"),
        **overall,
    }
    summary_rows.append(overall_row)

    tasks = []
    for seg_col in cols.cols_segment:
        if seg_col not in df.columns:
            continue
        grouped = safe_groupby(df, seg_col, dropna=False)
        sibling_ars = []
        parts = []
        for value, part in grouped:
            part = part.copy()
            parts.append((value, part))
            sibling_ars.append(approval_rate(part, cols))
        finite_ars = [a for a in sibling_ars if a == a]
        sibling_min_ar = min(finite_ars) if finite_ars else None
        for value, part in parts:
            tasks.append(
                (
                    seg_col,
                    value,
                    part,
                    cols,
                    gates,
                    overall_gini,
                    n_all,
                    defaults_all,
                    obs_all,
                    grouping,
                    sibling_min_ar,
                    overall_ar,
                )
            )

    n_jobs = getattr(gates, "n_jobs", 1)
    results = run_parallel(_evaluate_one_segment, tasks, n_jobs=n_jobs, prefer="threads")
    for item in results:
        summary_rows.extend(item["summary_rows"])
        decision_rows.append(item["decision"])
        refit_rows.append(item["hold"])
        if item["char"] is not None and not item["char"].empty:
            char_rows.append(item["char"])
        if item["vintage"] is not None and not item["vintage"].empty:
            vintage_rows.append(item["vintage"])

    return SegmentEvalResult(
        segment_summary=pd.DataFrame(summary_rows),
        decisions=pd.DataFrame(decision_rows),
        refit_comparison=pd.DataFrame(refit_rows),
        characteristics=pd.concat(char_rows, ignore_index=True) if char_rows else pd.DataFrame(),
        vintage=pd.concat(vintage_rows, ignore_index=True) if vintage_rows else pd.DataFrame(),
    )


# Alias requested in the product brief
evaluate_segment = evaluate_segments
