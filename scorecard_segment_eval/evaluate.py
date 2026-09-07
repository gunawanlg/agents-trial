from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from scorecard_segment_eval.bootstrap import bootstrap_delta_gini, bootstrap_gini_ci
from scorecard_segment_eval.characteristics import feature_diagnostics, shape_divergence
from scorecard_segment_eval.decision import is_important, q1_verdict, q2_action
from scorecard_segment_eval.metrics import brier, gini, logloss, performance_bundle
from scorecard_segment_eval.population import fantomas_mask, observable_mask, time_holdout_mask
from scorecard_segment_eval.refit import recalibrate_pd, refit_same_predictors
from scorecard_segment_eval.schema import Gates, ScorecardColumns


@dataclass
class SegmentEvalResult:
    segment_summary: pd.DataFrame
    decisions: pd.DataFrame
    refit_comparison: pd.DataFrame
    characteristics: pd.DataFrame
    vintage: pd.DataFrame


def _vintage_ratio(obs, cols):
    if obs.empty:
        return None, pd.DataFrame()
    dates = pd.to_datetime(obs[cols.col_date], errors="coerce")
    months = dates.dt.to_period("M")
    rows = []
    for period, part in obs.groupby(months, dropna=True):
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


def _run_holdout_models(obs_seg, cols, gates, do_refit, do_recal, **kwargs):
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
    }
    if len(obs_seg) < 40:
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
            p_refit, details = refit_same_predictors(
                train,
                holdout,
                cols.cols_pred,
                cols.col_target,
                gates,
                date_col=cols.col_date,
                return_details=True,
                **kwargs
            )
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
            empty["refit_mean_predictor_psi"] = details["mean_selected_psi"]
            empty["refit_selected_predictors"] = ",".join(details["selected_predictors"])
            empty["refit_selection_objective"] = details["selection_objective"]
    return empty


def _similar_approval_rate_analysis(part, portfolio, cols, gates):
    approval_seg = float(part[cols.col_obs].fillna(0).astype(float).mean())
    approval_all = float(portfolio[cols.col_obs].fillna(0).astype(float).mean())
    gap = abs(approval_seg - approval_all)
    output = {
        "approval_rate": approval_seg,
        "approval_rate_overall": approval_all,
        "approval_rate_gap": gap,
        "gini_similar_ar": float("nan"),
        "gini_overall_similar_ar": float("nan"),
        "similar_ar_analysis": "not_required",
    }
    if gap < gates.approval_rate_gap:
        return output
    common_ar = min(approval_seg, approval_all)
    ginis = []
    counts = []
    for frame in (part, portfolio):
        score = pd.to_numeric(frame[cols.col_score], errors="coerce")
        threshold = score.quantile(common_ar)
        simulated = frame.loc[score <= threshold]
        simulated = simulated.loc[observable_mask(simulated, cols)]
        ginis.append(gini(simulated[cols.col_target], simulated[cols.col_score]))
        counts.append(len(simulated))
    output.update(
        {
            "gini_similar_ar": ginis[0],
            "gini_overall_similar_ar": ginis[1],
            "similar_ar_analysis": "simulated_at_lower_ar=%.4f;n_segment=%d;n_overall=%d" % (
                common_ar,
                counts[0],
                counts[1],
            ),
        }
    )
    return output


def evaluate_segments(df, cols, gates=None, **kwargs):
    gates = gates or Gates()
    obs_all = df.loc[observable_mask(df, cols)].copy()
    fan_all = fantomas_mask(df, cols)
    overall = _slice_metrics(obs_all, cols, gates)
    overall_gini = overall["gini"]
    n_all = len(df)
    defaults_all = float(obs_all[cols.col_target].sum()) if len(obs_all) else 0.0

    summary_rows = []
    decision_rows = []
    refit_rows = []
    char_rows = []
    vintage_rows = []

    overall_row = {
        "segment_col": "__overall__",
        "segment_value": "ALL",
        "slice": "observable",
        "n_ttd": n_all,
        "volume_share": 1.0,
        "default_share": 1.0,
        "fantomas_rate": float(fan_all.mean()) if n_all else float("nan"),
        "important": True,
        **overall,
    }
    summary_rows.append(overall_row)

    for seg_col in cols.cols_segment:
        if seg_col not in df.columns:
            continue
        for value, part in df.groupby(seg_col, dropna=False):
            fan = fantomas_mask(part, cols)
            obs = part.loc[observable_mask(part, cols)]
            obs_ex = obs.loc[~fantomas_mask(obs, cols)]
            n_ttd = len(part)
            volume_share = n_ttd / n_all if n_all else float("nan")
            defaults = float(obs[cols.col_target].sum()) if len(obs) else 0.0
            default_share = defaults / defaults_all if defaults_all else 0.0
            important = is_important(volume_share, default_share, gates)
            similar_ar = _similar_approval_rate_analysis(part, df, cols, gates)

            m_obs = _slice_metrics(obs, cols, gates) if len(obs) else performance_bundle([], [])
            if len(obs) == 0:
                m_obs.update({"gini_se": float("nan"), "gini_ci_low": float("nan")})
            ratio, vint = _vintage_ratio(obs, cols)
            if not vint.empty:
                vint = vint.assign(segment_col=seg_col, segment_value=str(value))
                vintage_rows.append(vint)

            summary_rows.append(
                {
                    "segment_col": seg_col,
                    "segment_value": str(value),
                    "slice": "observable",
                    "n_ttd": n_ttd,
                    "volume_share": volume_share,
                    "default_share": default_share,
                    "fantomas_rate": float(fan.mean()) if n_ttd else float("nan"),
                    "important": important,
                    **similar_ar,
                    **m_obs,
                }
            )
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

            char = feature_diagnostics(obs, obs_all, cols, gates) if len(obs) and len(obs_all) else pd.DataFrame()
            divergent = shape_divergence(char, gates) if not char.empty else False
            if not char.empty:
                char = char.assign(segment_col=seg_col, segment_value=str(value))
                char_rows.append(char)

            do_recal = q1 == "WEAK" and "rank_order" not in failed and "calibration" in failed
            do_refit = (q1 == "WEAK" and "rank_order" in failed) or divergent
            hold = _run_holdout_models(
                obs,
                cols,
                gates,
                do_refit=do_refit,
                do_recal=do_recal or do_refit,
                **kwargs
            )
            hold.update(
                {
                    "segment_col": seg_col,
                    "segment_value": str(value),
                    "shape_divergent": divergent,
                }
            )
            refit_rows.append(hold)

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

            decision_rows.append(
                {
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
                    **similar_ar,
                }
            )

    return SegmentEvalResult(
        segment_summary=pd.DataFrame(summary_rows),
        decisions=pd.DataFrame(decision_rows),
        refit_comparison=pd.DataFrame(refit_rows),
        characteristics=pd.concat(char_rows, ignore_index=True) if char_rows else pd.DataFrame(),
        vintage=pd.concat(vintage_rows, ignore_index=True) if vintage_rows else pd.DataFrame(),
    )
