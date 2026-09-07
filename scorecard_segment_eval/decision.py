from typing import List, Optional, Tuple

from scorecard_segment_eval.schema import Gates


def has_power(n, defaults, gates):
    # type: (float, float, Gates) -> bool
    return n >= gates.min_n and defaults >= gates.min_events


def rank_order_pass(gini_seg, gini_ci_low, gini_overall, gates):
    # type: (float, float, float, Gates) -> bool
    if not (gini_seg == gini_seg):  # NaN
        return False
    if gini_overall and gini_overall == gini_overall and gini_overall != 0:
        ratio = gini_seg / gini_overall
    else:
        ratio = float("nan")
    absolute_ok = gini_seg >= gates.gini_floor
    relative_ok = ratio == ratio and ratio >= gates.gini_ratio_floor
    ci_clears_floor = gini_ci_low == gini_ci_low and gini_ci_low >= gates.gini_floor
    return absolute_ok and (relative_ok or ci_clears_floor)


def calibration_pass(oe, ece_val, gates):
    # type: (float, float, Gates) -> bool
    if not (oe == oe) or not (ece_val == ece_val):
        return False
    return gates.oe_lo <= oe <= gates.oe_hi and ece_val <= gates.ece_max


def stability_pass(latest_to_early_gini_ratio, gates):
    # type: (Optional[float], Gates) -> Tuple[bool, bool]
    """Returns (pass, is_warn_only). Fail only on a severe configured drop."""
    if latest_to_early_gini_ratio is None or not (latest_to_early_gini_ratio == latest_to_early_gini_ratio):
        return True, False
    if latest_to_early_gini_ratio < gates.vintage_gini_ratio_floor:
        return False, False
    return True, False


def q1_verdict(
    n,  # type: float
    defaults,  # type: float
    gini_seg,  # type: float
    gini_ci_low,  # type: float
    gini_overall,  # type: float
    oe,  # type: float
    ece_val,  # type: float
    vintage_ratio,  # type: Optional[float]
    gates,  # type: Gates
    **kwargs
):
    # type: (...) -> Tuple[str, List[str]]
    if not has_power(n, defaults, gates):
        return "INCONCLUSIVE", ["insufficient_power"]
    failed = []  # type: List[str]
    if not rank_order_pass(gini_seg, gini_ci_low, gini_overall, gates):
        failed.append("rank_order")
    if not calibration_pass(oe, ece_val, gates):
        failed.append("calibration")
    stab_ok, _unused = stability_pass(vintage_ratio, gates)
    if not stab_ok:
        failed.append("stability")
    if failed:
        return "WEAK", failed
    return "GOOD", []


def is_important(volume_share, default_share, gates):
    # type: (float, float, Gates) -> bool
    return volume_share >= gates.importance_share_floor or default_share >= gates.importance_share_floor


def q2_action(
    q1,  # type: str
    failed_pillars,  # type: List[str]
    important,  # type: bool
    shape_divergent,  # type: bool
    delta_gini,  # type: Optional[float]
    delta_gini_ci_low,  # type: Optional[float]
    brier_refit,  # type: Optional[float]
    brier_pooled,  # type: Optional[float]
    logloss_refit,  # type: Optional[float]
    logloss_pooled,  # type: Optional[float]
    gates,  # type: Gates
    **kwargs
):
    # type: (...) -> Tuple[str, str]
    if q1 == "INCONCLUSIVE":
        return "NONE", "insufficient_power"
    rank_ok = "rank_order" not in failed_pillars
    cal_ok = "calibration" not in failed_pillars
    if rank_ok and not cal_ok:
        return "RECALIBRATE", "rank_order_ok_calibration_fail"
    if q1 == "GOOD":
        return "KEEP_POOLED", "performance_good"
    should_test_split = (not rank_ok) or shape_divergent
    if not should_test_split:
        return "KEEP_POOLED", "no_split_trigger"
    if not important:
        return "KEEP_POOLED", "not_important_enough"
    if delta_gini is None or not (delta_gini == delta_gini):
        return "KEEP_POOLED", "need_new_information_not_new_coefficients"
    brier_ok = (
        brier_refit is not None
        and brier_pooled is not None
        and brier_refit == brier_refit
        and brier_pooled == brier_pooled
        and brier_refit < brier_pooled
    )
    logloss_ok = (
        logloss_refit is not None
        and logloss_pooled is not None
        and logloss_refit == logloss_refit
        and logloss_pooled == logloss_pooled
        and logloss_refit < logloss_pooled
    )
    proper_ok = brier_ok or logloss_ok
    ci_ok = delta_gini_ci_low is not None and delta_gini_ci_low == delta_gini_ci_low and delta_gini_ci_low > 0
    material = delta_gini >= gates.min_delta_gini
    if material and ci_ok and proper_ok and shape_divergent:
        return "SPLIT", "holdout_delta_gini_and_shape"
    if not rank_ok:
        return "KEEP_POOLED", "need_new_information_not_new_coefficients"
    return "KEEP_POOLED", "split_gates_failed"
