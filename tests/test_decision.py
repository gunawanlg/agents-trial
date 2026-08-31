from scorecard_segment_eval.decision import q1_verdict, q2_action, rank_order_pass
from scorecard_segment_eval.schema import Gates


def test_inconclusive_when_underpowered():
    gates = Gates(min_n=1000, min_events=50)
    verdict, reasons = q1_verdict(
        n=200,
        defaults=10,
        gini_seg=0.4,
        gini_ci_low=0.3,
        gini_overall=0.4,
        oe=1.0,
        ece_val=0.01,
        vintage_ratio=1.0,
        gates=gates,
    )
    assert verdict == "INCONCLUSIVE"
    assert reasons == ["insufficient_power"]


def test_good_when_all_pillars_pass():
    gates = Gates()
    verdict, reasons = q1_verdict(
        n=5000,
        defaults=200,
        gini_seg=0.42,
        gini_ci_low=0.38,
        gini_overall=0.45,
        oe=1.02,
        ece_val=0.01,
        vintage_ratio=0.95,
        gates=gates,
    )
    assert verdict == "GOOD"
    assert reasons == []


def test_weak_calibration_only():
    gates = Gates()
    verdict, failed = q1_verdict(
        n=5000,
        defaults=200,
        gini_seg=0.40,
        gini_ci_low=0.35,
        gini_overall=0.42,
        oe=0.5,
        ece_val=0.08,
        vintage_ratio=1.0,
        gates=gates,
    )
    assert verdict == "WEAK"
    assert failed == ["calibration"]


def test_rank_pass_if_ci_clears_floor_when_ratio_low():
    gates = Gates(gini_floor=0.25, gini_ratio_floor=0.80)
    assert rank_order_pass(gini_seg=0.30, gini_ci_low=0.27, gini_overall=0.50, gates=gates)


def test_q2_recalibrate_when_cal_only():
    action, reason = q2_action(
        q1="WEAK",
        failed_pillars=["calibration"],
        important=True,
        shape_divergent=False,
        delta_gini=None,
        delta_gini_ci_low=None,
        brier_refit=None,
        brier_pooled=None,
        logloss_refit=None,
        logloss_pooled=None,
        gates=Gates(),
    )
    assert action == "RECALIBRATE"
    assert "calibration" in reason


def test_q2_split_requires_material_ci_proper_and_shape():
    gates = Gates(min_delta_gini=0.03)
    action, _ = q2_action(
        q1="WEAK",
        failed_pillars=["rank_order"],
        important=True,
        shape_divergent=True,
        delta_gini=0.08,
        delta_gini_ci_low=0.04,
        brier_refit=0.10,
        brier_pooled=0.12,
        logloss_refit=0.40,
        logloss_pooled=0.45,
        gates=gates,
    )
    assert action == "SPLIT"


def test_q2_keep_when_lift_too_small():
    action, reason = q2_action(
        q1="WEAK",
        failed_pillars=["rank_order"],
        important=True,
        shape_divergent=True,
        delta_gini=0.01,
        delta_gini_ci_low=0.005,
        brier_refit=0.10,
        brier_pooled=0.12,
        logloss_refit=0.40,
        logloss_pooled=0.45,
        gates=Gates(min_delta_gini=0.03),
    )
    assert action == "KEEP_POOLED"
    assert reason == "need_new_information_not_new_coefficients"
