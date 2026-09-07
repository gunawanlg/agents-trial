import numpy as np
import pandas as pd
import pytest

from scorecard_segment_eval import Gates, evaluate_segments, make_synthetic_book
from scorecard_segment_eval.decision import ar_artifact_suspected
from scorecard_segment_eval.metrics import (
    MATCHED_AR_KEYS,
    approval_rate,
    matched_ar_comparison,
    score_cutoff_for_ar,
)


def _population(n=4000, shift=0.0, seed=0):
    """PD-like scores; ``shift`` moves the whole score distribution upward."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    logit = -2.2 + 1.5 * x + shift
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p)
    return y.astype(float), p


def test_cutoff_and_approval_rate_are_consistent():
    _y, p = _population(seed=1)
    cutoff = score_cutoff_for_ar(p, 0.8)
    assert approval_rate(p, cutoff) == pytest.approx(0.8, abs=0.01)


def test_all_keys_are_present_even_when_not_triggered():
    y_ref, p_ref = _population(seed=2)
    out = matched_ar_comparison(y_ref, p_ref, y_ref, p_ref, Gates())
    assert set(MATCHED_AR_KEYS).issubset(set(out))
    assert out["ar_gap_triggered"] is False
    assert np.isnan(out["gini_at_matched_ar"])
    assert np.isnan(out["matched_ar"])


def test_large_ar_gap_triggers_the_matched_simulation():
    y_ref, p_ref = _population(n=8000, seed=3)
    y_seg, p_seg = _population(n=3000, shift=1.6, seed=4)
    out = matched_ar_comparison(y_seg, p_seg, y_ref, p_ref, Gates())
    assert out["ar_gap_triggered"] is True
    assert out["ar_segment"] < out["ar_reference"]
    assert out["matched_ar_anchor"] == "segment"
    assert out["matched_ar"] == pytest.approx(out["ar_segment"])
    assert np.isfinite(out["gini_at_matched_ar"])
    assert np.isfinite(out["gini_reference_at_matched_ar"])
    assert out["matched_ar_n"] == pytest.approx(
        round(out["matched_ar"] * len(y_seg)), abs=2
    )


def test_anchor_is_the_lower_ar_side():
    y_ref, p_ref = _population(n=8000, seed=5)
    y_seg, p_seg = _population(n=3000, shift=-1.8, seed=6)
    out = matched_ar_comparison(y_seg, p_seg, y_ref, p_ref, Gates())
    assert out["ar_gap_triggered"] is True
    assert out["ar_segment"] > out["ar_reference"]
    assert out["matched_ar_anchor"] == "reference"
    assert out["matched_ar"] == pytest.approx(out["ar_reference"])


def test_columns_are_null_when_the_anchored_sample_is_too_small():
    y_ref, p_ref = _population(n=8000, seed=7)
    y_seg, p_seg = _population(n=120, shift=1.6, seed=8)
    out = matched_ar_comparison(y_seg, p_seg, y_ref, p_ref, Gates(matched_ar_min_n=500))
    assert out["ar_gap_triggered"] is True
    assert np.isfinite(out["matched_ar_n"])
    assert np.isnan(out["gini_at_matched_ar"])


def test_ratio_is_suppressed_when_the_denominator_is_noise():
    y_ref, p_ref = _population(n=6000, seed=9)
    y_seg, p_seg = _population(n=3000, shift=1.6, seed=10)
    # Destroy the reference ranking so its anchored Gini sits near zero.
    rng = np.random.default_rng(11)
    p_ref_shuffled = p_ref[rng.permutation(len(p_ref))]
    out = matched_ar_comparison(y_seg, p_seg, y_ref, p_ref_shuffled, Gates())
    if abs(out["gini_reference_at_matched_ar"]) < 0.05:
        assert np.isnan(out["gini_at_matched_ar_ratio"])
        assert np.isfinite(out["gini_at_matched_ar_gap"])


def test_trigger_threshold_is_configurable():
    y_ref, p_ref = _population(n=6000, seed=12)
    y_seg, p_seg = _population(n=3000, shift=0.5, seed=13)
    loose = matched_ar_comparison(y_seg, p_seg, y_ref, p_ref, Gates(ar_gap_trigger=0.9))
    tight = matched_ar_comparison(y_seg, p_seg, y_ref, p_ref, Gates(ar_gap_trigger=0.001))
    assert loose["ar_gap_triggered"] is False
    assert tight["ar_gap_triggered"] is True


def test_empty_inputs_return_nulls():
    out = matched_ar_comparison([], [], [], [], Gates())
    assert np.isnan(out["ar_segment"])
    assert out["ar_gap_triggered"] is False


def test_evaluate_segments_exposes_matched_ar_columns():
    df, cols = make_synthetic_book(n=9000, seed=3)
    gates = Gates(min_n=400, min_events=25, n_bootstrap=40, bootstrap_seed=0, holdout_frac=0.3)
    result = evaluate_segments(df, cols, gates)
    for key in ("ar_segment", "ar_reference", "ar_gap", "ar_gap_triggered", "gini_at_matched_ar"):
        assert key in result.decisions.columns
    assert key in result.decisions.columns
    triggered = result.decisions.loc[result.decisions["ar_gap_triggered"].astype(bool)]
    assert len(triggered) >= 1, "the miscalibrated segment should trip the AR gap"
    row = triggered.iloc[0]
    assert np.isfinite(row["gini_at_matched_ar"])
    not_triggered = result.decisions.loc[~result.decisions["ar_gap_triggered"].astype(bool)]
    assert pd.isna(not_triggered["gini_at_matched_ar"]).all()
    summary = result.segment_summary
    assert "gini_at_matched_ar" in summary.columns


def test_ar_artifact_flag_requires_a_rank_order_failure_and_a_trigger():
    gates = Gates()
    assert ar_artifact_suspected(["rank_order"], True, 0.95, gates)
    assert not ar_artifact_suspected(["calibration"], True, 0.95, gates)
    assert not ar_artifact_suspected(["rank_order"], False, 0.95, gates)
    assert not ar_artifact_suspected(["rank_order"], True, 0.2, gates)
    # Ratio unavailable: fall back to the absolute Gini floor.
    assert ar_artifact_suspected(["rank_order"], True, float("nan"), gates, gini_at_matched_ar=0.4)
    assert not ar_artifact_suspected(
        ["rank_order"], True, float("nan"), gates, gini_at_matched_ar=0.05
    )
