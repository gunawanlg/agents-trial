from scorecard_segment_eval import Gates, evaluate_segments, make_ar_imbalanced_book, make_synthetic_book


def _gates():
    return Gates(
        min_n=400,
        min_events=25,
        n_bootstrap=80,
        bootstrap_seed=0,
        holdout_frac=0.3,
        n_jobs=1,
        woe_strategy="tree",
    )


def test_synthetic_segment_verdicts():
    df, cols = make_synthetic_book(n=9000, seed=3)
    result = evaluate_segments(df, cols, _gates())
    d = result.decisions.set_index("segment_value")

    assert d.loc["tiny", "q1_verdict"] == "INCONCLUSIVE"
    assert d.loc["tiny", "q2_action"] == "NONE"

    assert d.loc["miscal", "q1_verdict"] == "WEAK"
    assert "calibration" in str(d.loc["miscal", "failed_pillars"])
    assert d.loc["miscal", "q2_action"] == "RECALIBRATE"

    assert d.loc["core", "q1_verdict"] == "GOOD"
    assert d.loc["core", "q2_action"] == "KEEP_POOLED"

    assert d.loc["inverted", "q1_verdict"] == "WEAK"
    assert "rank_order" in str(d.loc["inverted", "failed_pillars"])
    assert d.loc["inverted", "q2_action"] == "SPLIT"
    assert d.loc["inverted", "important"]

    ref = result.refit_comparison.set_index("segment_value")
    assert ref.loc["inverted", "delta_gini"] >= 0.03
    assert result.characteristics["feature"].isin(["x1", "x2", "cat"]).any()


def test_ar_aligned_gini_when_gap_is_large():
    df, cols = make_ar_imbalanced_book(n=6000, seed=11)
    gates = Gates(
        min_n=200,
        min_events=10,
        n_bootstrap=40,
        bootstrap_seed=1,
        ar_gap_material=0.10,
        n_jobs=1,
    )
    result = evaluate_segments(df, cols, gates)
    d = result.decisions.set_index("segment_value")
    assert "gini_ar_aligned" in d.columns
    assert "approval_rate" in d.columns
    high_ar = float(d.loc["high_ar", "approval_rate"])
    low_ar = float(d.loc["low_ar", "approval_rate"])
    assert high_ar - low_ar >= 0.10
    aligned = d.loc["high_ar", "gini_ar_aligned"]
    assert aligned == aligned  # not NaN
    assert d.loc["high_ar", "ar_aligned"] is not None
