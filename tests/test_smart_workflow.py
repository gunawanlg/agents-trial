import json

import numpy as np
import pandas as pd
import pytest

from scorecard_segment_eval import (
    Gates,
    analyst_report,
    evaluate_segments,
    infer_scorecard_data,
    make_synthetic_book,
)
from scorecard_segment_eval.characteristics import feature_diagnostics
from scorecard_segment_eval.evaluate import _similar_approval_rate_analysis
from scorecard_segment_eval.refit import WoEEncoder, refit_same_predictors


def test_smart_data_infers_observation_and_warns_when_predictors_absent():
    frame = pd.DataFrame(
        {
            "SKP_CREDIT_CASE": [1, 2, 3],
            "pd": [0.1, 0.2, 0.3],
            "TargetDefault": [0, 1, np.nan],
            "score_date": pd.date_range("2024-01-01", periods=3),
        }
    )
    with pytest.warns(UserWarning):
        smart = infer_scorecard_data(frame, col_score="pd", cols_pred_used=["x_woe"])
    assert smart.columns.col_obs == "TargetDefaultObs"
    assert smart.frame["TargetDefaultObs"].tolist() == [1, 1, 0]
    assert "predictor_refit" not in smart.analyses
    assert any("refit is unavailable" in message for message in smart.warnings)


def test_local_lookup_enriches_optional_metadata_but_not_predictors():
    frame = pd.DataFrame({"SKP_CREDIT_CASE": [1, 2], "pd": [0.1, 0.2]})
    lookup = pd.DataFrame(
        {
            "SKP_CREDIT_CASE": [1, 2],
            "TargetDefault": [0, 1],
            "TargetDefaultObs": [1, 1],
            "segment": ["A", "B"],
        }
    )
    with pytest.warns(UserWarning):
        smart = infer_scorecard_data(
            frame,
            col_score="pd",
            cols_pred_used=["x_woe"],
            sql_lookup=lookup,
        )
    assert smart.columns.cols_pred == []
    assert smart.columns.cols_segment == ["segment"]
    assert smart.frame["TargetDefault"].tolist() == [0, 1]


def test_tree_woe_refit_reports_vintage_stability():
    rng = np.random.RandomState(7)
    n = 500
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        }
    )
    frame["target"] = (frame["x"] + rng.normal(size=n) > 0.8).astype(int)
    train, holdout = frame.iloc[:400], frame.iloc[400:]
    pred, details = refit_same_predictors(
        train,
        holdout,
        ["x", "cat"],
        "target",
        Gates(n_woe_bins=5),
        date_col="date",
        return_details=True,
    )
    assert len(pred) == len(holdout)
    assert set(details["predictor_stability"]) == {"x", "cat"}
    assert details["selection_objective"] == details["selection_objective"]
    assert len(details["grouping"]["x"]["edges"]) <= 6


def test_characteristic_psi_uses_grouping_and_portfolio_deciles():
    frame, cols = make_synthetic_book(n=1200, seed=11)
    cols = cols.__class__(
        col_id=cols.col_id,
        col_date=cols.col_date,
        col_obs=cols.col_obs,
        col_target=cols.col_target,
        col_score=cols.col_score,
        cols_pred=["x1"],
        cols_segment=cols.cols_segment,
        cols_pred_woe=["x1_woe"],
        cols_pred_used=cols.cols_pred_used,
        grouping={"x1_woe": {"edges": [-0.5, 0.5]}},
    )
    out = feature_diagnostics(frame.iloc[:300], frame, cols, Gates())
    source = out.set_index("feature")["psi_binning"]
    assert source["x1"] == "portfolio_deciles"
    assert source["x1_woe"] == "grouping"


def test_large_approval_gap_adds_lower_rate_simulation():
    n = 200
    portfolio = pd.DataFrame(
        {
            "id": np.arange(n),
            "date": pd.date_range("2024-01-01", periods=n),
            "obs": [1] * 100 + [0] * 100,
            "target": ([0, 1] * 50) + ([0] * 100),
            "pd": np.linspace(0.01, 0.9, n),
        }
    )
    from scorecard_segment_eval import ScorecardColumns

    cols = ScorecardColumns(
        col_id="id",
        col_date="date",
        col_obs="obs",
        col_target="target",
        col_score="pd",
    )
    part = portfolio.iloc[:60].copy()
    analysis = _similar_approval_rate_analysis(part, portfolio, cols, Gates(approval_rate_gap=0.1))
    assert analysis["approval_rate_gap"] >= 0.1
    assert analysis["similar_ar_analysis"].startswith("simulated_at_lower_ar=")
    assert "gini_similar_ar" in analysis


def test_report_contains_prioritized_recommendations():
    frame, cols = make_synthetic_book(n=3000, seed=5)
    result = evaluate_segments(frame, cols, Gates(min_n=100, min_events=5, n_bootstrap=20))
    html = analyst_report(result)
    assert "Recommended actions" in html
    assert "approval-rate-adjusted Gini" in html
    assert "<!doctype html>" in html
