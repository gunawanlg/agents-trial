import warnings

import numpy as np
import pandas as pd
import pytest

from scorecard_segment_eval.binning import BinningModel, BinSpec
from scorecard_segment_eval.metrics import gini
from scorecard_segment_eval.refit import (
    MAX_SUBMODEL_DEPTH,
    FittedModelArtifact,
    WoEEncoder,
    fit_submodel,
    load_fitted_artifact,
    recalibrate_pd,
    recalibrate_with_diagnostics,
    refit_same_predictors,
    refit_with_diagnostics,
    submodel_params,
    xgboost_available,
)
from scorecard_segment_eval.schema import Gates


def _split_book(n=6000, seed=17):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 540, size=n), unit="D"
    )
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["A", "B", "C"], size=n)
    logit = -2.0 + 1.3 * x1 - 0.8 * x2 + np.where(cat == "B", 0.5, 0.0)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    frame = pd.DataFrame(
        {"date": dates, "x1": x1, "x2": x2, "cat": cat, "y": y}
    ).sort_values("date").reset_index(drop=True)
    cut = int(len(frame) * 0.7)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def test_woe_encoder_keeps_legacy_surface_and_exposes_grouping():
    train, holdout = _split_book()
    encoder = WoEEncoder(n_bins=8)
    encoder.fit(train[["x1", "x2", "cat"]], train["y"])
    assert encoder.columns_ == ["x1", "x2", "cat"]
    assert set(encoder.woe_) == {"x1", "x2", "cat"}
    assert encoder.edges_["cat"] is None
    assert encoder.edges_["x1"] is not None
    matrix = encoder.transform(holdout[["x1", "x2", "cat"]])
    assert matrix.shape == (len(holdout), 3)
    grouping = encoder.to_grouping()
    assert isinstance(grouping, BinningModel)
    rebuilt = WoEEncoder.from_grouping(grouping)
    np.testing.assert_allclose(rebuilt.transform(holdout[["x1", "x2", "cat"]]), matrix)


def test_woe_encoder_transform_before_fit_raises():
    with pytest.raises(ValueError):
        WoEEncoder().transform(pd.DataFrame({"x": [1.0]}))


def test_refit_beats_pooled_when_signal_is_present():
    train, holdout = _split_book()
    p = refit_same_predictors(train, holdout, ["x1", "x2", "cat"], "y", Gates())
    assert len(p) == len(holdout)
    assert gini(holdout["y"].to_numpy(dtype=float), p) > 0.3


def test_refit_with_diagnostics_returns_stability_and_coefficients():
    train, holdout = _split_book()
    result = refit_with_diagnostics(
        train, holdout, ["x1", "x2", "cat"], "y", Gates(), date_col="date"
    )
    assert result.method == "logistic_woe"
    assert "__intercept__" in result.coefficients
    assert set(result.stability["feature"]) == {"x1", "x2", "cat"}
    assert "stability_pass" in result.stability_summary
    assert isinstance(result.stability_pass(), bool)
    assert np.isfinite(result.stability_score()) or np.isnan(result.stability_score())


def test_refit_reuses_a_supplied_grouping():
    train, holdout = _split_book()
    grouping = BinningModel.fit(train[["x1", "x2", "cat"]], train["y"], Gates())
    result = refit_with_diagnostics(
        train, holdout, ["x1", "x2", "cat"], "y", Gates(), grouping=grouping
    )
    # A clone is used when the supplied object is also the portfolio baseline
    # so comparison can refresh stats without mutating the original.
    assert set(result.grouping.columns) == set(grouping.columns)
    for col in grouping.columns:
        assert result.grouping.specs[col].kind == grouping.specs[col].kind
    assert result.grouping is not grouping
    assert "segment_grouping_cloned_from_supplied" in result.messages
    assert not result.grouping_comparison.empty


def test_refit_without_usable_predictors_returns_nans():
    train, holdout = _split_book()
    result = refit_with_diagnostics(train, holdout, ["nope"], "y", Gates())
    assert result.method == "none"
    assert np.isnan(result.p_holdout).all()
    assert result.messages == ["no_usable_predictors"]


def test_submodel_params_hard_cap_max_depth():
    params = submodel_params(Gates(), {"max_depth": 12, "n_estimators": 10})
    assert params["max_depth"] == MAX_SUBMODEL_DEPTH
    assert params["n_estimators"] == 10
    lowered = submodel_params(Gates(submodel_max_depth=2))
    assert lowered["max_depth"] == 2
    raised = submodel_params(Gates(submodel_max_depth=99))
    assert raised["max_depth"] == MAX_SUBMODEL_DEPTH


def test_submodel_params_warns_when_depth_is_clamped():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        submodel_params(Gates(), {"max_depth": 7})
    assert any("interaction cap" in str(w.message) for w in caught)


def test_submodel_falls_back_with_a_warning_when_xgboost_is_absent(monkeypatch):
    """Exercised regardless of whether xgboost happens to be installed."""
    monkeypatch.setattr("scorecard_segment_eval.refit.xgboost_available", lambda: False)
    train, holdout = _split_book(n=3000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = refit_with_diagnostics(
            train, holdout, ["x1", "x2", "cat"], "y", Gates(), submodel=True, date_col="date"
        )
    assert result.method == "logistic_woe"
    assert "xgboost_unavailable_fallback_logistic" in result.messages
    assert any("xgboost" in str(w.message) for w in caught)
    assert np.isfinite(result.p_holdout).all()


def test_fit_submodel_returns_none_without_xgboost(monkeypatch):
    monkeypatch.setattr("scorecard_segment_eval.refit.xgboost_available", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_submodel(np.zeros((10, 2)), np.array([0, 1] * 5), Gates())
    assert model is None
    assert any("xgboost" in str(w.message) for w in caught)


@pytest.mark.skipif(not xgboost_available(), reason="optional xgboost extra not installed")
def test_submodel_uses_xgboost_and_respects_the_depth_cap():
    train, holdout = _split_book(n=4000)
    result = refit_with_diagnostics(
        train,
        holdout,
        ["x1", "x2", "cat"],
        "y",
        Gates(),
        submodel=True,
        xgb_params={"n_estimators": 60, "max_depth": 9},
        date_col="date",
    )
    assert result.method == "xgboost_submodel"
    assert np.isfinite(result.p_holdout).all()
    assert gini(holdout["y"].to_numpy(dtype=float), result.p_holdout) > 0.3
    assert set(result.coefficients) == {"x1", "x2", "cat"}
    booster = fit_submodel(
        np.column_stack([train["x1"], train["x2"]]),
        train["y"].to_numpy(dtype=int),
        Gates(),
        {"n_estimators": 10, "max_depth": 9},
    )
    assert int(booster.get_params()["max_depth"]) == MAX_SUBMODEL_DEPTH


def test_recalibration_fixes_the_level_without_touching_the_ranking():
    rng = np.random.default_rng(3)
    n = 4000
    p_true = rng.uniform(0.01, 0.4, size=n)
    y = rng.binomial(1, p_true)
    p_biased = np.clip(p_true * 2.5, 1e-4, 0.99)
    recalibrated = recalibrate_pd(p_biased, y, p_biased)
    assert abs(recalibrated.mean() - y.mean()) < abs(p_biased.mean() - y.mean())
    assert gini(y, recalibrated) == pytest.approx(gini(y, p_biased), abs=1e-9)


def test_recalibration_with_one_class_returns_input():
    p = np.array([0.1, 0.2, 0.3])
    out = recalibrate_pd(p, np.zeros(3, dtype=int), p)
    np.testing.assert_allclose(out, p)


def test_refit_is_identical_serially_and_in_parallel():
    train, holdout = _split_book()
    serial = refit_with_diagnostics(
        train, holdout, ["x1", "x2", "cat"], "y", Gates(), date_col="date", n_jobs=1
    )
    parallel = refit_with_diagnostics(
        train, holdout, ["x1", "x2", "cat"], "y", Gates(), date_col="date", n_jobs=-1
    )
    np.testing.assert_array_equal(serial.p_holdout, parallel.p_holdout)
    assert serial.stability.equals(parallel.stability)
    assert serial.stability_summary == parallel.stability_summary


def test_refit_keeps_the_model_and_grouping_and_round_trips(tmp_path):
    train, holdout = _split_book()
    result = refit_with_diagnostics(
        train, holdout, ["x1", "x2", "cat"], "y", Gates(), date_col="date"
    )
    assert result.model is not None
    assert result.grouping is not None
    assert result.artifact is not None
    assert result.artifact.kind == "refit"
    scored = result.artifact.predict_proba(holdout[["x1", "x2", "cat"]])
    np.testing.assert_allclose(scored, result.p_holdout)
    directory = str(tmp_path / "refit_art")
    result.save(directory)
    reloaded = load_fitted_artifact(directory)
    np.testing.assert_allclose(
        reloaded.predict_proba(holdout[["x1", "x2", "cat"]]), result.p_holdout
    )
    assert reloaded.grouping is not None
    assert reloaded.method == "logistic_woe"


def test_refit_notes_differences_against_the_portfolio_grouping():
    train, holdout = _split_book()
    portfolio = BinningModel.fit(train[["x1", "x2", "cat"]], train["y"], Gates())
    # A deliberately coarser segment grouping so the comparison has something to say.
    coarse_gates = Gates(binning_max_bins=2, binning_min_bin_frac=0.3)
    result = refit_with_diagnostics(
        train,
        holdout,
        ["x1", "x2", "cat"],
        "y",
        coarse_gates,
        date_col="date",
        portfolio_grouping=portfolio,
        pred_woe_map={"x1": "x1_woe"},
    )
    assert result.grouping is not None
    assert result.grouping is not portfolio
    assert not result.grouping_comparison.empty
    assert set(result.grouping_comparison["feature"]) <= {"x1", "x2", "cat"}
    notes = " ".join(result.grouping.specs["x1"].notes)
    # Either the bins differ (notes get portfolio_diff) or they happen to align.
    assert ("portfolio_diff:" in notes) or (result.grouping_comparison["kind"] == "aligned").any()


def test_recalibration_saves_the_logistic_model(tmp_path):
    rng = np.random.default_rng(3)
    n = 4000
    p_true = rng.uniform(0.01, 0.4, size=n)
    y = rng.binomial(1, p_true)
    p_biased = np.clip(p_true * 2.5, 1e-4, 0.99)
    result = recalibrate_with_diagnostics(p_biased, y, p_biased, score_col="pd")
    assert result.model is not None
    np.testing.assert_allclose(result.p_holdout, result.artifact.predict_proba(p_biased))
    directory = str(tmp_path / "recal_art")
    result.save(directory)
    reloaded = FittedModelArtifact.load(directory)
    assert reloaded.kind == "recalibrate"
    np.testing.assert_allclose(reloaded.predict_proba(p_biased), result.p_holdout)
    assert abs(result.intercept) > 0 or abs(result.slope - 1.0) > 0


def test_refit_keeps_logit_form_and_woe_bins_the_rest():
    rng = np.random.default_rng(21)
    n = 4000
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 400, size=n), unit="D"
    )
    x1 = rng.normal(size=n)
    feat_pd = np.clip(1.0 / (1.0 + np.exp(-(-1.5 + 1.2 * x1))), 0.01, 0.99)
    y = rng.binomial(1, feat_pd)
    frame = pd.DataFrame(
        {"date": dates, "x1": x1, "feat_pd": feat_pd, "y": y}
    ).sort_values("date").reset_index(drop=True)
    cut = int(len(frame) * 0.7)
    train, holdout = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
    portfolio = BinningModel(
        specs={
            "feat_pd": BinSpec(
                feature="feat_pd",
                kind="logit",
                method="sql_logit",
                transform="logit",
                impute=-2.5,
                labels=["logit"],
            )
        }
    )
    result = refit_with_diagnostics(
        train,
        holdout,
        ["x1", "feat_pd"],
        "y",
        Gates(),
        date_col="date",
        logit_cols=["feat_pd"],
        portfolio_grouping=portfolio,
    )
    assert result.method == "logistic_mixed"
    assert result.grouping.specs["feat_pd"].kind == "logit"
    assert result.grouping.specs["feat_pd"].transform == "logit"
    assert result.grouping.specs["feat_pd"].impute == pytest.approx(-2.5)
    assert "refit_keeps_logit_form" in result.grouping.specs["feat_pd"].notes
    assert result.grouping.specs["x1"].kind != "logit"
    assert any(m.startswith("refit_keeps_logit_form") for m in result.messages)
    assert np.isfinite(result.p_holdout).all()
    assert gini(holdout["y"].to_numpy(dtype=float), result.p_holdout) > 0.2

    only_logit = refit_with_diagnostics(
        train, holdout, ["feat_pd"], "y", Gates(), logit_cols=["feat_pd"]
    )
    assert only_logit.method == "logistic_logit"
    assert only_logit.grouping.specs["feat_pd"].kind == "logit"


def test_supplied_sql_grouping_still_produces_a_comparison():
    import os

    from scorecard_segment_eval.sql_model import parse_scorecard_sql_path

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_scorecard.sql")
    parsed = parse_scorecard_sql_path(fixture)
    rng = np.random.default_rng(8)
    n = 3000
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 400, size=n), unit="D")
    indosat = rng.uniform(0.0, 0.12, size=n)
    feat_e = np.clip(1.0 / (1.0 + np.exp(-rng.normal(size=n))), 0.02, 0.8)
    y = rng.binomial(1, 0.5 * feat_e + 0.05, size=n)
    frame = pd.DataFrame(
        {"date": dates, "indosat_v2": indosat, "featE": feat_e, "y": y}
    ).sort_values("date").reset_index(drop=True)
    cut = int(len(frame) * 0.7)
    train, holdout = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
    original_notes = list(parsed.grouping.specs["indosat_v2"].notes)
    result = refit_with_diagnostics(
        train,
        holdout,
        ["indosat_v2", "featE"],
        "y",
        Gates(),
        date_col="date",
        grouping=parsed.grouping,
        portfolio_grouping=parsed.grouping,
        logit_cols=["featE"],
    )
    assert result.grouping is not parsed.grouping
    assert parsed.grouping.specs["indosat_v2"].notes == original_notes
    assert not result.grouping_comparison.empty
    assert set(result.grouping_comparison["feature"]) >= {"indosat_v2", "featE"}
    assert result.grouping.specs["featE"].kind == "logit"
    assert np.isfinite(result.p_holdout).all()

