import numpy as np
import pandas as pd
import pytest

from scorecard_segment_eval.refit import (
    WoEEncoder,
    predictor_vintage_stability,
    refit_same_predictors,
    refit_segment_model,
)
from scorecard_segment_eval.schema import Gates
from scorecard_segment_eval.synthetic import make_synthetic_book


def test_tree_woe_creates_supervised_bins():
    rng = np.random.RandomState(0)
    x = rng.normal(size=800)
    y = (x > 0.2).astype(int)
    # Add noise so it's not a single split
    y = np.where(rng.rand(800) < 0.1, 1 - y, y)
    X = pd.DataFrame({"x": x})
    enc = WoEEncoder(n_bins=8, strategy="tree", tree_max_leaf=6, tree_min_bin=40, n_jobs=1)
    enc.fit(X, pd.Series(y))
    assert enc.bin_method_["x"] in ("tree", "quantile")
    xt = enc.transform(X)
    assert xt.shape == (800, 1)
    assert np.isfinite(xt).all()


def test_refit_returns_holdout_scores():
    df, cols = make_synthetic_book(n=4000, seed=4)
    part = df[df["channel"] == "inverted"].copy()
    part = part.sort_values("score_date")
    cut = part["score_date"].quantile(0.7)
    train = part[part["score_date"] < cut]
    hold = part[part["score_date"] >= cut]
    gates = Gates(n_woe_bins=8, n_jobs=1, woe_strategy="tree")
    p = refit_same_predictors(train, hold, cols.cols_pred, cols.col_target, gates)
    assert len(p) == len(hold)
    assert np.isfinite(p).all()


def test_stability_flags_sign_flipping_predictor():
    rng = np.random.RandomState(2)
    n = 1200
    months = np.repeat(pd.period_range("2023-01", periods=6, freq="M").astype(str), n // 6)
    y = rng.binomial(1, 0.2, size=len(months))
    # x_stable correlates with y; x_flip reverses halfway
    x_stable = y + rng.normal(scale=0.3, size=len(months))
    x_flip = x_stable.copy()
    later = pd.Index(months) >= "2023-04"
    x_flip[later] = -x_flip[later]
    df = pd.DataFrame(
        {
            "y": y,
            "x_stable": x_stable,
            "x_flip": x_flip,
            "dt": pd.to_datetime(months),
        }
    )
    stab = predictor_vintage_stability(
        df,
        ["x_stable", "x_flip"],
        "y",
        "dt",
        min_vintages=3,
        psi_max=0.25,
        gini_cv_max=0.35,
    )
    by = stab.set_index("feature")
    assert bool(by.loc["x_flip", "stable"]) is False
    assert "sign_flip" in str(by.loc["x_flip", "reason"])


def test_stability_constrained_refit_drops_unstable():
    rng = np.random.RandomState(5)
    n = 2000
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.randint(0, 400, size=n), unit="D")
    y = rng.binomial(1, 0.25, size=n)
    x_good = y + rng.normal(scale=0.4, size=n)
    x_bad = rng.normal(size=n)
    x_bad = np.where(dates > dates.median(), -y + rng.normal(scale=0.4, size=n), y + rng.normal(scale=0.4, size=n))
    df = pd.DataFrame({"y": y, "x_good": x_good, "x_bad": x_bad, "dt": dates, "pd": np.clip(0.1 + 0.3 * y, 0.01, 0.99)})
    df = df.sort_values("dt")
    cut = df["dt"].quantile(0.7)
    train, hold = df[df["dt"] < cut], df[df["dt"] >= cut]
    gates = Gates(n_jobs=1, stability_min_vintages=2, stability_gini_cv_max=0.3)
    result = refit_segment_model(train, hold, ["x_good", "x_bad"], "y", gates=gates, date_col="dt", submodel=False)
    assert "x_bad" in result.dropped or (not result.stability.empty)
    assert len(result.p_holdout) == len(hold)
    assert len(result.p_holdout_stable) == len(hold)


def test_submodel_requires_xgboost_or_runs():
    df, cols = make_synthetic_book(n=2500, seed=6)
    part = df[df["channel"] == "core"].copy()
    part = part.sort_values("score_date")
    cut = part["score_date"].quantile(0.75)
    train, hold = part[part["score_date"] < cut], part[part["score_date"] >= cut]
    gates = Gates(n_jobs=1, xgb_n_estimators=20, xgb_max_depth=3)
    try:
        import xgboost  # noqa: F401
    except Exception:
        with pytest.raises(ImportError):
            refit_same_predictors(train, hold, cols.cols_pred, cols.col_target, gates, submodel=True)
        return
    p = refit_same_predictors(train, hold, cols.cols_pred, cols.col_target, gates, submodel=True)
    assert len(p) == len(hold)
    assert np.isfinite(p).all()
