import numpy as np
import pandas as pd

from scorecard_segment_eval.binning import BinningModel
from scorecard_segment_eval.schema import Gates
from scorecard_segment_eval.stability import (
    gini_standard_error,
    predictor_stability,
    stability_summary,
    vintage_labels,
)


def _monthly_book(n_months=24, n_per=400, flip_from=None, drift_from=None, seed=0):
    """Monthly vintages of one predictor, optionally flipping sign or drifting."""
    rng = np.random.default_rng(seed)
    frames = []
    for month in range(n_months):
        shift = 0.0
        if drift_from is not None and month >= drift_from:
            shift = 2.5
        x = rng.normal(loc=shift, size=n_per)
        sign = -1.0 if (flip_from is not None and month >= flip_from) else 1.0
        logit = -2.0 + sign * 1.6 * (x - shift)
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp("2022-01-01") + pd.DateOffset(months=month),
                    "x": x,
                    "y": y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_stable_predictor_passes():
    book = _monthly_book(seed=1)
    table = predictor_stability(book, ["x"], "y", "date", Gates())
    row = table.set_index("feature").loc["x"]
    assert bool(row["assessed"])
    assert row["n_vintages"] == 24
    assert row["psi_max"] < 0.25
    assert row["sign_consistency"] == 1.0
    assert row["stability_score"] > 0.8
    assert bool(row["stable"])
    summary = stability_summary(table, Gates())
    assert summary["stability_pass"]
    assert summary["stability_reason"] == "stable"
    assert summary["unstable_predictors"] == ""


def test_sign_flip_in_recent_vintages_is_flagged():
    book = _monthly_book(flip_from=18, seed=2)
    table = predictor_stability(book, ["x"], "y", "date", Gates())
    row = table.set_index("feature").loc["x"]
    assert bool(row["assessed"])
    assert row["sign_consistency"] < 0.8
    assert "sign_flip" in row["stability_flags"]
    assert not bool(row["stable"])
    summary = stability_summary(table, Gates())
    assert not summary["stability_pass"]
    assert "sign_flip" in summary["stability_reason"]
    assert summary["unstable_predictors"] == "x"


def test_distribution_drift_is_flagged():
    book = _monthly_book(drift_from=14, seed=3)
    table = predictor_stability(book, ["x"], "y", "date", Gates())
    row = table.set_index("feature").loc["x"]
    assert row["psi_max"] > 0.25
    assert "distribution_drift" in row["stability_flags"]
    assert not bool(row["stable"])


def test_signalless_predictor_is_reported_but_never_vetoes():
    rng = np.random.default_rng(5)
    book = _monthly_book(seed=4)
    book["noise"] = rng.normal(size=len(book))
    table = predictor_stability(book, ["x", "noise"], "y", "date", Gates())
    noise = table.set_index("feature").loc["noise"]
    assert not bool(noise["assessed"])
    assert noise["stability_flags"] == "no_reference_signal"
    assert bool(noise["stable"])
    summary = stability_summary(table, Gates())
    assert summary["stability_pass"]
    assert summary["stability_n_predictors"] == 2


def test_too_few_vintages_is_not_treated_as_instability():
    book = _monthly_book(n_months=2, n_per=600, seed=6)
    table = predictor_stability(book, ["x"], "y", "date", Gates(stability_min_vintages=3))
    row = table.set_index("feature").loc["x"]
    assert not bool(row["assessed"])
    assert row["stability_flags"] == "insufficient_vintages"
    summary = stability_summary(table, Gates())
    assert summary["stability_pass"]
    assert summary["stability_reason"] == "insufficient_vintages"


def test_missing_date_column_degrades_gracefully():
    book = _monthly_book(n_months=6, seed=7).drop(columns=["date"])
    table = predictor_stability(book, ["x"], "y", None, Gates())
    assert list(table["stability_flags"]) == ["no_date_column"]
    assert bool(table["stable"].iloc[0])


def test_noise_correction_prevents_small_vintage_false_positives():
    """Tiny vintages are noisy; without the SE correction they all look weak."""
    book = _monthly_book(n_months=24, n_per=40, seed=8)
    table = predictor_stability(book, ["x"], "y", "date", Gates(), min_rows_per_vintage=30)
    row = table.set_index("feature").loc["x"]
    assert row["weak_vintage_share"] < 0.34
    assert "power_loss_in_vintage" not in str(row["stability_flags"])


def test_gini_standard_error_shrinks_with_sample_size():
    rng = np.random.default_rng(9)
    small = rng.normal(size=100)
    big = rng.normal(size=10000)
    y_small = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-1.5 + small))))
    y_big = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-1.5 + big))))
    se_small = gini_standard_error(y_small, small)
    se_big = gini_standard_error(y_big, big)
    assert np.isfinite(se_small) and np.isfinite(se_big)
    assert se_big < se_small


def test_gini_standard_error_is_nan_without_both_classes():
    assert np.isnan(gini_standard_error(np.zeros(50), np.arange(50, dtype=float)))


def test_vintage_labels_handles_unparseable_dates():
    labels = vintage_labels(pd.Series(["2024-01-05", "not-a-date", None]))
    assert labels.iloc[1] is None
    assert str(labels.iloc[0]) == "2024-01"


def test_stability_is_identical_serially_and_in_parallel():
    book = _monthly_book(n_months=12, seed=10)
    book["x2"] = book["x"] * 0.7 + np.random.default_rng(0).normal(size=len(book))
    serial = predictor_stability(book, ["x", "x2"], "y", "date", Gates(), n_jobs=1)
    parallel = predictor_stability(book, ["x", "x2"], "y", "date", Gates(), n_jobs=-1)
    assert serial.equals(parallel)


def test_overlapping_event_rate_bounds_are_flagged():
    book = _monthly_book(n_months=12, n_per=80, seed=11)
    grouping = BinningModel.fit(book[["x"]], book["y"], Gates())
    assert grouping.overlapping_event_rate_pairs(book, book["y"], "date", "x")
    table = predictor_stability(
        book, ["x"], "y", "date", Gates(), grouping=grouping
    )
    row = table.set_index("feature").loc["x"]
    assert "overlapping_event_rate_bounds" in str(row["stability_flags"])
    assert not bool(row["stable"])
