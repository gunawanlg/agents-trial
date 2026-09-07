import numpy as np
import pandas as pd
import pytest

from scorecard_segment_eval.binning import MISSING_LABEL, BinningModel, fit_bin_spec
from scorecard_segment_eval.characteristics import (
    METHOD_CATEGORICAL,
    METHOD_GROUPING,
    METHOD_NUMERIC_DECILES,
    PSI_KEYS,
    feature_diagnostics,
    portfolio_decile_edges,
    psi_for_feature,
    psi_from_labels,
    psi_table,
    shape_divergence,
)
from scorecard_segment_eval.schema import Gates, ScorecardColumns


def _reference(n=6000, seed=31):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    cat = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1])
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-2.0 + 1.2 * x))))
    return pd.DataFrame({"x": x, "cat": cat, "y": y})


def test_numeric_path_uses_portfolio_deciles():
    ref = _reference()
    out = psi_for_feature(ref["x"].iloc[:1000], ref["x"], n_bins=10)
    assert out["psi_method"] == METHOD_NUMERIC_DECILES
    assert out["psi_n_bins"] >= 10
    assert set(PSI_KEYS).issubset(set(out))


def test_categorical_path_uses_levels():
    ref = _reference()
    out = psi_for_feature(ref["cat"].iloc[:1000], ref["cat"])
    assert out["psi_method"] == METHOD_CATEGORICAL
    assert out["psi_n_bins"] >= 3


def test_grouping_path_reuses_the_existing_bins():
    ref = _reference()
    spec = fit_bin_spec(ref["x"], ref["y"], Gates(), feature="x")
    out = psi_for_feature(ref["x"].iloc[:2000], ref["x"], spec=spec)
    assert out["psi_method"] == METHOD_GROUPING
    assert out["psi_n_bins"] == len(spec.bin_order())


def test_identical_distributions_give_near_zero_psi():
    ref = _reference()
    numeric = psi_for_feature(ref["x"], ref["x"])
    categorical = psi_for_feature(ref["cat"], ref["cat"])
    assert numeric["psi"] == pytest.approx(0.0, abs=1e-9)
    assert categorical["psi"] == pytest.approx(0.0, abs=1e-9)


def test_shifted_distribution_gives_material_psi():
    ref = _reference()
    shifted = ref["x"] + 2.0
    out = psi_for_feature(shifted, ref["x"])
    assert out["psi"] > 0.25
    assert out["psi_worst_bin"] is not None
    assert out["psi_worst_bin_contrib"] > 0


def test_out_of_range_values_are_measured_not_dropped():
    ref = _reference()
    local = pd.Series(np.concatenate([ref["x"].to_numpy()[:500], np.full(500, 99.0)]))
    out = psi_for_feature(local, ref["x"])
    assert out["psi_out_of_range_share_segment"] == pytest.approx(0.5, abs=0.01)
    assert np.isfinite(out["psi"])


def test_missing_values_form_their_own_bin_and_are_reported():
    ref = _reference()
    local = ref["x"].copy()
    local.iloc[:600] = np.nan
    out = psi_for_feature(local, ref["x"])
    assert out["psi_missing_share_segment"] == pytest.approx(0.1, abs=1e-6)
    assert out["psi_missing_share_reference"] == 0.0
    assert out["psi"] > 0


def test_unseen_categories_count_as_out_of_range():
    ref = _reference()
    local = pd.Series(["A"] * 500 + ["brand_new"] * 500)
    out = psi_for_feature(local, ref["cat"])
    assert out["psi_out_of_range_share_segment"] == pytest.approx(0.5, abs=1e-6)


def test_return_shape_is_identical_across_branches():
    ref = _reference()
    spec = fit_bin_spec(ref["x"], ref["y"], Gates(), feature="x")
    numeric = psi_for_feature(ref["x"].iloc[:800], ref["x"])
    categorical = psi_for_feature(ref["cat"].iloc[:800], ref["cat"])
    grouped = psi_for_feature(ref["x"].iloc[:800], ref["x"], spec=spec)
    assert set(numeric) == set(categorical) == set(grouped)


def test_empty_bins_are_smoothed_not_dropped():
    out = psi_from_labels(["a"] * 100, ["a"] * 50 + ["b"] * 50)
    assert out["psi_empty_bins"] == 1
    assert np.isfinite(out["psi"]) and out["psi"] > 0


def test_portfolio_decile_edges_are_infinite_at_the_ends():
    ref = _reference()
    edges = portfolio_decile_edges(ref["x"], n_bins=10)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert portfolio_decile_edges(pd.Series([1.0, 1.0, 1.0])) is None


def test_psi_table_reports_one_row_per_feature_with_the_method_used():
    ref = _reference()
    seg = ref.iloc[:1500]
    grouping = BinningModel.fit(ref[["x"]], ref["y"], Gates())
    table = psi_table(seg, ref, ["x", "cat"], grouping=grouping, gates=Gates())
    assert list(table["feature"]) == ["x", "cat"]
    methods = dict(zip(table["feature"], table["psi_method"]))
    assert methods["x"] == METHOD_GROUPING
    assert methods["cat"] == METHOD_CATEGORICAL
    assert list(table.columns) == ["feature"] + list(PSI_KEYS)


def test_feature_diagnostics_records_the_method_and_grouping_use():
    ref = _reference()
    seg = ref.loc[ref["cat"].eq("A")]
    cols = ScorecardColumns(
        col_id="id", col_target="y", col_score="x", cols_pred=["x", "cat"], cols_segment=[]
    )
    plain = feature_diagnostics(seg, ref, cols, Gates())
    assert set(plain["feature"]) == {"x", "cat"}
    assert not plain["grouping_used"].any()
    assert set(plain["psi_method"]) == {METHOD_NUMERIC_DECILES, METHOD_CATEGORICAL}

    grouping = BinningModel.fit(ref[["x", "cat"]], ref["y"], Gates())
    grouped = feature_diagnostics(seg, ref, cols, Gates(), grouping=grouping)
    assert grouped["grouping_used"].all()
    assert set(grouped["psi_method"]) == {METHOD_GROUPING}
    assert set(plain.columns) == set(grouped.columns)


def test_shape_divergence_reacts_to_psi_and_reversal():
    frame = pd.DataFrame({"psi": [0.01, 0.02], "rank_reversal": [False, False]})
    assert not shape_divergence(frame, Gates())
    assert shape_divergence(frame.assign(psi=[0.01, 0.4]), Gates())
    assert shape_divergence(frame.assign(rank_reversal=[False, True]), Gates())
    assert not shape_divergence(pd.DataFrame(), Gates())


def test_psi_is_identical_serially_and_in_parallel():
    ref = _reference()
    seg = ref.iloc[:2000]
    serial = psi_table(seg, ref, ["x", "cat"], gates=Gates(), n_jobs=1)
    parallel = psi_table(seg, ref, ["x", "cat"], gates=Gates(), n_jobs=-1)
    assert serial.equals(parallel)


def test_missing_label_is_part_of_the_bin_order():
    ref = _reference()
    spec = fit_bin_spec(ref["x"], ref["y"], Gates(), feature="x")
    assert MISSING_LABEL in spec.bin_order()
