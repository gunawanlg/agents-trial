import json

import numpy as np
import pandas as pd
import pytest

from scorecard_segment_eval.binning import (
    MISSING_LABEL,
    OTHER_LABEL,
    BinningModel,
    edge_labels,
    fit_bin_spec,
    information_value,
    load_grouping,
    optbinning_available,
    save_grouping,
    woe_table,
)
from scorecard_segment_eval.schema import Gates


def _book(n=4000, seed=11, na_frac=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    logit = -2.0 + 1.4 * x
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    cat = rng.choice(["A", "B", "C", "rare"], size=n, p=[0.5, 0.3, 0.19, 0.01])
    frame = pd.DataFrame({"x": x.astype(float), "cat": cat, "y": y})
    if na_frac:
        idx = frame.index[: int(n * na_frac)]
        frame.loc[idx, "x"] = np.nan
    return frame


def test_numeric_binning_is_supervised_not_equal_frequency():
    frame = _book()
    spec = fit_bin_spec(frame["x"], frame["y"], Gates(), feature="x")
    assert spec.kind == "numeric"
    # Without optbinning installed the constrained-tree fallback must be used.
    expected = "optbinning" if optbinning_available() else "tree"
    assert spec.method == expected
    assert len(spec.labels) >= 2
    assert spec.edges[0] == -np.inf and spec.edges[-1] == np.inf
    assert spec.iv > 0.1


def test_bins_respect_minimum_size_and_event_floors():
    frame = _book(n=3000, seed=5)
    gates = Gates(binning_min_bin_frac=0.15, binning_min_bin_events=40)
    spec = fit_bin_spec(frame["x"], frame["y"], gates, feature="x")
    labels = spec.assign(frame["x"])
    counts = pd.Series(labels).value_counts()
    events = pd.DataFrame({"b": labels, "y": frame["y"].to_numpy()}).groupby("b")["y"].sum()
    for label in spec.labels:
        if counts.get(label, 0) == 0:
            continue
        assert counts[label] >= 0.15 * len(frame) * 0.99
        assert events[label] >= 40


def test_missing_values_get_their_own_bin():
    frame = _book(n=3000, seed=7, na_frac=0.12)
    spec = fit_bin_spec(frame["x"], frame["y"], Gates(), feature="x")
    labels = spec.assign(frame["x"])
    assert (pd.Series(labels) == MISSING_LABEL).sum() == frame["x"].isna().sum()
    assert MISSING_LABEL in spec.woe
    assert "missing_values_isolated" in spec.notes


def test_sparse_categories_are_merged():
    frame = _book(n=4000, seed=9)
    spec = fit_bin_spec(frame["cat"], frame["y"], Gates(binning_min_bin_events=25), feature="cat")
    assert spec.kind == "categorical"
    members = sorted(m for group in spec.groups.values() for m in group)
    assert members == ["A", "B", "C", "rare"]
    rare_group = [label for label, group in spec.groups.items() if "rare" in group][0]
    assert len(spec.groups[rare_group]) > 1, "the 1%-frequency level should be merged"


def test_unseen_category_maps_to_other_with_zero_woe():
    frame = _book(n=2000, seed=3)
    spec = fit_bin_spec(frame["cat"], frame["y"], Gates(), feature="cat")
    labels = spec.assign(pd.Series(["A", "totally_new"]))
    assert labels[1] == OTHER_LABEL
    assert spec.transform_woe(pd.Series(["totally_new"]))[0] == 0.0


def test_monotonicity_is_enforced_when_it_keeps_the_signal():
    frame = _book(n=6000, seed=13)
    spec = fit_bin_spec(frame["x"], frame["y"], Gates(binning_monotonic=True), feature="x")
    table = woe_table(spec.assign(frame["x"]), frame["y"].to_numpy(dtype=float))
    woes = [table["woe"][label] for label in spec.labels if label in table.index]
    diffs = np.diff(woes)
    assert spec.monotonic
    assert np.all(diffs >= 0) or np.all(diffs <= 0)


def test_monotonicity_is_skipped_when_it_would_destroy_iv():
    # V-shaped relation: forcing monotonicity would merge away the signal.
    rng = np.random.default_rng(4)
    n = 6000
    x = rng.normal(size=n)
    logit = -2.2 + 1.8 * np.abs(x)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    frame = pd.DataFrame({"x": x, "y": y})
    spec = fit_bin_spec(frame["x"], frame["y"], Gates(binning_monotonic=True), feature="x")
    assert not spec.monotonic
    assert "monotonicity_not_feasible" in spec.notes
    assert spec.iv > 0.1


def test_grouping_json_round_trip_is_lossless(tmp_path):
    frame = _book(n=3000, seed=21, na_frac=0.05)
    model = BinningModel.fit(frame[["x", "cat"]], frame["y"], Gates())
    path = str(tmp_path / "grouping.json")
    save_grouping(model, path)
    with open(path) as handle:
        payload = json.load(handle)
    assert payload["version"] == 1
    assert sorted(payload["features"]) == ["cat", "x"]
    reloaded = load_grouping(path)
    assert reloaded.columns == model.columns
    assert reloaded.to_dict() == model.to_dict()
    np.testing.assert_array_equal(
        reloaded.transform(frame[["x", "cat"]]), model.transform(frame[["x", "cat"]])
    )


def test_infinite_edges_survive_json_round_trip(tmp_path):
    frame = _book(n=2000, seed=2)
    model = BinningModel.fit(frame[["x"]], frame["y"], Gates())
    path = str(tmp_path / "g.json")
    model.save(path)
    reloaded = load_grouping(path)
    assert reloaded.specs["x"].edges[0] == float("-inf")
    assert reloaded.specs["x"].edges[-1] == float("inf")


def test_edge_labels_are_unique():
    labels = edge_labels([-np.inf, 1.0000000001, 1.0000000002, np.inf])
    assert len(set(labels)) == len(labels)


def test_left_closed_bins_match_sql_less_than_thresholds():
    from scorecard_segment_eval.binning import assign_numeric_bins

    edges = [-np.inf, 0.5, 1.5, np.inf]
    labels = edge_labels(edges, closed="left")
    assert labels[0].startswith("[")
    assigned = assign_numeric_bins([0.5, 1.5, 0.0], edges, labels, closed="left")
    # x < 0.5 is the first bin; x == 0.5 belongs to [0.5, 1.5).
    assert assigned[0] == labels[1]
    assert assigned[1] == labels[2]
    assert assigned[2] == labels[0]


def test_information_value_matches_manual_computation():
    y = np.array([1, 1, 0, 0, 0, 0], dtype=float)
    labels = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    table = woe_table(labels, y)
    assert information_value(labels, y) == pytest.approx(table["iv_part"].sum())


def test_degenerate_numeric_falls_back_to_categorical():
    frame = pd.DataFrame({"x": [1.0] * 50 + [2.0] * 50, "y": [0, 1] * 50})
    spec = fit_bin_spec(frame["x"], frame["y"], Gates(), feature="x")
    assert spec.kind == "categorical"


def test_binning_model_transform_is_order_stable_and_handles_absent_columns():
    frame = _book(n=1500, seed=6)
    model = BinningModel.fit(frame[["x", "cat"]], frame["y"], Gates())
    full = model.transform(frame[["x", "cat"]])
    partial = model.transform(frame[["x"]])
    assert full.shape == (len(frame), 2)
    np.testing.assert_allclose(partial[:, 0], full[:, 0])
    np.testing.assert_allclose(partial[:, 1], np.zeros(len(frame)))
