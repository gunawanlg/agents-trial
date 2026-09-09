import numpy as np
import pandas as pd

from scorecard_segment_eval.binning import (
    BinningModel,
    BinSpec,
    compare_groupings,
    edge_labels,
    grouping_from_woe_columns,
)
from scorecard_segment_eval.schema import Gates


def _numeric_spec(feature, edges, woes):
    labels = edge_labels(edges)
    return BinSpec(
        feature=feature,
        kind="numeric",
        method="manual",
        edges=edges,
        labels=labels,
        woe=dict(zip(labels, woes)),
    )


def test_numeric_woe_sign_flip_is_flagged_per_bin():
    portfolio = BinningModel(
        specs={
            "x": _numeric_spec("x", [-np.inf, 0.0, np.inf], [0.4, -0.4]),
        }
    )
    segment = BinningModel(
        specs={
            "x": _numeric_spec("x", [-np.inf, 0.0, np.inf], [-0.5, 0.5]),
        }
    )
    table = compare_groupings(segment, portfolio, gates=Gates())
    assert not table.empty
    flipped = table.loc[table["kind"].str.contains("woe_sign_flip")]
    assert len(flipped) == 2
    assert flipped["significant"].all()
    notes = " ".join(segment.specs["x"].notes)
    assert "portfolio_diff:" in notes
    assert "sign flipped" in notes


def test_merged_bins_are_noted_when_the_segment_is_coarser():
    portfolio = BinningModel(
        specs={
            "x": _numeric_spec("x", [-np.inf, -1.0, 1.0, np.inf], [0.5, 0.0, -0.5]),
        }
    )
    segment = BinningModel(
        specs={
            "x": _numeric_spec("x", [-np.inf, np.inf], [0.1]),
        }
    )
    table = compare_groupings(segment, portfolio, gates=Gates())
    assert (table["kind"] == "merged").any() or table["kind"].str.contains("merged").any()


def test_grouping_from_woe_columns_recovers_discrete_bins():
    raw = pd.Series(np.concatenate([np.full(40, -2.0), np.full(40, 0.0), np.full(40, 2.0)]))
    woe = pd.Series(np.concatenate([np.full(40, 0.8), np.full(40, 0.0), np.full(40, -0.7)]))
    frame = pd.DataFrame({"predA": raw, "predA_woe": woe})
    model = grouping_from_woe_columns(frame, {"predA": "predA_woe"}, Gates())
    assert model is not None
    spec = model.specs["predA"]
    assert spec.kind == "numeric"
    assert len(spec.labels) == 3
    assert "reconstructed_from_woe_column" in spec.notes


def test_continuous_woe_copy_is_not_treated_as_a_grouping():
    raw = pd.Series(np.linspace(-2, 2, 200))
    frame = pd.DataFrame({"predA": raw, "predA_woe": raw})
    assert grouping_from_woe_columns(frame, {"predA": "predA_woe"}, Gates()) is None


def test_categorical_membership_change_is_noted():
    portfolio = BinningModel(
        specs={
            "cat": BinSpec(
                feature="cat",
                kind="categorical",
                method="manual",
                groups={"A": ["A"], "B": ["B"], "C": ["C"]},
                labels=["A", "B", "C"],
                woe={"A": 0.4, "B": 0.0, "C": -0.3},
            )
        }
    )
    segment = BinningModel(
        specs={
            "cat": BinSpec(
                feature="cat",
                kind="categorical",
                method="manual",
                groups={"A|B": ["A", "B"], "C": ["C"]},
                labels=["A|B", "C"],
                woe={"A|B": 0.2, "C": -0.3},
            )
        }
    )
    table = compare_groupings(segment, portfolio, gates=Gates())
    mixed = table.loc[table["kind"].str.contains("membership_change")]
    assert not mixed.empty
    assert mixed["significant"].all()
