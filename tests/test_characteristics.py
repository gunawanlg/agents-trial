import json
import os

import numpy as np
import pandas as pd

from scorecard_segment_eval.characteristics import feature_diagnostics
from scorecard_segment_eval.schema import Gates, ScorecardColumns


def _mini(n=400, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, size=n),
            "num": rng.normal(size=n),
            "cat": rng.choice(["A", "B", "C"], size=n),
            "woe_x": rng.choice(["b1", "b2", "b3", "b4"], size=n),
        }
    )
    return df


def test_numeric_psi_uses_portfolio_deciles():
    all_df = _mini(800, 1)
    seg = all_df.iloc[:200].copy()
    seg["num"] = seg["num"] + 1.5
    cols = ScorecardColumns(
        col_id="id",
        col_date="d",
        col_obs="o",
        col_target="y",
        col_score="s",
        cols_pred=["num"],
        cols_segment=["g"],
    )
    gates = Gates(n_psi_deciles=10)
    table = feature_diagnostics(seg, all_df, cols, gates, grouping={})
    row = table.set_index("feature").loc["num"]
    assert row["psi_method"] == "decile"
    assert row["psi"] > 0.05


def test_woe_feature_uses_existing_bins():
    all_df = _mini(500, 2)
    seg = all_df.iloc[:180]
    cols = ScorecardColumns(
        col_id="id",
        col_date="d",
        col_obs="o",
        col_target="y",
        col_score="s",
        cols_pred=[],
        cols_pred_woe=["woe_x"],
        cols_segment=["g"],
    )
    table = feature_diagnostics(seg, all_df, cols, Gates(), grouping={})
    assert table.set_index("feature").loc["woe_x", "psi_method"] == "categorical"


def test_grouping_json_bins_drive_psi(tmp_path):
    all_df = _mini(500, 3)
    seg = all_df.iloc[:150].copy()
    path = str(tmp_path / "grouping.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "features": {
                    "num": {"type": "numerical", "bins": [-99, -0.5, 0.5, 99]},
                    "cat": {"type": "categorical", "groups": {"A": ["A"], "rest": ["B", "C"]}},
                }
            },
            handle,
        )
    cols = ScorecardColumns(
        col_id="id",
        col_date="d",
        col_obs="o",
        col_target="y",
        col_score="s",
        cols_pred=["num", "cat"],
        cols_segment=["g"],
        grouping_path=path,
    )
    table = feature_diagnostics(seg, all_df, cols, Gates(), grouping=None)
    by = table.set_index("feature")
    assert by.loc["num", "psi_method"] == "grouping"
    assert by.loc["cat", "psi_method"] == "grouping"
