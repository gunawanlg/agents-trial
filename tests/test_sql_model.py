# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from scorecard_segment_eval.binning import grouping_from_dict
from scorecard_segment_eval.evaluate import _coerce_grouping
from scorecard_segment_eval.smartdata import (
    MetadataInferenceWarning,
    build_analysis_frame,
    resolve_metadata,
)
from scorecard_segment_eval.sql_model import parse_scorecard_sql_path


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_scorecard.sql")


def _sample_model():
    return parse_scorecard_sql_path(FIXTURE)


def test_parse_extracts_column_lists():
    parsed = _sample_model()
    assert parsed.cols_pred == [
        "indosat_v2",
        "featureB",
        "featC",
        "feat_pred_D",
        "featE",
        "featF_v3_0",
        "featG_V2",
        "featH_v4_0",
        "feati_v3",
        "var_v2",
    ]
    assert parsed.cols_pred_woe == [
        "feature_a_WOE",
        "featureB_WOE",
        "featC_WOE",
        "feat_pred_D_WOE",
    ]
    assert parsed.cols_pred_used == [
        "feature_a_WOE",
        "featureB_WOE",
        "featC_WOE",
        "feat_pred_D_WOE",
        "featE_VAL",
        "featF_v3_0_VAL",
        "featG_V2_VAL",
        "featH_v4_0_VAL",
        "feati_v3_VAL",
        "var_v2_VAL",
    ]
    assert parsed.pred_map["indosat_v2"] == "feature_a_WOE"
    assert parsed.pred_map["featureB"] == "featureB_WOE"
    assert parsed.pred_map["featE"] == "featE_VAL"
    assert parsed.source_table == "_SOURCETABLENAME_"
    assert "LINEAR_SCORE = B^T X" in parsed.formula
    assert "Intercept * 7.864808103029582" in parsed.formula


def test_grouping_json_includes_null_impute_and_left_closed_bins():
    parsed = _sample_model()
    dumped = parsed.grouping.to_dict()
    feature_a = dumped["features"]["indosat_v2"]
    assert feature_a["closed"] == "left"
    assert "null_impute:" in feature_a["missing_note"]
    assert feature_a["impute"] == pytest.approx(-0.008113478679658392)
    assert feature_a["output_name"] == "feature_a_WOE"
    restored_a = grouping_from_dict(dumped).specs["indosat_v2"]
    assert restored_a.edges == [
        float("-inf"),
        0.030322997830808163,
        0.05057848058640957,
        float("inf"),
    ]

    feat_e = dumped["features"]["featE"]
    assert feat_e["kind"] == "logit"
    assert feat_e["impute"] == pytest.approx(-2.701124677318522)
    assert "null_impute:" in feat_e["missing_note"]

    feat_f = dumped["features"]["featF_v3_0"]
    assert feat_f["kind"] == "logit"
    assert feat_f["impute"] is None

    feat_d = dumped["features"]["feat_pred_D"]
    assert feat_d["kind"] == "categorical"
    assert feat_d["impute"] == pytest.approx(0.09577533753541978)
    assert feat_d["else_woe"] == pytest.approx(3.6602873860102516)

    feature_b = dumped["features"]["featureB"]
    assert feature_b["kind"] == "mixed"
    assert any(r["op"] == "eq" and r["value"] == "No Score" for r in feature_b["rules"])


def test_grouping_roundtrip_json(tmp_path):
    parsed = _sample_model()
    path = str(tmp_path / "grouping.json")
    parsed.save_grouping(path)
    with open(path) as handle:
        payload = json.load(handle)
    restored = grouping_from_dict(payload)
    assert restored.columns == parsed.grouping.columns
    assert restored.closed("indosat_v2") == "left"
    frame = pd.DataFrame(
        {
            "indosat_v2": [0.01, 0.030322997830808163, np.nan],
            "featureB": [600.0, "No Score", np.nan],
            "featC": [0.05, 0.2, np.nan],
            "feat_pred_D": ["0-1month", None, "unseen-bucket"],
            "featE": [0.5, np.nan, 0.8],
            "featF_v3_0": [0.5, 0.5, 0.5],
            "featG_V2": [0.5, 0.5, 0.5],
            "featH_v4_0": [0.5, 0.5, 0.5],
            "feati_v3": [0.5, 0.5, 0.5],
            "var_v2": [0.5, 0.5, 0.5],
        }
    )
    np.testing.assert_allclose(restored.transform(frame), parsed.grouping.transform(frame))


def _example_frame():
    return pd.DataFrame(
        {
            "indosat_v2": [0.01, 0.030322997830808163, 0.04, 0.05057848058640957, np.nan],
            "featureB": [600.0, "No Score", "No Match", 700.0, np.nan],
            "featC": [0.05, 0.1297735944390297, 0.2, np.nan, 0.0],
            "feat_pred_D": ["0-1month", "34-35month", "58-59month", None, "unseen-bucket"],
            "featE": [0.5, np.nan, 0.8, 0.2, 0.9],
            "featF_v3_0": [0.5, 0.5, 0.5, 0.5, 0.5],
            "featG_V2": [0.5, 0.5, 0.5, 0.5, 0.5],
            "featH_v4_0": [0.5, 0.5, 0.5, 0.5, 0.5],
            "feati_v3": [0.5, 0.5, 0.5, 0.5, 0.5],
            "var_v2": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )


def test_woe_bins_use_sql_threshold_semantics():
    parsed = _sample_model()
    mapped = parsed.grouping.transform_woe(_example_frame())

    assert mapped["indosat_v2"].iloc[0] == pytest.approx(0.8589883180461642)
    assert mapped["indosat_v2"].iloc[1] == pytest.approx(0.14506493043860003)
    assert mapped["indosat_v2"].iloc[3] == pytest.approx(-0.4838054685680173)
    assert mapped["indosat_v2"].iloc[4] == pytest.approx(-0.008113478679658392)

    assert mapped["featureB"].iloc[0] == pytest.approx(0.3509134596093779)
    assert mapped["featureB"].iloc[1] == pytest.approx(-0.19802888585164036)
    assert mapped["featureB"].iloc[2] == pytest.approx(-0.19802888585164036)
    assert mapped["featureB"].iloc[3] == pytest.approx(1.2771948599668872)
    assert mapped["featureB"].iloc[4] == pytest.approx(-0.19802888585164036)

    assert mapped["featC"].iloc[0] == pytest.approx(-0.31983068733407105)
    assert mapped["featC"].iloc[1] == pytest.approx(-0.9408524915552476)
    assert mapped["featC"].iloc[2] == pytest.approx(-0.9408524915552476)

    assert mapped["feat_pred_D"].iloc[0] == pytest.approx(-0.37442094147664373)
    assert mapped["feat_pred_D"].iloc[1] == pytest.approx(-0.13021532920078416)
    assert mapped["feat_pred_D"].iloc[2] == pytest.approx(3.6602873860102516)
    assert mapped["feat_pred_D"].iloc[3] == pytest.approx(0.09577533753541978)
    assert mapped["feat_pred_D"].iloc[4] == pytest.approx(3.6602873860102516)

    assert mapped["featE"].iloc[0] == pytest.approx(0.0)
    assert mapped["featE"].iloc[1] == pytest.approx(-2.701124677318522)


def test_sklearn_logistic_matches_sql_formula():
    parsed = _sample_model()
    model = parsed.model
    assert list(model.coef_[0]) == pytest.approx(
        [
            -0.6373764603049858,
            -0.5952350999512949,
            -0.8643582973975634,
            -0.7098298562716113,
            0.4582281228917229,
            0.576921144130071,
            0.5623909066531244,
            0.8358362941961892,
            0.6175631752093271,
            0.719355781611567,
        ]
    )
    assert float(model.intercept_[0]) == pytest.approx(7.864808103029582)

    frame = pd.DataFrame(
        {
            "indosat_v2": [0.01],
            "featureB": ["No Score"],
            "featC": [0.2],
            "feat_pred_D": ["0-1month"],
            "featE": [0.5],
            "featF_v3_0": [0.5],
            "featG_V2": [0.5],
            "featH_v4_0": [0.5],
            "feati_v3": [0.5],
            "var_v2": [0.5],
        }
    )
    mapped = parsed.grouping.transform_woe(frame)
    x = mapped[parsed.grouping.columns].to_numpy(dtype=float)
    linear = float(np.asarray(x.reshape(1, -1).dot(model.coef_[0]) + model.intercept_).reshape(-1)[0])
    expected_pd = 1.0 / (1.0 + math.exp(-linear))
    proba = parsed.predict_proba(frame)
    assert proba.shape == (1, 2)
    assert proba[0, 1] == pytest.approx(expected_pd)
    sklearn_pd = model.predict_proba(x)[0, 1]
    assert sklearn_pd == pytest.approx(expected_pd)

    artifact = parsed.to_artifact()
    assert artifact.kind == "pooled"
    assert artifact.method == "logistic_sql"
    assert artifact.predict_proba(frame)[0] == pytest.approx(expected_pd)

    scored = parsed.transform_frame(frame)
    assert scored["LINEAR_SCORE"].iloc[0] == pytest.approx(linear)
    assert scored["SCORE"].iloc[0] == pytest.approx(expected_pd)
    assert "feature_a_WOE" in scored.columns


def test_scorecard_columns_pred_map_wins_over_suffix():
    parsed = _sample_model()
    cols = parsed.to_columns(col_id="id")
    mapping = cols.pred_woe_map()
    assert mapping["indosat_v2"] == "feature_a_WOE"
    assert mapping["featureB"] == "featureB_WOE"
    assert "featE" not in mapping
    assert cols.pred_val_map()["featE"] == "featE_VAL"
    logit = cols.logit_pred_cols(grouping=parsed.grouping)
    assert "featE" in logit
    assert "featF_v3_0" in logit
    assert "indosat_v2" not in logit
    assert "featureB" not in logit


def test_evaluate_accepts_sql_model_as_grouping():
    parsed = _sample_model()
    assert _coerce_grouping(parsed) is parsed.grouping
    assert _coerce_grouping(parsed.grouping) is parsed.grouping
    assert _coerce_grouping(None) is None


def test_resolve_metadata_from_sql(tmp_path):
    grouping_path = str(tmp_path / "grouping.json")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        meta = resolve_metadata(
            table="risk.base",
            col_id="id",
            col_score="SCORE",
            col_target="y",
            model_sql_path=FIXTURE,
            grouping_path=grouping_path,
            cols_segment=["segment"],
        )
    assert meta.columns.cols_pred[0] == "indosat_v2"
    assert meta.columns.cols_pred_used[0] == "feature_a_WOE"
    assert meta.grouping is not None
    assert os.path.isfile(grouping_path)
    assert "psi_grouping" in meta.capabilities.available
    assert "LINEAR_SCORE" in (meta.formula or "")
    dumped = meta.to_dict()
    assert dumped["model_sql_path"] == FIXTURE
    restored = type(meta).from_dict(dumped)
    assert restored.columns.pred_map["indosat_v2"] == "feature_a_WOE"


def test_resolve_metadata_sql_text_without_path():
    with open(FIXTURE) as handle:
        sql = handle.read()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        meta = resolve_metadata(
            table="risk.base",
            col_id="id",
            col_score="SCORE",
            col_target="y",
            model_sql=sql,
        )
    assert meta.columns.cols_pred_used[-1] == "var_v2_VAL"
    assert meta.grouping.columns[-1] == "var_v2"


def test_build_analysis_frame_scores_from_sql():
    parsed = _sample_model()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        meta = resolve_metadata(
            table="risk.base",
            col_id="id",
            col_score="SCORE",
            col_target="y",
            col_obs="obs",
            model_sql_path=FIXTURE,
        )
    base = _example_frame()
    base["id"] = list(range(len(base)))
    base["y"] = [0, 1, 0, 1, 0]
    base["obs"] = [1, 1, 1, 1, 1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = build_analysis_frame(meta, executor=None, base=base)
    assert "SCORE" in frame.columns
    assert "feature_a_WOE" in frame.columns
    assert np.isfinite(frame["SCORE"].iloc[0])
    assert any(
        isinstance(w.message, MetadataInferenceWarning) or "scorecard SQL" in str(w.message)
        for w in caught
    )
    expected = parsed.predict_proba(base)[:, 1]
    np.testing.assert_allclose(frame["SCORE"].to_numpy(dtype=float), expected)
