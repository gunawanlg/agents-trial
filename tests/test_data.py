import os
import sqlite3

import pandas as pd
import pytest

from scorecard_segment_eval.data import (
    confirm_and_save_plan,
    infer_target_obs,
    load_saved_plan,
    load_scorecard_table,
)
from scorecard_segment_eval.synthetic import make_synthetic_book


def test_infer_target_from_obs_name():
    target, obs, src, warnings = infer_target_obs(
        columns=["TargetA", "TargetAObs"],
        col_id="SKP_CREDIT_CASE",
        col_target=None,
        col_obs="TargetAObs",
        mapping={},
    )
    assert target == "TargetA"
    assert obs == "TargetAObs"
    assert src["col_target"] == "from_col_obs"


def test_infer_obs_from_target_name():
    target, obs, src, _w = infer_target_obs(
        columns=["TargetA", "TargetAObs"],
        col_id="x",
        col_target="TargetA",
        col_obs=None,
        mapping={},
    )
    assert target == "TargetA"
    assert obs == "TargetAObs"
    assert src["col_obs"] == "from_col_target"


def test_infer_portfolio_mapping():
    target, obs, src, _w = infer_target_obs(
        columns=["TargetA", "TargetAObs"],
        col_id="SKP_CREDIT_CASE",
        portfolio="PortfolioA",
        mapping=None,
    )
    assert target == "TargetA"
    assert obs == "TargetAObs"


def test_default_target_when_nothing_matches():
    target, obs, src, warnings = infer_target_obs(columns=["pd"], col_id="id", mapping={})
    assert target == "TargetDefault"
    assert obs == "TargetDefaultObs"
    assert any("defaulting" in w.lower() or "not provided" in w.lower() for w in warnings)


def test_load_table_mandatory_only_warns_and_disables_refit():
    df, cols = make_synthetic_book(n=200, seed=1)
    base = df[["SKP_CREDIT_CASE", "pd", "x1_woe", "x2", "cat"]].copy()
    loaded, spec, plan = load_scorecard_table(
        base,
        col_id="SKP_CREDIT_CASE",
        col_score="pd",
        cols_pred_used=["x1_woe", "x2", "cat"],
        warn=False,
    )
    assert plan.capabilities["refit"] is False
    assert any("cols_pred" in w for w in plan.warnings)
    assert spec.col_target == "TargetDefault"


def test_sql_enrich_and_confirm_cache(tmp_path):
    df, _cols = make_synthetic_book(n=120, seed=2)
    base = df[["SKP_CREDIT_CASE", "pd", "x1_woe"]].copy()
    enrich = pd.DataFrame(
        {
            "SKP_CREDIT_CASE": df["SKP_CREDIT_CASE"],
            "DTIME_SCORE": df["score_date"],
            "CODE_CHANNEL": df["channel"],
            "CODE_PRODUCT": "P1",
            "TARGET_DEFAULT": df["default"],
            "TARGET_DEFAULT_OBS": df["obs"],
        }
    )
    sql_path = str(tmp_path / "enrich_by_id.sql")
    with open(sql_path, "w") as handle:
        handle.write(
            "SELECT SKP_CREDIT_CASE AS col_id, DTIME_SCORE AS col_date, "
            "CODE_CHANNEL AS segment_channel, TARGET_DEFAULT AS col_target, "
            "TARGET_DEFAULT_OBS AS col_obs FROM APP_ENRICH "
            "WHERE SKP_CREDIT_CASE IN ({id_list})"
        )
    con = sqlite3.connect(str(tmp_path / "t.db"))
    enrich.to_sql("APP_ENRICH", con, index=False, if_exists="replace")
    cache = str(tmp_path / "plan.json")
    loaded, spec, plan = load_scorecard_table(
        base,
        col_id="SKP_CREDIT_CASE",
        col_score="pd",
        cols_pred_used=["x1_woe"],
        connection=con,
        sql_path=sql_path,
        cols_pred=["x1_woe"],
        warn=False,
        auto_confirm=True,
        cache_filename=cache,
    )
    assert spec.col_date == "col_date"
    assert spec.col_target == "col_target"
    assert spec.col_obs == "col_obs"
    assert "segment_channel" in spec.cols_segment
    assert os.path.isfile(cache)
    restored = load_saved_plan(cache)
    assert restored.spec.col_id == "SKP_CREDIT_CASE"
    assert plan.capabilities["evaluate_segments"] is True
    con.close()


def test_confirm_saves_requested_filename(tmp_path):
    df, _cols = make_synthetic_book(n=50, seed=8)
    _loaded, spec, plan = load_scorecard_table(
        df,
        col_id="SKP_CREDIT_CASE",
        col_score="pd",
        cols_pred_used=["x1_woe", "x2", "cat"],
        col_date="score_date",
        col_target="default",
        col_obs="obs",
        cols_segment=["channel"],
        cols_pred=["x1", "x2", "cat"],
        cols_pred_woe=["x1_woe"],
        warn=False,
    )
    path = str(tmp_path / "my_settings.json")
    saved = confirm_and_save_plan(plan, confirmed=True, filename=path)
    assert saved == path
    assert os.path.isfile(path)
