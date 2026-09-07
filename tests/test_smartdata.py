import json
import os
import warnings

import pandas as pd
import pytest

from scorecard_segment_eval import config
from scorecard_segment_eval.dbio import (
    FakeSqlExecutor,
    available_queries,
    parse_sql_metadata,
    read_sql_template,
    render_sql,
    split_table,
)
from scorecard_segment_eval.schema import ANALYSIS_KEYS
from scorecard_segment_eval.smartdata import (
    MetadataInferenceWarning,
    build_analysis_frame,
    confirm_settings,
    load_settings,
    render_metadata_summary,
    required_columns,
    resolve_capabilities,
    resolve_metadata,
    save_settings,
)
from scorecard_segment_eval.synthetic import make_synthetic_db_table

TABLE = "risk.scorecard_base"


def _executor(portfolio="PortfolioA", hidden_columns=(), n=1200):
    frame = make_synthetic_db_table(n=n, seed=5, portfolio=portfolio)
    return FakeSqlExecutor(table=frame, hidden_columns=hidden_columns), frame


def _resolve(executor, **kwargs):
    kwargs.setdefault("table", TABLE)
    kwargs.setdefault("col_id", "SKP_CREDIT_CASE")
    kwargs.setdefault("col_score", "PD")
    kwargs.setdefault("cols_pred_used", ["x1_woe", "x2", "cat"])
    kwargs.setdefault("executor", executor)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        meta = resolve_metadata(**kwargs)
    return meta, [str(w.message) for w in caught if w.category is MetadataInferenceWarning]


# --------------------------------------------------------------------------
# SQL lives in files
# --------------------------------------------------------------------------
def test_every_query_is_a_file_with_a_tag():
    names = available_queries()
    assert set(names) == {
        "column_profile",
        "describe_table",
        "fetch_columns_by_id",
        "fetch_portfolio",
    }
    for name in names:
        text = read_sql_template(name)
        assert text.startswith("-- query: " + name)
        assert "SELECT" in text.upper()


def test_render_sql_reports_missing_parameters():
    with pytest.raises(ValueError) as excinfo:
        render_sql("column_profile", table=TABLE)
    assert "column" in str(excinfo.value)


def test_render_sql_round_trips_its_parameters():
    sql = render_sql("fetch_columns_by_id", table=TABLE, col_id="ID", columns=["A", "B"])
    meta = parse_sql_metadata(sql)
    assert meta["query"] == "fetch_columns_by_id"
    assert meta["params"]["col_id"] == "ID"
    assert "A" in sql and "B" in sql


def test_unknown_template_raises():
    with pytest.raises(KeyError):
        read_sql_template("does_not_exist")


def test_split_table_handles_qualified_names():
    assert split_table("a.b") == {"table": "a.b", "table_schema": "a", "table_name": "b"}
    assert split_table("b")["table_schema"] == ""


def test_fake_executor_rejects_unknown_queries():
    executor, _frame = _executor()
    with pytest.raises(KeyError):
        executor("-- query: nonsense\nSELECT 1\n-- params: {}\n")


# --------------------------------------------------------------------------
# Mandatory inputs
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "missing", ["table", "col_id", "col_score", "cols_pred_used"]
)
def test_mandatory_inputs_are_enforced(missing):
    executor, _frame = _executor()
    kwargs = {
        "table": TABLE,
        "col_id": "SKP_CREDIT_CASE",
        "col_score": "PD",
        "cols_pred_used": ["x1_woe"],
        "executor": executor,
    }
    kwargs[missing] = None
    with pytest.raises(ValueError) as excinfo:
        resolve_metadata(**kwargs)
    assert missing in str(excinfo.value)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def test_date_and_segments_are_inferred_with_warnings():
    executor, _frame = _executor()
    meta, messages = _resolve(executor)
    assert meta.columns.col_date == "DATE_DECISION"
    assert meta.columns.cols_segment == ["CHANNEL"]
    assert meta.sources["col_date"] == "inferred:catalogue_candidate"
    assert meta.sources["cols_segment"] == "inferred:catalogue_candidate"
    assert any("col_date was not supplied" in m for m in messages)
    assert any("cols_segment was not supplied" in m for m in messages)


def test_supplied_values_are_never_overwritten_and_emit_no_inference_warning():
    executor, _frame = _executor()
    meta, messages = _resolve(
        executor,
        col_date="DATE_DECISION",
        cols_segment=["CHANNEL"],
        col_target="TargetA",
        col_obs="TargetAObs",
        cols_pred=["x1", "x2", "cat"],
        cols_pred_woe=["x1_woe"],
    )
    assert meta.sources["col_date"] == "supplied"
    assert meta.sources["cols_segment"] == "supplied"
    assert meta.sources["col_target"] == "supplied"
    assert meta.supplied_fields()
    assert meta.inferred_fields() == []
    assert messages == []


def test_portfolio_drives_target_detection():
    executor, _frame = _executor(portfolio="PortfolioA")
    meta, messages = _resolve(executor)
    assert meta.portfolio == "PortfolioA"
    assert meta.columns.col_target == "TargetA"
    assert meta.columns.col_obs == "TargetAObs"
    assert meta.sources["col_target"] == "inferred:portfolio=PortfolioA"
    assert any("PortfolioA" in m and "implies target" in m for m in messages)


def test_target_is_inferred_from_the_observation_flag():
    executor, _frame = _executor()
    meta, messages = _resolve(executor, col_obs="TargetAObs")
    assert meta.columns.col_target == "TargetA"
    assert meta.sources["col_target"] == "inferred:from_col_obs"
    assert any("observation flag" in m for m in messages)


def test_observation_flag_is_inferred_from_the_target():
    executor, _frame = _executor()
    meta, messages = _resolve(executor, col_target="TargetA")
    assert meta.columns.col_obs == "TargetAObs"
    assert meta.sources["col_obs"] == "inferred:from_col_target"


def test_unknown_portfolio_falls_back_to_the_configured_default():
    frame = make_synthetic_db_table(n=800, seed=5, portfolio="SomethingElse")
    frame = frame.rename(
        columns={"TargetA": config.DEFAULT_TARGET, "TargetAObs": config.DEFAULT_TARGET_OBS}
    )
    executor = FakeSqlExecutor(table=frame)
    meta, messages = _resolve(executor)
    assert meta.columns.col_target == config.DEFAULT_TARGET
    assert meta.columns.col_obs == config.DEFAULT_TARGET_OBS
    assert meta.sources["col_target"] == "default"
    assert any("defaulting to" in m for m in messages)


def test_portfolio_map_is_extensible():
    assert config.target_for_portfolio("PortfolioB") == ("TargetB", "TargetBObs")
    assert config.target_from_obs("TargetBObs") == "TargetB"
    assert config.obs_from_target("TargetC") == "TargetCObs"
    assert config.target_for_portfolio("nope") is None
    assert config.target_from_obs(None) is None


def test_resolved_target_absent_from_the_table_is_reported_as_missing():
    frame = make_synthetic_db_table(n=600, seed=5, portfolio="PortfolioA")
    frame = frame.drop(columns=["TargetA", "TargetAObs"])
    executor = FakeSqlExecutor(table=frame)
    meta, messages = _resolve(executor)
    assert meta.columns.col_target is None
    assert meta.sources["col_target"] == "missing"
    assert any("no such column exists" in m for m in messages)
    assert "segment_performance" in meta.capabilities.blocked


def test_no_date_column_available_is_reported_as_missing():
    frame = make_synthetic_db_table(n=600, seed=5).drop(columns=["DATE_DECISION"])
    executor = FakeSqlExecutor(table=frame)
    meta, messages = _resolve(executor)
    assert meta.columns.col_date is None
    assert meta.sources["col_date"] == "missing"
    assert any("no date-like column" in m for m in messages)


def test_high_cardinality_candidate_is_not_adopted_as_a_segment():
    frame = make_synthetic_db_table(n=900, seed=5)
    frame["PRODUCT"] = ["p%d" % (i,) for i in range(len(frame))]
    executor = FakeSqlExecutor(table=frame)
    meta, messages = _resolve(executor)
    assert "PRODUCT" not in meta.columns.cols_segment
    assert any("distinct values" in m for m in messages)


def test_missing_executor_warns_and_uses_only_supplied_settings():
    meta, messages = _resolve(None, col_date="DATE_DECISION", cols_segment=["CHANNEL"])
    assert any("no SQL executor" in m for m in messages)
    assert meta.catalogue == []


# --------------------------------------------------------------------------
# Non-inferable inputs and capability resolution
# --------------------------------------------------------------------------
def test_missing_cols_pred_blocks_the_refit():
    executor, _frame = _executor()
    meta, messages = _resolve(executor)
    assert meta.sources["cols_pred"] == "missing"
    assert any("refit" in m and "cannot be performed" in m for m in messages)
    assert "refit" in meta.capabilities.blocked
    assert "cols_pred" in meta.capabilities.blocked["refit"]
    assert "submodel" in meta.capabilities.blocked


def test_missing_cols_pred_woe_blocks_recalibration():
    executor, _frame = _executor()
    meta, messages = _resolve(executor, cols_pred=["x1", "x2", "cat"])
    assert meta.sources["cols_pred_woe"] == "missing"
    assert any("recalibration" in m for m in messages)
    assert "recalibration" in meta.capabilities.blocked
    assert "refit" in meta.capabilities.available


def test_grouping_without_cols_pred_blocks_grouping_psi(tmp_path):
    executor, _frame = _executor()
    grouping = str(tmp_path / "grouping.json")
    with open(grouping, "w") as handle:
        json.dump({"version": 1, "features": {}}, handle)
    meta, messages = _resolve(executor, grouping_path=grouping)
    assert "psi_grouping" in meta.capabilities.blocked
    assert "cols_pred is missing" in meta.capabilities.blocked["psi_grouping"]
    assert any("grouping cannot be applied" in m for m in messages)


def test_grouping_with_cols_pred_enables_grouping_psi(tmp_path):
    executor, _frame = _executor()
    grouping = str(tmp_path / "grouping.json")
    with open(grouping, "w") as handle:
        json.dump({"version": 1, "features": {}}, handle)
    meta, _messages = _resolve(
        executor, grouping_path=grouping, cols_pred=["x1", "x2", "cat"], cols_pred_woe=["x1_woe"]
    )
    assert "psi_grouping" in meta.capabilities.available


def test_absent_grouping_file_is_reported():
    executor, _frame = _executor()
    meta, messages = _resolve(executor, grouping_path="/nonexistent/grouping.json")
    assert meta.sources["grouping_path"] == "missing"
    assert any("does not exist" in m for m in messages)


def test_capability_report_covers_every_known_analysis():
    executor, _frame = _executor()
    meta, _messages = _resolve(executor, cols_pred=["x1"], cols_pred_woe=["x1_woe"])
    report = resolve_capabilities(meta)
    covered = set(report.available) | set(report.blocked)
    assert covered == set(ANALYSIS_KEYS)
    frame = report.to_frame()
    assert set(frame["status"]) <= {"available", "blocked"}
    assert (frame.loc[frame["status"].eq("blocked"), "reason"] != "").all()
    text = report.render_text()
    assert "Available analyses:" in text and "Blocked analyses:" in text


# --------------------------------------------------------------------------
# Presentation, confirmation and persistence
# --------------------------------------------------------------------------
def test_summary_shows_provenance_and_warnings():
    executor, _frame = _executor()
    meta, _messages = _resolve(executor)
    summary = render_metadata_summary(meta)
    assert "risk.scorecard_base" in summary
    assert "[inferred:catalogue_candidate]" in summary
    assert "Warnings (" in summary
    assert "Blocked analyses:" in summary


def test_auto_confirm_bypasses_the_prompt_and_persists(tmp_path):
    executor, _frame = _executor()
    meta, _messages = _resolve(executor)
    path = str(tmp_path / "settings.json")

    def _explode(_message):
        raise AssertionError("prompt must not be used when auto-confirming")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = confirm_settings(
            meta, auto_confirm=True, prompt=_explode, settings_path=path, display_fn=lambda _t: None
        )
    assert result.confirmed and result.auto
    assert result.settings_path == path
    assert os.path.isfile(path)
    assert any("auto-confirming" in str(w.message) for w in caught)


def test_interactive_confirmation_asks_for_a_filename(tmp_path):
    executor, _frame = _executor()
    meta, _messages = _resolve(executor)
    target = str(tmp_path / "chosen.json")
    asked = []

    def _prompt(message):
        asked.append(message)
        return "y" if "Proceed" in message else target

    result = confirm_settings(
        meta, auto_confirm=False, prompt=_prompt, display_fn=lambda _t: None
    )
    assert result.confirmed and not result.auto
    assert result.settings_path == target
    assert os.path.isfile(target)
    assert any("Proceed" in m for m in asked)
    assert any("Save settings as" in m for m in asked)


def test_declining_stops_before_writing_anything(tmp_path):
    executor, _frame = _executor()
    meta, _messages = _resolve(executor)
    result = confirm_settings(
        meta, auto_confirm=False, prompt=lambda _m: "n", display_fn=lambda _t: None
    )
    assert not result.confirmed
    assert result.settings_path is None
    assert not os.listdir(str(tmp_path))


def test_settings_round_trip_is_lossless(tmp_path):
    executor, _frame = _executor()
    meta, _messages = _resolve(
        executor, cols_pred=["x1", "x2", "cat"], cols_pred_woe=["x1_woe"], col_fantomas="FLAG_FANTOMAS"
    )
    path = str(tmp_path / "settings.json")
    save_settings(meta, path)
    reloaded = load_settings(path)
    assert reloaded.to_dict() == meta.to_dict()
    assert reloaded.columns == meta.columns
    assert reloaded.capabilities.available == meta.capabilities.available
    assert reloaded.capabilities.blocked == meta.capabilities.blocked
    # Reloading again from the reloaded object must still be stable.
    second = str(tmp_path / "again.json")
    save_settings(reloaded, second)
    assert load_settings(second).to_dict() == meta.to_dict()


def test_newer_settings_version_warns_but_loads(tmp_path):
    path = str(tmp_path / "future.json")
    with open(path, "w") as handle:
        json.dump({"version": 99, "table": "t", "columns": {"col_id": "i"}}, handle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        meta = load_settings(path)
    assert meta.columns.col_id == "i"
    assert any("newer version" in str(w.message) for w in caught)


# --------------------------------------------------------------------------
# Frame assembly
# --------------------------------------------------------------------------
def test_build_analysis_frame_fetches_only_what_is_missing():
    executor, frame = _executor()
    meta, _messages = _resolve(executor, cols_pred=["x1", "x2", "cat"], cols_pred_woe=["x1_woe"])
    base = frame[["SKP_CREDIT_CASE", "PD"]].copy()
    executor.queries = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built = build_analysis_frame(meta, executor, base=base)
    assert set(required_columns(meta)).issubset(set(built.columns))
    assert len(built) == len(base)
    assert [q["query"] for q in executor.queries] == ["fetch_columns_by_id"]
    fetched = executor.queries[0]["params"]["columns"]
    assert "PD" not in fetched, "already-supplied columns must not be re-fetched"
    assert any("were not supplied by you" in str(w.message) for w in caught)


def test_build_analysis_frame_without_a_base_pulls_everything():
    executor, _frame = _executor()
    meta, _messages = _resolve(executor, cols_pred=["x1", "x2", "cat"])
    warnings.simplefilter("ignore")
    built = build_analysis_frame(meta, executor)
    assert set(required_columns(meta)).issubset(set(built.columns))


def test_build_analysis_frame_is_a_no_op_when_nothing_is_missing():
    executor, frame = _executor()
    meta, _messages = _resolve(executor, cols_pred=["x1", "x2", "cat"], cols_pred_woe=["x1_woe"])
    base = frame[required_columns(meta)].copy()
    executor.queries = []
    built = build_analysis_frame(meta, executor, base=base)
    assert executor.queries == []
    assert list(built.columns) == list(base.columns)


def test_build_analysis_frame_needs_an_executor_for_missing_columns():
    executor, frame = _executor()
    meta, _messages = _resolve(executor, cols_pred=["x1", "x2", "cat"])
    with pytest.raises(ValueError):
        build_analysis_frame(meta, None, base=frame[["SKP_CREDIT_CASE"]])


def test_resolved_metadata_feeds_evaluate_segments_end_to_end():
    from scorecard_segment_eval import Gates, evaluate_segments

    executor, _frame = _executor(n=6000)
    meta, _messages = _resolve(
        executor, cols_pred=["x1", "x2", "cat"], cols_pred_woe=["x1_woe"], col_fantomas="FLAG_FANTOMAS"
    )
    warnings.simplefilter("ignore")
    df = build_analysis_frame(meta, executor)
    result = evaluate_segments(
        df, meta.columns, Gates(min_n=300, min_events=20, n_bootstrap=30, bootstrap_seed=0)
    )
    assert not result.decisions.empty
    assert set(result.decisions["segment_col"]) == {"CHANNEL"}
