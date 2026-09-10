import os

import numpy as np
import pandas as pd

from scorecard_segment_eval import Gates, evaluate_segments, make_synthetic_book
from scorecard_segment_eval.binning import BinningModel
from scorecard_segment_eval.metrics import MATCHED_AR_KEYS


def _gates(**kwargs):
    base = {
        "min_n": 400,
        "min_events": 25,
        "n_bootstrap": 80,
        "bootstrap_seed": 0,
        "holdout_frac": 0.3,
    }
    base.update(kwargs)
    return Gates(**base)


def test_synthetic_segment_verdicts():
    df, cols = make_synthetic_book(n=9000, seed=3)
    result = evaluate_segments(df, cols, _gates())
    d = result.decisions.set_index("segment_value")

    assert d.loc["tiny", "q1_verdict"] == "INCONCLUSIVE"
    assert d.loc["tiny", "q2_action"] == "NONE"

    assert d.loc["miscal", "q1_verdict"] == "WEAK"
    assert "calibration" in str(d.loc["miscal", "failed_pillars"])
    assert d.loc["miscal", "q2_action"] == "RECALIBRATE"

    assert d.loc["core", "q1_verdict"] == "GOOD"
    assert d.loc["core", "q2_action"] == "KEEP_POOLED"

    assert d.loc["inverted", "q1_verdict"] == "WEAK"
    assert "rank_order" in str(d.loc["inverted", "failed_pillars"])
    assert d.loc["inverted", "q2_action"] == "SPLIT"
    assert d.loc["inverted", "important"]

    ref = result.refit_comparison.set_index("segment_value")
    assert ref.loc["inverted", "delta_gini"] >= 0.03
    assert result.characteristics["feature"].isin(["x1", "x2", "cat"]).any()


def test_split_requires_stable_predictors():
    """Same book, unsatisfiable stability gate: SPLIT downgrades to MONITOR."""
    df, cols = make_synthetic_book(n=9000, seed=3)
    strict = _gates(stability_score_min=1.01, stability_min_reference_gini=0.0)
    result = evaluate_segments(df, cols, strict)
    row = result.decisions.set_index("segment_value").loc["inverted"]
    assert row["q2_action"] == "MONITOR"
    assert row["q2_reason"].startswith("refit_predictors_unstable")
    assert not bool(row["stability_pass"])

    relaxed = evaluate_segments(df, cols, _gates(refit_requires_stability=False))
    assert relaxed.decisions.set_index("segment_value").loc["inverted", "q2_action"] == "SPLIT"


def test_result_carries_stability_and_meta():
    df, cols = make_synthetic_book(n=9000, seed=3)
    gates = _gates()
    result = evaluate_segments(df, cols, gates, n_jobs=1)
    assert not result.stability.empty
    assert set(result.stability["segment_col"]) >= {"__overall__"}
    assert set(result.vintage["segment_col"]) >= {"__overall__"}
    overall_vint = result.vintage.loc[result.vintage["segment_col"].eq("__overall__")]
    assert (overall_vint["segment_value"] == "ALL").all()
    assert result.meta["n_rows"] == len(df)
    assert result.meta["overall_gini"] == result.segment_summary.iloc[0]["gini"]
    assert result.meta["gates"]["min_n"] == gates.min_n
    assert result.meta["columns"]["col_id"] == "app_id"


def test_summary_and_decisions_expose_matched_ar_columns():
    df, cols = make_synthetic_book(n=9000, seed=3)
    result = evaluate_segments(df, cols, _gates())
    for key in MATCHED_AR_KEYS:
        assert key in result.segment_summary.columns
    overall = result.segment_summary.iloc[0]
    assert overall["segment_value"] == "ALL"
    assert pd.isna(overall["gini_at_matched_ar"])


def test_characteristics_record_the_psi_method():
    df, cols = make_synthetic_book(n=9000, seed=3)
    result = evaluate_segments(df, cols, _gates())
    assert "psi_method" in result.characteristics.columns
    assert not result.characteristics["psi_method"].isna().any()
    assert set(result.decisions["psi_method"]) == {"numeric_portfolio_deciles"}


def test_supplied_grouping_switches_psi_to_grouping_bins():
    df, cols = make_synthetic_book(n=9000, seed=3)
    obs = df.loc[df["obs"].eq(1)]
    grouping = BinningModel.fit(obs[list(cols.cols_pred)], obs[cols.col_target], _gates())
    result = evaluate_segments(df, cols, _gates(), grouping=grouping)
    char = result.characteristics
    grouped = char.loc[char["feature"].isin(cols.cols_pred)]
    assert set(grouped["psi_method"]) == {"grouping_bins"}
    assert grouped["grouping_used"].all()
    # x1_woe is not in the grouping, so it still falls back to portfolio deciles.
    ungrouped = char.loc[~char["feature"].isin(cols.cols_pred)]
    assert set(ungrouped["psi_method"]) == {"numeric_portfolio_deciles"}
    assert not ungrouped["grouping_used"].any()
    assert result.meta["grouping_supplied"]


def test_evaluation_survives_a_missing_observation_flag_and_fantomas():
    df, cols = make_synthetic_book(n=6000, seed=4)
    minimal = cols.__class__(
        col_id=cols.col_id,
        col_date=cols.col_date,
        col_target=cols.col_target,
        col_score=cols.col_score,
        cols_pred=list(cols.cols_pred),
        cols_segment=list(cols.cols_segment),
        cols_pred_used=list(cols.cols_pred_used),
    )
    result = evaluate_segments(df, minimal, _gates())
    assert not result.decisions.empty
    assert result.meta["n_observable"] == len(df)
    assert (result.segment_summary["slice"] == "observable").all()


def test_evaluation_without_a_date_column_skips_holdout_work():
    df, cols = make_synthetic_book(n=6000, seed=4)
    no_date = cols.__class__(
        col_id=cols.col_id,
        col_obs=cols.col_obs,
        col_target=cols.col_target,
        col_score=cols.col_score,
        cols_pred=list(cols.cols_pred),
        cols_segment=list(cols.cols_segment),
        cols_pred_used=list(cols.cols_pred_used),
    )
    result = evaluate_segments(df, no_date, _gates())
    assert result.vintage.empty
    assert result.refit_comparison["n_holdout"].isna().all()
    assert not result.decisions.empty


def test_segments_with_no_rows_of_a_category_do_not_crash():
    df, cols = make_synthetic_book(n=3000, seed=6)
    df.loc[df.index[:10], "channel"] = None
    result = evaluate_segments(df, cols, _gates())
    assert not result.decisions.empty
    assert np.isfinite(result.meta["overall_gini"])


def test_evaluate_keeps_fitted_artifacts_and_grouping_notes(tmp_path):
    df, cols = make_synthetic_book(n=9000, seed=3)
    grouping = BinningModel.fit(
        df.loc[df["obs"].eq(1), list(cols.cols_pred)],
        df.loc[df["obs"].eq(1), cols.col_target],
        _gates(),
    )
    result = evaluate_segments(df, cols, _gates(), grouping=grouping)
    kinds = set(art.kind for art in result.fitted_artifacts)
    assert "recalibrate" in kinds
    assert "refit" in kinds
    assert result.meta["pred_woe_map"] == {"x1": "x1_woe"}
    assert not result.grouping_comparison.empty
    assert result.grouping_comparison["significant"].any() or (
        result.grouping_comparison["kind"] == "aligned"
    ).any()
    written = result.save_artifacts(str(tmp_path / "artifacts"))
    assert written
    from scorecard_segment_eval.refit import SCORECARD_FILENAME, load_fitted_artifact
    from scorecard_segment_eval.sql_model import parse_scorecard_sql_path

    reloaded = load_fitted_artifact(written[0])
    assert reloaded.kind in ("refit", "recalibrate")
    assert reloaded.model is not None or reloaded.kind == "recalibrate"
    refit_sql = None
    for path in written:
        art = load_fitted_artifact(path)
        if art.kind == "refit" and str(art.method).startswith("logistic"):
            sql_path = os.path.join(path, SCORECARD_FILENAME)
            assert os.path.isfile(sql_path)
            refit_sql = parse_scorecard_sql_path(sql_path)
            assert refit_sql.grouping is not None
            break
    assert refit_sql is not None

