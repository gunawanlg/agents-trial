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

    overall = d.loc["ALL"]
    assert overall["segment_col"] == "__overall__"
    assert overall["important"]
    assert overall["q1_verdict"] in ("GOOD", "WEAK", "INCONCLUSIVE")
    assert overall["q2_action"] != "SPLIT"
    assert "gini" in overall and np.isfinite(overall["gini"])
    assert "oe" in overall and np.isfinite(overall["oe"])
    assert "ece" in overall
    assert overall["q2_action"] in ("KEEP_POOLED", "RECALIBRATE", "NONE", "MONITOR")

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
    for key in ("obs_rate", "mean_pd", "ar_reference_cutoff"):
        assert key in result.decisions.columns


def test_characteristics_record_the_psi_method():
    df, cols = make_synthetic_book(n=9000, seed=3)
    result = evaluate_segments(df, cols, _gates())
    assert "psi_method" in result.characteristics.columns
    assert not result.characteristics["psi_method"].isna().any()
    segment_decisions = result.decisions.loc[result.decisions["segment_col"].ne("__overall__")]
    assert set(segment_decisions["psi_method"]) == {"numeric_portfolio_deciles"}


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


def test_portfolio_row_is_always_in_decisions():
    df, cols = make_synthetic_book(n=6000, seed=4)
    result = evaluate_segments(df, cols, _gates(), n_jobs=1)
    port = result.decisions.loc[result.decisions["segment_col"].eq("__overall__")]
    assert len(port) == 1
    row = port.iloc[0]
    assert row["segment_value"] == "ALL"
    assert row["volume_share"] == 1.0
    assert row["q1_verdict"] in ("GOOD", "WEAK", "INCONCLUSIVE")
    assert np.isfinite(row["gini"])
    assert np.isfinite(row["oe"])
    assert result.meta["include_segments"] is True
    assert result.meta["portfolio_whatif"] is False
    # Without a grouping there is no portfolio WoE table.
    assert result.grouping_summary.empty


def test_portfolio_grouping_summary_when_grouping_supplied():
    df, cols = make_synthetic_book(n=6000, seed=4)
    obs = df.loc[df["obs"].eq(1)]
    grouping = BinningModel.fit(obs[list(cols.cols_pred)], obs[cols.col_target], _gates())
    result = evaluate_segments(df, cols, _gates(), grouping=grouping, n_jobs=1)
    summary = result.grouping_summary
    assert not summary.empty
    assert set(summary["feature"]) >= set(cols.cols_pred)
    assert summary["univariate_gini"].notna().any()
    assert (summary["n_bins"] >= 1).all()
    port = result.decisions.set_index("segment_value").loc["ALL"]
    assert port["psi_method"] == "grouping_bins"


def test_evaluate_without_segment_columns_still_scores_the_book():
    df, cols = make_synthetic_book(n=4000, seed=5)
    no_seg = cols.__class__(
        col_id=cols.col_id,
        col_date=cols.col_date,
        col_obs=cols.col_obs,
        col_target=cols.col_target,
        col_score=cols.col_score,
        cols_pred=list(cols.cols_pred),
        cols_pred_woe=list(cols.cols_pred_woe or []),
        cols_pred_used=list(cols.cols_pred_used or []),
    )
    result = evaluate_segments(df, no_seg, _gates(), n_jobs=1)
    assert set(result.decisions["segment_col"]) == {"__overall__"}
    assert result.decisions.iloc[0]["segment_value"] == "ALL"
    assert np.isfinite(result.meta["overall_gini"])


def test_portfolio_whatif_always_refits(tmp_path):
    from scorecard_segment_eval import evaluate_portfolio_whatif
    from scorecard_segment_eval.refit import SCORECARD_FILENAME, load_fitted_artifact
    from scorecard_segment_eval.sql_model import parse_scorecard_sql_path

    df, cols = make_synthetic_book(n=9000, seed=3)
    obs = df.loc[df["obs"].eq(1)]
    grouping = BinningModel.fit(obs[list(cols.cols_pred)], obs[cols.col_target], _gates())
    result = evaluate_portfolio_whatif(df, cols, _gates(), grouping=grouping, n_jobs=1)
    assert result.meta["portfolio_whatif"] is True
    assert result.meta["include_segments"] is False
    assert set(result.decisions["segment_col"]) == {"__overall__"}
    d = result.decisions.iloc[0]
    assert d["q2_action"] in ("KEEP_POOLED", "RECALIBRATE", "REFIT", "MONITOR", "NONE")
    assert d["q2_action"] != "SPLIT"
    ref = result.refit_comparison.iloc[0]
    assert pd.notna(ref["gini_pooled"])
    assert pd.notna(ref["gini_refit"])
    assert pd.notna(ref["gini_recal"])
    assert not result.grouping_summary.empty
    assert not result.grouping_comparison.empty
    assert any(art.kind == "refit" for art in result.fitted_artifacts)
    written = result.save_artifacts(str(tmp_path / "portfolio_whatif"))
    assert written
    found_sql = False
    for path in written:
        art = load_fitted_artifact(path)
        if art.kind == "refit" and str(art.method).startswith("logistic"):
            sql_path = os.path.join(path, SCORECARD_FILENAME)
            assert os.path.isfile(sql_path)
            parsed = parse_scorecard_sql_path(sql_path)
            assert parsed.grouping is not None
            found_sql = True
            break
    assert found_sql


def test_portfolio_whatif_recommends_refit_when_production_pd_is_inverted():
    from scorecard_segment_eval import ScorecardColumns, evaluate_portfolio_whatif

    rng = np.random.default_rng(11)
    n = 5000
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 400, size=n), unit="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = -1.6 + 1.5 * x1 + 1.2 * x2
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    p_wrong = 1.0 / (1.0 + np.exp(-(-1.6 - 1.5 * x1 - 1.2 * x2)))
    df = pd.DataFrame(
        {
            "app_id": np.arange(n),
            "score_date": dates,
            "obs": np.ones(n, dtype=int),
            "default": y.astype(int),
            "pd": p_wrong,
            "x1": x1,
            "x2": x2,
        }
    )
    cols = ScorecardColumns(
        col_id="app_id",
        col_date="score_date",
        col_obs="obs",
        col_target="default",
        col_score="pd",
        cols_pred=["x1", "x2"],
    )
    grouping = BinningModel.fit(df[["x1", "x2"]], df["default"], _gates())
    result = evaluate_portfolio_whatif(df, cols, _gates(), grouping=grouping, n_jobs=1)
    row = result.decisions.iloc[0]
    assert row["q1_verdict"] == "WEAK"
    assert "rank_order" in str(row["failed_pillars"])
    ref = result.refit_comparison.iloc[0]
    assert ref["delta_gini"] > 0.05
    assert row["q2_action"] in ("REFIT", "MONITOR")
    if bool(row["stability_pass"]):
        assert row["q2_action"] == "REFIT"
        assert row["q2_reason"] == "holdout_delta_gini_portfolio"

