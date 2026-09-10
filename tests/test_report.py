import os

import pandas as pd
import pytest

from scorecard_segment_eval import Gates, evaluate_segments, make_synthetic_book
from scorecard_segment_eval.evaluate import SegmentEvalResult
from scorecard_segment_eval.report import (
    ACTION_PRIORITY,
    action_list,
    decision_table,
    html_table,
    recommendations,
    render_html_report,
    render_markdown_report,
    save_report,
)
from scorecard_segment_eval.condensed_report import render_condensed_html_report


@pytest.fixture(scope="module")
def result():
    df, cols = make_synthetic_book(n=9000, seed=3)
    gates = Gates(min_n=400, min_events=25, n_bootstrap=60, bootstrap_seed=0, holdout_frac=0.3)
    return evaluate_segments(df, cols, gates, n_jobs=-1), gates


def test_decision_table_keeps_its_contract(result):
    res, _gates = result
    table = decision_table(res)
    for column in ("segment_col", "segment_value", "q1_verdict", "q2_action", "gini"):
        assert column in table.columns
    assert bool(table["important"].iloc[0])
    assert len(table) == len(res.decisions)


def test_decision_table_surfaces_the_new_columns(result):
    res, _gates = result
    table = decision_table(res)
    for column in ("ar_gap", "gini_at_matched_ar", "psi_method", "stability_score"):
        assert column in table.columns


def test_action_list_keeps_its_contract(result):
    res, _gates = result
    actions = action_list(res)
    assert set(actions["q1_verdict"]) == {"WEAK"}
    assert actions["important"].all()


def test_action_list_on_empty_input():
    empty = SegmentEvalResult(
        segment_summary=pd.DataFrame(),
        decisions=pd.DataFrame(),
        refit_comparison=pd.DataFrame(),
        characteristics=pd.DataFrame(),
        vintage=pd.DataFrame(),
    )
    assert action_list(empty).empty
    assert decision_table(empty).empty
    assert recommendations(empty).empty
    assert "No recommendations" in render_html_report(empty)


def test_recommendations_are_prioritised_and_concrete(result):
    res, gates = result
    recs = recommendations(res, gates)
    assert not recs.empty
    assert list(recs["rank"]) == list(range(1, len(recs) + 1))
    assert recs["priority"].is_monotonic_increasing
    top = recs.iloc[0]
    assert top["action"] == "SPLIT"
    assert top["priority"] == ACTION_PRIORITY["SPLIT"]
    assert "inverted" in top["recommendation"]
    for _idx, row in recs.iterrows():
        assert row["recommendation"] and row["why"] and row["evidence"]


def test_recommendations_cover_every_action_kind(result):
    res, gates = result
    recs = recommendations(res, gates)
    actions = set(recs["action"])
    assert "SPLIT" in actions
    assert "RECALIBRATE" in actions
    assert "NONE" in actions
    texts = " ".join(recs["recommendation"].tolist())
    assert "Recalibrate the PD level" in texts
    assert "insufficient power" in texts


def test_recommendations_explain_the_stability_veto():
    df, cols = make_synthetic_book(n=9000, seed=3)
    gates = Gates(
        min_n=400,
        min_events=25,
        n_bootstrap=40,
        bootstrap_seed=0,
        holdout_frac=0.3,
        # Impossible to satisfy, so any refit is vetoed on stability.
        stability_score_min=1.01,
        stability_min_reference_gini=0.0,
    )
    res = evaluate_segments(df, cols, gates)
    assert "MONITOR" in set(res.decisions["q2_action"])
    monitored = res.decisions.loc[res.decisions["q2_action"].eq("MONITOR")].iloc[0]
    assert monitored["q2_reason"].startswith("refit_predictors_unstable")
    recs = recommendations(res, gates)
    monitor_recs = recs.loc[recs["action"].eq("MONITOR")]
    assert not monitor_recs.empty
    assert any("stabilise" in text.lower() for text in monitor_recs["recommendation"])


def test_html_report_is_self_contained_and_has_every_section(result):
    res, gates = result
    html = render_html_report(res, gates=gates)
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html and "http://" not in html and "https://" not in html
    for heading in (
        "What to do next",
        "Per-segment verdicts",
        "matched approval rate",
        "Characteristic drift and PSI method",
        "Refit performance",
        "Segment vs portfolio grouping",
        "Predictor stability over vintages",
        "Vintage detail",
        "Gates used",
    ):
        assert heading in html, "missing section: " + heading
    assert "numeric_portfolio_deciles" in html
    assert "gini_at_matched_ar" in html
    assert "stability_score" in html
    assert "ar_gap_trigger" in html


def test_html_report_escapes_values():
    res = SegmentEvalResult(
        segment_summary=pd.DataFrame(),
        decisions=pd.DataFrame(
            [
                {
                    "segment_col": "channel",
                    "segment_value": "<script>alert(1)</script>",
                    "q1_verdict": "WEAK",
                    "failed_pillars": "rank_order",
                    "important": True,
                    "q2_action": "KEEP_POOLED",
                    "q2_reason": "need_new_information_not_new_coefficients",
                    "volume_share": 0.4,
                    "default_share": 0.4,
                    "gini": 0.1,
                }
            ]
        ),
        refit_comparison=pd.DataFrame(),
        characteristics=pd.DataFrame(),
        vintage=pd.DataFrame(),
    )
    html = render_html_report(res)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_markdown_report_covers_the_same_ground(result):
    res, gates = result
    text = render_markdown_report(res, gates=gates)
    assert text.startswith("# Segment scorecard evaluation")
    for heading in (
        "## 1. What to do next",
        "## 2. Per-segment verdicts",
        "## 3. Matched approval-rate comparison",
        "## 4. Characteristic drift (PSI)",
        "## 5. Refit performance",
        "## 5b. Segment vs portfolio grouping",
        "## 6. Predictor stability",
    ):
        assert heading in text
    assert "| segment_value |" in text


def test_save_report_picks_the_format_from_the_extension(result, tmp_path):
    res, gates = result
    html_path = save_report(res, str(tmp_path / "r.html"), gates=gates)
    md_path = save_report(res, str(tmp_path / "r.md"), gates=gates)
    assert os.path.getsize(html_path) > 5000
    with open(html_path) as handle:
        assert handle.read().startswith("<!DOCTYPE html>")
    with open(md_path) as handle:
        assert handle.read().startswith("# Segment")


def test_save_report_creates_missing_directories(result, tmp_path):
    res, gates = result
    path = save_report(res, str(tmp_path / "deep" / "nested" / "r.html"), gates=gates)
    assert os.path.isfile(path)


def test_html_table_handles_missing_and_boolean_values():
    frame = pd.DataFrame({"a": [1.5, float("nan")], "flag": [True, False], "s": ["x", None]})
    out = html_table(frame)
    assert "<td class=\"num\">1.500</td>" in out
    assert "<td class=\"num\">-</td>" in out
    assert ">yes<" in out and ">no<" in out
    assert html_table(pd.DataFrame(), empty_message="nothing").endswith("nothing</p>")


def test_condensed_html_report_has_toc_frozen_tables_and_filters(result, tmp_path):
    res, gates = result
    html = render_condensed_html_report(res, gates=gates)
    assert html.startswith("<!DOCTYPE html>")
    assert 'id="top"' in html
    assert 'href="#top"' in html
    assert 'class="frozen"' in html
    assert 'id="sec-next"' in html
    assert "Recalibrate the PD level for" in html
    assert "channel" in html
    assert "How the matched-AR cutoff is simulated" in html
    assert "gini_refit" in html
    assert "stability_reason" in html
    assert "scorecard.sql" in html
    assert "class=\"idx\"" in html
    assert "class=\"idx2\"" in html
    assert "class=\"frozen\"" in html
    vintage_part = html.split('id="sec-vintage"', 1)[1]
    assert "<table" not in vintage_part
    assert "portfolio event rate" in vintage_part
    path = save_report(
        res, str(tmp_path / "segment_evaluation_report_condensed.html"), gates=gates
    )
    with open(path) as handle:
        saved = handle.read()
    assert saved.startswith("<!DOCTYPE html>")
    assert "back to top" in saved
    fmt_path = save_report(res, str(tmp_path / "r.html"), fmt="condensed", gates=gates)
    with open(fmt_path) as handle:
        assert "back to top" in handle.read()


def test_condensed_html_escapes_values():
    res = SegmentEvalResult(
        segment_summary=pd.DataFrame(),
        decisions=pd.DataFrame(
            [
                {
                    "segment_col": "channel",
                    "segment_value": "<script>alert(1)</script>",
                    "q1_verdict": "WEAK",
                    "failed_pillars": "rank_order",
                    "important": True,
                    "q2_action": "KEEP_POOLED",
                    "q2_reason": "need_new_information_not_new_coefficients",
                    "volume_share": 0.4,
                    "default_share": 0.4,
                    "gini": 0.1,
                    "stability_reason": "stable",
                }
            ]
        ),
        refit_comparison=pd.DataFrame(),
        characteristics=pd.DataFrame(),
        vintage=pd.DataFrame(),
    )
    html = render_condensed_html_report(res)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_condensed_recalibrate_groups_values_under_segment_col():
    from scorecard_segment_eval.condensed_report import _grouped_next_actions

    recs = pd.DataFrame(
        [
            {
                "rank": 1,
                "priority": 1,
                "action": "RECALIBRATE",
                "segment_col": "channel",
                "segment_value": "miscal",
                "recommendation": "Recalibrate the PD level of channel = miscal.",
                "why": "why-a",
                "evidence": "e1",
            },
            {
                "rank": 2,
                "priority": 1,
                "action": "RECALIBRATE",
                "segment_col": "channel",
                "segment_value": "shift",
                "recommendation": "Recalibrate the PD level of channel = shift.",
                "why": "why-b",
                "evidence": "e2",
            },
        ]
    )
    html = _grouped_next_actions(recs)
    assert html.count("Recalibrate the PD level for") == 1
    assert "miscal, shift" in html
    assert html.count("RECALIBRATE") == 1
