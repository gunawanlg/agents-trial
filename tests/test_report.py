import os

from scorecard_segment_eval import Gates, evaluate_segments, make_synthetic_book, write_final_report
from scorecard_segment_eval.report import build_html_report, build_markdown_report, recommend_segment


def test_html_and_markdown_report(tmp_path):
    df, cols = make_synthetic_book(n=3500, seed=3)
    gates = Gates(min_n=400, min_events=25, n_bootstrap=30, bootstrap_seed=0, n_jobs=1)
    result = evaluate_segments(df, cols, gates)
    html = build_html_report(result)
    md = build_markdown_report(result)
    assert "SPLIT" in html or "RECALIBRATE" in html or "KEEP_POOLED" in html
    assert "Analyst recommendations" in html
    assert "Decision table" in md or "Decision" in md
    html_path = str(tmp_path / "segment_report.html")
    out, md_path = write_final_report(result, html_path)
    assert os.path.isfile(out)
    assert os.path.isfile(md_path)
    with open(out) as handle:
        body = handle.read()
    assert "badge" in body
    assert len(result.decisions) >= 1
    recs = recommend_segment(result.decisions.iloc[0])
    assert recs
