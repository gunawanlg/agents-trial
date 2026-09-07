import io
import pandas as pd

from scorecard_segment_eval.evaluate import SegmentEvalResult


def decision_table(result):
    cols = [
        "segment_col",
        "segment_value",
        "important",
        "q1_verdict",
        "failed_pillars",
        "q2_action",
        "q2_reason",
        "gini",
        "gini_ratio",
        "oe",
        "ece",
        "volume_share",
        "default_share",
    ]
    present = [c for c in cols if c in result.decisions.columns]
    return result.decisions[present].sort_values(["important", "volume_share"], ascending=[False, False])


def action_list(result):
    d = result.decisions
    if d.empty:
        return d
    mask = d["important"].eq(True) & d["q1_verdict"].eq("WEAK")
    return d.loc[mask].copy()


def recommendations(result):
    """Return prioritized, analyst-readable recommendations."""
    rows = []
    if result.decisions.empty:
        return pd.DataFrame(columns=["priority", "segment", "recommendation", "rationale"])
    priority = {"SPLIT": 1, "RECALIBRATE": 2, "KEEP_POOLED": 3, "NONE": 4}
    text = {
        "SPLIT": "Build and independently validate a segment-specific model.",
        "RECALIBRATE": "Recalibrate the pooled score intercept and slope for this segment.",
        "KEEP_POOLED": "Retain the pooled model and continue routine monitoring.",
        "NONE": "Collect more observed outcomes before changing the model.",
    }
    for _, row in result.decisions.iterrows():
        action = row.get("q2_action", "NONE")
        rationale = str(row.get("q2_reason", ""))
        ar_note = row.get("similar_ar_analysis", "not_required")
        if ar_note != "not_required":
            rationale += "; approval-rate-adjusted comparison: %s" % ar_note
        rows.append(
            {
                "priority": priority.get(action, 5),
                "segment": "%s = %s" % (row.get("segment_col"), row.get("segment_value")),
                "recommendation": text.get(action, action),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows).sort_values(["priority", "segment"])


def analyst_report(result, title="Scorecard segment evaluation"):
    """Build a self-contained polished HTML report."""
    decisions = decision_table(result)
    recs = recommendations(result)
    weak = int(result.decisions["q1_verdict"].eq("WEAK").sum()) if not result.decisions.empty else 0
    inconclusive = int(result.decisions["q1_verdict"].eq("INCONCLUSIVE").sum()) if not result.decisions.empty else 0
    split = int(result.decisions["q2_action"].eq("SPLIT").sum()) if not result.decisions.empty else 0
    style = """
    body{font:14px/1.45 Arial,sans-serif;color:#243447;margin:32px auto;max-width:1200px;padding:0 24px}
    h1,h2{color:#12355b}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:#eef5fb;border-radius:8px;padding:16px;min-width:150px}
    .card strong{display:block;font-size:28px;color:#087e8b}table{border-collapse:collapse;width:100%;margin:12px 0 28px}
    th{background:#12355b;color:white;text-align:left}th,td{padding:8px;border:1px solid #d8e1e8}tr:nth-child(even){background:#f7f9fb}
    .note{border-left:4px solid #087e8b;padding:10px 14px;background:#f1fbfc}
    """
    return """<!doctype html><html><head><meta charset="utf-8"><title>{0}</title><style>{1}</style></head>
    <body><h1>{0}</h1><p class="note">Recommendations combine segment performance, calibration,
    vintage stability, approval-rate-adjusted Gini, predictor shape, and forward-holdout refit evidence.</p>
    <div class="cards"><div class="card"><strong>{2}</strong>weak segments</div>
    <div class="card"><strong>{3}</strong>split candidates</div><div class="card"><strong>{4}</strong>inconclusive segments</div></div>
    <h2>Recommended actions</h2>{5}<h2>Decision evidence</h2>{6}
    <h2>Refit comparison</h2>{7}<h2>Characteristic stability</h2>{8}</body></html>""".format(
        title,
        style,
        weak,
        split,
        inconclusive,
        recs.to_html(index=False, border=0),
        decisions.to_html(index=False, border=0),
        result.refit_comparison.to_html(index=False, border=0),
        result.characteristics.to_html(index=False, border=0),
    )


def write_analyst_report(result, filename, title="Scorecard segment evaluation"):
    with io.open(filename, "w", encoding="utf-8") as handle:
        handle.write(analyst_report(result, title=title))
    return filename
