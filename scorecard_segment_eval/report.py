"""Final report: decision tables, prioritised recommendations and rendering.

Renders a :class:`~scorecard_segment_eval.evaluate.SegmentEvalResult` into a
self-contained HTML page (or Markdown) using nothing but string templating and
the stdlib, so there is no new dependency to install.

Sections, in the order an analyst reads them:

1. what was run and against what gates;
2. an executive summary and the prioritised recommendation list;
3. per-segment verdicts;
4. the matched approval-rate Gini comparison, when it triggered;
5. characteristic drift, including which PSI method each characteristic used;
6. refit performance *and* predictor-stability findings;
7. vintage detail and a methodology note.
"""

import datetime
import html
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scorecard_segment_eval.evaluate import SegmentEvalResult
from scorecard_segment_eval.schema import Gates

DECISION_COLUMNS = (
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
    "ar_gap",
    "gini_at_matched_ar",
    "ar_artifact_suspected",
    "psi_method",
    "stability_score",
    "unstable_predictors",
)

#: Action ordering used to prioritise recommendations (lower runs first).
ACTION_PRIORITY = {
    "SPLIT": 0,
    "RECALIBRATE": 1,
    "MONITOR": 2,
    "KEEP_POOLED": 4,
    "NONE": 5,
}

_ACTION_BADGE = {
    "SPLIT": "danger",
    "RECALIBRATE": "warn",
    "MONITOR": "warn",
    "KEEP_POOLED": "ok",
    "NONE": "muted",
}

_VERDICT_BADGE = {
    "GOOD": "ok",
    "WEAK": "warn",
    "INCONCLUSIVE": "muted",
}


# --------------------------------------------------------------------------
# Tables (unchanged public contract)
# --------------------------------------------------------------------------
def decision_table(result):
    # type: (SegmentEvalResult) -> pd.DataFrame
    """Per-segment decisions, most important first."""
    present = [c for c in DECISION_COLUMNS if c in result.decisions.columns]
    if not present or result.decisions.empty:
        return result.decisions
    return result.decisions[present].sort_values(
        ["important", "volume_share"], ascending=[False, False]
    )


def action_list(result):
    # type: (SegmentEvalResult) -> pd.DataFrame
    """Important segments where the pooled score is weak."""
    d = result.decisions
    if d.empty:
        return d
    mask = d["important"].eq(True) & d["q1_verdict"].eq("WEAK")
    return d.loc[mask].copy()


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
def _num(value, digits=3, default="-"):
    # type: (Any, int, str) -> str
    if value is None:
        return default
    try:
        val = float(value)
    except (TypeError, ValueError):
        return html_safe(str(value))
    if not np.isfinite(val):
        return default
    return ("%." + str(digits) + "f") % val


def _pct(value, digits=1, default="-"):
    # type: (Any, int, str) -> str
    if value is None:
        return default
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(val):
        return default
    return ("%." + str(digits) + "f%%") % (100.0 * val)


def html_safe(text):
    # type: (Any) -> str
    return html.escape("" if text is None else str(text), quote=True)


def _truthy(value):
    # type: (Any) -> bool
    if value is None:
        return False
    if isinstance(value, float) and not np.isfinite(value):
        return False
    return bool(value)


def _recommendation_for_row(row, gates):
    # type: (Dict[str, Any], Gates) -> Optional[Dict[str, Any]]
    action = str(row.get("q2_action") or "NONE")
    verdict = str(row.get("q1_verdict") or "")
    segment = "%s = %s" % (row.get("segment_col"), row.get("segment_value"))
    volume = row.get("volume_share")
    default_share = row.get("default_share")
    evidence = []  # type: List[str]
    evidence.append("Gini %s (ratio to portfolio %s)" % (_num(row.get("gini")), _num(row.get("gini_ratio"), 2)))
    evidence.append("O/E %s, ECE %s" % (_num(row.get("oe"), 2), _num(row.get("ece"))))
    evidence.append("volume share %s, default share %s" % (_pct(volume), _pct(default_share)))
    if _truthy(row.get("ar_gap_triggered")):
        evidence.append(
            "approval-rate gap %s, Gini at matched AR %s (portfolio %s)"
            % (
                _pct(row.get("ar_gap")),
                _num(row.get("gini_at_matched_ar")),
                _num(row.get("gini_reference_at_matched_ar")),
            )
        )
    if row.get("psi_method"):
        evidence.append("PSI method: %s" % (row.get("psi_method"),))
    if row.get("stability_score") is not None and _num(row.get("stability_score")) != "-":
        evidence.append("predictor stability score %s" % (_num(row.get("stability_score"), 2),))
    if row.get("unstable_predictors"):
        evidence.append("unstable predictors: %s" % (row.get("unstable_predictors"),))

    if action == "SPLIT":
        headline = "Split out a dedicated model for %s" % (segment,)
        why = (
            "The pooled score fails rank ordering here, the WoE shape diverges from the "
            "portfolio, and a same-predictor refit beats the pooled score on the forward "
            "holdout by %s Gini (lower CI bound %s) with a better proper score. Predictor "
            "stability over vintages also clears the gate, so the new coefficients are "
            "expected to hold."
            % (_num(row.get("delta_gini")), _num(row.get("delta_gini_ci_low")))
        )
    elif action == "RECALIBRATE":
        headline = "Recalibrate the PD level for %s" % (segment,)
        why = (
            "Ranking is intact but the PD level is off (O/E %s, ECE %s). A two-parameter "
            "intercept/slope rescale fixes the level without touching the ranking, so no "
            "split is warranted."
            % (_num(row.get("oe"), 2), _num(row.get("ece")))
        )
    elif action == "MONITOR":
        headline = "Hold the split for %s and stabilise its inputs first" % (segment,)
        why = (
            "The refit does beat the pooled score on the holdout, but its predictors are "
            "not stable over vintages (%s). Fitting new coefficients on drifting or "
            "sign-flipping inputs buys holdout Gini and loses it in production. Fix the "
            "inputs, or re-assess once more vintages are available."
            % (row.get("stability_reason") or "see stability table",)
        )
    elif action == "NONE" or verdict == "INCONCLUSIVE":
        headline = "Do not judge %s yet -- insufficient power" % (segment,)
        why = (
            "The segment has fewer than %d observable cases or %d defaults, so any verdict "
            "would be noise. Accumulate observations, or merge the segment with a "
            "neighbour for reporting." % (gates.min_n, gates.min_events)
        )
    elif _truthy(row.get("ar_artifact_suspected")):
        headline = "Treat the Gini shortfall on %s as an approval-rate artefact" % (segment,)
        why = (
            "The segment operates at a %s approval rate against the portfolio's %s. Once "
            "both sides are cut to the same approval rate its Gini is %s, back within "
            "tolerance -- so the raw gap reflects a different operating point, not a worse "
            "ranking."
            % (
                _pct(row.get("ar_segment")),
                _pct(row.get("ar_reference")),
                _num(row.get("gini_at_matched_ar")),
            )
        )
    elif verdict == "WEAK" and str(row.get("q2_reason")) == "need_new_information_not_new_coefficients":
        headline = "Source new information for %s rather than refitting" % (segment,)
        why = (
            "The segment is weak, but refitting the same predictors does not recover "
            "enough Gini to justify a split. The bottleneck is the information set, not "
            "the coefficients: look for predictors that discriminate inside this segment."
        )
    elif verdict == "WEAK":
        headline = "Keep %s pooled, with monitoring" % (segment,)
        why = (
            "The segment fails a pillar (%s) but no remediation clears its gates "
            "(%s). Keep it pooled and watch it."
            % (row.get("failed_pillars") or "-", row.get("q2_reason") or "-")
        )
    else:
        return None

    priority = ACTION_PRIORITY.get(action, 4)
    if verdict == "WEAK" and action == "KEEP_POOLED":
        priority = 3
    importance = 0.0
    for candidate in (volume, default_share):
        try:
            val = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            importance = max(importance, val)
    return {
        "priority": priority,
        "segment_col": row.get("segment_col"),
        "segment_value": row.get("segment_value"),
        "action": action,
        "q1_verdict": verdict,
        "important": bool(row.get("important")),
        "importance_share": importance,
        "recommendation": headline,
        "why": why,
        "evidence": "; ".join(evidence),
    }


def _portfolio_recommendations(result, gates):
    # type: (SegmentEvalResult, Gates) -> List[Dict[str, Any]]
    """Book-wide findings that are not tied to one segment."""
    out = []  # type: List[Dict[str, Any]]
    stability = result.stability
    if stability is not None and not stability.empty and "segment_value" in stability.columns:
        overall = stability.loc[stability["segment_col"].eq("__overall__")]
        if not overall.empty:
            flagged = overall.loc[~overall["stable"].fillna(True).astype(bool)]
            if not flagged.empty:
                names = ", ".join(str(f) for f in flagged["feature"].tolist())
                out.append(
                    {
                        "priority": 2,
                        "segment_col": "__overall__",
                        "segment_value": "ALL",
                        "action": "MONITOR",
                        "q1_verdict": "",
                        "important": True,
                        "importance_share": 1.0,
                        "recommendation": "Stabilise portfolio-level predictors: " + names,
                        "why": (
                            "These predictors drift, lose power or flip sign across vintages at "
                            "the portfolio level. Any segment refit that leans on them inherits "
                            "the problem, so fix them before considering splits."
                        ),
                        "evidence": "; ".join(
                            "%s: %s (score %s)"
                            % (r["feature"], r["stability_flags"] or "flagged", _num(r["stability_score"], 2))
                            for _i, r in flagged.iterrows()
                        ),
                    }
                )
    char = result.characteristics
    if char is not None and not char.empty and "psi" in char.columns:
        drifted = char.loc[pd.to_numeric(char["psi"], errors="coerce").fillna(0.0) >= gates.psi_woe_material]
        if not drifted.empty:
            grouped = drifted.groupby("feature")["psi"].max().sort_values(ascending=False)
            names = ", ".join("%s (max PSI %s)" % (f, _num(v, 2)) for f, v in grouped.items())
            out.append(
                {
                    "priority": 3,
                    "segment_col": "__overall__",
                    "segment_value": "ALL",
                    "action": "MONITOR",
                    "q1_verdict": "",
                    "important": True,
                    "importance_share": 0.9,
                    "recommendation": "Investigate characteristic drift: " + names,
                    "why": (
                        "These characteristics sit at or above the material PSI threshold of %s "
                        "in at least one segment, which is what triggers the split test in the "
                        "first place. Confirm the shift is real rather than a data-supply change."
                        % (_num(gates.psi_woe_material, 2),)
                    ),
                    "evidence": "PSI computed with: "
                    + ", ".join(sorted(set(str(m) for m in drifted["psi_method"].dropna().tolist()))),
                }
            )
    return out


def recommendations(result, gates=None):
    # type: (SegmentEvalResult, Optional[Gates]) -> pd.DataFrame
    """Prioritised, concrete recommendations derived from the evaluation.

    Priority 0 is act-now (split), rising to 5 for "no action possible".  Ties
    break on how much of the book the segment carries.
    """
    gates = gates or _gates_from_result(result)
    rows = []  # type: List[Dict[str, Any]]
    if result.decisions is not None and not result.decisions.empty:
        merged = _decisions_with_refit(result)
        for _idx, row in merged.iterrows():
            rec = _recommendation_for_row(dict(row), gates)
            if rec is not None:
                rows.append(rec)
    rows.extend(_portfolio_recommendations(result, gates))
    if not rows:
        return pd.DataFrame(
            columns=[
                "priority",
                "segment_col",
                "segment_value",
                "action",
                "q1_verdict",
                "important",
                "importance_share",
                "recommendation",
                "why",
                "evidence",
            ]
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["priority", "important", "importance_share"], ascending=[True, False, False]
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def _gates_from_result(result):
    # type: (SegmentEvalResult) -> Gates
    payload = (result.meta or {}).get("gates")
    if not payload:
        return Gates()
    known = set(Gates().__dict__.keys())
    return Gates(**dict((k, v) for k, v in payload.items() if k in known))


def _decisions_with_refit(result):
    # type: (SegmentEvalResult) -> pd.DataFrame
    decisions = result.decisions.copy()
    refit = result.refit_comparison
    if refit is None or refit.empty:
        return decisions
    keep = [
        c
        for c in (
            "segment_col",
            "segment_value",
            "delta_gini",
            "delta_gini_ci_low",
            "gini_pooled",
            "gini_refit",
            "gini_recal",
            "brier_refit",
            "brier_pooled",
            "logloss_refit",
            "logloss_pooled",
            "n_holdout",
            "refit_method",
        )
        if c in refit.columns
    ]
    overlap = [c for c in keep if c in decisions.columns and c not in ("segment_col", "segment_value")]
    return decisions.drop(columns=overlap, errors="ignore").merge(
        refit[keep], on=["segment_col", "segment_value"], how="left"
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
_CSS = """
:root { --ok:#1a7f37; --warn:#9a6700; --danger:#b42318; --muted:#57606a; --line:#d8dee4; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 32px; color: #1f2328; background: #f6f8fa; line-height: 1.5; }
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 32px 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h3 { font-size: 15px; margin: 18px 0 6px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; margin: 12px 0; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
.kpi { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; min-width: 150px; flex: 1; }
.kpi .v { font-size: 22px; font-weight: 600; }
.kpi .l { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }
th, td { border: 1px solid var(--line); padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #eef1f4; font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:nth-child(even) td { background: #fbfcfd; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px;
         font-weight: 700; letter-spacing: .03em; border: 1px solid currentColor; }
.badge.ok { color: var(--ok); } .badge.warn { color: var(--warn); }
.badge.danger { color: var(--danger); } .badge.muted { color: var(--muted); }
.rec { border-left: 4px solid var(--line); padding: 10px 14px; margin: 10px 0;
       background: #fff; border-radius: 0 8px 8px 0; border-top: 1px solid var(--line);
       border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.rec.p0 { border-left-color: var(--danger); } .rec.p1, .rec.p2 { border-left-color: var(--warn); }
.rec.p3, .rec.p4, .rec.p5 { border-left-color: var(--muted); }
.rec .h { font-weight: 600; font-size: 14px; }
.rec .w { font-size: 13px; margin-top: 4px; }
.rec .e { font-size: 12px; color: var(--muted); margin-top: 6px; font-family: ui-monospace, monospace; }
.note { font-size: 12px; color: var(--muted); }
.scroll { overflow-x: auto; }
dl.gates { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 2px 16px;
           font-size: 12px; margin: 0; }
dl.gates dt { font-weight: 600; color: var(--muted); }
dl.gates dd { margin: 0 0 4px; font-variant-numeric: tabular-nums; }
"""

_NUMERIC_DIGITS = {
    "volume_share": 3,
    "default_share": 3,
    "gini_ratio": 2,
    "oe": 2,
    "stability_score": 2,
    "importance_share": 3,
}


def _cell(value, column):
    # type: (Any, str) -> str
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _num(value, _NUMERIC_DIGITS.get(column, 3))
    return html_safe(value)


def _is_numeric_column(series):
    # type: (pd.Series) -> bool
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


def html_table(frame, columns=None, empty_message="No rows."):
    # type: (pd.DataFrame, Optional[Sequence[str]], str) -> str
    if frame is None or frame.empty:
        return '<p class="note">' + html_safe(empty_message) + "</p>"
    cols = [c for c in (columns or list(frame.columns)) if c in frame.columns]
    if not cols:
        return '<p class="note">' + html_safe(empty_message) + "</p>"
    numeric = dict((c, _is_numeric_column(frame[c])) for c in cols)
    head = "".join(
        '<th class="num">%s</th>' % (html_safe(c),) if numeric[c] else "<th>%s</th>" % (html_safe(c),)
        for c in cols
    )
    body = []
    for _idx, row in frame[cols].iterrows():
        cells = []
        for col in cols:
            klass = ' class="num"' if numeric[col] else ""
            cells.append("<td%s>%s</td>" % (klass, _cell(row[col], col)))
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _badge(value, mapping):
    # type: (Any, Dict[str, str]) -> str
    key = str(value or "")
    klass = mapping.get(key, "muted")
    return '<span class="badge %s">%s</span>' % (klass, html_safe(key or "-"))


def _kpis(result):
    # type: (SegmentEvalResult) -> str
    meta = result.meta or {}
    decisions = result.decisions
    counts = decisions["q2_action"].value_counts() if not decisions.empty else pd.Series(dtype=int)
    items = [
        ("Rows", "{:,}".format(int(meta.get("n_rows", 0)))),
        ("Observable", "{:,}".format(int(meta.get("n_observable", 0)))),
        ("Portfolio Gini", _num(meta.get("overall_gini"))),
        ("Segments", str(int(len(decisions))) if not decisions.empty else "0"),
        ("Split", str(int(counts.get("SPLIT", 0)))),
        ("Recalibrate", str(int(counts.get("RECALIBRATE", 0)))),
        ("Monitor", str(int(counts.get("MONITOR", 0)))),
    ]
    cells = "".join(
        '<div class="kpi"><div class="l">%s</div><div class="v">%s</div></div>'
        % (html_safe(label), html_safe(value))
        for label, value in items
    )
    return '<div class="grid">' + cells + "</div>"


def _recommendation_blocks(recs):
    # type: (pd.DataFrame) -> str
    if recs is None or recs.empty:
        return '<p class="note">No recommendations: every segment passed on the pooled score.</p>'
    blocks = []
    for _idx, row in recs.iterrows():
        blocks.append(
            '<div class="rec p%d"><div class="h">%d. %s &nbsp;%s</div>'
            '<div class="w">%s</div><div class="e">%s</div></div>'
            % (
                int(row["priority"]),
                int(row["rank"]),
                html_safe(row["recommendation"]),
                _badge(row["action"], _ACTION_BADGE),
                html_safe(row["why"]),
                html_safe(row["evidence"]),
            )
        )
    return "".join(blocks)


def _verdict_rows(result):
    # type: (SegmentEvalResult) -> str
    decisions = result.decisions
    if decisions.empty:
        return '<p class="note">No segments evaluated.</p>'
    head = (
        "<tr><th>Segment</th><th>Value</th><th>Important</th><th>Q1</th><th>Failed pillars</th>"
        '<th>Q2 action</th><th>Reason</th><th class="num">Gini</th><th class="num">Ratio</th>'
        '<th class="num">O/E</th><th class="num">ECE</th><th class="num">Volume</th></tr>'
    )
    body = []
    for _idx, row in decisions.iterrows():
        body.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            '<td class="num">%s</td><td class="num">%s</td><td class="num">%s</td>'
            '<td class="num">%s</td><td class="num">%s</td></tr>'
            % (
                html_safe(row.get("segment_col")),
                html_safe(row.get("segment_value")),
                "yes" if _truthy(row.get("important")) else "no",
                _badge(row.get("q1_verdict"), _VERDICT_BADGE),
                html_safe(row.get("failed_pillars") or "-"),
                _badge(row.get("q2_action"), _ACTION_BADGE),
                html_safe(row.get("q2_reason") or "-"),
                _num(row.get("gini")),
                _num(row.get("gini_ratio"), 2),
                _num(row.get("oe"), 2),
                _num(row.get("ece")),
                _pct(row.get("volume_share")),
            )
        )
    return '<div class="scroll"><table><thead>' + head + "</thead><tbody>" + "".join(body) + "</tbody></table></div>"


def _matched_ar_section(result, gates):
    # type: (SegmentEvalResult, Gates) -> str
    decisions = result.decisions
    intro = (
        '<p class="note">A segment scored on a different part of the risk spectrum cannot be '
        "compared to the portfolio at face value. When the approval-rate gap reaches %s, both "
        "sides are re-cut to the lower-AR anchor and Gini is recomputed there. Rows below show "
        "only the segments where that triggered; a null Gini at matched AR means the anchored "
        "sub-population was too small (&lt; %d rows or &lt; %d defaults).</p>"
        % (_pct(gates.ar_gap_trigger), gates.matched_ar_min_n, gates.matched_ar_min_events)
    )
    if decisions.empty or "ar_gap_triggered" not in decisions.columns:
        return intro + '<p class="note">Not computed.</p>'
    triggered = decisions.loc[decisions["ar_gap_triggered"].fillna(False).astype(bool)]
    cols = [
        "segment_col",
        "segment_value",
        "ar_segment",
        "ar_reference",
        "ar_gap",
        "matched_ar",
        "matched_ar_anchor",
        "gini",
        "gini_at_matched_ar",
        "gini_reference_at_matched_ar",
        "gini_at_matched_ar_gap",
        "gini_at_matched_ar_ratio",
        "ar_artifact_suspected",
    ]
    return intro + html_table(
        triggered,
        cols,
        empty_message="No segment reached the approval-rate gap trigger, so no matched-AR "
        "simulation was needed.",
    )


def _gates_block(gates):
    # type: (Gates) -> str
    items = []
    for name in sorted(gates.__dict__.keys()):
        value = getattr(gates, name)
        items.append("<dt>%s</dt><dd>%s</dd>" % (html_safe(name), html_safe(value)))
    return '<dl class="gates">' + "".join(items) + "</dl>"


def render_html_report(result, title="Segment scorecard evaluation", gates=None, recs=None):
    # type: (SegmentEvalResult, str, Optional[Gates], Optional[pd.DataFrame]) -> str
    """Self-contained HTML report (inline CSS, no external assets)."""
    gates = gates or _gates_from_result(result)
    recs = recommendations(result, gates) if recs is None else recs
    meta = result.meta or {}
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    stability = result.stability if result.stability is not None else pd.DataFrame()
    char = result.characteristics if result.characteristics is not None else pd.DataFrame()

    parts = []  # type: List[str]
    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>" + html_safe(title) + "</title><style>" + _CSS + "</style></head><body>")
    parts.append("<div class='wrap'>")
    parts.append("<h1>" + html_safe(title) + "</h1>")
    parts.append(
        '<div class="sub">Generated %s &middot; %s rows (%s observable) &middot; workers: %s '
        "&middot; sub-model: %s &middot; grouping supplied: %s</div>"
        % (
            html_safe(generated),
            "{:,}".format(int(meta.get("n_rows", 0))),
            "{:,}".format(int(meta.get("n_observable", 0))),
            html_safe(meta.get("n_jobs")),
            "yes" if meta.get("submodel") else "no",
            "yes" if meta.get("grouping_supplied") else "no",
        )
    )
    parts.append(_kpis(result))

    parts.append("<h2>1. What to do next</h2>")
    parts.append(_recommendation_blocks(recs))

    parts.append("<h2>2. Per-segment verdicts</h2>")
    parts.append(
        '<p class="note">Q1 asks whether the pooled score is good enough on the segment; Q2 asks '
        "what to do about it. <strong>SPLIT</strong> requires a material holdout Gini gain with a "
        "positive lower CI bound, a better proper score, divergent WoE shape <em>and</em> stable "
        "predictors. <strong>MONITOR</strong> means the performance case was made but stability "
        "was not.</p>"
    )
    parts.append(_verdict_rows(result))

    parts.append("<h2>3. Like-for-like comparison at matched approval rate</h2>")
    parts.append(_matched_ar_section(result, gates))

    parts.append("<h2>4. Characteristic drift and PSI method</h2>")
    parts.append(
        '<p class="note">Grouped and categorical characteristics are compared on the grouping\'s '
        "own bins (<code>grouping_bins</code> / <code>categorical_levels</code>); ungrouped "
        "numeric characteristics use portfolio-level decile edges applied unchanged to the "
        "segment (<code>numeric_portfolio_deciles</code>). Missing values always form their own "
        "bin and are counted in <code>psi_missing_share_*</code>.</p>"
    )
    char_cols = [
        "segment_col",
        "segment_value",
        "feature",
        "psi",
        "psi_method",
        "psi_n_bins",
        "psi_worst_bin",
        "psi_missing_share_segment",
        "psi_out_of_range_share_segment",
        "iv_segment",
        "iv_overall",
        "woe_spearman",
        "rank_reversal",
    ]
    if not char.empty and "psi" in char.columns:
        char = char.sort_values("psi", ascending=False)
    parts.append(html_table(char, char_cols, empty_message="No characteristic diagnostics available."))

    parts.append("<h2>5. Refit performance</h2>")
    refit_cols = [
        "segment_col",
        "segment_value",
        "shape_divergent",
        "n_holdout",
        "defaults_holdout",
        "gini_pooled",
        "gini_refit",
        "delta_gini",
        "delta_gini_ci_low",
        "gini_recal",
        "brier_pooled",
        "brier_refit",
        "logloss_pooled",
        "logloss_refit",
        "refit_method",
    ]
    parts.append(html_table(result.refit_comparison, refit_cols, empty_message="No refit was attempted."))

    parts.append("<h2>5b. Segment vs portfolio grouping</h2>")
    parts.append(
        '<p class="note">When a segment is refitted, its WoE bins are compared to the original '
        "portfolio grouping (the supplied grouping, the grouping reconstructed from "
        "<code>cols_pred_woe</code>, or a grouping fitted on the portfolio). "
        "Significant differences — edge movement, merge/split, WoE shift or sign flip — "
        "are noted per bin.</p>"
    )
    grouping_cols = [
        "segment_col",
        "segment_value",
        "feature",
        "segment_bin",
        "portfolio_bins",
        "woe_segment",
        "woe_portfolio",
        "woe_delta",
        "kind",
        "significant",
        "note",
    ]
    grouping_cmp = getattr(result, "grouping_comparison", None)
    if grouping_cmp is None:
        grouping_cmp = pd.DataFrame()
    parts.append(
        html_table(
            grouping_cmp,
            grouping_cols,
            empty_message="No grouping comparison was produced (no refit, or no portfolio grouping).",
        )
    )

    parts.append("<h2>6. Predictor stability over vintages</h2>")
    parts.append(
        '<p class="note">A refit is only accepted when its predictors are stable. Weak vintages '
        "are judged against the Hanley-McNeil standard error, so small monthly samples are not "
        "mistaken for drift, and a predictor with no univariate signal is reported as "
        "<code>no_reference_signal</code> and never vetoes a refit.</p>"
    )
    stability_cols = [
        "segment_col",
        "segment_value",
        "feature",
        "n_vintages",
        "assessed",
        "psi_max",
        "gini_reference",
        "gini_ratio_median",
        "weak_vintage_share",
        "sign_consistency",
        "stability_score",
        "stable",
        "stability_flags",
    ]
    parts.append(html_table(stability, stability_cols, empty_message="No stability diagnostics available."))

    parts.append("<h2>7. Vintage detail</h2>")
    parts.append(
        html_table(
            result.vintage,
            ["segment_col", "segment_value", "vintage", "n", "defaults", "gini", "oe"],
            empty_message="No vintage table available.",
        )
    )

    parts.append("<h2>8. Gates used</h2>")
    parts.append('<div class="card">' + _gates_block(gates) + "</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


def render_markdown_report(result, title="Segment scorecard evaluation", gates=None, recs=None):
    # type: (SegmentEvalResult, str, Optional[Gates], Optional[pd.DataFrame]) -> str
    """Plain-text/Markdown rendering of the same content."""
    gates = gates or _gates_from_result(result)
    recs = recommendations(result, gates) if recs is None else recs
    meta = result.meta or {}
    lines = ["# " + title, ""]
    lines.append(
        "Rows: %s | observable: %s | portfolio Gini: %s | workers: %s"
        % (
            meta.get("n_rows"),
            meta.get("n_observable"),
            _num(meta.get("overall_gini")),
            meta.get("n_jobs"),
        )
    )
    lines.append("")
    lines.append("## 1. What to do next")
    lines.append("")
    if recs is None or recs.empty:
        lines.append("_No recommendations: every segment passed on the pooled score._")
    else:
        for _idx, row in recs.iterrows():
            lines.append("%d. **%s** [%s]" % (int(row["rank"]), row["recommendation"], row["action"]))
            lines.append("   - Why: " + str(row["why"]))
            lines.append("   - Evidence: " + str(row["evidence"]))
    lines.append("")
    lines.append("## 2. Per-segment verdicts")
    lines.append("")
    lines.append(_markdown_table(decision_table(result)))
    lines.append("")
    lines.append("## 3. Matched approval-rate comparison")
    lines.append("")
    decisions = result.decisions
    if not decisions.empty and "ar_gap_triggered" in decisions.columns:
        triggered = decisions.loc[decisions["ar_gap_triggered"].fillna(False).astype(bool)]
        if triggered.empty:
            lines.append(
                "_No segment reached the %s approval-rate gap trigger._" % (_pct(gates.ar_gap_trigger),)
            )
        else:
            lines.append(
                _markdown_table(
                    triggered[
                        [
                            c
                            for c in (
                                "segment_value",
                                "ar_segment",
                                "ar_reference",
                                "ar_gap",
                                "matched_ar",
                                "gini",
                                "gini_at_matched_ar",
                                "gini_at_matched_ar_gap",
                            )
                            if c in triggered.columns
                        ]
                    ]
                )
            )
    else:
        lines.append("_Not computed._")
    lines.append("")
    lines.append("## 4. Characteristic drift (PSI)")
    lines.append("")
    lines.append(_markdown_table(result.characteristics, ["segment_value", "feature", "psi", "psi_method", "rank_reversal"]))
    lines.append("")
    lines.append("## 5. Refit performance")
    lines.append("")
    lines.append(
        _markdown_table(
            result.refit_comparison,
            ["segment_value", "n_holdout", "gini_pooled", "gini_refit", "delta_gini", "delta_gini_ci_low", "refit_method"],
        )
    )
    lines.append("")
    lines.append("## 5b. Segment vs portfolio grouping")
    lines.append("")
    grouping_cmp = getattr(result, "grouping_comparison", None)
    lines.append(
        _markdown_table(
            grouping_cmp,
            [
                "segment_value",
                "feature",
                "segment_bin",
                "kind",
                "significant",
                "note",
            ],
        )
    )
    lines.append("")
    lines.append("## 6. Predictor stability")
    lines.append("")
    lines.append(
        _markdown_table(
            result.stability,
            ["segment_value", "feature", "assessed", "psi_max", "weak_vintage_share", "sign_consistency", "stability_score", "stable", "stability_flags"],
        )
    )
    return "\n".join(lines)


def _markdown_table(frame, columns=None):
    # type: (Optional[pd.DataFrame], Optional[Sequence[str]]) -> str
    if frame is None or frame.empty:
        return "_No rows._"
    cols = [c for c in (columns or list(frame.columns)) if c in frame.columns]
    if not cols:
        return "_No rows._"
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    divider = "| " + " | ".join("---" for _c in cols) + " |"
    rows = []
    for _idx, row in frame[cols].iterrows():
        rows.append("| " + " | ".join(_cell(row[c], c).replace("|", "\\|") for c in cols) + " |")
    return "\n".join([header, divider] + rows)


def save_report(result, path, fmt=None, title="Segment scorecard evaluation", gates=None):
    # type: (SegmentEvalResult, str, Optional[str], str, Optional[Gates]) -> str
    """Write the report to ``path``.  Format follows the extension by default.

    ``fmt="condensed"`` (or a path whose name contains ``condensed.html``) writes
    the shorter analyst view from :func:`render_condensed_html_report`.
    """
    from scorecard_segment_eval.condensed_report import render_condensed_html_report

    lowered = str(path).lower()
    if fmt is None:
        if "condensed" in os.path.basename(lowered) and lowered.endswith(".html"):
            fmt = "condensed"
        elif lowered.endswith((".md", ".markdown", ".txt")):
            fmt = "md"
        else:
            fmt = "html"
    if fmt in ("condensed", "condensed-html", "condensed_html"):
        content = render_condensed_html_report(result, title=title, gates=gates)
    elif fmt in ("md", "markdown", "text", "txt"):
        content = render_markdown_report(result, title=title, gates=gates)
    else:
        content = render_html_report(result, title=title, gates=gates)
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        handle.write(content)
    return path
