"""Condensed, analyst-facing HTML report.

A second, shorter rendering of :class:`~scorecard_segment_eval.evaluate.SegmentEvalResult`
that keeps the full report untouched.  Tables freeze ``segment_col`` /
``segment_value``, every heading has a back-to-top link, and each section
keeps only the rows that need a decision.
"""

from __future__ import print_function, unicode_literals

import base64
import datetime
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd

from scorecard_segment_eval.evaluate import SegmentEvalResult
from scorecard_segment_eval.refit import _safe_path_token
from scorecard_segment_eval.report import (
    ACTION_PRIORITY,
    _ACTION_BADGE,
    _VERDICT_BADGE,
    _badge,
    _cell,
    _decisions_with_refit,
    _gates_block,
    _gates_from_result,
    _is_numeric_column,
    _kpis,
    _pct,
    html_safe,
    recommendations,
)
from scorecard_segment_eval.schema import Gates

_INDEX_COLS = ("segment_col", "segment_value")
_COMPARE_ACTIONS = ("KEEP_POOLED", "MONITOR", "SPLIT")
_AR_GAP_FLOOR = 0.10
_PSI_FLOOR = 0.25

_CSS = """
:root { --ok:#1a7f37; --warn:#9a6700; --danger:#b42318; --muted:#57606a; --line:#d8dee4; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 32px; color: #1f2328; background: #f6f8fa; line-height: 1.5; }
.wrap { max-width: 1220px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 32px 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h2 .back { float: right; font-size: 12px; font-weight: 500; }
h3 { font-size: 15px; margin: 18px 0 6px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; margin: 12px 0; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
.kpi { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; min-width: 150px; flex: 1; }
.kpi .v { font-size: 22px; font-weight: 600; }
.kpi .l { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
table { border-collapse: separate; border-spacing: 0; width: max-content; min-width: 100%; font-size: 13px; background: #fff; }
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
.scroll { overflow: auto; max-height: 460px; }
table.frozen thead th { position: sticky; top: 0; z-index: 3; }
table.frozen td.idx, table.frozen th.idx { position: sticky; left: 0; z-index: 1; min-width: 118px; background: #fff; }
table.frozen td.idx2, table.frozen th.idx2 { position: sticky; left: 118px; z-index: 1; min-width: 128px; background: #fff;
       box-shadow: 3px 0 6px rgba(31,35,40,.08); }
table.frozen thead th.idx, table.frozen thead th.idx2 { z-index: 4; background: #eef1f4; }
tr:nth-child(even) td.idx, tr:nth-child(even) td.idx2 { background: #fbfcfd; }
.toc { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px 18px 12px 18px; }
.toc ol { margin: 6px 0 0; padding-left: 22px; }
.toc a { text-decoration: none; }
a { color: #0969da; }
img.viz { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
.illus { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 10px 0 16px; overflow-x: auto; }
.illus svg { display: block; max-width: 100%; }
.values { font-family: ui-monospace, monospace; font-size: 12px; }
.calib { font-size: 12px; margin-top: 8px; }
table.frozen tr.clickable { cursor: pointer; }
table.frozen tr.clickable:hover td,
table.frozen tr.clickable:hover td.idx,
table.frozen tr.clickable:hover td.idx2 { background: #ddf4ff; }
table.frozen tr.clickable.selected td,
table.frozen tr.clickable.selected td.idx,
table.frozen tr.clickable.selected td.idx2 { background: #fff8c5; }
dl.gates { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 2px 16px;
           font-size: 12px; margin: 0; }
dl.gates dt { font-weight: 600; color: var(--muted); }
dl.gates dd { margin: 0 0 4px; font-variant-numeric: tabular-nums; }
.flag-viz { min-width: 220px; }
.flag-viz svg { display: block; }
.flag-mean { font-size: 11px; color: var(--muted); margin-top: 4px; }
"""

_TOC = (
    ("sec-next", "What to do next"),
    ("sec-verdicts", "Per-segment verdicts"),
    ("sec-ar", "Like-for-like comparison at matched approval rate"),
    ("sec-psi", "Characteristic drift"),
    ("sec-grouping", "Segment vs portfolio grouping"),
    ("sec-stability", "Predictor stability over vintages"),
    ("sec-vintage", "Vintage detail"),
    ("sec-gates", "Gates used"),
)

_STABILITY_FLAG_MEANING = {
    "distribution_drift": "Vintage PSI vs overall exceeds the PSI gate; the bin mix has shifted.",
    "power_loss_in_vintage": "Too many vintages have univariate Gini below the noise-adjusted floor.",
    "power_loss_median": "Median vintage Gini is below the reference Gini ratio floor.",
    "sign_flip": "Too few vintages keep the same univariate sign as the reference.",
    "low_stability_score": "The composite of drift, power persistence and sign agreement is below the gate.",
    "overlapping_event_rate_bounds": "Adjacent bins have overlapping vintage event-rate confidence bounds.",
    "insufficient_vintages": "Too few vintages to assess; this does not veto a refit.",
    "no_reference_signal": "No univariate signal on the pooled sample; not allowed to veto a refit.",
    "no_date_column": "No date column, so vintage stability was not assessed.",
    "no_grouping": "No grouping for this predictor.",
    "not_attempted": "Stability was not run for this segment.",
}


def _slug(*parts):
    # type: (Any) -> str
    text = "-".join("" if p is None else str(p) for p in parts)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text or "item")[:80]


def _finite(value):
    # type: (Any) -> bool
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _js_num(value):
    # type: (Any) -> str
    if not _finite(value):
        return "null"
    return "%.6g" % (float(value),)


def _calib_lookup(result):
    # type: (Optional[SegmentEvalResult]) -> Dict[Tuple[str, str], Dict[str, Any]]
    out = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    if result is None:
        return out
    frames = []
    if result.decisions is not None and not result.decisions.empty:
        frames.append(result.decisions)
    summary = result.segment_summary
    if summary is not None and not summary.empty:
        if "slice" in summary.columns:
            summary = summary.loc[summary["slice"].astype(str).eq("observable")]
        frames.append(summary)
    for frame in frames:
        for _idx, row in frame.iterrows():
            key = (str(row.get("segment_col")), str(row.get("segment_value")))
            rec = out.get(key, {})
            for col in ("oe", "obs_rate", "mean_pd"):
                if col in row.index and not _finite(rec.get(col)):
                    rec[col] = row.get(col)
            out[key] = rec
    return out


def _heading(anchor, title):
    # type: (str, str) -> str
    return (
        '<h2 id="%s">%s <a class="back" href="#top">back to top</a></h2>'
        % (html_safe(anchor), html_safe(title))
    )


def _toc():
    # type: () -> str
    items = "".join(
        '<li><a href="#%s">%s</a></li>' % (html_safe(anchor), html_safe(title))
        for anchor, title in _TOC
    )
    return '<nav id="top" class="toc"><strong>Contents</strong><ol>' + items + "</ol></nav>"


def frozen_html_table(frame, columns=None, empty_message="No rows.", index_cols=None):
    # type: (pd.DataFrame, Optional[Sequence[str]], str, Optional[Sequence[str]]) -> str
    """HTML table with sticky header and frozen ``segment_col`` / ``segment_value``."""
    if frame is None or frame.empty:
        return '<p class="note">' + html_safe(empty_message) + "</p>"
    cols = [c for c in (columns or list(frame.columns)) if c in frame.columns]
    if not cols:
        return '<p class="note">' + html_safe(empty_message) + "</p>"
    freeze = [c for c in (index_cols or _INDEX_COLS) if c in cols]
    numeric = dict((c, _is_numeric_column(frame[c])) for c in cols)
    head_cells = []
    for col in cols:
        klass = []
        if col == freeze[0] if freeze else False:
            klass.append("idx")
        elif len(freeze) > 1 and col == freeze[1]:
            klass.append("idx2")
        if numeric[col]:
            klass.append("num")
        attr = ' class="%s"' % (" ".join(klass),) if klass else ""
        head_cells.append("<th%s>%s</th>" % (attr, html_safe(col)))
    body = []
    for _idx, row in frame[cols].iterrows():
        cells = []
        for col in cols:
            klass = []
            if freeze and col == freeze[0]:
                klass.append("idx")
            elif len(freeze) > 1 and col == freeze[1]:
                klass.append("idx2")
            if numeric[col]:
                klass.append("num")
            attr = ' class="%s"' % (" ".join(klass),) if klass else ""
            cells.append("<td%s>%s</td>" % (attr, _cell(row[col], col)))
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table class="frozen"><thead><tr>'
        + "".join(head_cells)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _matplotlib():
    # type: () -> Any
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _fig_img(fig, alt=""):
    # type: (Any, str) -> str
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt = _matplotlib()
    if plt is not None:
        plt.close(fig)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return '<p><img class="viz" alt="%s" src="data:image/png;base64,%s" /></p>' % (
        html_safe(alt),
        payload,
    )


def _download_link(filename, content, label="scorecard.sql"):
    # type: (str, str, str) -> str
    if not content:
        return '<span class="note">-</span>'
    href = "data:text/plain;charset=utf-8," + quote(content)
    return '<a download="%s" href="%s">%s</a>' % (html_safe(filename), href, html_safe(label))


def _artifact_sql_map(result):
    # type: (SegmentEvalResult) -> Dict[Tuple[str, str], str]
    out = {}  # type: Dict[Tuple[str, str], str]
    for art in result.fitted_artifacts or []:
        if not getattr(art, "_should_write_scorecard_sql", lambda: False)():
            continue
        key = (str(art.segment_col), str(art.segment_value))
        kind = str(art.kind or "")
        if key in out and kind != "refit":
            continue
        try:
            out[key] = art.to_scorecard_sql()
        except Exception:
            continue
    return out


def _grouped_next_actions(recs, result=None):
    # type: (pd.DataFrame, Optional[SegmentEvalResult]) -> str
    if recs is None or recs.empty:
        return '<p class="note">No recommendations: every segment passed on the pooled score.</p>'
    calib = _calib_lookup(result)
    blocks = []
    grouped = recs.copy()
    grouped["_seg"] = grouped["segment_col"].astype(str)
    grouped["_act"] = grouped["action"].astype(str)
    rank = 1
    keys = []
    for (action, seg_col), part in grouped.groupby(["_act", "_seg"], sort=False):
        keys.append((int(part["priority"].min()), int(part["rank"].min()), action, seg_col, part))
    keys.sort(key=lambda item: (item[0], item[1], ACTION_PRIORITY.get(item[2], 9)))
    for priority, _r, action, seg_col, part in keys:
        values = [str(v) for v in part["segment_value"].tolist()]
        unique_vals = []
        for value in values:
            if value not in unique_vals:
                unique_vals.append(value)
        if str(seg_col) == "__overall__":
            headline = str(part.iloc[0]["recommendation"])
            why = str(part.iloc[0]["why"])
        elif action == "RECALIBRATE":
            headline = "Recalibrate the PD level for %s: %s" % (seg_col, ", ".join(unique_vals))
            why = (
                "Ranking is intact but the PD level is off on these values of %s. "
                "A two-parameter intercept/slope rescale fixes the level without a split."
                % (seg_col,)
            )
        else:
            headline = "%s — %s: %s" % (action, seg_col, ", ".join(unique_vals))
            why = str(part.iloc[0]["why"])
        evidence = " | ".join(str(e) for e in part["evidence"].tolist() if e)
        extra = ""
        if action == "RECALIBRATE":
            extra = _recalibrate_calib_table(str(seg_col), unique_vals, calib)
        blocks.append(
            '<div class="rec p%d"><div class="h">%d. %s &nbsp;%s</div>'
            '<div class="w">%s</div>'
            '<div class="e"><span class="values">%s = %s</span> &middot; %s</div>%s</div>'
            % (
                int(priority),
                rank,
                html_safe(headline),
                _badge(action, _ACTION_BADGE),
                html_safe(why),
                html_safe(seg_col),
                html_safe(", ".join(unique_vals)),
                html_safe(evidence),
                extra,
            )
        )
        rank += 1
    return "".join(blocks)


def _recalibrate_calib_table(seg_col, values, calib):
    # type: (str, Sequence[str], Dict[Tuple[str, str], Dict[str, Any]]) -> str
    rows = []
    for value in values:
        rec = calib.get((seg_col, value), {})
        rows.append(
            {
                "segment_col": seg_col,
                "segment_value": value,
                "oe": rec.get("oe"),
                "observed": _pct(rec.get("obs_rate")),
                "expected": _pct(rec.get("mean_pd")),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return ""
    return (
        '<div class="calib">O/E is observed event rate / expected mean PD.</div>'
        + frozen_html_table(
            table,
            ["segment_col", "segment_value", "oe", "observed", "expected"],
            empty_message="",
        )
    )


def _verdict_table(result, sql_map):
    # type: (SegmentEvalResult, Dict[Tuple[str, str], str]) -> str
    merged = _decisions_with_refit(result)
    if merged.empty:
        return '<p class="note">No segments evaluated.</p>'
    rows = []
    for _idx, row in merged.iterrows():
        action = str(row.get("q2_action") or "")
        key = (str(row.get("segment_col")), str(row.get("segment_value")))
        show_cmp = action in _COMPARE_ACTIONS
        sql = sql_map.get(key) if show_cmp else None
        filename = "%s_%s_scorecard.sql" % (
            _safe_path_token(row.get("segment_col")),
            _safe_path_token(row.get("segment_value")),
        )
        rec = {
            "segment_col": row.get("segment_col"),
            "segment_value": row.get("segment_value"),
            "q1_verdict": row.get("q1_verdict"),
            "q2_action": action,
            "failed_pillars": row.get("failed_pillars") or "-",
            "gini": row.get("gini"),
            "gini_pooled": row.get("gini_pooled") if show_cmp else float("nan"),
            "gini_refit": row.get("gini_refit") if show_cmp else float("nan"),
            "delta_gini": row.get("delta_gini") if show_cmp else float("nan"),
            "stability_pass": row.get("stability_pass") if show_cmp else None,
            "stability_reason": row.get("stability_reason") if show_cmp else "-",
            "scorecard.sql": _download_link(filename, sql or "", "scorecard.sql")
            if sql
            else '<span class="note">-</span>',
        }
        rows.append(rec)
    table = pd.DataFrame(rows)
    cols = [
        "segment_col",
        "segment_value",
        "q1_verdict",
        "q2_action",
        "failed_pillars",
        "gini",
        "gini_pooled",
        "gini_refit",
        "delta_gini",
        "stability_reason",
        "scorecard.sql",
    ]
    # Render badges / HTML cells by building the table ourselves so q2_action stays a badge.
    freeze = [c for c in _INDEX_COLS if c in table.columns]
    head = []
    for col in cols:
        klass = []
        if freeze and col == freeze[0]:
            klass.append("idx")
        elif len(freeze) > 1 and col == freeze[1]:
            klass.append("idx2")
        if col not in ("q1_verdict", "q2_action", "failed_pillars", "stability_reason", "scorecard.sql"):
            if col not in freeze:
                klass.append("num")
        attr = ' class="%s"' % (" ".join(klass),) if klass else ""
        head.append("<th%s>%s</th>" % (attr, html_safe(col)))
    body = []
    for _idx, row in table.iterrows():
        cells = []
        for col in cols:
            klass = []
            if freeze and col == freeze[0]:
                klass.append("idx")
            elif len(freeze) > 1 and col == freeze[1]:
                klass.append("idx2")
            if col == "q1_verdict":
                cells.append("<td%s>%s</td>" % (' class="%s"' % (" ".join(klass),) if klass else "", _badge(row[col], _VERDICT_BADGE)))
            elif col == "q2_action":
                cells.append("<td%s>%s</td>" % (' class="%s"' % (" ".join(klass),) if klass else "", _badge(row[col], _ACTION_BADGE)))
            elif col == "scorecard.sql":
                cells.append("<td%s>%s</td>" % (' class="%s"' % (" ".join(klass),) if klass else "", row[col]))
            else:
                if col not in ("failed_pillars", "stability_reason") and col not in freeze:
                    klass.append("num")
                attr = ' class="%s"' % (" ".join(klass),) if klass else ""
                cells.append("<td%s>%s</td>" % (attr, _cell(row[col], col) if col != "stability_reason" else html_safe(row[col] or "-")))
        body.append("<tr>" + "".join(cells) + "</tr>")
    note = (
        '<p class="note"><strong>KEEP_POOLED</strong>, <strong>MONITOR</strong> and '
        "<strong>SPLIT</strong> show holdout <code>gini_refit</code> vs <code>gini_pooled</code> "
        "and the stability reason. The last column downloads the refit <code>scorecard.sql</code> "
        "when a logistic artefact exists.</p>"
    )
    return note + (
        '<div class="scroll"><table class="frozen"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _matched_ar_illustration():
    # type: () -> str
    svg = """
<svg id="ar-illustration" viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Matched approval-rate cutoff simulation">
  <text x="12" y="18" font-size="13" font-weight="600" fill="#1f2328">How the matched-AR cutoff is simulated</text>
  <text x="12" y="38" font-size="11" fill="#57606a">1. Score (PD) on the x-axis. Approve when score &le; cutoff. Click a row to load that segment's numbers.</text>
  <rect x="20" y="55" width="220" height="80" fill="#ddf4ff" stroke="#0969da"/>
  <text x="30" y="74" font-size="11">Portfolio scores</text>
  <line id="ar-cut-port" x1="180" y1="55" x2="180" y2="135" stroke="#b42318" stroke-width="2"/>
  <text id="ar-label-port-cut" x="24" y="148" font-size="10" fill="#b42318">portfolio cutoff = (click a row)</text>
  <text id="ar-label-ar-ref" x="30" y="164" font-size="10">AR_ref</text>
  <rect x="260" y="55" width="220" height="80" fill="#fff8c5" stroke="#9a6700"/>
  <text x="270" y="74" font-size="11">Segment scores</text>
  <line id="ar-cut-seg" x1="420" y1="55" x2="420" y2="135" stroke="#b42318" stroke-width="2"/>
  <text id="ar-label-seg-same" x="264" y="148" font-size="10" fill="#b42318">same cutoff</text>
  <text id="ar-label-ar-seg" x="270" y="164" font-size="10">AR_seg at that cutoff</text>
  <line id="ar-cut-matched-port" x1="180" y1="55" x2="180" y2="135" stroke="#1a7f37" stroke-width="2" stroke-dasharray="5 3" opacity="0"/>
  <line id="ar-cut-matched-seg" x1="420" y1="55" x2="420" y2="135" stroke="#1a7f37" stroke-width="2" stroke-dasharray="5 3" opacity="0"/>
  <text x="500" y="70" font-size="11">2. If |AR_seg - AR_ref| &ge; 10%</text>
  <text x="500" y="88" font-size="11">anchor = min(AR_seg, AR_ref).</text>
  <text x="500" y="106" font-size="11">3. Re-derive a threshold inside</text>
  <text x="500" y="122" font-size="11">each group that hits that AR.</text>
  <text id="ar-label-matched" x="500" y="148" font-size="11" fill="#1a7f37">matched cutoffs appear as dashed lines</text>
  <text id="ar-label-matched2" x="500" y="164" font-size="11" fill="#1a7f37"></text>
  <text id="ar-label-row" x="12" y="198" font-size="11" fill="#57606a">No row selected yet. Click a 10% gap row below.</text>
  <text x="12" y="230" font-size="10" fill="#57606a">Solid red = portfolio cutoff applied to both sides. Dashed green = matched-AR cutoffs (one per group).</text>
</svg>
"""
    script = """
<script>
(function () {
  var BOX_PORT = 20, BOX_SEG = 260, BOX_W = 220;
  function clamp01(v) {
    if (v === null || v === undefined || isNaN(v)) return 0.5;
    return Math.max(0, Math.min(1, v));
  }
  function xAt(left, cutoff) { return left + clamp01(cutoff) * BOX_W; }
  function fmt(v, digits) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    var n = Number(v);
    if (digits === "%") return (100 * n).toFixed(1) + "%";
    return n.toFixed(digits || 3);
  }
  function setLine(id, left, cutoff, show) {
    var el = document.getElementById(id);
    if (!el) return;
    var x = xAt(left, cutoff);
    el.setAttribute("x1", x);
    el.setAttribute("x2", x);
    el.setAttribute("opacity", show ? "1" : "0");
  }
  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }
  function applyRow(tr) {
    var rows = document.querySelectorAll("#ar-table tr.clickable");
    for (var i = 0; i < rows.length; i++) rows[i].className = rows[i].className.replace(" selected", "");
    tr.className = tr.className + " selected";
    var cut = parseFloat(tr.getAttribute("data-cutoff"));
    var cutSeg = parseFloat(tr.getAttribute("data-cutoff-seg"));
    var cutRef = parseFloat(tr.getAttribute("data-cutoff-ref-matched"));
    var arSeg = parseFloat(tr.getAttribute("data-ar-seg"));
    var arRef = parseFloat(tr.getAttribute("data-ar-ref"));
    var matched = parseFloat(tr.getAttribute("data-matched-ar"));
    var label = tr.getAttribute("data-label") || "";
    setLine("ar-cut-port", BOX_PORT, cut, true);
    setLine("ar-cut-seg", BOX_SEG, cut, true);
    setLine("ar-cut-matched-port", BOX_PORT, cutRef, !isNaN(cutRef));
    setLine("ar-cut-matched-seg", BOX_SEG, cutSeg, !isNaN(cutSeg));
    setText("ar-label-port-cut", "portfolio cutoff = " + fmt(cut, 4));
    setText("ar-label-seg-same", "same cutoff = " + fmt(cut, 4));
    setText("ar-label-ar-ref", "AR_ref = " + fmt(arRef, "%"));
    setText("ar-label-ar-seg", "AR_seg at that cutoff = " + fmt(arSeg, "%"));
    setText("ar-label-matched", "matched AR " + fmt(matched, "%"));
    setText("ar-label-matched2", "cutoff_seg " + fmt(cutSeg, 4) + "   cutoff_ref " + fmt(cutRef, 4));
    setText("ar-label-row", "Showing cutoffs for " + label);
  }
  function bind() {
    var rows = document.querySelectorAll("#ar-table tr.clickable");
    for (var i = 0; i < rows.length; i++) {
      rows[i].addEventListener("click", function (ev) { applyRow(ev.currentTarget); });
    }
    if (rows.length) applyRow(rows[0]);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
</script>
"""
    return '<div class="illus">' + svg + "</div>" + script


def _matched_ar_section(result, gates):
    # type: (SegmentEvalResult, Gates) -> str
    intro = (
        '<p class="note">A segment scored on a different part of the risk spectrum cannot be '
        "compared to the portfolio at face value. When the approval-rate gap reaches 10%, "
        "both sides are re-cut to the lower-AR anchor and Gini is recomputed there. "
        "Click a row to move the cutoff lines to that segment's numbers.</p>"
    )
    decisions = result.decisions
    if decisions is None or decisions.empty or "ar_gap" not in decisions.columns:
        return intro + _matched_ar_illustration() + '<p class="note">Not computed.</p>'
    gap = pd.to_numeric(decisions["ar_gap"], errors="coerce").abs()
    triggered = decisions.loc[gap.fillna(0.0) >= _AR_GAP_FLOOR]
    cols = [
        "segment_col",
        "segment_value",
        "ar_reference_cutoff",
        "ar_segment",
        "ar_reference",
        "ar_gap",
        "matched_ar",
        "matched_ar_anchor",
        "matched_ar_threshold_segment",
        "matched_ar_threshold_reference",
        "gini",
        "gini_at_matched_ar",
        "gini_reference_at_matched_ar",
        "gini_at_matched_ar_gap",
        "ar_artifact_suspected",
    ]
    if triggered.empty:
        return (
            intro
            + _matched_ar_illustration()
            + '<p class="note">No segment has an approval-rate gap of 10% or more.</p>'
        )
    freeze = [c for c in _INDEX_COLS if c in triggered.columns]
    show = [c for c in cols if c in triggered.columns]
    numeric = dict((c, _is_numeric_column(triggered[c])) for c in show)
    head = []
    for col in show:
        klass = []
        if freeze and col == freeze[0]:
            klass.append("idx")
        elif len(freeze) > 1 and col == freeze[1]:
            klass.append("idx2")
        if numeric.get(col):
            klass.append("num")
        attr = ' class="%s"' % (" ".join(klass),) if klass else ""
        head.append("<th%s>%s</th>" % (attr, html_safe(col)))
    body = []
    for _idx, row in triggered.iterrows():
        label = "%s = %s" % (row.get("segment_col"), row.get("segment_value"))
        attrs = (
            ' class="clickable" data-label="%s" data-cutoff="%s" data-cutoff-seg="%s" '
            'data-cutoff-ref-matched="%s" data-ar-seg="%s" data-ar-ref="%s" data-matched-ar="%s"'
            % (
                html_safe(label),
                _js_num(row.get("ar_reference_cutoff")),
                _js_num(row.get("matched_ar_threshold_segment")),
                _js_num(row.get("matched_ar_threshold_reference")),
                _js_num(row.get("ar_segment")),
                _js_num(row.get("ar_reference")),
                _js_num(row.get("matched_ar")),
            )
        )
        cells = []
        for col in show:
            klass = []
            if freeze and col == freeze[0]:
                klass.append("idx")
            elif len(freeze) > 1 and col == freeze[1]:
                klass.append("idx2")
            if numeric.get(col):
                klass.append("num")
            attr = ' class="%s"' % (" ".join(klass),) if klass else ""
            cells.append("<td%s>%s</td>" % (attr, _cell(row[col], col)))
        body.append("<tr%s>%s</tr>" % (attrs, "".join(cells)))
    table = (
        '<div class="scroll"><table id="ar-table" class="frozen"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )
    return intro + _matched_ar_illustration() + table


def _psi_link(row):
    # type: (pd.Series) -> str
    anchor = "psi-" + _slug(row.get("segment_col"), row.get("segment_value"), row.get("feature"))
    return '<a href="#%s">view</a>' % (html_safe(anchor),)


def _plot_share_bars(bin_table, title):
    # type: (Sequence[Dict[str, Any]], str) -> str
    plt = _matplotlib()
    if plt is None or not bin_table:
        return ""
    labels = [str(r.get("bin")) for r in bin_table]
    seg = [float(r.get("share_segment") or 0.0) for r in bin_table]
    ref = [float(r.get("share_reference") or 0.0) for r in bin_table]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels) + 2), 3.2))
    ax.bar(x - 0.2, ref, width=0.4, label="overall")
    ax.bar(x + 0.2, seg, width=0.4, label="segment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("share")
    ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return _fig_img(fig, title)


def _characteristics_section(result, gates):
    # type: (SegmentEvalResult, Gates) -> str
    char = result.characteristics if result.characteristics is not None else pd.DataFrame()
    intro = (
        '<p class="note">Only characteristics with PSI &gt; 0.25 or a WoE rank reversal '
        "are listed. The comparison plot is segment vs overall bin share, including "
        "<code>__missing__</code> when it is present.</p>"
    )
    if char.empty:
        return intro + '<p class="note">No characteristic diagnostics available.</p>'
    psi = pd.to_numeric(char["psi"], errors="coerce") if "psi" in char.columns else pd.Series(dtype=float)
    reversal = char["rank_reversal"].fillna(False).astype(bool) if "rank_reversal" in char.columns else False
    keep = char.loc[(psi.fillna(0.0) > _PSI_FLOOR) | reversal]
    if keep.empty:
        return intro + '<p class="note">No characteristic exceeded PSI 0.25 or reversed rank.</p>'
    display = keep.copy()
    display["comparison"] = [_psi_link(row) for _i, row in display.iterrows()]
    cols = [
        "segment_col",
        "segment_value",
        "feature",
        "psi",
        "psi_method",
        "psi_worst_bin",
        "rank_reversal",
        "comparison",
    ]
    # comparison is already HTML; build table then append figures
    freeze = [c for c in _INDEX_COLS if c in display.columns]
    head = []
    for col in cols:
        klass = []
        if freeze and col == freeze[0]:
            klass.append("idx")
        elif len(freeze) > 1 and col == freeze[1]:
            klass.append("idx2")
        if col == "psi":
            klass.append("num")
        attr = ' class="%s"' % (" ".join(klass),) if klass else ""
        head.append("<th%s>%s</th>" % (attr, html_safe(col)))
    body = []
    figures = []
    for _idx, row in display.iterrows():
        cells = []
        for col in cols:
            klass = []
            if freeze and col == freeze[0]:
                klass.append("idx")
            elif len(freeze) > 1 and col == freeze[1]:
                klass.append("idx2")
            if col == "psi":
                klass.append("num")
            attr = ' class="%s"' % (" ".join(klass),) if klass else ""
            if col == "comparison":
                cells.append("<td%s>%s</td>" % (attr, row[col]))
            else:
                cells.append("<td%s>%s</td>" % (attr, _cell(row[col], col)))
        body.append("<tr>" + "".join(cells) + "</tr>")
        bins = row.get("psi_bin_table") if "psi_bin_table" in display.columns else None
        if not bins:
            bins = []
        anchor = "psi-" + _slug(row.get("segment_col"), row.get("segment_value"), row.get("feature"))
        title = "%s = %s / %s" % (row.get("segment_col"), row.get("segment_value"), row.get("feature"))
        figures.append(
            '<h3 id="%s">%s</h3>' % (html_safe(anchor), html_safe(title))
            + (_plot_share_bars(bins, title) if bins else '<p class="note">No bin shares to plot.</p>')
        )
    table = (
        '<div class="scroll"><table class="frozen"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )
    return intro + table + "".join(figures)


def _grouping_link(row):
    # type: (pd.Series) -> str
    anchor = "woe-" + _slug(row.get("segment_col"), row.get("segment_value"), row.get("feature"))
    return '<a href="#%s">view</a>' % (html_safe(anchor),)


def _plot_woe_bars(part, title):
    # type: (pd.DataFrame, str) -> str
    plt = _matplotlib()
    if plt is None or part.empty:
        return ""
    labels = [str(v) for v in part["segment_bin"].tolist()]
    seg = pd.to_numeric(part["woe_segment"], errors="coerce").fillna(0.0).tolist()
    port = pd.to_numeric(part["woe_portfolio"], errors="coerce").fillna(0.0).tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels) + 2), 3.2))
    ax.bar(x - 0.2, port, width=0.4, label="portfolio WoE")
    ax.bar(x + 0.2, seg, width=0.4, label="segment WoE")
    ax.axhline(0.0, color="#57606a", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("WoE")
    ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return _fig_img(fig, title)


def _grouping_section(result):
    # type: (SegmentEvalResult) -> str
    cmp_ = getattr(result, "grouping_comparison", None)
    intro = (
        '<p class="note">Only bins whose grouping comparison is not <code>aligned</code> and '
        "is marked significant are shown. The plot is segment WoE vs portfolio WoE for that "
        "<code>(segment_col, segment_value)</code> feature.</p>"
    )
    if cmp_ is None or cmp_.empty:
        return intro + '<p class="note">No grouping comparison was produced.</p>'
    kind = cmp_["kind"].astype(str) if "kind" in cmp_.columns else pd.Series([""] * len(cmp_))
    if "significant" in cmp_.columns:
        sig = cmp_["significant"].map(lambda v: str(v).lower() in ("true", "yes", "1"))
    else:
        sig = pd.Series([False] * len(cmp_))
    keep = cmp_.loc[(kind != "aligned") & sig]
    if keep.empty:
        return intro + '<p class="note">Every compared bin is aligned; nothing significant to show.</p>'
    display = keep.copy()
    if "portfolio_bins" in display.columns and "portfolio_bin" not in display.columns:
        display["portfolio_bin"] = display["portfolio_bins"]
    display["comparison"] = [_grouping_link(row) for _i, row in display.iterrows()]
    cols = [
        "segment_col",
        "segment_value",
        "feature",
        "segment_bin",
        "portfolio_bin",
        "kind",
        "significant",
        "woe_segment",
        "woe_portfolio",
        "woe_delta",
        "comparison",
    ]
    freeze = [c for c in _INDEX_COLS if c in display.columns]
    head = []
    for col in cols:
        klass = []
        if freeze and col == freeze[0]:
            klass.append("idx")
        elif len(freeze) > 1 and col == freeze[1]:
            klass.append("idx2")
        if col in ("woe_segment", "woe_portfolio", "woe_delta"):
            klass.append("num")
        attr = ' class="%s"' % (" ".join(klass),) if klass else ""
        head.append("<th%s>%s</th>" % (attr, html_safe(col)))
    body = []
    for _idx, row in display.iterrows():
        cells = []
        for col in cols:
            klass = []
            if freeze and col == freeze[0]:
                klass.append("idx")
            elif len(freeze) > 1 and col == freeze[1]:
                klass.append("idx2")
            if col in ("woe_segment", "woe_portfolio", "woe_delta"):
                klass.append("num")
            attr = ' class="%s"' % (" ".join(klass),) if klass else ""
            if col == "comparison":
                cells.append("<td%s>%s</td>" % (attr, row[col]))
            else:
                cells.append("<td%s>%s</td>" % (attr, _cell(row[col], col)))
        body.append("<tr>" + "".join(cells) + "</tr>")
    table = (
        '<div class="scroll"><table class="frozen"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )
    figures = []
    keys = display.drop_duplicates(["segment_col", "segment_value", "feature"])
    for _idx, row in keys.iterrows():
        mask = (
            (cmp_["segment_col"].astype(str) == str(row["segment_col"]))
            & (cmp_["segment_value"].astype(str) == str(row["segment_value"]))
            & (cmp_["feature"].astype(str) == str(row["feature"]))
        )
        part = cmp_.loc[mask]
        anchor = "woe-" + _slug(row.get("segment_col"), row.get("segment_value"), row.get("feature"))
        title = "%s = %s / %s" % (row.get("segment_col"), row.get("segment_value"), row.get("feature"))
        figures.append(
            '<h3 id="%s">%s</h3>' % (html_safe(anchor), html_safe(title)) + _plot_woe_bars(part, title)
        )
    return intro + table + "".join(figures)


def _stability_flag_viz(row, gates):
    # type: (pd.Series, Gates) -> str
    raw = str(row.get("stability_flags") or "")
    flags = [f for f in raw.split(",") if f]
    if not flags:
        meaning = "No stability flags: the predictor cleared every vintage gate."
        return '<div class="flag-viz"><span class="note">stable</span><div class="flag-mean">%s</div></div>' % (
            html_safe(meaning),
        )
    meaning = " ".join(_STABILITY_FLAG_MEANING.get(f, f) for f in flags)
    meters = [
        ("PSI", row.get("psi_max"), gates.stability_psi_max, True, "distribution_drift" in flags),
        (
            "weak vint.",
            row.get("weak_vintage_share"),
            gates.stability_max_weak_vintage_share,
            True,
            "power_loss_in_vintage" in flags,
        ),
        (
            "Gini ratio",
            row.get("gini_ratio_median"),
            gates.stability_gini_ratio_min,
            False,
            "power_loss_median" in flags,
        ),
        (
            "sign",
            row.get("sign_consistency"),
            gates.stability_sign_consistency_min,
            False,
            "sign_flip" in flags,
        ),
        (
            "score",
            row.get("stability_score"),
            gates.stability_score_min,
            False,
            "low_stability_score" in flags,
        ),
    ]
    height = 16 * len(meters) + 8
    parts = [
        '<svg width="240" height="%d" viewBox="0 0 240 %d" xmlns="http://www.w3.org/2000/svg">'
        % (height, height)
    ]
    for i, item in enumerate(meters):
        label, value, gate, higher_worse, fired = item
        y = 4 + i * 16
        color = "#b42318" if fired else "#57606a"
        bar_color = "#b42318" if fired else "#0969da"
        val = float(value) if _finite(value) else 0.0
        gate_f = float(gate) if _finite(gate) else 0.0
        scale = max(val, gate_f, 1e-6)
        if not higher_worse:
            scale = max(scale, 1.0)
        bar_w = 90.0 * min(val / scale, 1.0)
        gate_x = 70.0 + 90.0 * min(gate_f / scale, 1.0)
        parts.append(
            '<text x="0" y="%d" font-size="9" fill="%s">%s</text>'
            % (y + 10, color, html_safe(label))
        )
        parts.append(
            '<rect x="70" y="%d" width="90" height="8" fill="#eef1f4" rx="1"/>' % (y + 3,)
        )
        parts.append(
            '<rect x="70" y="%d" width="%.1f" height="8" fill="%s" rx="1"/>'
            % (y + 3, bar_w, bar_color)
        )
        parts.append(
            '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#1f2328" stroke-width="1"/>'
            % (gate_x, y + 2, gate_x, y + 12)
        )
        shown = "-" if not _finite(value) else ("%.2f" % float(value))
        parts.append(
            '<text x="166" y="%d" font-size="9" fill="%s">%s</text>'
            % (y + 10, color, html_safe(shown))
        )
    parts.append("</svg>")
    flag_list = ", ".join(flags)
    return (
        '<div class="flag-viz">%s<div class="flag-mean"><strong>%s</strong> — %s</div></div>'
        % ("".join(parts), html_safe(flag_list), html_safe(meaning))
    )


def _stability_section(result, gates=None):
    # type: (SegmentEvalResult, Optional[Gates]) -> str
    gates = gates or _gates_from_result(result)
    stability = result.stability if result.stability is not None else pd.DataFrame()
    intro = (
        '<p class="note">Logit / VAL / LIN predictors are assessed on portfolio-quantile bins '
        "(segment vs overall), with <code>__missing__</code> kept as its own category. Ordinary "
        "WoE predictors still use the grouping bins. The flag column draws each metric against "
        "its gate (black tick); a red bar is a fired flag.</p>"
    )
    if stability is None or stability.empty:
        return intro + '<p class="note">No stability diagnostics available.</p>'
    display = stability.copy()
    display["flag_meaning"] = [_stability_flag_viz(row, gates) for _i, row in display.iterrows()]
    cols = [
        "segment_col",
        "segment_value",
        "feature",
        "stable",
        "stability_flags",
        "flag_meaning",
        "n_vintages",
        "assessed",
        "psi_max",
        "gini_reference",
        "sign_consistency",
        "stability_score",
        "stability_binning",
    ]
    freeze = [c for c in _INDEX_COLS if c in display.columns]
    show = [c for c in cols if c in display.columns]
    head = []
    for col in show:
        klass = []
        if freeze and col == freeze[0]:
            klass.append("idx")
        elif len(freeze) > 1 and col == freeze[1]:
            klass.append("idx2")
        attr = ' class="%s"' % (" ".join(klass),) if klass else ""
        head.append("<th%s>%s</th>" % (attr, html_safe(col)))
    body = []
    for _idx, row in display.iterrows():
        cells = []
        for col in show:
            klass = []
            if freeze and col == freeze[0]:
                klass.append("idx")
            elif len(freeze) > 1 and col == freeze[1]:
                klass.append("idx2")
            attr = ' class="%s"' % (" ".join(klass),) if klass else ""
            if col == "flag_meaning":
                cells.append("<td%s>%s</td>" % (attr, row[col]))
            else:
                cells.append("<td%s>%s</td>" % (attr, _cell(row[col], col)))
        body.append("<tr>" + "".join(cells) + "</tr>")
    return intro + (
        '<div class="scroll"><table class="frozen"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _portfolio_rate_lookup(vintage):
    # type: (pd.DataFrame) -> Dict[str, float]
    if vintage is None or vintage.empty:
        return {}
    overall = vintage.loc[vintage["segment_col"].astype(str).eq("__overall__")]
    if overall.empty:
        return {}
    out = {}
    for _idx, row in overall.iterrows():
        n = float(row.get("n") or 0.0)
        defaults = float(row.get("defaults") or 0.0)
        out[str(row.get("vintage"))] = defaults / n if n else float("nan")
    return out


def _plot_vintage_col(part, segment_col, portfolio_rates):
    # type: (pd.DataFrame, str, Dict[str, float]) -> str
    plt = _matplotlib()
    if plt is None:
        return '<p class="note">Vintage plot requires matplotlib (install the plot extra).</p>'
    vintages = []
    for value in part["vintage"].tolist():
        text = str(value)
        if text not in vintages:
            vintages.append(text)
    hues = []
    for value in part["segment_value"].astype(str).tolist():
        if value not in hues:
            hues.append(value)
    x = np.arange(len(vintages))
    width = 0.8 / max(len(hues), 1)
    fig, (ax_rate, ax_gini) = plt.subplots(
        2, 1, sharex=True, figsize=(max(8.0, 0.42 * len(vintages) + 3.5), 7.2)
    )
    ax_rate_r = ax_rate.twinx()
    ax_gini_r = ax_gini.twinx()
    cmap = plt.get_cmap("tab10")
    for i, sval in enumerate(hues):
        sub = part.loc[part["segment_value"].astype(str).eq(sval)]
        by_v = dict((str(r["vintage"]), r) for _, r in sub.iterrows())
        ns, rates, ginis = [], [], []
        for v in vintages:
            row = by_v.get(v)
            n = float(row["n"]) if row is not None else 0.0
            defaults = float(row["defaults"]) if row is not None else 0.0
            ns.append(n)
            rates.append(defaults / n if n else float("nan"))
            ginis.append(float(row["gini"]) if row is not None else float("nan"))
        color = cmap(i % 10)
        offset = (i - (len(hues) - 1) / 2.0) * width
        ax_rate.bar(x + offset, ns, width=width * 0.9, color=color, alpha=0.28, label="%s n" % sval)
        ax_gini.bar(x + offset, ns, width=width * 0.9, color=color, alpha=0.28, label="%s n" % sval)
        ax_rate_r.plot(x, rates, marker="o", color=color, label="%s event rate" % sval)
        ax_gini_r.plot(x, ginis, marker="s", linestyle="--", color=color, label="%s gini" % sval)
    if portfolio_rates:
        pr = [portfolio_rates.get(v, float("nan")) for v in vintages]
        ax_rate_r.plot(
            x,
            pr,
            color="black",
            linewidth=2.4,
            linestyle=(0, (5, 2, 1, 2)),
            label="portfolio event rate",
        )
    ax_rate.set_ylabel("n")
    ax_rate_r.set_ylabel("event rate")
    ax_gini.set_ylabel("n")
    ax_gini_r.set_ylabel("Gini")
    ax_rate.set_title("%s — volume and event rate" % segment_col)
    ax_gini.set_title("%s — volume and Gini" % segment_col)
    ax_gini.set_xticks(x)
    ax_gini.set_xticklabels(vintages, rotation=45, ha="right")
    for ax, ax_r in ((ax_rate, ax_rate_r), (ax_gini, ax_gini_r)):
        handles, labels = ax.get_legend_handles_labels()
        h2, lab2 = ax_r.get_legend_handles_labels()
        ax.legend(handles + h2, labels + lab2, fontsize="small", ncol=2, loc="upper left")
    fig.tight_layout()
    return _fig_img(fig, "vintage %s" % segment_col)


def _vintage_section(result):
    # type: (SegmentEvalResult) -> str
    vintage = result.vintage if result.vintage is not None else pd.DataFrame()
    intro = (
        '<p class="note">One chart per <code>segment_col</code>, hue = <code>segment_value</code>. '
        "Each panel keeps the volume bars. The top panel is event rate (the black striped "
        "line is the portfolio event rate); the bottom panel is Gini.</p>"
    )
    if vintage.empty:
        return intro + '<p class="note">No vintage table available.</p>'
    portfolio_rates = _portfolio_rate_lookup(vintage)
    parts = []
    cols = [c for c in vintage["segment_col"].astype(str).unique() if c != "__overall__"]
    if not cols:
        cols = list(vintage["segment_col"].astype(str).unique())
    for col in cols:
        part = vintage.loc[vintage["segment_col"].astype(str).eq(col)]
        if part.empty:
            continue
        parts.append("<h3>%s</h3>" % html_safe(col))
        parts.append(_plot_vintage_col(part, col, portfolio_rates))
    return intro + "".join(parts)


def render_condensed_html_report(result, title="Segment scorecard evaluation (condensed)", gates=None, recs=None):
    # type: (SegmentEvalResult, str, Optional[Gates], Optional[pd.DataFrame]) -> str
    """Self-contained condensed HTML report (inline CSS and figures)."""
    gates = gates or _gates_from_result(result)
    recs = recommendations(result, gates) if recs is None else recs
    meta = result.meta or {}
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sql_map = _artifact_sql_map(result)

    parts = []  # type: List[str]
    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>" + html_safe(title) + "</title><style>" + _CSS + "</style></head><body>")
    parts.append("<div class='wrap'>")
    parts.append("<h1>" + html_safe(title) + "</h1>")
    parts.append(
        '<div class="sub">Generated %s &middot; %s rows (%s observable) &middot; condensed view</div>'
        % (
            html_safe(generated),
            "{:,}".format(int(meta.get("n_rows", 0))),
            "{:,}".format(int(meta.get("n_observable", 0))),
        )
    )
    parts.append(_kpis(result))
    parts.append(_toc())

    parts.append(_heading("sec-next", "1. What to do next"))
    parts.append(
        '<p class="note">RECALIBRATE (and other triggered actions) are grouped by '
        "<code>segment_col</code>, listing every triggered <code>segment_value</code>.</p>"
    )
    parts.append(_grouped_next_actions(recs, result))

    parts.append(_heading("sec-verdicts", "2. Per-segment verdicts"))
    parts.append(_verdict_table(result, sql_map))

    parts.append(_heading("sec-ar", "3. Like-for-like comparison at matched approval rate"))
    parts.append(_matched_ar_section(result, gates))

    parts.append(_heading("sec-psi", "4. Characteristic drift"))
    parts.append(_characteristics_section(result, gates))

    parts.append(_heading("sec-grouping", "5. Segment vs portfolio grouping"))
    parts.append(_grouping_section(result))

    parts.append(_heading("sec-stability", "6. Predictor stability over vintages"))
    parts.append(_stability_section(result, gates))

    parts.append(_heading("sec-vintage", "7. Vintage detail"))
    parts.append(_vintage_section(result))

    parts.append(_heading("sec-gates", "8. Gates used"))
    parts.append('<div class="card">' + _gates_block(gates) + "</div>")

    parts.append("</div></body></html>")
    return "".join(parts)
