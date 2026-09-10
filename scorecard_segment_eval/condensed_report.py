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
    _gates_from_result,
    _is_numeric_column,
    _kpis,
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
"""

_TOC = (
    ("sec-next", "What to do next"),
    ("sec-verdicts", "Per-segment verdicts"),
    ("sec-ar", "Like-for-like comparison at matched approval rate"),
    ("sec-psi", "Characteristic drift"),
    ("sec-grouping", "Segment vs portfolio grouping"),
    ("sec-stability", "Predictor stability over vintages"),
    ("sec-vintage", "Vintage detail"),
)


def _slug(*parts):
    # type: (Any) -> str
    text = "-".join("" if p is None else str(p) for p in parts)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text or "item")[:80]


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


def _grouped_next_actions(recs):
    # type: (pd.DataFrame) -> str
    if recs is None or recs.empty:
        return '<p class="note">No recommendations: every segment passed on the pooled score.</p>'
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
        blocks.append(
            '<div class="rec p%d"><div class="h">%d. %s &nbsp;%s</div>'
            '<div class="w">%s</div>'
            '<div class="e"><span class="values">%s = %s</span> &middot; %s</div></div>'
            % (
                int(priority),
                rank,
                html_safe(headline),
                _badge(action, _ACTION_BADGE),
                html_safe(why),
                html_safe(seg_col),
                html_safe(", ".join(unique_vals)),
                html_safe(evidence),
            )
        )
        rank += 1
    return "".join(blocks)


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
<svg viewBox="0 0 720 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Matched approval-rate cutoff simulation">
  <text x="12" y="18" font-size="13" font-weight="600" fill="#1f2328">How the matched-AR cutoff is simulated</text>
  <text x="12" y="40" font-size="11" fill="#57606a">1. Score (PD) on the x-axis. Approve when score &le; cutoff.</text>
  <rect x="20" y="55" width="200" height="70" fill="#ddf4ff" stroke="#0969da"/>
  <text x="30" y="75" font-size="11">Portfolio scores</text>
  <line x1="160" y1="55" x2="160" y2="125" stroke="#b42318" stroke-width="2"/>
  <text x="128" y="140" font-size="10" fill="#b42318">portfolio cutoff</text>
  <text x="30" y="148" font-size="10">AR_ref (e.g. 85%)</text>
  <rect x="260" y="55" width="200" height="70" fill="#fff8c5" stroke="#9a6700"/>
  <text x="270" y="75" font-size="11">Segment scores (shifted)</text>
  <line x1="400" y1="55" x2="400" y2="125" stroke="#b42318" stroke-width="2"/>
  <text x="270" y="148" font-size="10">AR_seg at the same cutoff (e.g. 55%)</text>
  <text x="480" y="70" font-size="11">2. If |AR_seg - AR_ref| &ge; 10%</text>
  <text x="480" y="88" font-size="11">anchor = min(AR_seg, AR_ref).</text>
  <text x="480" y="106" font-size="11">3. Re-derive a threshold inside</text>
  <text x="480" y="122" font-size="11">each group that hits that AR,</text>
  <text x="480" y="138" font-size="11">keep approved rows, recompute Gini.</text>
  <line x1="20" y1="175" x2="700" y2="175" stroke="#d8dee4"/>
  <text x="12" y="198" font-size="11" fill="#57606a">Only rows with |approval-rate gap| of at least 10% are shown below.</text>
</svg>
"""
    return '<div class="illus">' + svg + "</div>"


def _matched_ar_section(result, gates):
    # type: (SegmentEvalResult, Gates) -> str
    intro = (
        '<p class="note">A segment scored on a different part of the risk spectrum cannot be '
        "compared to the portfolio at face value. When the approval-rate gap reaches 10%, "
        "both sides are re-cut to the lower-AR anchor and Gini is recomputed there.</p>"
    )
    decisions = result.decisions
    if decisions is None or decisions.empty or "ar_gap" not in decisions.columns:
        return intro + _matched_ar_illustration() + '<p class="note">Not computed.</p>'
    gap = pd.to_numeric(decisions["ar_gap"], errors="coerce").abs()
    triggered = decisions.loc[gap.fillna(0.0) >= _AR_GAP_FLOOR]
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
        "ar_artifact_suspected",
    ]
    return (
        intro
        + _matched_ar_illustration()
        + frozen_html_table(
            triggered,
            cols,
            empty_message="No segment has an approval-rate gap of 10% or more.",
        )
    )


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
    display["comparison"] = [_grouping_link(row) for _i, row in display.iterrows()]
    cols = [
        "segment_col",
        "segment_value",
        "feature",
        "segment_bin",
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


def _stability_section(result):
    # type: (SegmentEvalResult) -> str
    stability = result.stability if result.stability is not None else pd.DataFrame()
    intro = (
        '<p class="note">Logit / VAL / LIN predictors are assessed on portfolio-quantile bins '
        "(segment vs overall), with <code>__missing__</code> kept as its own category. Ordinary "
        "WoE predictors still use the grouping bins. <code>stability_binning</code> records "
        "which ruler was used.</p>"
    )
    cols = [
        "segment_col",
        "segment_value",
        "feature",
        "n_vintages",
        "assessed",
        "psi_max",
        "gini_reference",
        "sign_consistency",
        "stability_score",
        "stable",
        "stability_flags",
        "stability_binning",
    ]
    return intro + frozen_html_table(
        stability, cols, empty_message="No stability diagnostics available."
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
    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(vintages) + 3.5), 4.4))
    ax2 = ax.twinx()
    cmap = plt.get_cmap("tab10")
    for i, sval in enumerate(hues):
        sub = part.loc[part["segment_value"].astype(str).eq(sval)]
        by_v = dict((str(r["vintage"]), r) for _, r in sub.iterrows())
        ns, rates, ginis, oes = [], [], [], []
        for v in vintages:
            row = by_v.get(v)
            n = float(row["n"]) if row is not None else 0.0
            defaults = float(row["defaults"]) if row is not None else 0.0
            ns.append(n)
            rates.append(defaults / n if n else float("nan"))
            ginis.append(float(row["gini"]) if row is not None else float("nan"))
            oes.append(float(row["oe"]) if row is not None else float("nan"))
        color = cmap(i % 10)
        offset = (i - (len(hues) - 1) / 2.0) * width
        ax.bar(x + offset, ns, width=width * 0.9, color=color, alpha=0.35, label="%s n" % sval)
        ax2.plot(x, rates, marker="o", color=color, label="%s default" % sval)
        ax2.plot(x, ginis, marker="s", linestyle="--", color=color, label="%s gini" % sval)
        ax2.plot(x, oes, marker="^", linestyle=":", color=color, label="%s oe" % sval)
    if portfolio_rates:
        pr = [portfolio_rates.get(v, float("nan")) for v in vintages]
        ax2.plot(
            x,
            pr,
            color="black",
            linewidth=2.4,
            linestyle=(0, (5, 2, 1, 2)),
            label="portfolio event rate",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(vintages, rotation=45, ha="right")
    ax.set_ylabel("n")
    ax2.set_ylabel("default / gini / O/E")
    ax.set_title("%s — vintage n, default, Gini, O/E" % segment_col)
    handles, labels = ax.get_legend_handles_labels()
    h2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + lab2, fontsize="small", ncol=2, loc="upper left")
    fig.tight_layout()
    return _fig_img(fig, "vintage %s" % segment_col)


def _vintage_section(result):
    # type: (SegmentEvalResult) -> str
    vintage = result.vintage if result.vintage is not None else pd.DataFrame()
    intro = (
        '<p class="note">One chart per <code>segment_col</code>, hue = <code>segment_value</code>. '
        "Bars are volume (<code>n</code>); lines are default rate, Gini and O/E. The black "
        "striped line is the portfolio event rate over the same vintages.</p>"
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
    parts.append(_grouped_next_actions(recs))

    parts.append(_heading("sec-verdicts", "2. Per-segment verdicts"))
    parts.append(_verdict_table(result, sql_map))

    parts.append(_heading("sec-ar", "3. Like-for-like comparison at matched approval rate"))
    parts.append(_matched_ar_section(result, gates))

    parts.append(_heading("sec-psi", "4. Characteristic drift"))
    parts.append(_characteristics_section(result, gates))

    parts.append(_heading("sec-grouping", "5. Segment vs portfolio grouping"))
    parts.append(_grouping_section(result))

    parts.append(_heading("sec-stability", "6. Predictor stability over vintages"))
    parts.append(_stability_section(result))

    parts.append(_heading("sec-vintage", "7. Vintage detail"))
    parts.append(_vintage_section(result))

    parts.append("</div></body></html>")
    return "".join(parts)
