"""Analyst-facing reports from ``evaluate_segments`` results."""

import datetime
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from scorecard_segment_eval.compat import ensure_dir
from scorecard_segment_eval.evaluate import SegmentEvalResult


def decision_table(result):
    # type: (SegmentEvalResult) -> pd.DataFrame
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
        "gini_ar_aligned",
        "approval_rate",
        "oe",
        "ece",
        "volume_share",
        "default_share",
    ]
    present = [c for c in cols if c in result.decisions.columns]
    if not present:
        return result.decisions
    return result.decisions[present].sort_values(["important", "volume_share"], ascending=[False, False])


def action_list(result):
    # type: (SegmentEvalResult) -> pd.DataFrame
    d = result.decisions
    if d.empty:
        return d
    mask = d["important"].eq(True) & d["q1_verdict"].eq("WEAK")
    return d.loc[mask].copy()


def _fmt(value, nd=3):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if isinstance(value, float):
        return ("%." + str(nd) + "f") % value
    return str(value)


def _fmt_pct(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    try:
        return "%.1f%%" % (float(value) * 100.0)
    except Exception:
        return str(value)


def _badge_class(action_or_verdict):
    mapping = {
        "GOOD": "ok",
        "WEAK": "warn",
        "INCONCLUSIVE": "muted",
        "SPLIT": "alert",
        "RECALIBRATE": "info",
        "KEEP_POOLED": "ok",
        "NONE": "muted",
    }
    return mapping.get(str(action_or_verdict), "muted")


def recommend_segment(decision_row, refit_row=None, char_table=None):
    # type: (pd.Series, Optional[pd.Series], Optional[pd.DataFrame]) -> List[str]
    recs = []  # type: List[str]
    action = decision_row.get("q2_action")
    verdict = decision_row.get("q1_verdict")
    failed = str(decision_row.get("failed_pillars") or "")
    gini_raw = decision_row.get("gini")
    gini_al = decision_row.get("gini_ar_aligned")
    ar = decision_row.get("approval_rate")
    ar_gap = decision_row.get("ar_gap_vs_overall")

    if verdict == "GOOD":
        recs.append("Pooled score looks adequate on this slice. Keep monitoring vintage Gini and O/E.")
    elif verdict == "INCONCLUSIVE":
        recs.append("Sample is under-powered. Do not split or recalibrate until more defaults accumulate.")
    elif verdict == "WEAK":
        recs.append("Pooled score is weak on this slice (%s)." % (failed.replace(",", ", ") or "see pillars"))

    if action == "SPLIT":
        recs.append(
            "Holdout same-predictor refit shows a material Gini lift with shape divergence. "
            "Consider a segment scorecard, but only after reviewing predictor vintage stability."
        )
        if refit_row is not None:
            g_ref = refit_row.get("gini_refit")
            g_st = refit_row.get("gini_refit_stable")
            dropped = str(refit_row.get("refit_dropped") or "")
            if dropped:
                recs.append("Stability screen dropped: %s. Prefer the stability-constrained refit if lift remains." % dropped)
            if g_ref is not None and g_st is not None:
                try:
                    if pd.notna(g_ref) and pd.notna(g_st) and float(g_ref) - float(g_st) >= 0.02:
                        recs.append(
                            "Performance-max Gini (%.3f) beats the stable-predictor Gini (%.3f). "
                            "The extra lift may be vintage-fragile — do not chase it blindly."
                            % (float(g_ref), float(g_st))
                        )
                    elif pd.notna(g_st):
                        recs.append("Stable-predictor refit Gini is %.3f; lift is not coming from unstable variables." % float(g_st))
                except Exception:
                    pass
    elif action == "RECALIBRATE":
        recs.append(
            "Rank order is acceptable; calibration is not. Recalibrate intercept/slope on this segment "
            "rather than fitting a new scorecard."
        )
    elif action == "KEEP_POOLED" and verdict == "WEAK":
        recs.append(
            "A same-predictor coefficient refit is not justified (need new information, not new coefficients). "
            "Investigate characteristic mix, policy cut-offs, and missing predictors."
        )

    try:
        if gini_al is not None and pd.notna(gini_al) and gini_raw is not None and pd.notna(gini_raw):
            recs.append(
                "Approval rates differ enough to confound Gini. At the lower-AR aligned threshold, Gini is %s "
                "(raw Gini %s, AR %s, gap vs overall %s). Interpret rank-order gaps cautiously."
                % (_fmt(gini_al), _fmt(gini_raw), _fmt_pct(ar), _fmt_pct(ar_gap))
            )
    except Exception:
        pass

    if char_table is not None and not char_table.empty and "psi" in char_table.columns:
        hot = char_table.loc[char_table["psi"] >= 0.25]
        if not hot.empty:
            names = ", ".join(hot.sort_values("psi", ascending=False)["feature"].astype(str).head(6).tolist())
            recs.append("Material PSI (≥0.25) versus portfolio on: %s." % names)
        if "rank_reversal" in char_table.columns and bool(char_table["rank_reversal"].fillna(False).any()):
            recs.append("WoE rank reversal versus the portfolio — local relationship is inverted or non-monotonic.")
    if not recs:
        recs.append("No additional flags.")
    return recs


def _html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _kpi_counts(decisions):
    if decisions is None or decisions.empty:
        return {"n": 0, "good": 0, "weak": 0, "inconclusive": 0, "split": 0, "recal": 0}
    v = decisions["q1_verdict"] if "q1_verdict" in decisions.columns else pd.Series(dtype=object)
    a = decisions["q2_action"] if "q2_action" in decisions.columns else pd.Series(dtype=object)
    return {
        "n": int(len(decisions)),
        "good": int((v == "GOOD").sum()),
        "weak": int((v == "WEAK").sum()),
        "inconclusive": int((v == "INCONCLUSIVE").sum()),
        "split": int((a == "SPLIT").sum()),
        "recal": int((a == "RECALIBRATE").sum()),
    }


def build_markdown_report(result, title="Segment scorecard evaluation"):
    # type: (SegmentEvalResult, str) -> str
    k = _kpi_counts(result.decisions)
    lines = [
        "# " + title,
        "",
        "Generated %s." % datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "",
        "## Snapshot",
        "",
        "- Segments reviewed: **%s**" % k["n"],
        "- GOOD / WEAK / INCONCLUSIVE: **%s / %s / %s**" % (k["good"], k["weak"], k["inconclusive"]),
        "- Recommended SPLIT: **%s** · RECALIBRATE: **%s**" % (k["split"], k["recal"]),
        "",
        "## Decision table",
        "",
    ]
    table = decision_table(result)
    if table.empty:
        lines.append("_No segment decisions._")
    else:
        show = table.copy()
        lines.append(show.to_markdown(index=False) if hasattr(show, "to_markdown") else show.to_string(index=False))
    lines.extend(["", "## Recommendations", ""])
    if result.decisions.empty:
        lines.append("_None._")
        return "\n".join(lines)

    refit = result.refit_comparison
    chars = result.characteristics
    for _, row in result.decisions.iterrows():
        ref_row = None
        char_seg = None
        if refit is not None and not refit.empty:
            hit = refit[(refit["segment_col"] == row["segment_col"]) & (refit["segment_value"].astype(str) == str(row["segment_value"]))]
            if len(hit):
                ref_row = hit.iloc[0]
        if chars is not None and not chars.empty and "segment_value" in chars.columns:
            hitc = chars[(chars["segment_col"] == row["segment_col"]) & (chars["segment_value"].astype(str) == str(row["segment_value"]))]
            char_seg = hitc
        recs = recommend_segment(row, ref_row, char_seg)
        lines.append("### %s = %s · %s → %s" % (row.get("segment_col"), row.get("segment_value"), row.get("q1_verdict"), row.get("q2_action")))
        for rec in recs:
            lines.append("- %s" % rec)
        lines.append("")
    lines.extend(
        [
            "## How to read this",
            "",
            "- **GOOD**: rank order, calibration, and vintage stability all clear the gates.",
            "- **RECALIBRATE**: rank order is fine; intercept/slope is not.",
            "- **SPLIT**: same-predictor refit beats pooled on holdout *and* characteristics diverge; still check vintage stability.",
            "- **gini_ar_aligned**: Gini after matching the *lower* approval-rate threshold when AR gaps are large.",
            "",
        ]
    )
    return "\n".join(lines)


def build_html_report(result, title="Segment scorecard evaluation"):
    # type: (SegmentEvalResult, str) -> str
    k = _kpi_counts(result.decisions)
    table = decision_table(result)
    rows_html = []
    if not table.empty:
        for _, row in table.iterrows():
            v_cls = _badge_class(row.get("q1_verdict"))
            a_cls = _badge_class(row.get("q2_action"))
            rows_html.append(
                "<tr>"
                "<td>%s</td><td>%s</td><td>%s</td>"
                "<td><span class='badge %s'>%s</span></td>"
                "<td>%s</td>"
                "<td><span class='badge %s'>%s</span></td>"
                "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                "</tr>"
                % (
                    _html_escape(row.get("segment_col", "")),
                    _html_escape(row.get("segment_value", "")),
                    "yes" if row.get("important") else "no",
                    v_cls,
                    _html_escape(row.get("q1_verdict", "")),
                    _html_escape(row.get("failed_pillars", "") or "—"),
                    a_cls,
                    _html_escape(row.get("q2_action", "")),
                    _fmt(row.get("gini")),
                    _fmt(row.get("gini_ar_aligned")),
                    _fmt_pct(row.get("approval_rate")) if "approval_rate" in row else "—",
                    _fmt(row.get("oe")),
                    _fmt_pct(row.get("volume_share")),
                )
            )

    cards = []
    refit = result.refit_comparison
    chars = result.characteristics
    if not result.decisions.empty:
        ordered = result.decisions.sort_values(["important", "volume_share"], ascending=[False, False])
        for _, row in ordered.iterrows():
            ref_row = None
            char_seg = None
            if refit is not None and not refit.empty:
                hit = refit[(refit["segment_col"] == row["segment_col"]) & (refit["segment_value"].astype(str) == str(row["segment_value"]))]
                if len(hit):
                    ref_row = hit.iloc[0]
            if chars is not None and not chars.empty and "segment_value" in chars.columns:
                char_seg = chars[(chars["segment_col"] == row["segment_col"]) & (chars["segment_value"].astype(str) == str(row["segment_value"]))]
            recs = recommend_segment(row, ref_row, char_seg)
            rec_lis = "".join("<li>%s</li>" % _html_escape(r) for r in recs)
            psi_note = ""
            if char_seg is not None and not char_seg.empty and "psi" in char_seg.columns:
                top = char_seg.sort_values("psi", ascending=False).head(5)
                psi_cells = "".join(
                    "<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (
                        _html_escape(r["feature"]),
                        _fmt(r["psi"]),
                        _html_escape(r["psi_method"] if "psi_method" in r.index else ""),
                    )
                    for _, r in top.iterrows()
                )
                psi_note = "<table class='mini'><thead><tr><th>Feature</th><th>PSI</th><th>Method</th></tr></thead><tbody>%s</tbody></table>" % psi_cells
            cards.append(
                """
<article class="card">
  <header>
    <h3>%s <span class="sep">/</span> %s</h3>
    <div class="tags">
      <span class="badge %s">%s</span>
      <span class="badge %s">%s</span>
    </div>
  </header>
  <p class="meta">volume %s · default share %s · Gini %s · AR %s · aligned Gini %s</p>
  <ul class="recs">%s</ul>
  %s
</article>
"""
                % (
                    _html_escape(row.get("segment_col")),
                    _html_escape(row.get("segment_value")),
                    _badge_class(row.get("q1_verdict")),
                    _html_escape(row.get("q1_verdict")),
                    _badge_class(row.get("q2_action")),
                    _html_escape(row.get("q2_action")),
                    _fmt_pct(row.get("volume_share")),
                    _fmt_pct(row.get("default_share")),
                    _fmt(row.get("gini")),
                    _fmt_pct(row.get("approval_rate")) if "approval_rate" in row.index else "—",
                    _fmt(row.get("gini_ar_aligned")),
                    rec_lis,
                    psi_note,
                )
            )

    generated = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>%(title)s</title>
  <style>
    :root { --navy:#14213d; --ink:#1b2838; --paper:#f6f3ee; --card:#fffdf8;
            --ok:#1b7f5a; --warn:#b45309; --alert:#9f1239; --info:#1d4ed8; --muted:#64748b; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: "Source Sans 3", "Segoe UI", sans-serif; background: var(--paper); color: var(--ink); }
    header.hero { background: linear-gradient(135deg, #14213d, #1d3557); color:#f8fafc; padding:36px 48px 28px; }
    header.hero h1 { margin:0 0 6px; font-weight:650; letter-spacing:-0.02em; }
    header.hero p { margin:0; opacity:0.8; }
    .kpis { display:flex; gap:12px; flex-wrap:wrap; padding:20px 48px 0; }
    .kpi { background:var(--card); border:1px solid #e7e0d6; border-radius:12px; padding:14px 18px; min-width:120px; }
    .kpi b { display:block; font-size:22px; }
    .kpi span { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.06em; }
    main { padding:24px 48px 64px; }
    h2 { margin:28px 0 12px; font-size:18px; }
    table.grid { width:100%%; border-collapse:collapse; background:var(--card); border-radius:12px; overflow:hidden; }
    table.grid th, table.grid td { padding:8px 10px; text-align:left; font-size:13px; border-bottom:1px solid #eee6dc; }
    table.grid th { background:#efe8dd; font-size:11px; text-transform:uppercase; letter-spacing:0.04em; }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:650; }
    .badge.ok { background:#d1fae5; color:var(--ok); }
    .badge.warn { background:#ffedd5; color:var(--warn); }
    .badge.alert { background:#ffe4e6; color:var(--alert); }
    .badge.info { background:#dbeafe; color:var(--info); }
    .badge.muted { background:#e2e8f0; color:var(--muted); }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }
    .card { background:var(--card); border:1px solid #e7e0d6; border-radius:14px; padding:16px 18px; }
    .card h3 { margin:0; font-size:16px; }
    .card .sep { color:#c4b8a5; }
    .card header { display:flex; justify-content:space-between; gap:8px; align-items:center; }
    .card .meta { color:var(--muted); font-size:12px; margin:8px 0 10px; }
    .card ul.recs { margin:0; padding-left:18px; }
    .card ul.recs li { margin:0 0 6px; font-size:13.5px; line-height:1.45; }
    table.mini { width:100%%; margin-top:10px; font-size:12px; border-collapse:collapse; }
    table.mini th, table.mini td { padding:4px 6px; border-bottom:1px solid #f0eae2; text-align:left; }
    footer { padding:0 48px 40px; color:var(--muted); font-size:12px; }
  </style>
</head>
<body>
  <header class="hero">
    <h1>%(title)s</h1>
    <p>Recommendations for split vs pooled vs recalibrate · generated %(generated)s</p>
  </header>
  <section class="kpis">
    <div class="kpi"><span>Segments</span><b>%(n)s</b></div>
    <div class="kpi"><span>GOOD</span><b>%(good)s</b></div>
    <div class="kpi"><span>WEAK</span><b>%(weak)s</b></div>
    <div class="kpi"><span>SPLIT</span><b>%(split)s</b></div>
    <div class="kpi"><span>Recalibrate</span><b>%(recal)s</b></div>
  </section>
  <main>
    <h2>Decision overview</h2>
    <table class="grid">
      <thead>
        <tr>
          <th>Segment</th><th>Value</th><th>Important</th><th>Q1</th><th>Failed</th>
          <th>Q2</th><th>Gini</th><th>Gini AR-aligned</th><th>AR</th><th>O/E</th><th>Volume</th>
        </tr>
      </thead>
      <tbody>%(rows)s</tbody>
    </table>
    <h2>Analyst recommendations</h2>
    <div class="cards">%(cards)s</div>
  </main>
  <footer>
    Gini AR-aligned is only populated when the approval-rate gap is material; it simulates the
    lower-AR threshold on the higher-AR side. Refit uses tree (or opt-binning) WoE and optionally an
    XGBoost submodel with max_depth=3. Stability-constrained refits drop vintage-unstable predictors.
  </footer>
</body>
</html>
""" % {
        "title": _html_escape(title),
        "generated": generated,
        "n": k["n"],
        "good": k["good"],
        "weak": k["weak"],
        "split": k["split"],
        "recal": k["recal"],
        "rows": "".join(rows_html) or "<tr><td colspan='11'>No segments</td></tr>",
        "cards": "".join(cards) or "<p>No recommendations.</p>",
    }
    return html


def write_final_report(result, path, title="Segment scorecard evaluation", also_markdown=True):
    """Write a professional HTML report (and optional sibling .md) from evaluate_segments."""
    html = build_html_report(result, title=title)
    ensure_dir(path)
    with open(path, "w") as handle:
        handle.write(html)
    md_path = None
    if also_markdown:
        root, _ext = os.path.splitext(path)
        md_path = root + ".md"
        with open(md_path, "w") as handle:
            handle.write(build_markdown_report(result, title=title))
    return path, md_path


def build_final_report(result, path=None, title="Segment scorecard evaluation", also_markdown=True):
    html = build_html_report(result, title=title)
    if path:
        write_final_report(result, path, title=title, also_markdown=also_markdown)
    return html
