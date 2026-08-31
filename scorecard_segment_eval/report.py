from __future__ import annotations

import pandas as pd

from scorecard_segment_eval.evaluate import SegmentEvalResult


def decision_table(result: SegmentEvalResult) -> pd.DataFrame:
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


def action_list(result: SegmentEvalResult) -> pd.DataFrame:
    d = result.decisions
    if d.empty:
        return d
    mask = d["important"].eq(True) & d["q1_verdict"].eq("WEAK")
    return d.loc[mask].copy()
