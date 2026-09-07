"""Population masks: observability, fantomas exclusion and time holdout."""

from typing import Any

import pandas as pd

from scorecard_segment_eval.schema import ScorecardColumns


def observable_mask(df, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> pd.Series
    """Rows whose target is observable.

    With no observation flag resolved, every row with a non-null target counts
    as observable so the analysis still runs.
    """
    if cols.col_obs is None or cols.col_obs not in df.columns:
        if cols.col_target is not None and cols.col_target in df.columns:
            return df[cols.col_target].notna()
        return pd.Series(True, index=df.index)
    return df[cols.col_obs].fillna(0).astype(int).eq(1)


def fantomas_mask(df, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> pd.Series
    if cols.col_fantomas is None or cols.col_fantomas not in df.columns:
        return pd.Series(False, index=df.index)
    return df[cols.col_fantomas].fillna(0).astype(int).eq(1)


def time_holdout_mask(dates, holdout_frac=0.25):
    # type: (Any, float) -> pd.Series
    """Forward holdout: last ``holdout_frac`` of unique scoring dates."""
    parsed = pd.to_datetime(dates, errors="coerce")
    uniq = parsed.dropna().sort_values().unique()
    if len(uniq) < 2:
        return pd.Series(False, index=dates.index)
    cut_idx = int(len(uniq) * (1.0 - holdout_frac))
    cut_idx = min(max(cut_idx, 1), len(uniq) - 1)
    cutoff = uniq[cut_idx]
    return parsed >= cutoff
