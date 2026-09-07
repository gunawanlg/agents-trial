import pandas as pd

from scorecard_segment_eval.schema import ScorecardColumns


def observable_mask(df, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> pd.Series
    if cols.col_obs is None or cols.col_obs not in df.columns:
        return pd.Series(True, index=df.index)
    return df[cols.col_obs].fillna(0).astype(int).eq(1)


def fantomas_mask(df, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> pd.Series
    if cols.col_fantomas is None or cols.col_fantomas not in df.columns:
        return pd.Series(False, index=df.index)
    return df[cols.col_fantomas].fillna(0).astype(int).eq(1)


def approval_mask(df, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> pd.Series
    """Through-the-door approvals: observable outcomes stand in for booked/approved."""
    return observable_mask(df, cols)


def approval_rate(df, cols):
    # type: (pd.DataFrame, ScorecardColumns) -> float
    if df is None or len(df) == 0:
        return float("nan")
    return float(approval_mask(df, cols).mean())


def time_holdout_mask(dates, holdout_frac=0.25):
    # type: (pd.Series, float) -> pd.Series
    """Forward holdout: last `holdout_frac` of unique scoring dates."""
    parsed = pd.to_datetime(dates, errors="coerce")
    uniq = parsed.dropna().sort_values().unique()
    if len(uniq) < 2:
        return pd.Series(False, index=dates.index)
    cut_idx = int(len(uniq) * (1.0 - holdout_frac))
    cut_idx = min(max(cut_idx, 1), len(uniq) - 1)
    cutoff = uniq[cut_idx]
    return parsed >= cutoff
