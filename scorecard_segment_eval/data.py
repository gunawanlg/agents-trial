"""Smart scorecard metadata discovery and interactive analysis entry points."""

import json
import os
import warnings

import pandas as pd

from scorecard_segment_eval.schema import ScorecardColumns


_DATE_NAMES = ("score_date", "application_date", "date", "dt")
_SEGMENT_NAMES = ("segment", "channel", "portfolio", "product")
_TARGET_NAMES = ("TargetDefault", "target_default", "default", "target")
_OBS_NAMES = ("TargetDefaultObs", "target_default_obs", "obs", "observable")


class SmartData(object):
    def __init__(self, frame, columns, analyses, warnings_list):
        self.frame = frame
        self.columns = columns
        self.analyses = analyses
        self.warnings = warnings_list

    def metadata(self):
        return {
            "col_id": self.columns.col_id,
            "col_date": self.columns.col_date,
            "col_obs": self.columns.col_obs,
            "col_target": self.columns.col_target,
            "col_score": self.columns.col_score,
            "cols_pred": list(self.columns.cols_pred),
            "cols_pred_used": list(self.columns.cols_pred_used),
            "cols_pred_woe": list(self.columns.cols_pred_woe),
            "cols_segment": list(self.columns.cols_segment),
        }


def _first_existing(frame, provided, candidates):
    if provided:
        return provided
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _warn(messages, text):
    messages.append(text)
    warnings.warn(text, UserWarning)


def _load_grouping(grouping):
    if grouping is None:
        return {}
    if isinstance(grouping, dict):
        return grouping
    with open(grouping, "r") as handle:
        return json.load(handle)


def infer_scorecard_data(
    table,
    col_score,
    cols_pred_used,
    col_id="SKP_CREDIT_CASE",
    col_date=None,
    cols_segment=None,
    col_target=None,
    target_str=None,
    col_obs=None,
    target_obs=None,
    cols_pred=None,
    cols_pred_woe=None,
    grouping=None,
    sql_lookup=None,
    portfolio=None,
):
    """Infer optional metadata, enriching only through an explicit local lookup.

    ``sql_lookup`` may be a DataFrame or a callable receiving IDs. Predictor
    names are deliberately never inferred from SQL.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame loaded from the DB table")
    if col_id not in table or col_score not in table:
        raise ValueError("table must contain col_id '%s' and col_score '%s'" % (col_id, col_score))
    if not cols_pred_used:
        raise ValueError("cols_pred_used is mandatory")

    frame = table.copy()
    messages = []
    lookup = sql_lookup(frame[col_id].dropna().unique()) if callable(sql_lookup) else sql_lookup
    if lookup is not None:
        if not isinstance(lookup, pd.DataFrame) or col_id not in lookup:
            raise ValueError("sql_lookup must return a DataFrame containing col_id")
        add = [name for name in lookup.columns if name == col_id or name not in frame.columns]
        frame = frame.merge(lookup[add].drop_duplicates(col_id), on=col_id, how="left")
        _warn(messages, "Optional metadata was enriched using the local SQL lookup keyed by %s." % col_id)

    date_name = _first_existing(frame, col_date, _DATE_NAMES)
    target_name = _first_existing(frame, col_target or target_str, _TARGET_NAMES)
    obs_name = _first_existing(frame, col_obs or target_obs, _OBS_NAMES)
    segments = list(cols_segment or [name for name in _SEGMENT_NAMES if name in frame.columns])
    raw_predictors = list(cols_pred or [])
    woe_predictors = list(cols_pred_woe or [])
    grouping_data = _load_grouping(grouping)

    if target_name is None:
        target_name = "TargetDefault"
    if obs_name is None:
        obs_name = "TargetDefaultObs"
    if target_name not in frame and portfolio is not None:
        mapping = portfolio.get(target_name) if isinstance(portfolio, dict) else None
        if mapping and mapping in frame:
            frame[target_name] = frame[mapping]
    if target_name in frame and obs_name not in frame:
        frame[obs_name] = frame[target_name].notna().astype(int)
        _warn(messages, "%s was inferred from non-null %s values." % (obs_name, target_name))
    if obs_name in frame and target_name not in frame:
        _warn(messages, "%s is present but target outcomes cannot be derived safely; provide col_target or SQL lookup." % obs_name)
    if date_name is None:
        _warn(messages, "No date column was inferred; vintage and forward-holdout analyses are unavailable.")
        date_name = "date"
    if not raw_predictors:
        _warn(messages, "cols_pred cannot be guessed from SQL; predictor refit is unavailable.")
    if not woe_predictors:
        _warn(messages, "cols_pred_woe is absent; recalibration remains available but grouped WoE PSI is unavailable.")
    if grouping_data and not raw_predictors:
        _warn(messages, "grouping.json exists without cols_pred; grouped PSI is limited and refit is unavailable.")
    elif woe_predictors and not grouping_data:
        _warn(messages, "WoE predictors have no grouping.json; grouped PSI may be less informative.")

    available = ["segment_performance", "approval_rate_adjusted_gini", "recalibration"]
    if date_name in frame:
        available.append("vintage_stability")
    if raw_predictors and all(name in frame for name in raw_predictors):
        available.extend(["predictor_refit", "characteristic_psi"])
    columns = ScorecardColumns(
        col_id=col_id,
        col_date=date_name,
        col_obs=obs_name,
        col_target=target_name,
        col_score=col_score,
        cols_pred=raw_predictors,
        cols_segment=segments,
        cols_pred_woe=woe_predictors,
        cols_pred_used=list(cols_pred_used),
        grouping=grouping_data,
    )
    return SmartData(frame, columns, available, messages)


def confirm_and_evaluate(smart_data, gates=None, output_filename=None, cache_dir=".scorecard_segment_eval", input_fn=input, **kwargs):
    """Display inferred settings and require confirmation before evaluation."""
    from scorecard_segment_eval.evaluate import evaluate_segments
    from scorecard_segment_eval.report import write_analyst_report

    print("Inferred metadata:")
    print(json.dumps(smart_data.metadata(), indent=2, sort_keys=True))
    print("Available analyses: %s" % ", ".join(smart_data.analyses))
    answer = input_fn("Continue with core analysis? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError("Analysis cancelled; update inferred metadata and confirm again.")
    if not output_filename:
        output_filename = input_fn("Output report filename: ").strip()
    if not output_filename:
        raise ValueError("An output filename is required")
    if not output_filename.lower().endswith(".html"):
        output_filename += ".html"

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    settings_path = os.path.join(cache_dir, "settings.json")
    settings = smart_data.metadata()
    settings["output_filename"] = output_filename
    with open(settings_path, "w") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)

    result = evaluate_segments(smart_data.frame, smart_data.columns, gates, **kwargs)
    write_analyst_report(result, output_filename)
    return result
