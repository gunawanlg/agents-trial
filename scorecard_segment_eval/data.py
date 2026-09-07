"""Smart loading of a scorecard base table: infer optional columns, SQL enrich, plan cache."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from scorecard_segment_eval.compat import dump_json, load_json, warn_user
from scorecard_segment_eval.schema import ScorecardColumns


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SQL_DIR = os.path.join(PACKAGE_ROOT, "sql")
REPO_SQL_DIR = os.path.join(os.path.dirname(PACKAGE_ROOT), "sql")

DEFAULT_TARGET = "TargetDefault"
DEFAULT_OBS = "TargetDefaultObs"

DATE_CANDIDATES = [
    "col_date",
    "DTIME_SCORE",
    "SCORE_DATE",
    "score_date",
    "APPLICATION_DATE",
    "DTIME_APPLICATION",
    "date",
]
SEGMENT_CANDIDATES = [
    "CODE_CHANNEL",
    "CODE_PRODUCT",
    "channel",
    "product",
    "segment",
    "SEGMENT",
]
ID_CANDIDATES = ["SKP_CREDIT_CASE", "col_id", "app_id"]


def _read_text(path):
    with open(path, "r") as handle:
        return handle.read()


def load_target_mapping(path=None):
    # type: (Optional[str]) -> Dict[str, Dict[str, str]]
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            os.path.join(REPO_SQL_DIR, "target_mapping.json"),
            os.path.join(DEFAULT_SQL_DIR, "target_mapping.json"),
            "sql/target_mapping.json",
        ]
    )
    for cand in candidates:
        if cand and os.path.isfile(cand):
            raw = load_json(cand)
            return raw if isinstance(raw, dict) else {}
    return {
        "SKP_CREDIT_CASE": {"col_target": DEFAULT_TARGET, "col_obs": DEFAULT_OBS},
        "PortfolioA": {"col_target": "TargetA", "col_obs": "TargetAObs"},
        "PortfolioB": {"col_target": "TargetB", "col_obs": "TargetBObs"},
    }


def resolve_sql_path(sql_path=None, sql_dir=None):
    # type: (Optional[str], Optional[str]) -> Optional[str]
    if sql_path and os.path.isfile(sql_path):
        return sql_path
    search_dirs = []
    if sql_dir:
        search_dirs.append(sql_dir)
    search_dirs.extend([REPO_SQL_DIR, DEFAULT_SQL_DIR, "sql", os.getcwd()])
    names = []
    if sql_path:
        names.append(os.path.basename(sql_path))
    names.extend(["enrich_by_id.sql", "lookup_by_id.sql"])
    for directory in search_dirs:
        if not directory:
            continue
        for name in names:
            cand = name if os.path.isabs(name) else os.path.join(directory, name)
            if os.path.isfile(cand):
                return cand
    return None


def _sql_in_list(ids):
    parts = []
    for item in ids:
        text = str(item).replace("'", "''")
        parts.append("'" + text + "'")
    if not parts:
        return "NULL"
    return ",".join(parts)


def render_enrich_sql(template, ids, col_id="SKP_CREDIT_CASE"):
    rendered = template.replace("{id_list}", _sql_in_list(ids))
    rendered = rendered.replace("{col_id}", str(col_id))
    return rendered


def _strip_obs_suffix(name):
    text = str(name)
    for suffix in ("Obs", "_obs", "_OBS", "OBS"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return None


def infer_target_obs(columns, col_id, col_target=None, col_obs=None, portfolio=None, mapping=None):
    """Infer target / observation flags from names, mapping, or defaults.

    Rules:
    - mapping keyed by col_id name (e.g. SKP_CREDIT_CASE) or portfolio (PortfolioA → TargetA)
    - col_obs=TargetAObs → col_target=TargetA
    - col_target=TargetA → col_obs=TargetAObs
    - else TargetDefault / TargetDefaultObs
    """
    columns = list(columns or [])
    mapping = mapping if mapping is not None else load_target_mapping()
    warnings = []  # type: List[str]
    source = {}  # type: Dict[str, str]

    mapped = None
    for key in (portfolio, col_id, str(col_id).upper() if col_id else None):
        if key and key in mapping:
            mapped = mapping[key]
            break
    if mapped:
        if col_target is None:
            col_target = mapped.get("col_target") or mapped.get("target")
            source["col_target"] = "mapping:" + str(key)
        if col_obs is None:
            col_obs = mapped.get("col_obs") or mapped.get("target_obs")
            source["col_obs"] = "mapping:" + str(key)

    if col_obs and not col_target:
        inferred = _strip_obs_suffix(col_obs)
        if inferred:
            col_target = inferred
            source["col_target"] = "from_col_obs"
    if col_target and not col_obs:
        col_obs = str(col_target) + "Obs"
        source["col_obs"] = "from_col_target"

    if col_target is None:
        for cand in (DEFAULT_TARGET, "TargetDefault", "default", "TARGET", "y"):
            if cand in columns:
                col_target = cand
                source["col_target"] = "column_present"
                break
    if col_obs is None:
        for cand in (DEFAULT_OBS, "TargetDefaultObs", "obs", "TARGET_OBS"):
            if cand in columns:
                col_obs = cand
                source["col_obs"] = "column_present"
                break

    if col_target is None:
        col_target = DEFAULT_TARGET
        source["col_target"] = "default"
        warnings.append(
            "col_target was not provided; defaulting to %s. Pass col_target or a portfolio mapping if this is wrong."
            % DEFAULT_TARGET
        )
    if col_obs is None:
        col_obs = DEFAULT_OBS
        source["col_obs"] = "default"
        warnings.append(
            "col_obs was not provided; defaulting to %s. Pass col_obs or a portfolio mapping if this is wrong."
            % DEFAULT_OBS
        )

    if col_target not in columns:
        warnings.append("Inferred col_target=%s is not in the current table columns." % col_target)
    if col_obs not in columns:
        warnings.append("Inferred col_obs=%s is not in the current table columns." % col_obs)
    return col_target, col_obs, source, warnings


def _pick_first_present(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    for col in columns:
        if str(col).startswith("segment_"):
            return col
    return None


def _pick_all_present(columns, candidates):
    found = []
    for name in candidates:
        if name in columns:
            found.append(name)
    for col in columns:
        if str(col).startswith("segment_") and col not in found:
            found.append(col)
    return found


@dataclass
class DataSpec:
    col_id: str
    col_score: str
    cols_pred_used: List[str]
    col_date: Optional[str] = None
    cols_segment: List[str] = field(default_factory=list)
    col_target: Optional[str] = None
    col_obs: Optional[str] = None
    cols_pred: List[str] = field(default_factory=list)
    cols_pred_woe: List[str] = field(default_factory=list)
    col_fantomas: Optional[str] = None
    grouping_path: Optional[str] = None
    portfolio: Optional[str] = None
    score_higher_is_risk: bool = True

    def to_columns(self):
        if not self.col_date or not self.col_obs or not self.col_target:
            missing = [n for n, v in (("col_date", self.col_date), ("col_obs", self.col_obs), ("col_target", self.col_target)) if not v]
            raise ValueError("Cannot build ScorecardColumns; missing " + ", ".join(missing))
        return ScorecardColumns(
            col_id=self.col_id,
            col_date=self.col_date,
            col_obs=self.col_obs,
            col_target=self.col_target,
            col_score=self.col_score,
            cols_pred=list(self.cols_pred or []),
            cols_segment=list(self.cols_segment or []),
            col_fantomas=self.col_fantomas,
            cols_pred_woe=list(self.cols_pred_woe or []),
            cols_pred_used=list(self.cols_pred_used or []),
            grouping_path=self.grouping_path,
            score_higher_is_risk=self.score_higher_is_risk,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        known = {k: payload[k] for k in cls.__dataclass_fields__ if k in payload}  # type: ignore
        return cls(**known)


@dataclass
class AnalysisPlan:
    spec: DataSpec
    capabilities: Dict[str, bool]
    warnings: List[str]
    sources: Dict[str, str]
    n_rows: Optional[int] = None
    columns: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "spec": self.spec.to_dict(),
            "capabilities": dict(self.capabilities),
            "warnings": list(self.warnings),
            "sources": dict(self.sources),
            "n_rows": self.n_rows,
            "columns": list(self.columns),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            spec=DataSpec.from_dict(payload.get("spec") or {}),
            capabilities=payload.get("capabilities") or {},
            warnings=payload.get("warnings") or [],
            sources=payload.get("sources") or {},
            n_rows=payload.get("n_rows"),
            columns=payload.get("columns") or [],
        )


def compute_capabilities(spec, columns=None, grouping_exists=None):
    # type: (DataSpec, Optional[Sequence[str]], Optional[bool]) -> Tuple[Dict[str, bool], List[str]]
    columns = list(columns or [])
    warnings = []  # type: List[str]
    grouping_exists = bool(grouping_exists)
    if spec.grouping_path and os.path.isfile(spec.grouping_path):
        grouping_exists = True

    has_target = bool(spec.col_target) and (not columns or spec.col_target in columns)
    has_obs = bool(spec.col_obs) and (not columns or spec.col_obs in columns)
    has_date = bool(spec.col_date) and (not columns or spec.col_date in columns)
    has_score = bool(spec.col_score) and (not columns or spec.col_score in columns)
    has_id = bool(spec.col_id) and (not columns or spec.col_id in columns)
    has_pred = bool(spec.cols_pred)
    has_pred_woe = bool(spec.cols_pred_woe)
    has_used = bool(spec.cols_pred_used)
    grouping_psi = bool(has_pred_woe) or (grouping_exists and has_pred)

    if not has_pred:
        warnings.append("cols_pred not provided: cannot perform same-predictor refit.")
    if not has_pred_woe and not (grouping_exists and has_pred):
        warnings.append(
            "cols_pred_woe is missing"
            + (" and grouping.json cannot be applied without cols_pred" if grouping_exists and not has_pred else "")
            + ": cannot perform grouping-based PSI / WoE recalibration diagnostics."
        )
    if grouping_exists and not has_pred and not has_pred_woe:
        warnings.append("grouping.json is present but cols_pred was not provided; grouping PSI is disabled.")
    if not has_date:
        warnings.append("col_date missing: vintage stability analysis cannot be run.")
    if not spec.cols_segment:
        warnings.append("cols_segment missing: only the overall population will be evaluated.")
    if not has_target or not has_obs or not has_score or not has_id:
        warnings.append("Mandatory evaluation fields are incomplete; segment evaluation may fail.")

    caps = {
        "evaluate_segments": bool(has_id and has_score and has_target and has_obs),
        "segment_split": bool(spec.cols_segment),
        "vintage_stability": bool(has_date and has_target and has_score),
        "refit": bool(has_pred and has_target),
        "recalibrate_pd": bool(has_score and has_target),
        "psi_grouping": bool(grouping_psi),
        "psi_decile": bool(has_used or has_pred),
        "ar_aligned_gini": bool(has_score and has_target and has_obs),
        "report": bool(has_id and has_score and has_target),
    }
    return caps, warnings


def format_plan_text(plan):
    # type: (AnalysisPlan) -> str
    spec = plan.spec
    lines = []
    lines.append("=" * 72)
    lines.append("Scorecard analysis plan")
    lines.append("=" * 72)
    if plan.n_rows is not None:
        lines.append("Rows: %s" % plan.n_rows)
    lines.append("")
    lines.append("Mandatory")
    lines.append("  col_id:          %s" % spec.col_id)
    lines.append("  col_score:       %s" % spec.col_score)
    lines.append("  cols_pred_used:  %s" % (", ".join(spec.cols_pred_used) or "(none)"))
    lines.append("")
    lines.append("Optional / inferred")
    lines.append("  col_date:        %s  [%s]" % (spec.col_date or "(missing)", plan.sources.get("col_date", "-")))
    lines.append("  col_target:      %s  [%s]" % (spec.col_target or "(missing)", plan.sources.get("col_target", "-")))
    lines.append("  col_obs:         %s  [%s]" % (spec.col_obs or "(missing)", plan.sources.get("col_obs", "-")))
    lines.append("  cols_segment:    %s  [%s]" % (", ".join(spec.cols_segment) or "(missing)", plan.sources.get("cols_segment", "-")))
    lines.append("  cols_pred:       %s" % (", ".join(spec.cols_pred) or "(missing — refit disabled)"))
    lines.append("  cols_pred_woe:   %s" % (", ".join(spec.cols_pred_woe) or "(missing)"))
    lines.append("  grouping_path:   %s" % (spec.grouping_path or "(none)"))
    lines.append("")
    lines.append("Capabilities")
    for name in sorted(plan.capabilities.keys()):
        flag = "yes" if plan.capabilities[name] else "no "
        lines.append("  [%s] %s" % (flag, name))
    if plan.warnings:
        lines.append("")
        lines.append("Warnings")
        for w in plan.warnings:
            lines.append("  * %s" % w)
    lines.append("=" * 72)
    return "\n".join(lines)


def _try_ipython_display(text):
    try:
        from IPython.display import HTML, display

        html = "<pre style='font-family:ui-monospace,monospace;font-size:13px;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;white-space:pre-wrap;'>%s</pre>" % (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        display(HTML(html))
        return True
    except Exception:
        print(text)
        return False


def _prompt(message, default=None):
    try:
        raw = input(message)
    except Exception:
        return default
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def confirm_and_save_plan(plan, confirmed=None, filename=None, auto_confirm=False):
    """Notebook-friendly confirmation.

    * Displays the capability matrix.
    * Asks the user to confirm (ipywidgets if available, else input()).
    * Asks for a filename and writes the plan JSON cache.

    For tests / batch jobs pass ``confirmed=True`` and ``filename=...``.
    """
    text = format_plan_text(plan)
    used_ipy = _try_ipython_display(text)
    if auto_confirm:
        confirmed = True

    if confirmed is None:
        confirmed = _confirm_interactive(used_ipy)

    if not confirmed:
        warn_user("Analysis plan was not confirmed. Settings were not saved.")
        return None

    if not filename:
        filename = _prompt("Save settings as filename (e.g. analysis_plan.json): ", default="analysis_plan.json")
    if not filename:
        filename = "analysis_plan.json"
    dump_json(filename, plan.to_dict())
    print("Saved analysis plan to %s" % os.path.abspath(filename))
    return filename


def _confirm_interactive(used_ipy):
    try:
        import ipywidgets as widgets
        from IPython.display import display

        box = {"value": None}
        yes = widgets.Button(description="Confirm", button_style="success")
        no = widgets.Button(description="Cancel", button_style="danger")

        def _yes(_b):
            box["value"] = True

        def _no(_b):
            box["value"] = False

        yes.on_click(_yes)
        no.on_click(_no)
        display(widgets.HBox([yes, no]))
        # widgets are async in notebooks; also accept a typed fallback
        typed = _prompt("Type y to confirm this plan (or use the buttons): ", default="n")
        if box["value"] is not None:
            return bool(box["value"])
        return str(typed).lower() in ("y", "yes", "1", "true")
    except Exception:
        typed = _prompt("Proceed with this plan? [y/N]: ", default="n")
        return str(typed).lower() in ("y", "yes", "1", "true")


def load_saved_plan(path):
    payload = load_json(path)
    return AnalysisPlan.from_dict(payload)


def _read_table(source, connection=None):
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if connection is None:
        raise ValueError("A database connection is required when source is a table name.")
    sql = "SELECT * FROM %s" % source
    return pd.read_sql(sql, connection)


def _merge_enrich(base, extra, col_id):
    if extra is None or extra.empty:
        return base
    extra = extra.copy()
    if col_id not in extra.columns and extra.columns[0] != col_id:
        # try first column as id
        if "col_id" in extra.columns:
            extra = extra.rename(columns={"col_id": col_id})
    overlap = [c for c in extra.columns if c in base.columns and c != col_id]
    extra = extra.drop(columns=overlap, errors="ignore")
    return base.merge(extra, how="left", on=col_id)


def enrich_from_sql(base, col_id, connection, sql_path, warn=True):
    template = _read_text(sql_path)
    ids = list(base[col_id].dropna().unique())
    sql = render_enrich_sql(template, ids, col_id=col_id)
    if warn:
        warn_user(
            "Optional columns were not all present on the base table. "
            "Looking them up via %s using %s (%s ids). Review the join before relying on results."
            % (sql_path, col_id, len(ids))
        )
    extra = pd.read_sql(sql, connection)
    if "col_id" in extra.columns and col_id not in extra.columns:
        extra = extra.rename(columns={"col_id": col_id})
    return extra, sql


def infer_optional_columns(df, spec, mapping=None):
    # type: (pd.DataFrame, DataSpec, Optional[dict]) -> Tuple[DataSpec, Dict[str, str], List[str]]
    columns = list(df.columns)
    sources = {}  # type: Dict[str, str]
    warnings = []  # type: List[str]

    if spec.col_date is None:
        picked = _pick_first_present(columns, DATE_CANDIDATES)
        if picked:
            spec.col_date = picked
            sources["col_date"] = "column_present"
        else:
            sources["col_date"] = "missing"
    else:
        sources["col_date"] = "provided"

    if not spec.cols_segment:
        picked_seg = _pick_all_present(columns, SEGMENT_CANDIDATES)
        if picked_seg:
            spec.cols_segment = picked_seg
            sources["cols_segment"] = "column_present"
        else:
            sources["cols_segment"] = "missing"
    else:
        sources["cols_segment"] = "provided"

    target, obs, tsrc, twarn = infer_target_obs(
        columns,
        spec.col_id,
        col_target=spec.col_target,
        col_obs=spec.col_obs,
        portfolio=spec.portfolio,
        mapping=mapping,
    )
    if spec.col_target is None:
        spec.col_target = target
        sources["col_target"] = tsrc.get("col_target", "inferred")
    else:
        sources["col_target"] = "provided"
    if spec.col_obs is None:
        spec.col_obs = obs
        sources["col_obs"] = tsrc.get("col_obs", "inferred")
    else:
        sources["col_obs"] = "provided"
    warnings.extend(twarn)
    return spec, sources, warnings


def load_scorecard_table(
    source,
    col_id,
    col_score,
    cols_pred_used,
    connection=None,
    col_date=None,
    cols_segment=None,
    col_target=None,
    col_obs=None,
    cols_pred=None,
    cols_pred_woe=None,
    col_fantomas=None,
    grouping_path=None,
    portfolio=None,
    sql_path=None,
    sql_dir=None,
    enrich_df=None,
    target_mapping=None,
    warn=True,
    confirm=False,
    cache_filename=None,
    auto_confirm=False,
    score_higher_is_risk=True,
):
    """Load a user table, infer optional fields, optionally confirm in a notebook.

    Mandatory: col_id, col_score, cols_pred_used.
    Optional columns are used when present; otherwise we look them up via a local
    ``.sql`` file keyed by col_id (with a warning).
    """
    if isinstance(cols_pred_used, str):
        cols_pred_used = [cols_pred_used]
    if isinstance(cols_segment, str):
        cols_segment = [cols_segment]
    if isinstance(cols_pred, str):
        cols_pred = [cols_pred]
    if isinstance(cols_pred_woe, str):
        cols_pred_woe = [cols_pred_woe]

    df = _read_table(source, connection=connection)
    missing_mandatory = [c for c in [col_id, col_score] + list(cols_pred_used or []) if c not in df.columns]
    if missing_mandatory:
        raise ValueError("Base table is missing mandatory columns: " + ", ".join(missing_mandatory))

    spec = DataSpec(
        col_id=col_id,
        col_score=col_score,
        cols_pred_used=list(cols_pred_used or []),
        col_date=col_date,
        cols_segment=list(cols_segment or []),
        col_target=col_target,
        col_obs=col_obs,
        cols_pred=list(cols_pred or []),
        cols_pred_woe=list(cols_pred_woe or []),
        col_fantomas=col_fantomas,
        grouping_path=grouping_path,
        portfolio=portfolio,
        score_higher_is_risk=score_higher_is_risk,
    )

    need_date = spec.col_date is None or spec.col_date not in df.columns
    need_target = spec.col_target is None or spec.col_target not in df.columns
    need_obs = spec.col_obs is None or spec.col_obs not in df.columns
    need_seg = not spec.cols_segment or any(c not in df.columns for c in spec.cols_segment)
    needs_enrich = need_date or need_target or need_obs or need_seg

    sql_file = resolve_sql_path(sql_path, sql_dir=sql_dir)
    if needs_enrich and enrich_df is not None:
        if warn:
            warn_user("Optional columns missing on the base table; joining enrich_df on %s." % col_id)
        df = _merge_enrich(df, enrich_df, col_id)
    elif needs_enrich and connection is not None and sql_file:
        extra, _sql = enrich_from_sql(df, col_id, connection, sql_file, warn=warn)
        df = _merge_enrich(df, extra, col_id)
        # Map generic aliases from the SQL template
        rename = {}
        if "col_date" in extra.columns and (spec.col_date is None):
            rename["col_date"] = "col_date"
            spec.col_date = "col_date"
        if "col_target" in extra.columns and spec.col_target is None:
            spec.col_target = "col_target"
        if "col_obs" in extra.columns and spec.col_obs is None:
            spec.col_obs = "col_obs"
        seg_from_sql = [c for c in extra.columns if str(c).startswith("segment_")]
        if seg_from_sql and not spec.cols_segment:
            spec.cols_segment = seg_from_sql
    elif needs_enrich:
        if warn:
            warn_user(
                "Optional columns are missing and no enrich lookup ran. "
                "Place a query in sql/enrich_by_id.sql (or pass sql_path / enrich_df / connection) "
                "to join attributes by %s." % col_id
            )

    spec, sources, infer_warnings = infer_optional_columns(df, spec, mapping=target_mapping)
    grouping_exists = bool(spec.grouping_path and os.path.isfile(spec.grouping_path)) or os.path.isfile("grouping.json")
    if grouping_exists and not spec.grouping_path and os.path.isfile("grouping.json"):
        spec.grouping_path = os.path.abspath("grouping.json")
        sources["grouping_path"] = "local_file"
    caps, cap_warnings = compute_capabilities(spec, columns=list(df.columns), grouping_exists=grouping_exists)
    warnings = infer_warnings + cap_warnings
    if warn:
        for message in warnings:
            warn_user(message)

    plan = AnalysisPlan(
        spec=spec,
        capabilities=caps,
        warnings=warnings,
        sources=sources,
        n_rows=len(df),
        columns=list(df.columns),
    )
    if confirm or auto_confirm or cache_filename:
        confirm_and_save_plan(plan, confirmed=True if auto_confirm else None, filename=cache_filename, auto_confirm=auto_confirm)
    return df, spec, plan
