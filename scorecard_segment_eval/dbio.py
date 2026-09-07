"""Database access behind an injectable executor, with SQL kept in ``.sql`` files.

No SQL is hardcoded in Python.  Every statement lives in
``scorecard_segment_eval/sql/*.sql`` as a parameterised template and is
rendered by :func:`render_sql`.

An *executor* is any callable taking a SQL string and returning a
:class:`pandas.DataFrame`.  That is the whole contract, so a production
connection, a cached wrapper or the :class:`FakeSqlExecutor` used by the tests
and the demo notebook are interchangeable.

Rendered statements carry two machine-readable comment lines::

    -- query: fetch_portfolio
    -- params: {"table": "risk.base", "col_id": "SKP_CREDIT_CASE", ...}

A real database ignores comments.  The fake executor reads them instead of
parsing SQL, which is what keeps it honest without shipping a SQL parser.
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

SQL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql")

#: The executor contract.  ``typing.Protocol`` is 3.8+, so this is a plain
#: callable alias; see :class:`SqlExecutor` for the documented base class.
SqlExecutorType = Callable[[str], pd.DataFrame]

_TAG_RE = re.compile(r"^--\s*query:\s*(\S+)\s*$", re.MULTILINE)
_PARAMS_RE = re.compile(r"^--\s*params:\s*(\{.*\})\s*$", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class SqlExecutor(object):
    """Documented base class for SQL executors.

    Subclassing is optional -- any callable ``(str) -> DataFrame`` works.
    """

    def __call__(self, sql):
        # type: (str) -> pd.DataFrame
        raise NotImplementedError

    def execute(self, sql):
        # type: (str) -> pd.DataFrame
        return self(sql)


def available_queries():
    # type: () -> List[str]
    """Names of the shipped query templates."""
    if not os.path.isdir(SQL_DIR):
        return []
    return sorted(
        os.path.splitext(name)[0] for name in os.listdir(SQL_DIR) if name.endswith(".sql")
    )


def read_sql_template(name):
    # type: (str) -> str
    path = os.path.join(SQL_DIR, name + ".sql")
    if not os.path.isfile(path):
        raise KeyError(
            "unknown SQL template %r; available: %s" % (name, ", ".join(available_queries()))
        )
    with open(path, "r") as handle:
        return handle.read()


def split_table(table):
    # type: (str) -> Dict[str, str]
    """Split ``schema.name`` (or ``db.schema.name``) into its parts."""
    parts = str(table).split(".")
    if len(parts) == 1:
        return {"table": table, "table_schema": "", "table_name": parts[0]}
    return {"table": table, "table_schema": parts[-2], "table_name": parts[-1]}


def render_sql(name, **params):
    # type: (str, Any) -> str
    """Render a template, appending the resolved parameters as a comment."""
    template = read_sql_template(name)
    if "table" in params:
        for key, value in split_table(params["table"]).items():
            params.setdefault(key, value)
    if "columns" in params and not isinstance(params["columns"], str):
        params["columns"] = ",\n    ".join(str(c) for c in params["columns"])
    required = set(_PLACEHOLDER_RE.findall(template))
    missing = sorted(required - set(params.keys()))
    if missing:
        raise ValueError("SQL template %r is missing parameters: %s" % (name, ", ".join(missing)))
    body = template.format(**params)
    serialisable = dict((k, _jsonable(v)) for k, v in params.items())
    return body.rstrip("\n") + "\n-- params: " + json.dumps(serialisable, sort_keys=True) + "\n"


def _jsonable(value):
    # type: (Any) -> Any
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def parse_sql_metadata(sql):
    # type: (str) -> Dict[str, Any]
    """Recover ``{"query": name, "params": {...}}`` from a rendered statement."""
    tag = _TAG_RE.search(sql)
    params = _PARAMS_RE.search(sql)
    return {
        "query": tag.group(1) if tag else None,
        "params": json.loads(params.group(1)) if params else {},
    }


def _dtype_name(series):
    # type: (pd.Series) -> str
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    return "character varying"


class FakeSqlExecutor(SqlExecutor):
    """In-memory executor for tests, demos and notebooks.

    Register the frame that stands in for the database table and the fake
    answers every shipped query template against it.  Extra tables can be
    registered by name for multi-table setups.  Every executed statement is
    recorded on :attr:`queries`, so tests can assert *which* lookups happened.
    """

    def __init__(self, table=None, tables=None, hidden_columns=()):
        # type: (Optional[pd.DataFrame], Optional[Dict[str, pd.DataFrame]], Sequence[str]) -> None
        self.tables = dict(tables or {})
        if table is not None:
            self.tables.setdefault("__default__", table)
        #: Columns present in the frame but deliberately invisible to the
        #: catalogue, used to simulate a column the user must supply.
        self.hidden_columns = set(hidden_columns or ())
        self.queries = []  # type: List[Dict[str, Any]]

    # -- helpers -----------------------------------------------------------
    def register(self, name, frame):
        # type: (str, pd.DataFrame) -> None
        self.tables[str(name)] = frame

    def frame_for(self, table):
        # type: (Optional[str]) -> pd.DataFrame
        if table is not None and str(table) in self.tables:
            return self.tables[str(table)]
        if "__default__" in self.tables:
            return self.tables["__default__"]
        if len(self.tables) == 1:
            return list(self.tables.values())[0]
        raise KeyError("no registered frame for table %r" % (table,))

    def visible_columns(self, frame):
        # type: (pd.DataFrame) -> List[str]
        return [c for c in frame.columns if c not in self.hidden_columns]

    # -- executor ----------------------------------------------------------
    def __call__(self, sql):
        # type: (str) -> pd.DataFrame
        meta = parse_sql_metadata(sql)
        name = meta["query"]
        params = meta["params"]
        self.queries.append({"query": name, "params": params, "sql": sql})
        handler = getattr(self, "_q_" + str(name), None)
        if handler is None:
            raise KeyError("FakeSqlExecutor cannot answer query %r" % (name,))
        return handler(params)

    # -- query handlers ----------------------------------------------------
    def _q_describe_table(self, params):
        # type: (Dict[str, Any]) -> pd.DataFrame
        frame = self.frame_for(params.get("table"))
        rows = [
            {"column_name": col, "data_type": _dtype_name(frame[col])}
            for col in self.visible_columns(frame)
        ]
        return pd.DataFrame(rows, columns=["column_name", "data_type"])

    def _q_column_profile(self, params):
        # type: (Dict[str, Any]) -> pd.DataFrame
        frame = self.frame_for(params.get("table"))
        column = str(params.get("column"))
        col_id = str(params.get("col_id"))
        limit = int(params.get("sample_limit") or len(frame))
        if column not in frame.columns:
            return pd.DataFrame(columns=["value", "n_rows", "n_cases"])
        sample = frame.head(limit)
        grouped = sample.groupby(column, dropna=False)
        out = pd.DataFrame(
            {
                "value": [k for k, _ in grouped],
                "n_rows": grouped.size().to_numpy(),
                "n_cases": (
                    grouped[col_id].nunique().to_numpy()
                    if col_id in sample.columns
                    else grouped.size().to_numpy()
                ),
            }
        )
        return out.sort_values("n_rows", ascending=False).reset_index(drop=True)

    def _q_fetch_portfolio(self, params):
        # type: (Dict[str, Any]) -> pd.DataFrame
        frame = self.frame_for(params.get("table"))
        col_portfolio = str(params.get("col_portfolio"))
        col_id = str(params.get("col_id"))
        limit = int(params.get("sample_limit") or len(frame))
        if col_portfolio not in frame.columns:
            return pd.DataFrame(columns=["portfolio", "n_cases"])
        sample = frame.head(limit)
        grouped = sample.groupby(col_portfolio, dropna=False)
        out = pd.DataFrame(
            {
                "portfolio": [k for k, _ in grouped],
                "n_cases": (
                    grouped[col_id].nunique().to_numpy()
                    if col_id in sample.columns
                    else grouped.size().to_numpy()
                ),
            }
        )
        return out.sort_values("n_cases", ascending=False).reset_index(drop=True)

    def _q_fetch_columns_by_id(self, params):
        # type: (Dict[str, Any]) -> pd.DataFrame
        frame = self.frame_for(params.get("table"))
        col_id = str(params.get("col_id"))
        raw = params.get("columns") or ""
        wanted = [c.strip() for c in str(raw).replace("\n", "").split(",") if c.strip()]
        keep = [col_id] + [c for c in wanted if c in frame.columns and c != col_id]
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise KeyError("FakeSqlExecutor: table has no column(s) %s" % (", ".join(missing),))
        return frame.loc[:, keep].copy()
