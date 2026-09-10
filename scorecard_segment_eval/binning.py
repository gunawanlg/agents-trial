"""Optimal WoE binning with a serialisable grouping definition.

Three paths, chosen by ``Gates.binning_method`` (``"auto"`` by default):

``optbinning``
    Used when the optional :mod:`optbinning` package is importable.  It solves
    the constrained binning problem directly (monotonic trend, minimum bin
    size, maximum bin count).
``tree``
    The always-available fallback.  A shallow :class:`DecisionTreeClassifier`
    is fitted on the single predictor against the target, and its split
    thresholds become candidate bin edges.  Depth and leaf size are constrained
    from the gates, so the result is a supervised, coarse grouping rather than
    an arbitrary equal-frequency cut.
``quantile``
    The historical equal-frequency behaviour, kept as an explicit escape hatch
    and as the last resort when a tree cannot find any split.

Whichever path runs, the same post-processing applies: missing values get their
own bin, bins below the minimum size fraction or minimum event count are merged
into their smallest neighbour, and adjacent WoE violations are merged until the
WoE profile is monotone (when ``Gates.binning_monotonic`` is set and the
feature is numeric).

The resulting :class:`BinningModel` round-trips through JSON, which is the
``grouping.json`` artefact consumed by recalibration and grouping-based PSI.
"""

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scorecard_segment_eval.parallel import map_jobs
from scorecard_segment_eval.schema import Gates, _norm, _strip_woe_affix

MISSING_LABEL = "__missing__"
OTHER_LABEL = "__other__"
GROUPING_VERSION = 1
_WOE_EPS = 0.5


def optbinning_available():
    # type: () -> bool
    try:
        import optbinning  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# WoE / IV primitives
# --------------------------------------------------------------------------
def woe_table(labels, y, eps=_WOE_EPS):
    # type: (Sequence[Any], Sequence[float], float) -> pd.DataFrame
    """Per-bin count / event / WoE table.

    WoE follows the package convention ``log(good_share / bad_share)``, so a
    positive WoE means the bin is better than average.
    """
    y_arr = np.asarray(y, dtype=float)
    frame = pd.DataFrame({"bin": pd.Series(list(labels), dtype="object"), "y": y_arr})
    grouped = frame.groupby("bin", dropna=False)
    total = grouped.size().astype(float)
    bad = grouped["y"].sum().astype(float)
    good = total - bad
    n_bad = float(y_arr.sum())
    n_good = float(len(y_arr) - n_bad)
    bad_share = (bad + eps) / max(n_bad + 2.0 * eps, 1.0)
    good_share = (good + eps) / max(n_good + 2.0 * eps, 1.0)
    out = pd.DataFrame(
        {
            "count": total,
            "events": bad,
            "non_events": good,
            "event_rate": np.where(total > 0, bad / np.maximum(total, 1.0), np.nan),
            "woe": np.log(good_share / bad_share),
            "iv_part": (good_share - bad_share) * np.log(good_share / bad_share),
        }
    )
    out.index.name = "bin"
    return out


def information_value(labels, y, eps=_WOE_EPS):
    # type: (Sequence[Any], Sequence[float], float) -> float
    table = woe_table(labels, y, eps=eps)
    if table.empty:
        return float("nan")
    return float(table["iv_part"].sum())


def binomial_rate_bound(events, n, z=1.64):
    # type: (float, float, float) -> Tuple[float, float]
    """Wilson score interval for a binomial event rate."""
    n = float(n)
    if n <= 0:
        return float("nan"), float("nan")
    k = float(np.clip(events, 0.0, n))
    p = k / n
    z = float(z)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    inner = p * (1.0 - p) / n + z2 / (4.0 * n * n)
    if inner < 0:
        inner = 0.0
    margin = z * math.sqrt(inner) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return lo, hi


def _is_logit_spec(spec):
    # type: (BinSpec) -> bool
    return spec.kind == "logit" or spec.transform == "logit"


def _has_numeric_edges(spec):
    # type: (BinSpec) -> bool
    return spec.edges is not None and len(spec.edges) >= 2


def _numeric_bin_count(spec):
    # type: (BinSpec) -> int
    if spec.edges is None:
        return 0
    return max(len(spec.edges) - 1, 0)


# --------------------------------------------------------------------------
# Edge helpers
# --------------------------------------------------------------------------
def _fmt(value, precision):
    # type: (float, int) -> str
    if value == -np.inf:
        return "-inf"
    if value == np.inf:
        return "inf"
    return ("%." + str(precision) + "g") % value


def edge_labels(edges, closed="right"):
    # type: (Sequence[float], str) -> List[str]
    """Human-readable, guaranteed-unique labels for numeric bins.

    ``closed="right"`` (the historical default) uses ``(a, b]``.
    ``closed="left"`` uses ``[a, b)``, matching SQL ``WHEN x < t`` chains.
    """
    left_closed = closed == "left"
    for precision in (6, 12):
        labels = []
        for i in range(len(edges) - 1):
            lo = _fmt(float(edges[i]), precision)
            hi = _fmt(float(edges[i + 1]), precision)
            left = "[" if left_closed else "("
            right = ")" if left_closed or edges[i + 1] == np.inf else "]"
            if (not left_closed) and edges[i + 1] == np.inf:
                right = ")"
            labels.append(left + lo + ", " + hi + right)
        if len(set(labels)) == len(labels):
            return labels
    return ["bin_%02d" % i for i in range(len(edges) - 1)]


def assign_numeric_bins(values, edges, labels=None, missing_label=MISSING_LABEL, closed="right"):
    # type: (Any, Sequence[float], Optional[Sequence[str]], str, str) -> np.ndarray
    """Map numeric values onto ``edges``.

    ``closed="right"`` uses ``(a, b]`` (``searchsorted`` side ``left``).
    ``closed="left"`` uses ``[a, b)``, matching SQL ``x < t`` / ``x >= t``.
    Missing values (including non-coercible ones) get ``missing_label``.
    """
    numeric = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    names = list(labels) if labels is not None else edge_labels(edges, closed=closed)
    inner = np.asarray(edges, dtype=float)[1:-1]
    out = np.empty(len(arr), dtype=object)
    isnan = ~np.isfinite(arr)
    side = "left" if closed != "left" else "right"
    if len(inner):
        idx = np.searchsorted(inner, np.where(isnan, 0.0, arr), side=side)
    else:
        idx = np.zeros(len(arr), dtype=int)
    idx = np.clip(idx, 0, max(len(names) - 1, 0))
    for i in range(len(arr)):
        out[i] = missing_label if isnan[i] else names[int(idx[i])]
    return out


def _as_str_levels(values, missing_label=MISSING_LABEL):
    # type: (Any, str) -> np.ndarray
    series = pd.Series(values).reset_index(drop=True)
    out = series.astype("object").where(series.notna(), missing_label)
    return out.map(lambda v: missing_label if v is None else str(v)).to_numpy(dtype=object)


# --------------------------------------------------------------------------
# Bin specification
# --------------------------------------------------------------------------
class BinSpec(object):
    """Serialisable binning definition for one predictor."""

    def __init__(
        self,
        feature,
        kind,
        method,
        edges=None,
        groups=None,
        labels=None,
        woe=None,
        counts=None,
        events=None,
        iv=float("nan"),
        monotonic=False,
        missing_label=MISSING_LABEL,
        notes=None,
        closed="right",
        impute=None,
        transform=None,
        output_name=None,
        rules=None,
    ):
        # type: (str, str, str, Optional[Sequence[float]], Optional[Dict[str, List[str]]], Optional[Sequence[str]], Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[Dict[str, float]], float, bool, str, Optional[List[str]], str, Optional[float], Optional[str], Optional[str], Optional[Sequence[Dict[str, Any]]]) -> None
        self.feature = feature
        self.kind = kind
        self.method = method
        self.edges = [float(e) for e in edges] if edges is not None else None
        self.groups = {str(k): [str(v) for v in vals] for k, vals in (groups or {}).items()} or None
        self.labels = list(labels) if labels is not None else []
        self.woe = dict(woe or {})
        self.counts = dict(counts or {})
        self.events = dict(events or {})
        self.iv = float(iv)
        self.monotonic = bool(monotonic)
        self.missing_label = missing_label
        self.notes = list(notes or [])
        self.closed = closed or "right"
        self.impute = None if impute is None else float(impute)
        self.transform = transform
        self.output_name = output_name
        self.rules = [dict(r) for r in (rules or [])]

    # -- application -------------------------------------------------------
    def assign(self, values):
        # type: (Any) -> np.ndarray
        """Return the bin label of every value."""
        if self.kind == "logit" or self.transform == "logit":
            return _assign_logit_labels(values, self.missing_label)
        if self.rules:
            return apply_sql_rules(values, self.rules, self.missing_label, return_woe=False)
        if self.kind in ("numeric", "mixed") and self.edges is not None:
            if self.kind == "mixed":
                return _assign_mixed(
                    values,
                    self.edges,
                    self.labels,
                    self.groups,
                    self.missing_label,
                    self.closed,
                )
            return assign_numeric_bins(
                values, self.edges, self.labels, self.missing_label, closed=self.closed
            )
        levels = _as_str_levels(values, self.missing_label)
        lookup = {}
        for label, members in (self.groups or {}).items():
            for member in members:
                lookup[member] = label
        out = np.empty(len(levels), dtype=object)
        for i, level in enumerate(levels):
            if level == self.missing_label:
                out[i] = self.missing_label
            else:
                out[i] = lookup.get(level, OTHER_LABEL)
        return out

    def transform_woe(self, values):
        # type: (Any) -> np.ndarray
        if self.kind == "logit" or self.transform == "logit":
            return logit_transform(values, impute=self.impute)
        if self.rules:
            return apply_sql_rules(values, self.rules, self.missing_label, return_woe=True)
        labels = self.assign(values)
        default = float(self.woe.get(OTHER_LABEL, 0.0))
        return np.asarray([float(self.woe.get(label, default)) for label in labels], dtype=float)

    def bin_order(self):
        # type: () -> List[str]
        """All labels in reporting order (bins first, missing/other last)."""
        order = list(self.labels)
        for extra in (self.missing_label, OTHER_LABEL):
            if extra in self.woe and extra not in order:
                order.append(extra)
        return order

    # -- serialisation -----------------------------------------------------
    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "feature": self.feature,
            "kind": self.kind,
            "method": self.method,
            "edges": None if self.edges is None else [_json_float(e) for e in self.edges],
            "groups": self.groups,
            "labels": list(self.labels),
            "woe": {k: _json_float(v) for k, v in self.woe.items()},
            "counts": {k: _json_float(v) for k, v in self.counts.items()},
            "events": {k: _json_float(v) for k, v in self.events.items()},
            "iv": _json_float(self.iv),
            "monotonic": self.monotonic,
            "missing_label": self.missing_label,
            "notes": list(self.notes),
            "missing_note": "; ".join(self.notes),
            "closed": self.closed,
            "impute": None if self.impute is None else _json_float(self.impute),
            "else_woe": _json_float(self.woe[OTHER_LABEL]) if OTHER_LABEL in self.woe else None,
            "transform": self.transform,
            "output_name": self.output_name,
            "rules": [_rule_to_dict(r) for r in self.rules],
        }

    @classmethod
    def from_dict(cls, payload):
        # type: (Dict[str, Any]) -> "BinSpec"
        edges = payload.get("edges")
        if edges is not None:
            edges = [_from_json_float(e) for e in edges]
        return cls(
            feature=payload["feature"],
            kind=payload.get("kind", "numeric"),
            method=payload.get("method", "unknown"),
            edges=edges,
            groups=payload.get("groups"),
            labels=payload.get("labels") or [],
            woe={k: _from_json_float(v) for k, v in (payload.get("woe") or {}).items()},
            counts={k: _from_json_float(v) for k, v in (payload.get("counts") or {}).items()},
            events={k: _from_json_float(v) for k, v in (payload.get("events") or {}).items()},
            iv=_from_json_float(payload.get("iv", float("nan"))),
            monotonic=bool(payload.get("monotonic", False)),
            missing_label=payload.get("missing_label", MISSING_LABEL),
            notes=payload.get("notes") or [],
            closed=payload.get("closed") or "right",
            impute=_optional_json_float(payload.get("impute")),
            transform=payload.get("transform"),
            output_name=payload.get("output_name"),
            rules=payload.get("rules") or [],
        )


def _json_float(value):
    # type: (Any) -> Any
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return "nan"
    if val == float("inf"):
        return "inf"
    if val == float("-inf"):
        return "-inf"
    return val


def _from_json_float(value):
    # type: (Any) -> float
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "nan":
            return float("nan")
        if lowered in ("inf", "+inf", "infinity"):
            return float("inf")
        if lowered in ("-inf", "-infinity"):
            return float("-inf")
        return float(value)
    if value is None:
        return float("nan")
    return float(value)


def _optional_json_float(value):
    # type: (Any) -> Optional[float]
    if value is None:
        return None
    return _from_json_float(value)


def _rule_to_dict(rule):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    out = dict(rule)
    if "woe" in out:
        out["woe"] = _json_float(out["woe"])
    if "threshold" in out and out["threshold"] is not None:
        out["threshold"] = _json_float(out["threshold"])
    return out


def logit_transform(values, impute=None, eps=1e-6):
    # type: (Any, Optional[float], float) -> np.ndarray
    """``log(p / (1-p))`` with optional imputation for null / invalid p."""
    numeric = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    clipped = np.clip(arr, eps, 1.0 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(clipped / (1.0 - clipped))
    invalid = ~np.isfinite(arr) | (arr <= 0.0) | (arr >= 1.0) | ~np.isfinite(out)
    if impute is None:
        out[invalid] = np.nan
    else:
        out[invalid] = float(impute)
    return out.astype(float)


def _assign_logit_labels(values, missing_label=MISSING_LABEL):
    # type: (Any, str) -> np.ndarray
    numeric = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    out = np.empty(len(arr), dtype=object)
    for i in range(len(arr)):
        out[i] = missing_label if not np.isfinite(arr[i]) else "logit"
    return out


def _assign_mixed(values, edges, labels, groups, missing_label, closed):
    # type: (Any, Sequence[float], Sequence[str], Optional[Dict[str, List[str]]], str, str) -> np.ndarray
    series = pd.Series(values).reset_index(drop=True)
    lookup = {}
    for label, members in (groups or {}).items():
        for member in members:
            lookup[str(member)] = label
    numeric_labels = assign_numeric_bins(series, edges, labels, missing_label, closed=closed)
    out = np.empty(len(series), dtype=object)
    for i, value in enumerate(series.tolist()):
        if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
            out[i] = missing_label
            continue
        key = str(value)
        if key in lookup:
            out[i] = lookup[key]
        elif numeric_labels[i] != missing_label:
            out[i] = numeric_labels[i]
        else:
            out[i] = OTHER_LABEL
    return out


def apply_sql_rules(values, rules, missing_label=MISSING_LABEL, return_woe=False):
    # type: (Any, Sequence[Dict[str, Any]], str, bool) -> np.ndarray
    """Apply ordered SQL CASE WHEN rules; first match wins."""
    series = pd.Series(values).reset_index(drop=True)
    n = len(series)
    matched = np.zeros(n, dtype=bool)
    if return_woe:
        out = np.zeros(n, dtype=float)  # type: Any
    else:
        out = np.empty(n, dtype=object)
    numeric = pd.to_numeric(series, errors="coerce")
    num = numeric.to_numpy(dtype=float)
    is_null = series.isna().to_numpy()
    str_values = series.map(lambda v: missing_label if pd.isna(v) else str(v)).to_numpy(dtype=object)
    for rule in rules:
        op = str(rule.get("op", "")).lower()
        remaining = ~matched
        hit = np.zeros(n, dtype=bool)
        if op == "null":
            hit = is_null & remaining
        elif op == "eq":
            want = str(rule.get("value", ""))
            hit = (str_values == want) & (~is_null) & remaining
        elif op in ("lt", "le", "gt", "ge"):
            threshold = float(rule["threshold"])
            finite = np.isfinite(num) & remaining
            if op == "lt":
                hit = finite & (num < threshold)
            elif op == "le":
                hit = finite & (num <= threshold)
            elif op == "gt":
                hit = finite & (num > threshold)
            else:
                hit = finite & (num >= threshold)
        elif op == "else":
            hit = remaining
        else:
            continue
        if return_woe:
            out[hit] = float(rule.get("woe", 0.0))
        else:
            label = rule.get("label") or (missing_label if op == "null" else OTHER_LABEL)
            for i in np.where(hit)[0]:
                out[i] = label
        matched[hit] = True
    if not return_woe:
        for i in range(n):
            if not matched[i]:
                out[i] = missing_label if is_null[i] else OTHER_LABEL
    return out


# --------------------------------------------------------------------------
# Candidate edge search
# --------------------------------------------------------------------------
def _quantile_edges(values, max_bins):
    # type: (np.ndarray, int) -> Optional[np.ndarray]
    if len(values) == 0:
        return None
    qs = np.linspace(0.0, 1.0, int(max_bins) + 1)
    edges = np.unique(np.quantile(values, qs))
    if len(edges) < 3:
        return None
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _tree_edges(values, y, gates):
    # type: (np.ndarray, np.ndarray, Gates) -> Optional[np.ndarray]
    """Split thresholds of a depth/leaf-constrained decision tree."""
    from sklearn.tree import DecisionTreeClassifier

    n = len(values)
    if n < 10 or len(np.unique(y)) < 2 or len(np.unique(values)) < 3:
        return None
    max_bins = max(int(gates.binning_max_bins), 2)
    min_leaf = max(int(math.ceil(gates.binning_min_bin_frac * n)), int(gates.binning_min_bin_events), 1)
    min_leaf = min(min_leaf, max(n // 2, 1))
    max_depth = max(int(math.ceil(math.log(max_bins, 2))), 1)
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_leaf,
        random_state=0,
    )
    tree.fit(values.reshape(-1, 1), y)
    thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
    thresholds = np.unique(np.asarray(thresholds, dtype=float))
    thresholds = thresholds[np.isfinite(thresholds)]
    if len(thresholds) == 0:
        return None
    return np.concatenate(([-np.inf], thresholds, [np.inf]))


def _optbinning_edges(values, y, gates):
    # type: (np.ndarray, np.ndarray, Gates) -> Optional[np.ndarray]
    from optbinning import OptimalBinning

    trend = "auto_asc_desc" if gates.binning_monotonic else "auto"
    binner = OptimalBinning(
        name="x",
        dtype="numerical",
        monotonic_trend=trend,
        min_bin_size=float(gates.binning_min_bin_frac),
        max_n_bins=int(gates.binning_max_bins),
        min_bin_n_event=int(gates.binning_min_bin_events),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binner.fit(values, y)
    splits = np.unique(np.asarray(getattr(binner, "splits", []), dtype=float))
    splits = splits[np.isfinite(splits)]
    if len(splits) == 0:
        return None
    return np.concatenate(([-np.inf], splits, [np.inf]))


# --------------------------------------------------------------------------
# Post-processing: sparse-bin merge and monotonicity
# --------------------------------------------------------------------------
def _merge_sparse_numeric(edges, values, y, gates):
    # type: (np.ndarray, np.ndarray, np.ndarray, Gates) -> Tuple[np.ndarray, List[str]]
    """Drop interior edges until every bin clears the size/event floors."""
    notes = []  # type: List[str]
    edges = np.asarray(edges, dtype=float)
    n = max(len(values), 1)
    min_count = max(gates.binning_min_bin_frac * n, 1.0)
    min_events = max(float(gates.binning_min_bin_events), 0.0)
    guard = 0
    while len(edges) > 3 and guard < 100:
        guard += 1
        idx = np.clip(np.searchsorted(edges[1:-1], values, side="left"), 0, len(edges) - 2)
        counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        events = np.bincount(idx, weights=y, minlength=len(edges) - 1).astype(float)
        bad = [
            i
            for i in range(len(counts))
            if counts[i] < min_count or events[i] < min_events or (counts[i] - events[i]) < min_events
        ]
        if not bad:
            break
        # Merge the weakest bin into its smaller neighbour by dropping an edge.
        weakest = int(min(bad, key=lambda i: (counts[i], events[i])))
        if weakest == 0:
            drop = 1
        elif weakest == len(counts) - 1:
            drop = len(edges) - 2
        else:
            left_edge, right_edge = weakest, weakest + 1
            drop = left_edge if counts[weakest - 1] <= counts[weakest + 1] else right_edge
        edges = np.delete(edges, drop)
        notes.append("merged_sparse_bin")
    return edges, notes


def _bin_index(values, edges):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    inner = np.asarray(edges, dtype=float)[1:-1]
    if len(inner) == 0:
        return np.zeros(len(values), dtype=int)
    return np.clip(np.searchsorted(inner, values, side="left"), 0, len(edges) - 2)


def _woe_profile(edges, values, y):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]
    idx = _bin_index(values, edges)
    table = woe_table(idx.astype(object), y)
    n_bins = len(edges) - 1
    woes = np.asarray([float(table["woe"].get(i, float("nan"))) for i in range(n_bins)], dtype=float)
    counts = np.asarray([float(table["count"].get(i, 0.0)) for i in range(n_bins)], dtype=float)
    return woes, counts, float(table["iv_part"].sum())


def _monotonic_direction(woe_by_index, counts=None):
    # type: (np.ndarray, Optional[np.ndarray]) -> int
    finite = np.isfinite(woe_by_index)
    if finite.sum() < 2:
        return 0
    xs = np.arange(len(woe_by_index), dtype=float)[finite]
    ys = np.asarray(woe_by_index, dtype=float)[finite]
    weights = None
    if counts is not None:
        weights = np.asarray(counts, dtype=float)[finite]
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = None
    slope = float(np.polyfit(xs, ys, 1, w=weights)[0])
    if slope > 0:
        return 1
    if slope < 0:
        return -1
    return 0


def _enforce_monotonic_numeric(edges, values, y, gates):
    # type: (np.ndarray, np.ndarray, np.ndarray, Gates) -> Tuple[np.ndarray, bool, List[str]]
    """Pool adjacent WoE violators, but only while it stays *feasible*.

    A pooled book can genuinely be non-monotone in a predictor -- that is
    exactly the sign-flip situation this package hunts for.  Forcing
    monotonicity there would merge away the very bins that carry the signal, so
    the constrained solution is accepted only when it retains at least
    ``1 - binning_monotonic_iv_tolerance`` of the unconstrained IV.  Otherwise
    the unconstrained grouping is kept and flagged.
    """
    notes = []  # type: List[str]
    original = np.asarray(edges, dtype=float)
    _, _, iv_original = _woe_profile(original, values, y)
    candidate = original.copy()
    merged = 0
    guard = 0
    monotone = False
    while len(candidate) > 3 and guard < 100:
        guard += 1
        woes, counts, _ = _woe_profile(candidate, values, y)
        direction = _monotonic_direction(woes, counts)
        if direction == 0:
            break
        diffs = np.diff(woes) * direction
        violations = np.where(~(diffs >= 0) | ~np.isfinite(diffs))[0]
        if len(violations) == 0:
            monotone = True
            break
        worst = int(violations[int(np.argmin(np.nan_to_num(diffs[violations], nan=-np.inf)))])
        candidate = np.delete(candidate, worst + 1)
        merged += 1
    if merged == 0:
        return original, monotone, notes
    _, _, iv_candidate = _woe_profile(candidate, values, y)
    tolerance = float(getattr(gates, "binning_monotonic_iv_tolerance", 0.2))
    keeps_signal = (
        not np.isfinite(iv_original)
        or iv_original <= 0
        or iv_candidate >= (1.0 - tolerance) * iv_original
    )
    if monotone and keeps_signal:
        notes.extend(["merged_monotonicity_violation"] * merged)
        return candidate, True, notes
    notes.append("monotonicity_not_feasible")
    return original, False, notes


def _group_categorical(levels, y, gates):
    # type: (np.ndarray, np.ndarray, Gates) -> Tuple[Dict[str, List[str]], List[str], List[str]]
    """Merge sparse categories into WoE-adjacent groups."""
    notes = []  # type: List[str]
    table = woe_table(levels, y).sort_values("woe")
    n = max(len(levels), 1)
    min_count = max(gates.binning_min_bin_frac * n, 1.0)
    min_events = max(float(gates.binning_min_bin_events), 0.0)
    buckets = []  # type: List[Dict[str, Any]]
    for level, row in table.iterrows():
        if level == MISSING_LABEL:
            continue
        entry = {"members": [str(level)], "count": float(row["count"]), "events": float(row["events"])}
        if buckets and (
            buckets[-1]["count"] < min_count
            or buckets[-1]["events"] < min_events
            or (buckets[-1]["count"] - buckets[-1]["events"]) < min_events
        ):
            buckets[-1]["members"].extend(entry["members"])
            buckets[-1]["count"] += entry["count"]
            buckets[-1]["events"] += entry["events"]
            notes.append("merged_sparse_category")
        else:
            buckets.append(entry)
    while len(buckets) > 1 and (
        buckets[-1]["count"] < min_count
        or buckets[-1]["events"] < min_events
        or (buckets[-1]["count"] - buckets[-1]["events"]) < min_events
    ):
        tail = buckets.pop()
        buckets[-1]["members"].extend(tail["members"])
        buckets[-1]["count"] += tail["count"]
        buckets[-1]["events"] += tail["events"]
        notes.append("merged_sparse_category")
    while len(buckets) > max(int(gates.binning_max_bins), 2):
        smallest = int(min(range(len(buckets)), key=lambda i: buckets[i]["count"]))
        neighbour = smallest - 1 if smallest > 0 else 1
        buckets[neighbour]["members"].extend(buckets[smallest]["members"])
        buckets[neighbour]["count"] += buckets[smallest]["count"]
        buckets[neighbour]["events"] += buckets[smallest]["events"]
        buckets.pop(smallest)
        notes.append("merged_to_max_bins")
    groups = {}  # type: Dict[str, List[str]]
    labels = []  # type: List[str]
    for i, bucket in enumerate(buckets):
        members = sorted(bucket["members"])
        label = "|".join(members) if len("|".join(members)) <= 60 else "grp_%02d" % i
        if label in groups:
            label = "grp_%02d" % i
        groups[label] = members
        labels.append(label)
    return groups, labels, notes


# --------------------------------------------------------------------------
# Fitting one predictor
# --------------------------------------------------------------------------
def fit_bin_spec(series, y, gates=None, feature=None):
    # type: (Any, Any, Optional[Gates], Optional[str]) -> BinSpec
    """Fit an optimal :class:`BinSpec` for one predictor."""
    gates = gates or Gates()
    name = feature or getattr(series, "name", None) or "feature"
    series = pd.Series(series).reset_index(drop=True)
    y_arr = np.asarray(pd.Series(y).reset_index(drop=True).to_numpy(), dtype=float)
    notes = []  # type: List[str]

    numeric_like = pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2
    if not numeric_like:
        levels = _as_str_levels(series)
        groups, labels, cat_notes = _group_categorical(levels, y_arr, gates)
        spec = BinSpec(
            feature=str(name),
            kind="categorical",
            method="categorical_woe",
            groups=groups,
            labels=labels,
            notes=notes + cat_notes,
        )
        _finalise_stats(spec, series, y_arr)
        return spec

    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.notna().to_numpy()
    values = numeric.to_numpy(dtype=float)[mask]
    y_valid = y_arr[mask]
    if len(values) and (~mask).any():
        notes.append("missing_values_isolated")

    method = str(gates.binning_method or "auto").lower()
    edges = None  # type: Optional[np.ndarray]
    used = "quantile"
    if method in ("auto", "optbinning") and optbinning_available():
        try:
            edges = _optbinning_edges(values, y_valid, gates)
            used = "optbinning"
        except Exception as exc:  # pragma: no cover - only with optbinning present
            warnings.warn("optbinning failed for %r (%s); falling back to tree binning" % (name, exc))
            edges = None
    if edges is None and method in ("auto", "optbinning", "tree"):
        if method == "optbinning" and not optbinning_available():
            warnings.warn(
                "binning_method='optbinning' requested but optbinning is not installed; "
                "using the decision-tree fallback for %r" % (name,)
            )
        try:
            edges = _tree_edges(values, y_valid, gates)
            used = "tree"
        except Exception as exc:
            warnings.warn("tree binning failed for %r (%s); falling back to quantiles" % (name, exc))
            edges = None
    if edges is None:
        edges = _quantile_edges(values, gates.binning_max_bins)
        used = "quantile"
        notes.append("quantile_fallback")
    if edges is None:
        levels = _as_str_levels(series)
        groups, labels, cat_notes = _group_categorical(levels, y_arr, gates)
        spec = BinSpec(
            feature=str(name),
            kind="categorical",
            method="categorical_degenerate_numeric",
            groups=groups,
            labels=labels,
            notes=notes + cat_notes + ["numeric_had_too_few_distinct_values"],
        )
        _finalise_stats(spec, series, y_arr)
        return spec

    edges, merge_notes = _merge_sparse_numeric(edges, values, y_valid, gates)
    notes.extend(merge_notes)
    monotone = False
    if gates.binning_monotonic:
        edges, monotone, mono_notes = _enforce_monotonic_numeric(edges, values, y_valid, gates)
        notes.extend(mono_notes)

    spec = BinSpec(
        feature=str(name),
        kind="numeric",
        method=used,
        edges=[float(e) for e in edges],
        labels=edge_labels(edges),
        monotonic=monotone,
        notes=notes,
    )
    _finalise_stats(spec, series, y_arr)
    return spec


def _finalise_stats(spec, series, y_arr):
    # type: (BinSpec, Any, np.ndarray) -> None
    labels = spec.assign(series)
    table = woe_table(labels, y_arr)
    spec.woe = {str(k): float(v) for k, v in table["woe"].items()}
    spec.counts = {str(k): float(v) for k, v in table["count"].items()}
    spec.events = {str(k): float(v) for k, v in table["events"].items()}
    spec.iv = float(table["iv_part"].sum())
    # Empty bins still need a WoE so that transform never produces a surprise.
    for label in spec.labels:
        spec.woe.setdefault(label, 0.0)
        spec.counts.setdefault(label, 0.0)
        spec.events.setdefault(label, 0.0)
    for label in (spec.missing_label, OTHER_LABEL):
        spec.woe.setdefault(label, 0.0)
        spec.counts.setdefault(label, 0.0)
        spec.events.setdefault(label, 0.0)


def _vintage_axis_values(vintages):
    # type: (Sequence[str]) -> Tuple[List[Any], bool]
    """Parse vintage labels to timestamps when every label is date-like."""
    parsed = []  # type: List[Any]
    for value in vintages:
        ts = None  # type: Any
        try:
            ts = pd.Period(str(value)).to_timestamp()
        except Exception:
            ts = pd.to_datetime(str(value), errors="coerce")
        if ts is None or pd.isna(ts):
            return list(range(len(vintages))), False
        parsed.append(pd.Timestamp(ts).to_pydatetime())
    return parsed, True


# --------------------------------------------------------------------------
# Model over many predictors
# --------------------------------------------------------------------------
class BinningModel(object):
    """A collection of :class:`BinSpec` objects, i.e. a grouping definition."""

    def __init__(self, specs=None, gates=None):
        # type: (Optional[Dict[str, BinSpec]], Optional[Gates]) -> None
        self.specs = dict(specs or {})
        self.gates = gates or Gates()
        self.columns = list(self.specs.keys())  # type: List[str]

    def closed(self, feature):
        # type: (str) -> str
        spec = self.specs.get(feature)
        if spec is None:
            return "right"
        return spec.closed or "right"

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(cls, X, y, gates=None, n_jobs=None, date_col=None):
        # type: (pd.DataFrame, Any, Optional[Gates], Optional[int], Optional[str]) -> "BinningModel"
        gates = gates or Gates()
        frame = pd.DataFrame(X).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)
        columns = list(frame.columns)
        fit_cols = [c for c in columns if c != date_col]

        def _one(col):
            # type: (str) -> Tuple[str, BinSpec]
            return col, fit_bin_spec(frame[col], y_series, gates=gates, feature=str(col))

        jobs = n_jobs if n_jobs is not None else gates.n_jobs
        pairs = map_jobs(_one, fit_cols, n_jobs=jobs, cap=gates.max_workers_cap)
        model = cls(specs=dict(pairs), gates=gates)
        model.columns = [str(c) for c in fit_cols]
        if date_col is not None and date_col in frame.columns:
            model.merge_overlapping_event_rate_bounds(
                frame, y_series, date_col, gates=gates
            )
        return model

    # -- application -------------------------------------------------------
    def bin_frame(self, X):
        # type: (pd.DataFrame) -> pd.DataFrame
        """Bin labels for every known column present in ``X``."""
        frame = pd.DataFrame(X).reset_index(drop=True)
        data = {}
        for col in self.columns:
            if col in frame.columns and col in self.specs:
                data[col] = self.specs[col].assign(frame[col])
        return pd.DataFrame(data, columns=[c for c in self.columns if c in data])

    def transform(self, X):
        # type: (pd.DataFrame) -> np.ndarray
        """WoE matrix in ``self.columns`` order.  Unseen levels map to 0.0."""
        mapped = self.transform_woe(X)
        if mapped.empty:
            return np.zeros((len(pd.DataFrame(X)), 0))
        return mapped.to_numpy(dtype=float)

    def transform_woe(self, X):
        # type: (pd.DataFrame) -> pd.DataFrame
        """Transformed columns (WoE / logit) in ``self.columns`` order."""
        frame = pd.DataFrame(X).reset_index(drop=True)
        data = {}
        for col in self.columns:
            if col not in self.specs:
                continue
            if col in frame.columns:
                data[col] = self.specs[col].transform_woe(frame[col])
            else:
                data[col] = np.zeros(len(frame), dtype=float)
        return pd.DataFrame(data, columns=[c for c in self.columns if c in data])

    def iv_table(self):
        # type: () -> pd.DataFrame
        rows = []
        for col in self.columns:
            spec = self.specs.get(col)
            if spec is None:
                continue
            rows.append(
                {
                    "feature": col,
                    "kind": spec.kind,
                    "method": spec.method,
                    "n_bins": len(spec.labels),
                    "iv": spec.iv,
                    "monotonic": spec.monotonic,
                    "notes": ",".join(sorted(set(spec.notes))),
                }
            )
        return pd.DataFrame(rows)

    def vintage_stability_table(self, X, y, date_col, feature, freq="M", min_rows=30):
        # type: (pd.DataFrame, Any, str, str, str, int) -> pd.DataFrame
        """Per-bin event rate, share and univariate Gini over scoring vintages.

        One row per ``(vintage, bin)``.  ``univariate_gini`` is the vintage-level
        Gini of the transformed predictor (repeated on every bin row of that
        vintage).  WoE features use ``gini(y, -woe)`` so a positive WoE (safer
        than average) is scored in the risk direction; logit / VAL features use
        ``gini(y, logit)`` because a larger log-odds is already higher risk.

        Imported lazily so :mod:`scorecard_segment_eval.stability` can keep
        importing :class:`BinningModel` at module level.
        """
        from scorecard_segment_eval.metrics import gini
        from scorecard_segment_eval.stability import vintage_labels

        columns = [
            "feature",
            "vintage",
            "bin",
            "n",
            "events",
            "share",
            "event_rate",
            "univariate_gini",
        ]
        spec = self.specs.get(feature)
        frame = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(pd.Series(y).reset_index(drop=True), dtype=float)
        if spec is None or feature not in frame.columns:
            return pd.DataFrame(columns=columns)
        if date_col not in frame.columns:
            return pd.DataFrame(columns=columns)

        periods = vintage_labels(frame[date_col], freq=freq)
        labels_all = spec.assign(frame[feature])
        woe_all = spec.transform_woe(frame[feature])
        is_logit = spec.kind == "logit" or spec.transform == "logit"
        order = list(spec.bin_order())
        seen = set(order)
        for label in labels_all:
            if label not in seen:
                order.append(label)
                seen.add(label)

        counts = periods.value_counts(dropna=True)
        kept = sorted([p for p in counts.index if counts[p] >= min_rows], key=str)
        rows = []
        for period in kept:
            mask = (periods == period).to_numpy()
            n_v = int(mask.sum())
            if n_v < min_rows:
                continue
            y_v = y_arr[mask]
            labels_v = labels_all[mask]
            woe_v = woe_all[mask]
            score = woe_v if is_logit else -woe_v
            g_v = gini(y_v, score)
            for label in order:
                bin_mask = labels_v == label
                n_bin = int(bin_mask.sum())
                if n_bin <= 0:
                    continue
                events = float(y_v[bin_mask].sum())
                rows.append(
                    {
                        "feature": feature,
                        "vintage": str(period),
                        "bin": str(label),
                        "n": float(n_bin),
                        "events": events,
                        "share": float(n_bin) / float(n_v),
                        "event_rate": events / float(n_bin) if n_bin else float("nan"),
                        "univariate_gini": g_v,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)

    def plot_vintage_stability(
        self,
        X,
        y,
        date_col,
        feature,
        freq="M",
        min_rows=30,
        figsize=None,
    ):
        # type: (pd.DataFrame, Any, str, str, str, int, Optional[Tuple[float, float]]) -> Tuple[Any, Any]
        """One row of three subplots of WoE grouping stability over vintages.

        Left: true event rate by bin.  Middle: bin share.  Right: univariate
        Gini of the transformed predictor.  The ``__missing__`` bin stays in
        the legend even when every vintage has ``n=0`` for it.  Requires the
        optional ``plot`` extra (``matplotlib``).
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.lines import Line2D
        except Exception:
            raise ImportError(
                "plot_vintage_stability requires matplotlib; install with: "
                "pip install 'scorecard-segment-eval[plot]'"
            )
        table = self.vintage_stability_table(
            X, y, date_col, feature, freq=freq, min_rows=min_rows
        )
        if table.empty:
            raise ValueError(
                "no vintage/bin rows to plot for feature %r (missing spec, "
                "date column, or vintages below min_rows=%s)" % (feature, min_rows)
            )
        vintages = []  # type: List[str]
        for value in table["vintage"].tolist():
            text = str(value)
            if text not in vintages:
                vintages.append(text)
        spec = self.specs.get(feature)
        bins = []  # type: List[str]
        if spec is not None:
            for label in spec.bin_order():
                text = str(label)
                if text not in bins:
                    bins.append(text)
            missing = str(spec.missing_label or MISSING_LABEL)
            if missing not in bins:
                bins.append(missing)
        for value in table["bin"].tolist():
            text = str(value)
            if text not in bins:
                bins.append(text)
        if MISSING_LABEL not in bins:
            bins.append(MISSING_LABEL)
        x_vals, use_dates = _vintage_axis_values(vintages)
        fig, axes = plt.subplots(
            1, 3, sharex=True, figsize=figsize or (14.0, 4.0)
        )
        axes = np.atleast_1d(axes).ravel()
        legend_handles = []
        legend_labels = []
        for bin_label in bins:
            part = table.loc[table["bin"].astype(str) == bin_label]
            by_v = dict(
                (str(row["vintage"]), row)
                for _, row in part.iterrows()
            )
            rates = [
                float(by_v[v]["event_rate"]) if v in by_v else float("nan")
                for v in vintages
            ]
            shares = []
            for v in vintages:
                if v in by_v:
                    shares.append(float(by_v[v]["share"]))
                elif bin_label == MISSING_LABEL:
                    shares.append(0.0)
                else:
                    shares.append(float("nan"))
            line = axes[0].plot(x_vals, rates, marker="o", label=bin_label)[0]
            axes[1].plot(x_vals, shares, marker="o", label=bin_label)
            handle = line
            y_arr = np.asarray(rates, dtype=float)
            if not np.any(np.isfinite(y_arr)):
                handle = Line2D(
                    [0],
                    [0],
                    color=line.get_color(),
                    marker="o",
                    linestyle=line.get_linestyle(),
                )
            legend_handles.append(handle)
            legend_labels.append(bin_label)
        gini_by_v = table.drop_duplicates("vintage")
        gini_lookup = dict(
            (str(row["vintage"]), float(row["univariate_gini"]))
            for _, row in gini_by_v.iterrows()
        )
        ginis = [
            gini_lookup[v] if v in gini_lookup else float("nan") for v in vintages
        ]
        axes[2].plot(x_vals, ginis, marker="o", color="black")
        axes[0].set_ylabel("event rate")
        axes[1].set_ylabel("share")
        axes[2].set_ylabel("univariate Gini")
        axes[0].set_title("%s — true event rate" % feature)
        axes[1].set_title("bin share")
        axes[2].set_title("univariate Gini")
        axes[1].set_ylim(0.0, 1.0)
        for ax in axes:
            ax.set_xlabel("vintage")
        if use_dates:
            locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
            try:
                formatter = mdates.ConciseDateFormatter(locator)
            except Exception:
                formatter = mdates.DateFormatter("%Y-%m")
            for ax in axes:
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                ax.tick_params(axis="x", labelrotation=0)
        else:
            axes[2].set_xticks(x_vals)
            axes[2].set_xticklabels(vintages, rotation=45, ha="right")
        if MISSING_LABEL not in [str(lab) for lab in legend_labels]:
            legend_handles.append(Line2D([0], [0], marker="o"))
            legend_labels.append(MISSING_LABEL)
        if len(legend_labels) <= 12:
            axes[0].legend(
                legend_handles, legend_labels, loc="best", fontsize="small", ncol=2
            )
        fig.tight_layout()
        return fig, axes

    def clone(self):
        # type: () -> "BinningModel"
        """Deep copy via the JSON round-trip, so SQL specs are never mutated."""
        return BinningModel.from_dict(self.to_dict())

    def refresh_stats(self, X, y):
        # type: (pd.DataFrame, Any) -> "BinningModel"
        """Recompute per-bin counts / WoE on ``X`` (logit specs are left as-is)."""
        frame = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(pd.Series(y).reset_index(drop=True).to_numpy(), dtype=float)
        for col in self.columns:
            spec = self.specs.get(col)
            if spec is None or col not in frame.columns:
                continue
            if _is_logit_spec(spec):
                continue
            _finalise_stats(spec, frame[col], y_arr)
        return self

    def event_rate_bounds(self, X, y, date_col, feature, freq="M", min_rows=30, z=1.64):
        # type: (pd.DataFrame, Any, str, str, str, int, float) -> Dict[str, Tuple[float, float]]
        """Per-bin event-rate envelope across vintages (Wilson CI union)."""
        table = self.vintage_stability_table(
            X, y, date_col, feature, freq=freq, min_rows=min_rows
        )
        return _bounds_from_vintage_table(table, z=z)

    def overlapping_event_rate_pairs(self, X, y, date_col, feature, freq="M", min_rows=30, z=1.64):
        # type: (pd.DataFrame, Any, str, str, str, int, float) -> List[Tuple[str, str]]
        """Adjacent bins whose vintage event-rate bounds overlap."""
        spec = self.specs.get(feature)
        if spec is None or _is_logit_spec(spec):
            return []
        bounds = self.event_rate_bounds(
            X, y, date_col, feature, freq=freq, min_rows=min_rows, z=z
        )
        order = _adjacent_bin_labels(spec)
        pairs = []  # type: List[Tuple[str, str]]
        for i in range(len(order) - 1):
            a = order[i]
            b = order[i + 1]
            if a not in bounds or b not in bounds:
                continue
            if _intervals_overlap(bounds[a], bounds[b]):
                pairs.append((a, b))
        return pairs

    def merge_overlapping_event_rate_bounds(
        self, X, y, date_col, gates=None, freq="M", min_rows=30
    ):
        # type: (pd.DataFrame, Any, str, Optional[Gates], str, int) -> "BinningModel"
        """Merge adjacent bins whose vintage event-rate CIs overlap.

        Numeric features drop the shared edge.  Categorical features merge the
        two groups.  Logit / VAL specs are left untouched.  Remaining overlaps
        (typically two bins that still overlap) are noted on the spec.
        """
        gates = gates or self.gates or Gates()
        if not bool(getattr(gates, "stability_merge_overlapping_rates", True)):
            return self
        z = float(getattr(gates, "delta_gini_z", 1.64) or 1.64)
        frame = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(pd.Series(y).reset_index(drop=True).to_numpy(), dtype=float)
        for col in list(self.columns):
            spec = self.specs.get(col)
            if spec is None or col not in frame.columns or _is_logit_spec(spec):
                continue
            _merge_spec_overlapping_rates(
                spec,
                frame[col],
                y_arr,
                frame,
                date_col,
                freq=freq,
                min_rows=min_rows,
                z=z,
            )
        return self

    # -- serialisation -----------------------------------------------------
    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "version": GROUPING_VERSION,
            "columns": list(self.columns),
            "gates": {
                "binning_method": self.gates.binning_method,
                "binning_min_bin_frac": self.gates.binning_min_bin_frac,
                "binning_min_bin_events": self.gates.binning_min_bin_events,
                "binning_max_bins": self.gates.binning_max_bins,
                "binning_monotonic": self.gates.binning_monotonic,
            },
            "features": dict((name, spec.to_dict()) for name, spec in self.specs.items()),
        }

    @classmethod
    def from_dict(cls, payload):
        # type: (Dict[str, Any]) -> "BinningModel"
        features = payload.get("features") or {}
        specs = dict((name, BinSpec.from_dict(spec)) for name, spec in features.items())
        gate_payload = payload.get("gates") or {}
        gates = Gates(
            binning_method=gate_payload.get("binning_method", "auto"),
            binning_min_bin_frac=gate_payload.get("binning_min_bin_frac", 0.05),
            binning_min_bin_events=gate_payload.get("binning_min_bin_events", 10),
            binning_max_bins=gate_payload.get("binning_max_bins", 8),
            binning_monotonic=gate_payload.get("binning_monotonic", True),
        )
        model = cls(specs=specs, gates=gates)
        columns = payload.get("columns")
        model.columns = [str(c) for c in (columns if columns else sorted(specs.keys()))]
        return model

    def save(self, path):
        # type: (str) -> str
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path):
        # type: (str) -> "BinningModel"
        with open(path, "r") as handle:
            return cls.from_dict(json.load(handle))


def save_grouping(model, path):
    # type: (BinningModel, str) -> str
    """Write a ``grouping.json`` definition."""
    return model.save(path)


def load_grouping(path):
    # type: (str) -> BinningModel
    """Read a ``grouping.json`` definition."""
    return BinningModel.load(path)


def grouping_from_dict(payload):
    # type: (Dict[str, Any]) -> BinningModel
    """Rebuild a grouping from a ``grouping.json`` payload."""
    return BinningModel.from_dict(payload)


def _intervals_overlap(a, b):
    # type: (Tuple[float, float], Tuple[float, float]) -> bool
    a_lo, a_hi = a
    b_lo, b_hi = b
    if not (np.isfinite(a_lo) and np.isfinite(a_hi) and np.isfinite(b_lo) and np.isfinite(b_hi)):
        return False
    return not (a_hi < b_lo or b_hi < a_lo)


def _bounds_from_vintage_table(table, z=1.64):
    # type: (pd.DataFrame, float) -> Dict[str, Tuple[float, float]]
    """Event-rate envelope per bin: the range of vintage point rates.

    Wilson intervals are used only to ignore vintages whose CI is so wide it
    is uninformative; the bound itself is min/max of the point event rates so
    a strong monotone grouping is not collapsed by the union of 24 CIs.
    """
    bounds = {}  # type: Dict[str, Tuple[float, float]]
    if table is None or table.empty or "bin" not in table.columns:
        return bounds
    z = float(z) if z is not None else 1.64
    for bin_label, part in table.groupby("bin"):
        rates = []  # type: List[float]
        for _idx, row in part.iterrows():
            n = float(row["n"])
            events = float(row["events"])
            if n <= 0:
                continue
            rate = float(row["event_rate"])
            lo, hi = binomial_rate_bound(events, n, z=z)
            # Drop a vintage whose interval covers almost the whole [0, 1]
            # range: it carries no ordering information.
            if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) >= 0.9:
                continue
            if np.isfinite(rate):
                rates.append(rate)
        if len(rates) >= 2:
            bounds[str(bin_label)] = (float(min(rates)), float(max(rates)))
        elif len(rates) == 1:
            lo, hi = binomial_rate_bound(
                float(part["events"].iloc[0]), float(part["n"].iloc[0]), z=z
            )
            if np.isfinite(lo) and np.isfinite(hi):
                bounds[str(bin_label)] = (lo, hi)
    return bounds


def _adjacent_bin_labels(spec):
    # type: (BinSpec) -> List[str]
    """Reporting-order labels excluding missing/other, numeric bins first."""
    skip = set([spec.missing_label, OTHER_LABEL])
    if _has_numeric_edges(spec):
        n_num = _numeric_bin_count(spec)
        labels = [str(lab) for lab in spec.labels[:n_num] if str(lab) not in skip]
        extra = [
            str(lab)
            for lab in spec.labels[n_num:]
            if str(lab) not in skip
        ]
        return labels + extra
    return [str(lab) for lab in spec.bin_order() if str(lab) not in skip]


def _merge_spec_overlapping_rates(spec, series, y_arr, frame, date_col, freq, min_rows, z):
    # type: (BinSpec, Any, np.ndarray, pd.DataFrame, str, str, int, float) -> None
    """In-place merge until adjacent vintage event-rate bounds no longer overlap."""
    dummy = BinningModel(specs={spec.feature: spec}, gates=Gates())
    dummy.columns = [spec.feature]
    guard = 0
    merged = 0
    while guard < 40:
        guard += 1
        pairs = dummy.overlapping_event_rate_pairs(
            frame, y_arr, date_col, spec.feature, freq=freq, min_rows=min_rows, z=z
        )
        if not pairs:
            break
        a, b = pairs[0]
        if not _merge_two_bins(spec, a, b, series, y_arr):
            if "overlapping_event_rate_bounds" not in spec.notes:
                spec.notes.append("overlapping_event_rate_bounds")
            break
        merged += 1
        spec.notes.append("merged_overlapping_event_rate_bounds")
    if merged and "merged_overlapping_event_rate_bounds" not in spec.notes:
        spec.notes.append("merged_overlapping_event_rate_bounds")


def _merge_two_bins(spec, label_a, label_b, series, y_arr):
    # type: (BinSpec, str, str, Any, np.ndarray) -> bool
    """Merge ``label_a`` into ``label_b``'s neighbour.  Returns False if impossible."""
    if _has_numeric_edges(spec):
        n_num = _numeric_bin_count(spec)
        labels = [str(lab) for lab in spec.labels[:n_num]]
        try:
            i = labels.index(str(label_a))
            j = labels.index(str(label_b))
        except ValueError:
            return False
        if abs(i - j) != 1:
            return False
        drop = max(i, j)  # drop the shared interior edge
        if drop <= 0 or drop >= len(spec.edges) - 1:
            return False
        if len(spec.edges) <= 3:
            return False
        spec.edges = [float(e) for k, e in enumerate(spec.edges) if k != drop]
        extra = list(spec.labels[n_num:])
        spec.labels = edge_labels(spec.edges, closed=spec.closed) + extra
        _finalise_stats(spec, series, y_arr)
        return True
    groups = spec.groups or {}
    if str(label_a) not in groups or str(label_b) not in groups:
        return False
    if len(spec.labels) <= 2:
        return False
    members = list(groups.get(str(label_a), [])) + list(groups.get(str(label_b), []))
    new_label = "|".join(sorted(set(members)))
    if len(new_label) > 60:
        new_label = "grp_merged"
    new_groups = {}  # type: Dict[str, List[str]]
    new_labels = []  # type: List[str]
    replaced = False
    for label in spec.labels:
        if str(label) in (str(label_a), str(label_b)):
            if replaced:
                continue
            new_groups[new_label] = members
            new_labels.append(new_label)
            replaced = True
        else:
            new_groups[str(label)] = list(groups.get(str(label), []))
            new_labels.append(str(label))
    spec.groups = new_groups
    spec.labels = new_labels
    _finalise_stats(spec, series, y_arr)
    return True


# --------------------------------------------------------------------------
# Reconstruct a grouping from already-transformed WoE columns
# --------------------------------------------------------------------------
def grouping_from_woe_columns(frame, pred_woe_map, gates=None):
    # type: (pd.DataFrame, Dict[str, str], Optional[Gates]) -> Optional[BinningModel]
    """Rebuild a :class:`BinningModel` from raw predictors plus their WoE columns.

    Unique values of each WoE column are treated as the original bins.  When
    a column is not discrete (more unique values than the binning budget) the
    predictor is skipped, so a continuous copy of the raw feature is not
    mistaken for a grouping.
    """
    gates = gates or Gates()
    if frame is None or not pred_woe_map:
        return None
    specs = {}  # type: Dict[str, BinSpec]
    for raw_col, woe_col in pred_woe_map.items():
        if raw_col not in frame.columns or woe_col not in frame.columns:
            continue
        spec = bin_spec_from_woe_column(frame[raw_col], frame[woe_col], feature=str(raw_col), gates=gates)
        if spec is not None:
            specs[str(raw_col)] = spec
    if not specs:
        return None
    model = BinningModel(specs=specs, gates=gates)
    model.columns = [c for c in pred_woe_map if c in specs]
    return model


def bin_spec_from_woe_column(raw, woe, feature="feature", gates=None):
    # type: (Any, Any, str, Optional[Gates]) -> Optional[BinSpec]
    """Recover one :class:`BinSpec` from a (raw, WoE) pair of columns."""
    gates = gates or Gates()
    raw_s = pd.Series(raw).reset_index(drop=True)
    woe_s = pd.to_numeric(pd.Series(woe).reset_index(drop=True), errors="coerce")
    paired = pd.DataFrame({"raw": raw_s, "woe": woe_s})
    valid = paired.dropna(subset=["woe"])
    n_unique = int(valid["woe"].nunique())
    budget = max(int(gates.binning_max_bins) * 3, 4)
    if n_unique < 2 or n_unique > budget:
        return None
    if n_unique > max(int(0.5 * max(len(valid), 1)), budget):
        return None

    numeric_like = pd.api.types.is_numeric_dtype(raw_s) and raw_s.nunique(dropna=True) > 2
    if numeric_like:
        tmp = pd.DataFrame(
            {
                "raw": pd.to_numeric(valid["raw"], errors="coerce"),
                "woe": valid["woe"],
            }
        ).dropna()
        if tmp.empty or int(tmp["woe"].nunique()) < 2:
            return None
        grouped = tmp.groupby("woe").agg(lo=("raw", "min"), hi=("raw", "max"), n=("raw", "size"))
        grouped = grouped.sort_values("lo")
        los = grouped["lo"].to_numpy(dtype=float)
        his = grouped["hi"].to_numpy(dtype=float)
        edges = [-np.inf]
        for i in range(len(grouped) - 1):
            edges.append(0.5 * (float(his[i]) + float(los[i + 1])))
        edges.append(np.inf)
        labels = edge_labels(edges)
        woe_map = {}  # type: Dict[str, float]
        counts = {}  # type: Dict[str, float]
        for i, (woe_val, row) in enumerate(grouped.iterrows()):
            if i < len(labels):
                woe_map[labels[i]] = float(woe_val)
                counts[labels[i]] = float(row["n"])
        spec = BinSpec(
            feature=str(feature),
            kind="numeric",
            method="from_woe_column",
            edges=edges,
            labels=labels,
            woe=woe_map,
            counts=counts,
            notes=["reconstructed_from_woe_column"],
        )
        spec.woe.setdefault(MISSING_LABEL, 0.0)
        spec.counts.setdefault(MISSING_LABEL, 0.0)
        spec.woe.setdefault(OTHER_LABEL, 0.0)
        spec.counts.setdefault(OTHER_LABEL, 0.0)
        return spec

    tmp = pd.DataFrame(
        {
            "level": valid["raw"].map(lambda v: MISSING_LABEL if pd.isna(v) else str(v)),
            "woe": valid["woe"],
        }
    )
    groups = {}  # type: Dict[str, List[str]]
    labels = []  # type: List[str]
    woe_map = {}  # type: Dict[str, float]
    counts = {}  # type: Dict[str, float]
    for i, (woe_val, part) in enumerate(tmp.groupby("woe")):
        members = sorted(set(str(v) for v in part["level"].tolist() if v != MISSING_LABEL))
        if not members:
            continue
        label = "|".join(members) if len("|".join(members)) <= 60 else "grp_%02d" % i
        if label in groups:
            label = "grp_%02d" % i
        groups[label] = members
        labels.append(label)
        woe_map[label] = float(woe_val)
        counts[label] = float(len(part))
    if not groups:
        return None
    spec = BinSpec(
        feature=str(feature),
        kind="categorical",
        method="from_woe_column",
        groups=groups,
        labels=labels,
        woe=woe_map,
        counts=counts,
        notes=["reconstructed_from_woe_column"],
    )
    spec.woe.setdefault(MISSING_LABEL, 0.0)
    spec.counts.setdefault(MISSING_LABEL, 0.0)
    spec.woe.setdefault(OTHER_LABEL, 0.0)
    spec.counts.setdefault(OTHER_LABEL, 0.0)
    return spec


# --------------------------------------------------------------------------
# Segment grouping vs portfolio grouping
# --------------------------------------------------------------------------
def resolve_portfolio_spec(feature, portfolio_grouping, pred_woe_map=None):
    # type: (str, Optional[BinningModel], Optional[Dict[str, str]]) -> Optional[BinSpec]
    """Find the portfolio :class:`BinSpec` that corresponds to ``feature``."""
    if portfolio_grouping is None:
        return None
    specs = portfolio_grouping.specs or {}
    if feature in specs:
        return specs[feature]
    for spec in specs.values():
        if getattr(spec, "output_name", None) == feature or spec.feature == feature:
            return spec
    pred_woe_map = pred_woe_map or {}
    woe_name = pred_woe_map.get(feature)
    if woe_name and woe_name in specs:
        return specs[woe_name]
    reverse = dict((v, k) for k, v in pred_woe_map.items())
    if feature in reverse and reverse[feature] in specs:
        return specs[reverse[feature]]
    want = _norm(_strip_woe_affix(feature))
    hits = [
        spec
        for name, spec in specs.items()
        if _norm(name) == _norm(feature)
        or _norm(_strip_woe_affix(name)) == want
        or _norm(_strip_woe_affix(name)) == _norm(feature)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def compare_groupings(segment_grouping, portfolio_grouping, pred_woe_map=None, gates=None):
    # type: (Optional[BinningModel], Optional[BinningModel], Optional[Dict[str, str]], Optional[Gates]) -> pd.DataFrame
    """Per-bin notes on how a segment grouping differs from the portfolio one.

    Significant differences (WoE shift, sign flip, edge movement, merge/split,
    category membership change) are also appended onto each segment
    :class:`BinSpec` ``notes`` list so the saved grouping carries the commentary.
    """
    gates = gates or Gates()
    rows = []  # type: List[Dict[str, Any]]
    if segment_grouping is None or portfolio_grouping is None:
        return pd.DataFrame(columns=_COMPARISON_COLUMNS)
    pred_woe_map = pred_woe_map or {}
    for feature in segment_grouping.columns:
        seg_spec = segment_grouping.specs.get(feature)
        if seg_spec is None:
            continue
        port_spec = resolve_portfolio_spec(feature, portfolio_grouping, pred_woe_map)
        if port_spec is None:
            row = _comparison_row(
                feature=feature,
                segment_bin="__feature__",
                portfolio_bins="",
                woe_segment=float("nan"),
                woe_portfolio=float("nan"),
                kind="no_portfolio_spec",
                significant=False,
                note="no portfolio grouping found for %s" % (feature,),
            )
            rows.append(row)
            continue
        feature_rows = compare_bin_specs(seg_spec, port_spec, gates=gates)
        rows.extend(feature_rows)
        significant_notes = [r["note"] for r in feature_rows if r.get("significant")]
        for note in significant_notes:
            tagged = "portfolio_diff: " + note
            if tagged not in seg_spec.notes:
                seg_spec.notes.append(tagged)
    if not rows:
        return pd.DataFrame(columns=_COMPARISON_COLUMNS)
    return pd.DataFrame(rows)[_COMPARISON_COLUMNS]


_COMPARISON_COLUMNS = [
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


def _comparison_row(feature, segment_bin, portfolio_bins, woe_segment, woe_portfolio, kind, significant, note):
    # type: (str, str, str, float, float, str, bool, str) -> Dict[str, Any]
    woe_s = float(woe_segment) if woe_segment == woe_segment else float("nan")
    woe_p = float(woe_portfolio) if woe_portfolio == woe_portfolio else float("nan")
    delta = woe_s - woe_p if (np.isfinite(woe_s) and np.isfinite(woe_p)) else float("nan")
    return {
        "feature": feature,
        "segment_bin": segment_bin,
        "portfolio_bins": portfolio_bins,
        "woe_segment": woe_s,
        "woe_portfolio": woe_p,
        "woe_delta": delta,
        "kind": kind,
        "significant": bool(significant),
        "note": note,
    }


def compare_bin_specs(segment_spec, portfolio_spec, gates=None):
    # type: (BinSpec, BinSpec, Optional[Gates]) -> List[Dict[str, Any]]
    """Per-bin comparison of two grouping definitions for one predictor."""
    gates = gates or Gates()
    if _is_logit_spec(segment_spec) and _is_logit_spec(portfolio_spec):
        return [
            _comparison_row(
                feature=segment_spec.feature,
                segment_bin="logit",
                portfolio_bins="logit",
                woe_segment=float("nan"),
                woe_portfolio=float("nan"),
                kind="aligned",
                significant=False,
                note="%s kept in logit form (VAL/LIN)" % (segment_spec.feature,),
            )
        ]
    if _is_logit_spec(segment_spec) != _is_logit_spec(portfolio_spec):
        return [
            _comparison_row(
                feature=segment_spec.feature,
                segment_bin="__feature__",
                portfolio_bins="",
                woe_segment=float("nan"),
                woe_portfolio=float("nan"),
                kind="form_change",
                significant=True,
                note="%s logit form vs WoE bins (segment kind=%s, portfolio kind=%s)"
                % (segment_spec.feature, segment_spec.kind, portfolio_spec.kind),
            )
        ]
    if _has_numeric_edges(segment_spec) and _has_numeric_edges(portfolio_spec):
        return _compare_numeric_specs(segment_spec, portfolio_spec, gates)
    return _compare_categorical_specs(segment_spec, portfolio_spec, gates)


def _woe_of(spec, label):
    # type: (BinSpec, str) -> float
    if label in spec.woe:
        return float(spec.woe[label])
    return float("nan")


def _material_woe_shift(delta, gates):
    # type: (float, Gates) -> bool
    return bool(np.isfinite(delta) and abs(delta) >= float(gates.grouping_woe_shift_material))


def _sign_flip(woe_s, woe_p, gates):
    # type: (float, float, Gates) -> bool
    floor = float(gates.grouping_sign_flip_floor)
    if not (np.isfinite(woe_s) and np.isfinite(woe_p)):
        return False
    if abs(woe_s) < floor or abs(woe_p) < floor:
        return False
    return (woe_s > 0) != (woe_p > 0)


def _finite_span(lo, hi):
    # type: (float, float) -> float
    if np.isfinite(lo) and np.isfinite(hi):
        return abs(hi - lo)
    return float("nan")


def _edge_shifted(s_lo, s_hi, p_lo, p_hi, gates):
    # type: (float, float, float, float, Gates) -> bool
    frac = float(gates.grouping_edge_shift_frac)
    width = _finite_span(p_lo, p_hi)
    if not np.isfinite(width) or width <= 0:
        width = _finite_span(s_lo, s_hi)
    if not np.isfinite(width) or width <= 0:
        return (np.isfinite(s_lo) != np.isfinite(p_lo)) or (np.isfinite(s_hi) != np.isfinite(p_hi))
    lo_shift = abs(s_lo - p_lo) if (np.isfinite(s_lo) and np.isfinite(p_lo)) else 0.0
    hi_shift = abs(s_hi - p_hi) if (np.isfinite(s_hi) and np.isfinite(p_hi)) else 0.0
    if not np.isfinite(s_lo) or not np.isfinite(p_lo):
        lo_shift = 0.0 if (np.isneginf(s_lo) and np.isneginf(p_lo)) else max(lo_shift, width)
    if not np.isfinite(s_hi) or not np.isfinite(p_hi):
        hi_shift = 0.0 if (np.isposinf(s_hi) and np.isposinf(p_hi)) else max(hi_shift, width)
    return (lo_shift >= frac * width) or (hi_shift >= frac * width)


def _compare_numeric_specs(seg_spec, port_spec, gates):
    # type: (BinSpec, BinSpec, Gates) -> List[Dict[str, Any]]
    rows = []  # type: List[Dict[str, Any]]
    s_edges = [float(e) for e in (seg_spec.edges or [])]
    p_edges = [float(e) for e in (port_spec.edges or [])]
    if len(s_edges) < 2 or len(p_edges) < 2:
        return _compare_categorical_specs(seg_spec, port_spec, gates)
    n_seg = _numeric_bin_count(seg_spec)
    n_port = _numeric_bin_count(port_spec)
    seg_labels = list(seg_spec.labels[:n_seg]) if seg_spec.labels else edge_labels(s_edges, closed=seg_spec.closed)
    port_labels = list(port_spec.labels[:n_port]) if port_spec.labels else edge_labels(p_edges, closed=port_spec.closed)

    def _overlaps(s_lo, s_hi):
        # type: (float, float) -> List[Tuple[int, str, float, float]]
        hits = []  # type: List[Tuple[int, str, float, float]]
        for j, plabel in enumerate(port_labels):
            p_lo, p_hi = p_edges[j], p_edges[j + 1]
            if s_lo < p_hi and p_lo < s_hi:
                hits.append((j, plabel, p_lo, p_hi))
        return hits

    for i, label in enumerate(seg_labels):
        s_lo, s_hi = s_edges[i], s_edges[i + 1]
        woe_s = _woe_of(seg_spec, label)
        hits = _overlaps(s_lo, s_hi)
        hit_names = [h[1] for h in hits]
        port_joined = ",".join(hit_names)
        if not hits:
            rows.append(
                _comparison_row(
                    feature=seg_spec.feature,
                    segment_bin=label,
                    portfolio_bins="",
                    woe_segment=woe_s,
                    woe_portfolio=float("nan"),
                    kind="unmatched",
                    significant=True,
                    note="%s bin %s does not overlap any portfolio bin" % (seg_spec.feature, label),
                )
            )
            continue
        # Primary overlap: largest overlapping span, falling back to first.
        primary = hits[0]
        best_span = -1.0
        for hit in hits:
            span = min(s_hi, hit[3]) - max(s_lo, hit[2])
            if not np.isfinite(span):
                span = 1.0
            if span > best_span:
                best_span = span
                primary = hit
        woe_p = _woe_of(port_spec, primary[1])
        kinds = []  # type: List[str]
        if len(hits) > 1:
            kinds.append("merged")
        if _edge_shifted(s_lo, s_hi, primary[2], primary[3], gates):
            kinds.append("edge_shift")
        if _sign_flip(woe_s, woe_p, gates):
            kinds.append("woe_sign_flip")
        elif _material_woe_shift(woe_s - woe_p if np.isfinite(woe_s) and np.isfinite(woe_p) else float("nan"), gates):
            kinds.append("woe_shift")
        kind = "+".join(kinds) if kinds else "aligned"
        significant = kind != "aligned"
        if kind == "aligned":
            note = "%s bin %s aligns with portfolio %s (WoE %.3f vs %.3f)" % (
                seg_spec.feature,
                label,
                primary[1],
                woe_s,
                woe_p,
            )
        elif "merged" in kinds:
            note = (
                "%s bin %s spans portfolio bins [%s] (segment WoE %.3f vs primary %.3f)"
                % (seg_spec.feature, label, port_joined, woe_s, woe_p)
            )
        elif "woe_sign_flip" in kinds:
            note = "%s bin %s WoE sign flipped vs portfolio %s (%.3f vs %.3f)" % (
                seg_spec.feature,
                label,
                primary[1],
                woe_s,
                woe_p,
            )
        elif "woe_shift" in kinds:
            note = "%s bin %s WoE shifted vs portfolio %s (%.3f vs %.3f)" % (
                seg_spec.feature,
                label,
                primary[1],
                woe_s,
                woe_p,
            )
        else:
            note = "%s bin %s edges moved vs portfolio %s (%s vs %s)" % (
                seg_spec.feature,
                label,
                primary[1],
                label,
                primary[1],
            )
        rows.append(
            _comparison_row(
                feature=seg_spec.feature,
                segment_bin=label,
                portfolio_bins=port_joined,
                woe_segment=woe_s,
                woe_portfolio=woe_p,
                kind=kind,
                significant=significant,
                note=note,
            )
        )

    # Flag portfolio bins that were split across several segment bins.
    for j, plabel in enumerate(port_labels):
        p_lo, p_hi = p_edges[j], p_edges[j + 1]
        covering = []
        for i, label in enumerate(seg_labels):
            s_lo, s_hi = s_edges[i], s_edges[i + 1]
            if p_lo < s_hi and s_lo < p_hi:
                covering.append(label)
        if len(covering) > 1:
            woe_p = _woe_of(port_spec, plabel)
            rows.append(
                _comparison_row(
                    feature=seg_spec.feature,
                    segment_bin=",".join(covering),
                    portfolio_bins=plabel,
                    woe_segment=float("nan"),
                    woe_portfolio=_woe_of(port_spec, plabel),
                    kind="split",
                    significant=True,
                    note="%s portfolio bin %s is split across segment bins [%s]"
                    % (seg_spec.feature, plabel, ",".join(covering)),
                )
            )
    return rows


def _member_lookup(spec):
    # type: (BinSpec) -> Dict[str, str]
    lookup = {}  # type: Dict[str, str]
    for label, members in (spec.groups or {}).items():
        for member in members:
            lookup[str(member)] = label
    return lookup


def _compare_categorical_specs(seg_spec, port_spec, gates):
    # type: (BinSpec, BinSpec, Gates) -> List[Dict[str, Any]]
    rows = []  # type: List[Dict[str, Any]]
    port_lookup = _member_lookup(port_spec)
    seg_labels = list(seg_spec.labels) if seg_spec.labels else list((seg_spec.groups or {}).keys())
    if not seg_labels:
        seg_labels = [k for k in seg_spec.woe if k not in (seg_spec.missing_label, OTHER_LABEL)]
    for label in seg_labels:
        members = list((seg_spec.groups or {}).get(label) or [])
        if not members and label in (seg_spec.woe or {}):
            members = [label]
        sources = []  # type: List[str]
        seen = []  # type: List[str]
        for member in members:
            src = port_lookup.get(str(member))
            if src is None:
                # Numeric fallback: treat the label itself as the portfolio key.
                if label in port_spec.woe:
                    src = label
                else:
                    src = OTHER_LABEL
            if src not in seen:
                seen.append(src)
            sources.append(src)
        woe_s = _woe_of(seg_spec, label)
        primary = seen[0] if seen else ""
        woe_p = _woe_of(port_spec, primary) if primary else float("nan")
        kinds = []  # type: List[str]
        if len(seen) > 1:
            kinds.append("membership_change")
        if _sign_flip(woe_s, woe_p, gates):
            kinds.append("woe_sign_flip")
        elif _material_woe_shift(
            (woe_s - woe_p) if (np.isfinite(woe_s) and np.isfinite(woe_p)) else float("nan"),
            gates,
        ):
            kinds.append("woe_shift")
        kind = "+".join(kinds) if kinds else "aligned"
        significant = kind != "aligned"
        if kind == "aligned":
            note = "%s bin %s matches portfolio %s (WoE %.3f vs %.3f)" % (
                seg_spec.feature,
                label,
                primary,
                woe_s,
                woe_p,
            )
        elif "membership_change" in kinds:
            note = "%s bin %s mixes portfolio groups [%s] (WoE %.3f vs primary %.3f)" % (
                seg_spec.feature,
                label,
                ",".join(seen),
                woe_s,
                woe_p,
            )
        elif "woe_sign_flip" in kinds:
            note = "%s bin %s WoE sign flipped vs portfolio %s (%.3f vs %.3f)" % (
                seg_spec.feature,
                label,
                primary,
                woe_s,
                woe_p,
            )
        else:
            note = "%s bin %s WoE shifted vs portfolio %s (%.3f vs %.3f)" % (
                seg_spec.feature,
                label,
                primary,
                woe_s,
                woe_p,
            )
        rows.append(
            _comparison_row(
                feature=seg_spec.feature,
                segment_bin=label,
                portfolio_bins=",".join(seen),
                woe_segment=woe_s,
                woe_portfolio=woe_p,
                kind=kind,
                significant=significant,
                note=note,
            )
        )
    return rows
