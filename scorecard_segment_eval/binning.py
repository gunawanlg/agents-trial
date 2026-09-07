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
from scorecard_segment_eval.schema import Gates

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


def edge_labels(edges):
    # type: (Sequence[float]) -> List[str]
    """Human-readable, guaranteed-unique labels for ``(a, b]`` bins."""
    for precision in (6, 12):
        labels = []
        for i in range(len(edges) - 1):
            lo = _fmt(float(edges[i]), precision)
            hi = _fmt(float(edges[i + 1]), precision)
            right = ")" if edges[i + 1] == np.inf else "]"
            labels.append("(" + lo + ", " + hi + right)
        if len(set(labels)) == len(labels):
            return labels
    return ["bin_%02d" % i for i in range(len(edges) - 1)]


def assign_numeric_bins(values, edges, labels=None, missing_label=MISSING_LABEL):
    # type: (Any, Sequence[float], Optional[Sequence[str]], str) -> np.ndarray
    """Map numeric values onto ``edges`` using ``(a, b]`` semantics.

    Values outside the outer edges cannot occur because the outer edges are
    infinite, so out-of-range handling reduces to clamping into the extreme
    bins.  Missing values (including non-coercible ones) get ``missing_label``.
    """
    numeric = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    names = list(labels) if labels is not None else edge_labels(edges)
    inner = np.asarray(edges, dtype=float)[1:-1]
    out = np.empty(len(arr), dtype=object)
    isnan = ~np.isfinite(arr)
    if len(inner):
        idx = np.searchsorted(inner, np.where(isnan, 0.0, arr), side="left")
    else:
        idx = np.zeros(len(arr), dtype=int)
    idx = np.clip(idx, 0, len(names) - 1)
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
    ):
        # type: (str, str, str, Optional[Sequence[float]], Optional[Dict[str, List[str]]], Optional[Sequence[str]], Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[Dict[str, float]], float, bool, str, Optional[List[str]]) -> None
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

    # -- application -------------------------------------------------------
    def assign(self, values):
        # type: (Any) -> np.ndarray
        """Return the bin label of every value."""
        if self.kind == "numeric" and self.edges is not None:
            return assign_numeric_bins(values, self.edges, self.labels, self.missing_label)
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
        labels = self.assign(values)
        default = 0.0
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

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(cls, X, y, gates=None, n_jobs=None):
        # type: (pd.DataFrame, Any, Optional[Gates], Optional[int]) -> "BinningModel"
        gates = gates or Gates()
        frame = pd.DataFrame(X).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)
        columns = list(frame.columns)

        def _one(col):
            # type: (str) -> Tuple[str, BinSpec]
            return col, fit_bin_spec(frame[col], y_series, gates=gates, feature=str(col))

        jobs = n_jobs if n_jobs is not None else gates.n_jobs
        pairs = map_jobs(_one, columns, n_jobs=jobs, cap=gates.max_workers_cap)
        model = cls(specs=dict(pairs), gates=gates)
        model.columns = [str(c) for c in columns]
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
        frame = pd.DataFrame(X).reset_index(drop=True)
        cols = []
        for col in self.columns:
            if col not in self.specs:
                continue
            if col in frame.columns:
                cols.append(self.specs[col].transform_woe(frame[col]))
            else:
                cols.append(np.zeros(len(frame), dtype=float))
        if not cols:
            return np.zeros((len(frame), 0))
        return np.column_stack(cols)

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
