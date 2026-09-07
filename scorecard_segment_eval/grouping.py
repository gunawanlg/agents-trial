"""Load and apply grouping.json bin maps for WoE/PSI."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scorecard_segment_eval.compat import as_str_keys, load_json


GroupingSpec = Dict[str, Any]


def load_grouping(path):
    # type: (Optional[str]) -> Dict[str, GroupingSpec]
    if path is None or not path:
        return {}
    if not os.path.isfile(path):
        raise IOError("grouping file not found: " + str(path))
    raw = load_json(path)
    features = raw.get("features", raw)
    parsed = {}  # type: Dict[str, GroupingSpec]
    for name, spec in features.items():
        parsed[name] = _normalize_spec(spec)
    return parsed


def _normalize_spec(spec):
    # type: (Any) -> GroupingSpec
    if isinstance(spec, dict):
        kind = spec.get("type") or spec.get("kind")
        if kind is None:
            if "bins" in spec or "edges" in spec:
                kind = "numerical"
            elif "groups" in spec or "mapping" in spec:
                kind = "categorical"
            else:
                kind = "categorical"
        bins = spec.get("bins", spec.get("edges"))
        groups = spec.get("groups", spec.get("mapping"))
        return {"type": kind, "bins": bins, "groups": groups}
    if isinstance(spec, (list, tuple)):
        numeric_like = True
        for item in spec:
            if not isinstance(item, (int, float)):
                numeric_like = False
                break
        if numeric_like:
            return {"type": "numerical", "bins": list(spec), "groups": None}
        return {"type": "categorical", "bins": None, "groups": {str(v): [v] for v in spec}}
    return {"type": "categorical", "bins": None, "groups": {str(spec): [spec]}}


def apply_grouping(series, spec):
    # type: (pd.Series, GroupingSpec) -> pd.Series
    kind = (spec or {}).get("type") or "categorical"
    if kind in ("numerical", "numeric", "decile") and spec.get("bins"):
        edges = np.asarray(spec["bins"], dtype=float)
        edges = np.unique(edges)
        if len(edges) < 2:
            return as_str_keys(series)
        if np.isfinite(edges[0]):
            edges = np.concatenate(([-np.inf], edges))
        if np.isfinite(edges[-1]):
            edges = np.concatenate((edges, [np.inf]))
        edges = np.unique(edges)
        cut = pd.cut(pd.to_numeric(series, errors="coerce"), bins=edges, include_lowest=True)
        return as_str_keys(cut)
    groups = spec.get("groups") or {}
    reverse = {}
    for label, members in groups.items():
        if not isinstance(members, (list, tuple)):
            members = [members]
        for member in members:
            reverse[str(member)] = str(label)
    keys = as_str_keys(series)
    mapped = keys.map(lambda k: reverse.get(k, k))
    return pd.Series(mapped.to_numpy() if hasattr(mapped, "to_numpy") else np.asarray(mapped), index=series.index)
