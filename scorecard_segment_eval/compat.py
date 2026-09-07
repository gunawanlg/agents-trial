"""Helpers that keep the package usable on Python 3.6 and mixed pandas versions."""

from __future__ import print_function

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

PY36 = sys.version_info[:2] == (3, 6)
PY_LT_37 = sys.version_info < (3, 7)


def as_str_keys(series):
    """Map a series to string bin keys; missing values become ``__na__``."""
    values = series.astype(object)
    na = pd.isna(values)
    text = values.astype(str)
    if hasattr(text, "to_numpy"):
        arr = text.to_numpy()
        na_arr = na.to_numpy() if hasattr(na, "to_numpy") else np.asarray(na)
    else:
        arr = np.asarray(text)
        na_arr = np.asarray(na)
    arr = np.where(na_arr, "__na__", arr)
    return pd.Series(arr, index=series.index)


def to_numpy(obj, dtype=None):
    if hasattr(obj, "to_numpy"):
        if dtype is None:
            return obj.to_numpy()
        return obj.to_numpy(dtype=dtype)
    arr = np.asarray(obj)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def safe_groupby(obj, by, dropna=True, observed=True, **kwargs):
    """``groupby`` with kwargs tolerated by pandas 0.24 through 3.x."""
    attempts = [
        dict(dropna=dropna, observed=observed),
        dict(dropna=dropna),
        dict(observed=observed),
        dict(),
    ]
    last_err = None
    for extra in attempts:
        params = dict(kwargs)
        params.update(extra)
        try:
            return obj.groupby(by, **params)
        except TypeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return obj.groupby(by, **kwargs)


def is_numeric_series(s):
    try:
        return bool(pd.api.types.is_numeric_dtype(s))
    except Exception:
        return bool(np.issubdtype(np.asarray(s).dtype, np.number))


def ensure_dir(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)


def dump_json(path, payload):
    ensure_dir(path)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    return str(obj)


def warn_user(message, stacklevel=2):
    warnings.warn(message, UserWarning, stacklevel=stacklevel)
    print("WARNING: " + str(message))
