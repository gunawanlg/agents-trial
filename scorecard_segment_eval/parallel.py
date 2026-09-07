"""Deterministic, 3.6-safe parallel helpers.

Design contract (section D of the upgrade brief):

* ``n_jobs=1`` runs strictly serially in input order.
* Results are *identical* for any ``n_jobs``.  Randomness is therefore never
  drawn from a shared global RNG: callers ask :func:`seed_sequence_for` /
  :func:`spawn_seeds` for a per-work-item seed derived only from a base seed
  and the item index.
* Nested parallelism is suppressed.  Once a mapped call is running, any inner
  :func:`map_jobs` degrades to serial execution, so a parallel per-segment loop
  cannot multiply into a thread explosion in the per-predictor loop underneath.

A thread pool is used rather than processes: the payloads are pandas frames and
locally-defined closures (neither reliably picklable), and the hot inner work is
numpy/sklearn code that releases the GIL.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

_STATE = threading.local()

#: Environment variables that BLAS backends read; pinned inside workers so a
#: parallel outer loop does not fight an implicitly parallel inner BLAS call.
_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def cpu_count():
    # type: () -> int
    try:
        found = os.cpu_count()
    except Exception:  # pragma: no cover - platform dependent
        found = None
    return int(found) if found else 1


def resolve_n_jobs(n_jobs=None, cap=8, n_items=None):
    # type: (Optional[int], int, Optional[int]) -> int
    """Normalise an ``n_jobs`` request into a concrete worker count >= 1.

    ``None`` and ``0`` mean serial.  Negative values follow the joblib
    convention: ``-1`` is every core, ``-2`` all but one, and so on.  The result
    is capped by ``cap`` and by the number of work items.
    """
    if n_jobs is None:
        workers = 1
    elif n_jobs < 0:
        workers = max(cpu_count() + 1 + int(n_jobs), 1)
    else:
        workers = max(int(n_jobs), 1)
    if cap and cap > 0:
        workers = min(workers, int(cap))
    if n_items is not None:
        workers = min(workers, max(int(n_items), 1))
    return max(workers, 1)


def in_parallel_context():
    # type: () -> bool
    """True while a :func:`map_jobs` call is executing above us."""
    return bool(getattr(_STATE, "depth", 0) > 0)


class _NestingGuard(object):
    """Increment/decrement the nesting depth of the *calling* thread."""

    def __enter__(self):
        _STATE.depth = getattr(_STATE, "depth", 0) + 1
        return self

    def __exit__(self, exc_type, exc, tb):
        _STATE.depth = max(getattr(_STATE, "depth", 1) - 1, 0)
        return False


def _worker(func, item, index):
    # type: (Callable[..., Any], Any, int) -> Any
    # Workers inherit "we are already parallel" so inner maps stay serial.
    _STATE.depth = 1
    try:
        return func(item)
    finally:
        _STATE.depth = 0


def map_jobs(func, items, n_jobs=None, cap=8, allow_nested=False):
    # type: (Callable[[Any], Any], Iterable[Any], Optional[int], int, bool) -> List[Any]
    """Apply ``func`` to every item, returning results in input order.

    Falls back to serial execution when only one worker is warranted or when a
    parallel map is already in flight (unless ``allow_nested``).
    """
    work = list(items)
    if not work:
        return []
    workers = resolve_n_jobs(n_jobs, cap=cap, n_items=len(work))
    if workers <= 1 or (in_parallel_context() and not allow_nested):
        return [func(item) for item in work]
    with _NestingGuard():
        with _pinned_blas_threads():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_worker, func, item, i) for i, item in enumerate(work)]
                return [f.result() for f in futures]


class _pinned_blas_threads(object):
    """Temporarily set the BLAS thread env vars to 1 (3.6-safe, no contextlib)."""

    def __init__(self):
        self._previous = {}

    def __enter__(self):
        for name in _BLAS_ENV_VARS:
            self._previous[name] = os.environ.get(name)
            os.environ[name] = "1"
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


def spawn_seeds(base_seed, n):
    # type: (int, int) -> List[int]
    """Derive ``n`` independent integer seeds from ``base_seed``.

    Deterministic in ``(base_seed, n, position)`` only, so replicate ``i`` gets
    the same stream whether it ran first, last or on another thread.
    """
    if n <= 0:
        return []
    seq = np.random.SeedSequence(int(base_seed))
    children = seq.spawn(int(n))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def seed_sequence_for(base_seed, index):
    # type: (int, int) -> np.random.SeedSequence
    """A :class:`numpy.random.SeedSequence` for one indexed work item."""
    return np.random.SeedSequence([int(base_seed), int(index)])


def rng_for(base_seed, index):
    # type: (int, int) -> Any
    """A fresh generator for one indexed work item."""
    return np.random.default_rng(seed_sequence_for(base_seed, index))


def ordered_concat(frames):
    # type: (Sequence[Any]) -> Any
    """Concatenate non-empty frames preserving input order (import-light)."""
    import pandas as pd

    kept = [f for f in frames if f is not None and len(f) > 0]
    if not kept:
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True)
