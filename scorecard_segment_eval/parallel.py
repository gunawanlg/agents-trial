"""Lightweight parallel map. Thread pool by default so DataFrames need not pickle."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os

from typing import Any, Callable, Iterable, List, Optional


def resolve_n_jobs(n_jobs, n_tasks=None):
    # type: (Optional[int], Optional[int]) -> int
    cpu = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    if n_jobs < 0:
        n_jobs = max(1, cpu + 1 + n_jobs)
    n_jobs = max(1, int(n_jobs))
    if n_tasks is not None:
        n_jobs = min(n_jobs, max(1, int(n_tasks)))
    return n_jobs


def run_parallel(fn, items, n_jobs=1, prefer="threads"):
    # type: (Callable[[Any], Any], Iterable[Any], Optional[int], str) -> List[Any]
    """Map ``fn`` over ``items``. ``n_jobs<=1`` stays serial and deterministic."""
    seq = list(items)
    if not seq:
        return []
    workers = resolve_n_jobs(n_jobs, len(seq))
    if workers == 1 or len(seq) == 1:
        return [fn(item) for item in seq]
    pool_cls = ProcessPoolExecutor if prefer == "processes" else ThreadPoolExecutor
    with pool_cls(max_workers=workers) as pool:
        return list(pool.map(fn, seq))
