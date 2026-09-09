import threading

import numpy as np
import pytest

from scorecard_segment_eval import Gates, evaluate_segments, make_synthetic_book
from scorecard_segment_eval.bootstrap import bootstrap_delta_gini, bootstrap_gini_ci
from scorecard_segment_eval.parallel import (
    cpu_count,
    in_parallel_context,
    map_jobs,
    resolve_n_jobs,
    rng_for,
    spawn_seeds,
)


def _gates(**kwargs):
    base = {
        "min_n": 400,
        "min_events": 25,
        "n_bootstrap": 60,
        "bootstrap_seed": 0,
        "holdout_frac": 0.3,
    }
    base.update(kwargs)
    return Gates(**base)


# --------------------------------------------------------------------------
# Worker-count resolution
# --------------------------------------------------------------------------
def test_resolve_n_jobs_conventions():
    assert resolve_n_jobs(None) == 1
    assert resolve_n_jobs(0) == 1
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(3, cap=8, n_items=100) == 3
    assert resolve_n_jobs(-1, cap=64, n_items=1000) == cpu_count()
    assert resolve_n_jobs(-2, cap=64, n_items=1000) == max(cpu_count() - 1, 1)
    assert resolve_n_jobs(100, cap=4, n_items=1000) == 4
    assert resolve_n_jobs(100, cap=64, n_items=2) == 2
    assert resolve_n_jobs(-100, cap=8, n_items=10) == 1


# --------------------------------------------------------------------------
# map_jobs semantics
# --------------------------------------------------------------------------
def test_map_jobs_preserves_input_order():
    items = list(range(50))
    assert map_jobs(lambda i: i * i, items, n_jobs=1) == [i * i for i in items]
    assert map_jobs(lambda i: i * i, items, n_jobs=-1) == [i * i for i in items]


def test_map_jobs_on_empty_input():
    assert map_jobs(lambda i: i, [], n_jobs=-1) == []


def test_serial_mode_runs_in_the_calling_thread():
    seen = set()
    map_jobs(lambda i: seen.add(threading.current_thread().name), range(8), n_jobs=1)
    assert seen == {threading.current_thread().name}


def test_nested_maps_do_not_multiply_workers():
    """The inner map must degrade to serial so threads cannot explode."""
    nested_contexts = []

    def _inner(_i):
        return in_parallel_context()

    def _outer(_i):
        threads_before = threading.active_count()
        results = map_jobs(_inner, range(4), n_jobs=-1)
        nested_contexts.append((threads_before, threading.active_count(), results))
        return 1

    map_jobs(_outer, range(4), n_jobs=-1)
    for before, after, results in nested_contexts:
        assert after <= before, "inner map spawned extra threads"
        assert all(results), "inner work should know it is already parallel"


def test_allow_nested_is_opt_in():
    def _inner(_i):
        return in_parallel_context()

    def _outer(_i):
        return map_jobs(_inner, range(2), n_jobs=2, allow_nested=True)

    assert map_jobs(_outer, range(2), n_jobs=2) == [[True, True], [True, True]]


def test_exceptions_propagate_from_workers():
    def _boom(i):
        if i == 3:
            raise RuntimeError("boom")
        return i

    with pytest.raises(RuntimeError):
        map_jobs(_boom, range(8), n_jobs=-1)


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------
def test_spawn_seeds_are_deterministic_and_distinct():
    first = spawn_seeds(42, 16)
    second = spawn_seeds(42, 16)
    assert first == second
    assert len(set(first)) == 16
    assert spawn_seeds(43, 16) != first
    assert spawn_seeds(42, 0) == []


def test_rng_for_depends_only_on_seed_and_index():
    a = rng_for(7, 3).integers(0, 1000, size=5)
    b = rng_for(7, 3).integers(0, 1000, size=5)
    c = rng_for(7, 4).integers(0, 1000, size=5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --------------------------------------------------------------------------
# Identical results, serial vs parallel
# --------------------------------------------------------------------------
def test_bootstrap_gini_ci_is_identical_serially_and_in_parallel():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-2.0 + x)))).astype(float)
    serial = bootstrap_gini_ci(y, x, n_bootstrap=200, seed=11, n_jobs=1)
    parallel = bootstrap_gini_ci(y, x, n_bootstrap=200, seed=11, n_jobs=-1)
    assert serial == parallel
    assert np.isfinite(serial["gini_se"])


def test_bootstrap_delta_gini_is_identical_serially_and_in_parallel():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2500)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-2.0 + x)))).astype(float)
    noisy = x + rng.normal(size=2500)
    serial = bootstrap_delta_gini(y, x, noisy, n_bootstrap=150, seed=5, n_jobs=1)
    parallel = bootstrap_delta_gini(y, x, noisy, n_bootstrap=150, seed=5, n_jobs=-1)
    assert serial == parallel


def test_bootstrap_result_is_independent_of_the_worker_count():
    rng = np.random.default_rng(2)
    x = rng.normal(size=1500)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-1.8 + x)))).astype(float)
    results = [
        bootstrap_gini_ci(y, x, n_bootstrap=120, seed=3, n_jobs=jobs) for jobs in (1, 2, 3, 4, -1)
    ]
    assert all(r == results[0] for r in results)


def test_evaluate_segments_is_identical_serially_and_in_parallel():
    df, cols = make_synthetic_book(n=9000, seed=3)
    gates = _gates()
    serial = evaluate_segments(df, cols, gates, n_jobs=1)
    parallel = evaluate_segments(df, cols, gates, n_jobs=-1)
    for name in (
        "segment_summary",
        "decisions",
        "refit_comparison",
        "characteristics",
        "vintage",
        "stability",
        "grouping_comparison",
    ):
        left = getattr(serial, name)
        right = getattr(parallel, name)
        assert left.equals(right), name + " differs between serial and parallel runs"


def test_evaluate_segments_worker_count_comes_from_gates_by_default():
    df, cols = make_synthetic_book(n=4000, seed=8)
    from_gates = evaluate_segments(df, cols, _gates(n_jobs=-1))
    explicit = evaluate_segments(df, cols, _gates(n_jobs=1), n_jobs=-1)
    assert from_gates.meta["n_jobs"] == -1
    assert explicit.meta["n_jobs"] == -1
    assert from_gates.decisions.equals(explicit.decisions)
