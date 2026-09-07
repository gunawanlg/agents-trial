from scorecard_segment_eval.parallel import resolve_n_jobs, run_parallel


def test_run_parallel_serial_and_threaded():
    def add_one(x):
        return x + 1

    serial = run_parallel(add_one, [1, 2, 3], n_jobs=1)
    threaded = run_parallel(add_one, [1, 2, 3], n_jobs=2, prefer="threads")
    assert serial == [2, 3, 4]
    assert threaded == [2, 3, 4]


def test_resolve_n_jobs():
    assert resolve_n_jobs(1, 10) == 1
    assert resolve_n_jobs(8, 3) == 3
    assert resolve_n_jobs(-1, 2) >= 1
