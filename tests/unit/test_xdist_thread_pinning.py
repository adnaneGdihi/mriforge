"""Torch gets one intra-op thread per xdist worker, and the whole box serially (#945).

Torch sizes its thread pool from the HOST core count, not from its share of it.
Under ``-n 8`` on a 24-core machine that was 8 x 16 = 128 compute threads, which
cost a 62x wall-clock blowup *and* flipped six seeded
``loss after training < loss before`` assertions by changing reduction order
inside the matmuls. The failing subset varied run to run, so the reds could not
be baselined -- and while CI is disabled a baseline is the only gate there is.

Measured on an idle machine over the five affected strategy files (79 tests):

===============  ===================  =========
run              before               after
===============  ===================  =========
serial           79 passed / 6.7 s    unchanged
``-n 8``         5 FAILED / 412.1 s   79 passed / 14.4 s
===============  ===================  =========

This test pins the *mechanism*, not the symptom: asserting on the six
convergence tests directly would be asserting on a coin-flip.
"""

import os

import torch


def test_torch_threads_match_the_execution_mode():
    """One thread inside a worker; unconstrained in a serial session.

    ``PYTEST_XDIST_WORKER`` is set by ``pytest-xdist`` in each worker process
    and absent in a serial run, so this single test covers both directions --
    whichever way the suite is invoked, one branch is live. Both branches are
    exercised in CI, which runs serially in ``pr-required.yml`` and
    ``-n auto --dist loadfile`` in ``pr-advisory.yml``.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    threads = torch.get_num_threads()

    if worker:
        workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or 1)
        expected = max(1, (os.cpu_count() or 1) // max(1, workers))
        assert threads == expected, (
            f"xdist worker {worker} has {threads} torch threads, expected "
            f"{expected} ({os.cpu_count()} cores / {workers} workers). Each "
            "worker sizes its pool from the HOST core count unless told "
            "otherwise, so N workers oversubscribe by N-fold -- see "
            "tests/conftest.py"
        )
    else:
        assert threads > 1, (
            f"a serial session was pinned to {threads} thread(s). The pin is "
            "meant to apply only under xdist; serially the suite should use "
            "the whole machine"
        )


def test_the_thread_env_is_pinned_too_under_xdist():
    """The env vars matter for libraries that read them at import, not just torch.

    ``torch.set_num_threads`` covers torch's own pool. ``OMP_NUM_THREADS`` /
    ``MKL_NUM_THREADS`` are what BLAS backends and any subprocess consult, and
    they are read once -- so they are set alongside rather than instead.
    """
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return  # serial session: nothing is pinned, by design

    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or 1)
    expected = str(max(1, (os.cpu_count() or 1) // max(1, workers)))
    assert os.environ.get("OMP_NUM_THREADS") == expected
    assert os.environ.get("MKL_NUM_THREADS") == expected
