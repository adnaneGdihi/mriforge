"""Convergence test harness.

Tests in this package verify that each pillar training paradigm converges
within a bounded number of steps (≤200) on a small deterministic phantom.

The loss bounds are stored in ``bounds.json`` alongside this file.  If a
paradigm has no entry in ``bounds.json`` the test is skipped with an
explanatory message so that calibration can be done on the cluster.

Markers used::

    @pytest.mark.convergence  — declared in pyproject.toml
    @pytest.mark.slow          — declared in pyproject.toml
    @pytest.mark.gpu           — declared in pyproject.toml (skipped w/o CUDA)
"""
