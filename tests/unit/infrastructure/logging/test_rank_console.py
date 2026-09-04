"""Tests for :mod:`spectramr.infrastructure.logging.rank_console`.

Regression (2026-08-15): a 4-GPU ``train-distributed`` launch opened with four
interleaved copies of every startup line. ``setup_distributed`` did clamp
non-zero ranks to WARNING — but it runs *after* the config loads and the backend
resolves, and all four duplicated lines are emitted before that. The clamp moved
to the entry point, keyed off the rank torchrun exports into the environment
before the process even starts.
"""

from __future__ import annotations

import logging

import pytest

from spectramr.infrastructure.logging.rank_console import (
    quiet_secondary_ranks,
    rank_floor,
)


@pytest.fixture(autouse=True)
def _isolated_root_level(monkeypatch):
    """Restore the root level; these tests mutate global logging state."""
    original = logging.root.level
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    yield
    logging.root.setLevel(original)


# NOTE: every assertion below reads ``logging.root`` (the module-level object),
# never ``logging.getLogger()``. That is not style — under a leaked
# ``LoggingService`` patch, ``getLogger`` is not a read at all: its replacement
# does ``if logger.level < self._current_level: logger.setLevel(...)`` on every
# logger it hands out (``logging_service.py:500``). Asserting through it made
# these tests report the patch's level instead of the one under test, so a
# preceding ``tests/unit/infrastructure/services/`` file turned them red on an
# ordering that has nothing to do with rank. ``logging.root`` is a plain
# attribute lookup and cannot be intercepted. (The same leak is what reddens
# ``test_banners_and_progress.py`` in that ordering, on ``dev`` too -- filed
# separately; it is not this module's bug to fix, only to be immune to.)


def test_no_op_when_not_distributed():
    """A single-process run must be provably untouched."""
    logging.root.setLevel(logging.INFO)
    assert quiet_secondary_ranks() is False
    assert logging.root.level == logging.INFO


def test_no_op_on_rank_zero(monkeypatch):
    """Rank 0 is the rank whose narration we want to keep."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    logging.root.setLevel(logging.INFO)
    assert quiet_secondary_ranks() is False
    assert logging.root.level == logging.INFO


@pytest.mark.parametrize("rank", ["1", "2", "3"])
def test_clamps_secondary_rank(rank, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", rank)
    logging.root.setLevel(logging.INFO)
    assert quiet_secondary_ranks() is True
    assert logging.root.level == logging.WARNING


def test_warnings_survive_the_clamp(monkeypatch):
    """Only the narration is de-duplicated, not the diagnostics.

    A per-rank WARNING (DataLoader worker oversubscription, a Triton cache on
    NFS) is a fact about *that* rank; printing it once per rank is correct. If
    this ever clamps to ERROR, four ranks silently lose their warnings.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    quiet_secondary_ranks()
    assert logging.root.isEnabledFor(logging.WARNING)
    assert not logging.root.isEnabledFor(logging.INFO)


def test_is_idempotent(monkeypatch):
    """It runs at the entry point AND inside ``setup_distributed``."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    assert quiet_secondary_ranks() is True
    assert quiet_secondary_ranks() is True
    assert logging.root.level == logging.WARNING


def test_setup_distributed_uses_the_shared_helper():
    """``python -m spectramr.pipelines.distributed`` bypasses the CLI entry point.

    Pinned as source inspection because exercising it would require a real
    process group. The point is that there is ONE policy, not two that drift.
    """
    import inspect

    from spectramr.pipelines import distributed

    src = inspect.getsource(distributed)
    assert "quiet_secondary_ranks()" in src
    assert "logging.getLogger().setLevel" not in src, (
        "the inline clamp is back; it duplicates rank_console's policy"
    )


def test_the_bypass_path_is_clamped_before_anything_logs():
    """Merely *calling* the helper somewhere is not enough — WHERE decides.

    ``setup_distributed`` cannot run until the backend is known, and the
    backend comes from the config. So on the ``python -m`` path the config
    load and ``[Parallel] process-group backend=...`` both log at INFO
    *before* the only clamp that used to exist. The clamp therefore has to sit
    at the top of ``run_distributed_training`` — the funnel BOTH entry points
    share — not merely somewhere in the module.

    Pinned by source order, the same technique the startup-notice test uses:
    a refactor that moves the call down is exactly the regression, and it
    would leave the coarse ``in src`` assertion above still passing.
    """
    import inspect

    from spectramr.pipelines import distributed

    body = inspect.getsource(distributed.run_distributed_training)

    clamp = body.index("quiet_secondary_ranks()")
    assert clamp < body.index("TrainingSettings.from_yaml"), (
        "the config load logs before the clamp; rank 1-3 will echo it"
    )
    assert clamp < body.index("resolve_distributed_backend("), (
        "[Parallel] process-group backend=... logs before the clamp"
    )
    assert clamp < body.index("setup_distributed("), (
        "the clamp must precede the process group, not ride along inside it"
    )


class TestRankFloor:
    """``rank_floor`` is the half of the policy for code that RE-SETS a level.

    Clamping the root logger once is not enough: ``LoggingService.setup``
    resolves a level from the config and pushes it onto the root logger, every
    logger in ``loggerDict``, and every handler. That undoes an earlier
    ``quiet_secondary_ranks()`` the moment training configures logging — so the
    4x duplication would return for everything logged from that point on, which
    is most of a training run.
    """

    def test_single_process_is_untouched(self):
        assert rank_floor(logging.DEBUG) == logging.DEBUG
        assert rank_floor(logging.INFO) == logging.INFO

    def test_rank_zero_is_untouched(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "0")
        assert rank_floor(logging.INFO) == logging.INFO

    @pytest.mark.parametrize("rank", ["1", "2", "3"])
    def test_secondary_rank_is_floored(self, monkeypatch, rank):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", rank)
        assert rank_floor(logging.INFO) == logging.WARNING
        assert rank_floor(logging.DEBUG) == logging.WARNING

    def test_a_stricter_level_is_not_loosened(self, monkeypatch):
        """It is a FLOOR, not an assignment.

        A config asking for ERROR on a secondary rank must stay ERROR; clamping
        it down to WARNING would make a rank noisier than the operator asked.
        """
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        assert rank_floor(logging.ERROR) == logging.ERROR
        assert rank_floor(logging.CRITICAL) == logging.CRITICAL


class TestTheClampSurvivesLoggingConfiguration:
    """Both writers of the root level must go through the floor.

    Source-inspected rather than exercised: ``LoggingService.setup`` installs
    handlers and patches ``logging.getLogger`` globally, so running it for real
    inside the suite leaks into every later test. What must not regress is
    structural — that neither writer resolves a level the floor has not seen.
    """

    def test_logging_service_floors_the_level_it_resolves(self):
        import inspect

        from spectramr.infrastructure.services.logging_service import LoggingService

        body = inspect.getsource(LoggingService.setup)
        assert "rank_floor(self._resolve_log_level(log_level))" in body, (
            "setup() resolves a raw level again; it will re-verbose "
            "secondary ranks the moment training configures logging"
        )

    def test_bootstrap_floors_the_level_it_resolves(self):
        """``bootstrap_console_logging`` deliberately LOWERS an existing level.

        Its ``has_colored`` branch exists to never silence handlers a caller
        already installed, so without the floor it would happily undo the rank
        clamp. Today ``cli/app.py`` bootstraps before clamping, which hides
        this — the floor is what makes the two order-independent.
        """
        import inspect

        from spectramr.infrastructure.services import logging_service

        body = inspect.getsource(logging_service.bootstrap_console_logging)
        floor = body.index("rank_floor(")
        assert floor < body.index("root.setLevel("), (
            "the floor must be applied before any setLevel in this function"
        )
