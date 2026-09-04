"""Unit tests for the ``LoopState`` seam (WS-3 PR-3).

``LoopState`` is the mutable iteration/epoch counter the training loop owns and
the strategy reads through ``strategy.loop_state`` — the live-iteration SSOT
that replaces the frozen, perpetually-zero ``env.step``.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

from spectramr.infrastructure.training.loop_state import (
    LoopState,
    resolve_loop_iteration,
)


def test_defaults_match_historical_zero():
    """A fresh LoopState reports iteration/epoch 0 — the same default a direct
    ``strategy.train_step`` call (no loop) used to get from ``kwargs.get(
    'iteration', 0)``, so constructing a strategy outside the loop is safe."""
    state = LoopState()
    assert state.iteration == 0
    assert state.epoch == 0


def test_is_mutable_so_the_loop_can_advance_it():
    """Unlike the frozen ``TrainingEnvironment.step``, LoopState must be
    writable — that mutability is the entire point of the seam."""
    state = LoopState()
    state.iteration = 30_000
    state.epoch = 12
    assert state.iteration == 30_000
    assert state.epoch == 12


def test_only_consumed_counters_are_exposed():
    """Guard against re-introducing unwired state (facade, pitfall #16): the
    dataclass exposes exactly the fields a strategy/loop reads today —
    iteration/epoch plus the loss-schedule override map (consumed by the loss
    computer via ``reconstruction.py`` and written by LossScheduleController)."""
    assert {f.name for f in fields(LoopState)} == {
        "iteration",
        "epoch",
        "loss_weight_overrides",
    }


def test_loss_weight_overrides_defaults_empty_and_is_mutable():
    """The override map defaults empty (no scheduling => static behavior) and is
    independent per instance (default_factory, not a shared mutable default)."""
    a = LoopState()
    b = LoopState()
    assert a.loss_weight_overrides == {}
    a.loss_weight_overrides["l1"] = 0.5  # the controller writes here each step
    assert a.loss_weight_overrides == {"l1": 0.5}
    assert b.loss_weight_overrides == {}  # not shared across instances


# --- resolve_loop_iteration (the step-gated-logic accessor) -------------------


def test_resolve_reads_the_live_iteration():
    """The accessor returns the loop_state's live iteration — so step-gated
    logic (metric-interval throttle, debug gates) sees the real step, not the
    frozen ``env.step`` 0 (pitfall #16)."""
    strat = SimpleNamespace(loop_state=LoopState(iteration=4200, epoch=3))
    assert resolve_loop_iteration(strat) == 4200
    strat.loop_state.iteration = 4201  # the loop advances it in place
    assert resolve_loop_iteration(strat) == 4201


def test_resolve_falls_back_to_zero_without_loop_state():
    """A strategy/mixin constructed standalone (no loop, e.g. a bare unit-test
    stub) has no ``loop_state`` — the accessor must degrade to 0, mirroring the
    historical ``self.env.step if self.env else 0`` defensiveness rather than
    raising AttributeError."""
    assert resolve_loop_iteration(SimpleNamespace()) == 0
    assert resolve_loop_iteration(SimpleNamespace(loop_state=None)) == 0
