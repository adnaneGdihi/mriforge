"""Behaviour of :mod:`mriforge.infrastructure.training.strategies.lifecycle`.

Issue #1353 / audit dossier D12 §3.1: the four lifecycle hooks were declared on
``BaseTrainingStrategy``, overridden by real strategies, and **never called**.
The interesting failure mode of a fix for that is not "the hook is missing" — it
is "the hook fires at the wrong boundary, or twice, or with the wrong epoch
index", none of which raises. So this module drives the driver through the exact
iteration sequence a real loop produces (``epoch = iteration // len``, boundary
at ``iteration % len == 0``) and asserts on the recorded call *transcript*, not
on the fact that something was called.

The loop shell's side of the wiring is pinned in
``tests/unit/pipelines/test_training_loop.py`` — a full ``_execute_training_loop``
OOM-kills a dev box, which is why this file owns the behaviour and that one owns
the call sites.
"""

from __future__ import annotations

import logging

import pytest

from mriforge.infrastructure.training.strategies.lifecycle import (
    LIFECYCLE_HOOKS,
    StrategyLifecycleDriver,
)


class _SpyStrategy:
    """Records the lifecycle transcript in fire order."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def on_epoch_start(self, epoch: int) -> None:
        self.calls.append(("on_epoch_start", epoch))

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        self.calls.append(("on_epoch_end", epoch, metrics))

    def on_validation_start(self) -> None:
        self.calls.append(("on_validation_start",))

    def on_validation_end(self, metrics: dict) -> None:
        self.calls.append(("on_validation_end", metrics))


def _drive(driver: StrategyLifecycleDriver, *, steps: int, loader_len: int, start: int = 0):
    """Replay the loop's own boundary arithmetic against the driver.

    Mirrors ``_execute_training_loop``: ``begin_epoch`` every step before the
    train step; at ``iteration % loader_len == 0`` the boundary validation runs
    and then ``end_epoch`` closes the epoch that finished.
    """
    for iteration in range(start, start + steps):
        epoch = iteration // loader_len
        driver.begin_epoch(epoch)
        is_epoch_end = iteration % loader_len == 0
        metrics: dict = {}
        if is_epoch_end:
            driver.begin_validation()
            metrics = {"val_psnr": float(iteration)}
            driver.end_validation(metrics)
        if is_epoch_end:
            driver.end_epoch(metrics)


# --------------------------------------------------------------------------- #
# Epoch pair
# --------------------------------------------------------------------------- #


def test_the_first_begin_epoch_opens_the_epoch():
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)

    assert driver.begin_epoch(0) is True
    assert spy.calls == [("on_epoch_start", 0)]


def test_begin_epoch_is_idempotent_within_an_epoch():
    """The loop calls it every single step; only an index change may fire."""
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)

    for _ in range(25):
        driver.begin_epoch(0)

    assert spy.calls == [("on_epoch_start", 0)]


def test_no_epoch_has_ended_at_the_very_first_boundary():
    """Iteration 0 satisfies ``iteration % len == 0`` but nothing has finished.

    Without the driver's pending-index guard this is a phantom
    ``on_epoch_end(-1)`` — or, worse, ``on_epoch_end(0)`` for an epoch that has
    run exactly one step, spending a patience tick on it.
    """
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)

    driver.begin_epoch(0)
    assert driver.end_epoch({"val_psnr": 1.0}) is None
    assert [c[0] for c in spy.calls] == ["on_epoch_start"]


def test_end_epoch_reports_the_epoch_that_completed_not_the_current_one():
    """``on_epoch_end``'s docstring says "the epoch that just finished".

    At the boundary iteration the loop's ``epoch`` has already advanced to
    ``N + 1``; passing that through would mislabel every epoch by one.
    """
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)

    driver.begin_epoch(0)
    driver.begin_epoch(1)

    assert driver.end_epoch({"val_l1": 0.5}) == 0
    ends = [c for c in spy.calls if c[0] == "on_epoch_end"]
    assert ends == [("on_epoch_end", 0, {"val_l1": 0.5})]


def test_epoch_start_of_the_new_epoch_precedes_epoch_end_of_the_old_one():
    """Pinned as a DECISION, not left as an accident.

    The boundary iteration's ``train_step`` is the first step of the new epoch
    and runs before the boundary validation, and that validation is the only
    source of fresh end-of-epoch metrics. So the order is forced. A reviewer
    surprised by it should find this test rather than a silent surprise.
    """
    spy = _SpyStrategy()
    _drive(StrategyLifecycleDriver(spy), steps=9, loader_len=4)

    names = [(c[0], c[1] if len(c) > 1 and isinstance(c[1], int) else None) for c in spy.calls]
    start_1 = names.index(("on_epoch_start", 1))
    end_0 = names.index(("on_epoch_end", 0))
    assert start_1 < end_0


def test_a_replayed_loop_opens_and_closes_every_epoch_exactly_once():
    spy = _SpyStrategy()
    _drive(StrategyLifecycleDriver(spy), steps=9, loader_len=4)

    starts = [c[1] for c in spy.calls if c[0] == "on_epoch_start"]
    ends = [c[1] for c in spy.calls if c[0] == "on_epoch_end"]
    # iterations 0..8, len 4 -> epochs 0,1,2 opened; 0 and 1 completed.
    assert starts == [0, 1, 2]
    assert ends == [0, 1]


def test_the_closed_epoch_carries_its_own_boundary_validation_metrics():
    """Epoch 0 closes on the validation taken at iteration 4, not at 0."""
    spy = _SpyStrategy()
    _drive(StrategyLifecycleDriver(spy), steps=9, loader_len=4)

    ends = {c[1]: c[2] for c in spy.calls if c[0] == "on_epoch_end"}
    assert ends[0] == {"val_psnr": 4.0}
    assert ends[1] == {"val_psnr": 8.0}


def test_a_resumed_run_opens_its_epoch_without_inventing_an_ended_one():
    """Resume at iteration 25 with len 10 lands mid-epoch-2.

    Epochs 0 and 1 belong to the previous process; closing them here would run
    early stopping against metrics this run never measured. The first epoch this
    process may close is the first one it opened.
    """
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)
    _drive(driver, steps=6, loader_len=10, start=25)  # 25..30, boundary at 30

    starts = [c[1] for c in spy.calls if c[0] == "on_epoch_start"]
    ends = [c[1] for c in spy.calls if c[0] == "on_epoch_end"]
    assert starts == [2, 3]
    assert ends == [2]


def test_end_epoch_called_twice_in_one_epoch_does_not_double_count():
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)
    driver.begin_epoch(0)
    driver.begin_epoch(1)

    assert driver.end_epoch({}) == 0
    assert driver.end_epoch({}) is None
    assert len([c for c in spy.calls if c[0] == "on_epoch_end"]) == 1


# --------------------------------------------------------------------------- #
# Validation pair
# --------------------------------------------------------------------------- #


def test_the_validation_pair_brackets_the_pass_with_its_metrics():
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)

    driver.begin_validation()
    driver.end_validation({"val_psnr": 31.5})

    assert spy.calls == [("on_validation_start",), ("on_validation_end", {"val_psnr": 31.5})]


def test_absent_metrics_reach_the_hook_as_an_empty_dict_not_none():
    """The hook is typed ``dict[str, float]``; ``None`` would ``AttributeError``
    inside any implementation that does the obvious ``metrics.get(...)``."""
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)

    driver.end_validation(None)
    driver.begin_epoch(0)
    driver.begin_epoch(1)
    driver.end_epoch(None)

    assert ("on_validation_end", {}) in spy.calls
    assert ("on_epoch_end", 0, {}) in spy.calls


def test_a_hook_cannot_mutate_the_loops_own_metrics_dict():
    """The loop reuses ``val_metrics`` for early stopping and checkpoint
    selection immediately after; a hook popping a key would silently disarm
    both."""

    class _Vandal:
        def on_epoch_end(self, epoch, metrics):
            metrics["injected"] = 1.0
            metrics.clear()

    live = {"val_psnr": 30.0}
    driver = StrategyLifecycleDriver(_Vandal())
    driver.begin_epoch(0)
    driver.begin_epoch(1)
    driver.end_epoch(live)

    assert live == {"val_psnr": 30.0}


# --------------------------------------------------------------------------- #
# Reporting, not inferring
# --------------------------------------------------------------------------- #


def test_driven_reports_exactly_which_hooks_actually_fired():
    """NN16's observation: "declared" is not delivery, "fired" is."""
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)
    assert driver.driven == frozenset()

    driver.begin_epoch(0)
    driver.begin_validation()
    driver.end_validation({})

    assert driver.driven == {"on_epoch_start", "on_validation_start", "on_validation_end"}


def test_the_first_fire_of_each_hook_is_logged_at_info(caplog):
    """A run's own log is the operator-visible proof the contract is driven."""
    spy = _SpyStrategy()
    driver = StrategyLifecycleDriver(spy)

    logger_name = "mriforge.infrastructure.training.strategies.lifecycle"
    with caplog.at_level(logging.INFO, logger=logger_name):
        driver.begin_epoch(0)
        driver.begin_epoch(1)

    # getMessage(), not .message: a handler-stripping sibling test makes the
    # latter vanish in a wide run (#1290).
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("on_epoch_start" in m and "first fire" in m for m in messages)
    assert len([m for m in messages if "on_epoch_start" in m]) == 1


def test_a_strategy_missing_a_hook_is_reported_once_and_skipped(caplog):
    """Absent is a state to REPORT, never one to infer (non-negotiable 18)."""

    class _Partial:
        def on_epoch_start(self, epoch):
            pass

    driver = StrategyLifecycleDriver(_Partial())
    with caplog.at_level(logging.WARNING):
        driver.begin_validation()
        driver.begin_validation()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len([m for m in warnings if "on_validation_start" in m]) == 1
    assert "on_validation_start" not in driver.driven


def test_a_raising_hook_propagates_rather_than_being_swallowed():
    """An exception in ``on_epoch_end`` means early stopping did not evaluate;
    continuing would report a guarantee the run no longer has (NN3)."""

    class _Boom:
        def on_epoch_start(self, epoch):
            raise RuntimeError("stage unfreeze failed")

    with pytest.raises(RuntimeError, match="stage unfreeze failed"):
        StrategyLifecycleDriver(_Boom()).begin_epoch(0)


def test_the_hook_census_matches_what_the_base_strategy_declares():
    """A fifth hook added to ``BaseTrainingStrategy`` must be driven, not merely
    declared — which is the exact defect #1353 exists to close. This is the
    ratchet that makes the next one visible."""
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

    declared = {
        name
        for name in vars(BaseTrainingStrategy)
        if name.startswith("on_") and callable(vars(BaseTrainingStrategy)[name])
    }
    assert declared == set(LIFECYCLE_HOOKS), (
        "BaseTrainingStrategy declares lifecycle hooks the driver does not "
        f"dispatch: {sorted(declared - set(LIFECYCLE_HOOKS))}"
    )
