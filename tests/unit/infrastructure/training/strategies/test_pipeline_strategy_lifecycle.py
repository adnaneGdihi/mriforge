"""The two schema-declared features that live only inside lifecycle hooks.

``training.pipeline.end_to_end_finetune_epoch`` and per-stage ``early_stopping``
are both read at construction, both summarised in the startup log as active, and
both have their entire behavioural implementation inside
``MultiTrainingStrategy.on_epoch_start`` / ``on_epoch_end`` — hooks that no driver
called until #1353. Wiring the driver is only half the delivery: non-negotiable
16 is satisfied when the *feature* is observed to engage, so this module drives
the real hook bodies and asserts on ``requires_grad`` and on the freeze.

The hooks are exercised on an instance built with ``object.__new__``: a real
``MultiTrainingStrategy.__init__`` builds stage models, losses and optimizers from a
full ``TrainingSettings``, none of which either hook reads.
"""

from __future__ import annotations

import inspect
import logging
import re
from types import SimpleNamespace

import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.pipeline_strategy import MultiTrainingStrategy


def _es(*, patience: int, enabled: bool = True, metric: str | None = None):
    return SimpleNamespace(enabled=enabled, patience=patience, metric=metric)


def _strategy(
    *,
    stages: dict[str, nn.Module] | None = None,
    finetune_epoch: int | None = None,
    early_stopping: dict | None = None,
) -> MultiTrainingStrategy:
    strategy = object.__new__(MultiTrainingStrategy)
    strategy.multi_stages = stages if stages is not None else {"stage1": nn.Linear(2, 2)}
    strategy._finetune_epoch = finetune_epoch
    strategy._stage_early_stopping = early_stopping or {}
    strategy._es_unresolved_monitors = set()
    strategy.collected = []
    strategy.registered = 0

    def _collect():
        strategy.collected.append(True)
        return []

    def _register():
        strategy.registered += 1

    strategy._collect_trainable_params = _collect
    strategy._register_global_params = _register
    return strategy


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _all_require_grad(module: nn.Module) -> bool:
    return all(p.requires_grad for p in module.parameters())


# --------------------------------------------------------------------------- #
# end_to_end_finetune_epoch  (on_epoch_start)
# --------------------------------------------------------------------------- #


def test_stages_stay_frozen_before_the_finetune_epoch():
    stage = nn.Linear(2, 2)
    _freeze(stage)
    strategy = _strategy(stages={"s": stage}, finetune_epoch=5)

    strategy.on_epoch_start(4)

    assert not _all_require_grad(stage)
    assert strategy.registered == 0


def test_the_finetune_epoch_unfreezes_every_stage_and_hands_them_to_the_optimizer():
    """The unfreeze is only half of it: newly-unfrozen parameters that are never
    registered keep ``requires_grad=True`` and receive no updates — a
    fine-tuning phase that silently does not fine-tune."""
    a, b = nn.Linear(2, 2), nn.Linear(2, 2)
    _freeze(a)
    _freeze(b)
    strategy = _strategy(stages={"a": a, "b": b}, finetune_epoch=5)

    strategy.on_epoch_start(5)

    assert _all_require_grad(a) and _all_require_grad(b)
    assert strategy.registered == 1


def test_a_resumed_run_unfreezes_on_its_first_hook_call_past_the_threshold():
    """Resume drops the loop straight into epoch 9; the driver opens that epoch
    and no earlier one. A threshold test written as ``epoch == finetune_epoch``
    would leave such a run frozen for its whole remaining budget."""
    stage = nn.Linear(2, 2)
    _freeze(stage)
    strategy = _strategy(stages={"s": stage}, finetune_epoch=3)

    strategy.on_epoch_start(9)

    assert _all_require_grad(stage)


def test_unfreezing_happens_exactly_once():
    stage = nn.Linear(2, 2)
    _freeze(stage)
    strategy = _strategy(stages={"s": stage}, finetune_epoch=1)

    strategy.on_epoch_start(1)
    strategy.on_epoch_start(2)
    strategy.on_epoch_start(3)

    assert strategy.registered == 1


def test_an_unset_finetune_epoch_is_inert():
    stage = nn.Linear(2, 2)
    _freeze(stage)
    strategy = _strategy(stages={"s": stage}, finetune_epoch=None)

    strategy.on_epoch_start(1000)

    assert not _all_require_grad(stage)


# --------------------------------------------------------------------------- #
# per-stage early_stopping  (on_epoch_end)
# --------------------------------------------------------------------------- #


def test_the_default_monitor_is_the_key_validation_step_actually_emits():
    """Anti-vacuity: the fixture shape comes from the PRODUCER, not the docstring.

    ``on_epoch_end`` used to default to ``val_{stage}/l1``. Nothing in the
    repository emits a slash-separated metric key — ``validation_step`` writes
    ``val_{stage}_l1`` — so the primary lookup could never hit and only a
    hand-written second candidate kept the feature alive at all.
    """
    produced = re.findall(
        r'metrics\[f"(val_\{\w+\}_l1)"\]', inspect.getsource(MultiTrainingStrategy.validation_step)
    )
    assert produced, "validation_step no longer emits a per-stage L1 key"

    consumed = inspect.getsource(MultiTrainingStrategy.on_epoch_end)
    assert 'f"val_{stage_name}_l1"' in consumed
    assert 'f"val_{stage_name}/l1"' not in consumed


def test_the_monitor_resolves_through_the_frameworks_elected_resolver():
    """One owner per invariant (#17).

    ``metric_keys.resolve_metric_key`` is the framework's owner of monitor-key
    aliasing — the training loop's own early stopping already routes through it.
    A validator emitting the unprefixed ``stage1_l1`` is one of the four mismatch
    classes it handles; the retired two-candidate chain missed it.
    """
    stage = nn.Linear(2, 2)
    strategy = _strategy(stages={"stage1": stage}, early_stopping={"stage1": _es(patience=1)})

    strategy.on_epoch_end(0, {"stage1_l1": 0.5})
    strategy.on_epoch_end(1, {"stage1_l1": 0.9})

    assert not _all_require_grad(stage), "patience exceeded but the stage was not frozen"


def test_a_stage_freezes_only_after_patience_epochs_without_improvement():
    stage = nn.Linear(2, 2)
    strategy = _strategy(stages={"stage1": stage}, early_stopping={"stage1": _es(patience=2)})

    for value in (0.50, 0.60):  # best=0.50, then one epoch of no improvement
        strategy.on_epoch_end(0, {"val_stage1_l1": value})
    assert _all_require_grad(stage)

    strategy.on_epoch_end(2, {"val_stage1_l1": 0.61})  # second — patience reached
    assert not _all_require_grad(stage)


def test_an_improving_stage_is_never_frozen():
    stage = nn.Linear(2, 2)
    strategy = _strategy(stages={"stage1": stage}, early_stopping={"stage1": _es(patience=1)})

    for value in (0.9, 0.8, 0.7, 0.6, 0.5):
        strategy.on_epoch_end(0, {"val_stage1_l1": value})

    assert _all_require_grad(stage)


def test_an_explicit_monitor_overrides_the_default():
    stage = nn.Linear(2, 2)
    strategy = _strategy(
        stages={"stage1": stage},
        early_stopping={"stage1": _es(patience=1, metric="val_psnr")},
    )

    strategy.on_epoch_end(0, {"val_psnr": 0.4, "val_stage1_l1": 9.0})
    strategy.on_epoch_end(1, {"val_psnr": 0.5, "val_stage1_l1": 0.1})

    assert not _all_require_grad(stage)


def test_a_disabled_block_is_inert():
    stage = nn.Linear(2, 2)
    strategy = _strategy(
        stages={"stage1": stage}, early_stopping={"stage1": _es(patience=1, enabled=False)}
    )

    strategy.on_epoch_end(0, {"val_stage1_l1": 5.0})
    strategy.on_epoch_end(1, {"val_stage1_l1": 9.0})

    assert _all_require_grad(stage)


def test_an_unresolvable_monitor_is_reported_once_not_silently_skipped(caplog):
    """The wired-but-dead shape this whole issue is about.

    Per-stage L1 keys are emitted only when ``evaluate_intermediates`` is on. An
    arm that declares ``early_stopping`` without it gets a startup log calling
    the feature active and a hook that ``continue``s forever. Absent is a state
    to report (#18) — and once per stage, not once per epoch.
    """
    strategy = _strategy(early_stopping={"stage1": _es(patience=1)})

    with caplog.at_level(logging.WARNING):
        strategy.on_epoch_end(0, {"val_psnr": 30.0})
        strategy.on_epoch_end(1, {"val_psnr": 31.0})

    hits = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "stage1" in r.getMessage()
    ]
    assert len(hits) == 1, hits
    assert "evaluate_intermediates" in hits[0]


def test_no_metrics_at_all_is_a_no_op_rather_than_a_warning(caplog):
    """An epoch that closed without a validation event has nothing to say about
    any stage; warning there would fire on every non-validating boundary."""
    strategy = _strategy(early_stopping={"stage1": _es(patience=1)})

    with caplog.at_level(logging.WARNING):
        strategy.on_epoch_end(0, {})

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert strategy._es_unresolved_monitors == set()


def test_a_non_float_metric_value_does_not_reach_the_comparison():
    """``validation_step`` writes host floats, but a strategy that hands the hook
    a 0-dim tensor must not silently compare tensors — the freeze decision would
    become a tensor truthiness sync in a training loop (#9)."""
    stage = nn.Linear(2, 2)
    strategy = _strategy(stages={"stage1": stage}, early_stopping={"stage1": _es(patience=1)})

    strategy.on_epoch_end(0, {"val_stage1_l1": torch.tensor(0.5)})
    strategy.on_epoch_end(1, {"val_stage1_l1": torch.tensor(0.9)})

    assert not _all_require_grad(stage)
    assert isinstance(strategy._es_best["stage1"], float)
