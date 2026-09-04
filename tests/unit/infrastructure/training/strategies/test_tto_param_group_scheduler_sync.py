"""Regression test for F7d (smoke audit 2026-06-03).

``BaseTrainingStrategy._sync_schedulers_after_param_group_addition`` keeps an
LR scheduler's per-group state in sync when a strategy registers a new
param-group AFTER the scheduler is built. PyTorch schedulers snapshot
``base_lrs`` (one entry per param group) at construction; TTO adds
``motion_traj`` to ``opt_g`` after the scheduler exists, so the next
``scheduler.step()`` raised ``zip() argument 2 is longer than argument 1``
(experiment_vf_tto_v2, run 20260603_182243).

These tests drive the REAL helper (bound to a minimal stub holding
``env.schedulers``) and assert step() succeeds for both a raw
``CosineAnnealingWarmRestarts`` and a ``WarmupScheduler`` wrapper — the latter
keeps its own ``base_lrs`` / ``warmup_start_lr`` / ``warmup_end_lr`` lists that
must ALSO grow (the bug the first proposed patch missed).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # noqa: E402

from spectramr.infrastructure.training.scheduler_system import (  # noqa: E402
    WarmupScheduler,
)
from spectramr.infrastructure.training.strategies.base import (  # noqa: E402
    BaseTrainingStrategy,
)


def _sync(schedulers: dict | None, new_lr: float) -> None:
    """Invoke the real helper bound to a minimal stub self."""
    stub = SimpleNamespace(env=SimpleNamespace(schedulers=schedulers))
    BaseTrainingStrategy._sync_schedulers_after_param_group_addition(stub, new_lr)


def _model_opt():
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, opt


def test_raw_scheduler_base_lrs_synced_and_step_ok() -> None:
    model, opt = _model_opt()
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
    assert len(sched.base_lrs) == 1

    extra = torch.nn.Parameter(torch.randn(3))
    opt.add_param_group({"params": [extra], "lr": 1e-3})
    _sync({"opt_g": sched}, 1e-3)

    assert len(sched.base_lrs) == len(opt.param_groups) == 2
    # step() must not raise the zip()-length mismatch.
    for _ in range(7):
        opt.zero_grad(set_to_none=True)
        model(torch.randn(2, 8)).sum().backward()
        opt.step()
        sched.step()


def test_warmup_wrapper_inner_and_outer_lists_synced() -> None:
    model, opt = _model_opt()
    inner = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
    sched = WarmupScheduler(optimizer=opt, main_scheduler=inner, warmup_steps=3)
    assert len(sched.base_lrs) == 1
    assert len(inner.base_lrs) == 1

    extra = torch.nn.Parameter(torch.randn(3))
    opt.add_param_group({"params": [extra], "lr": 1e-3})
    _sync({"opt_g": sched}, 1e-3)

    # Outer WarmupScheduler keeps its own per-group lists — all must grow.
    assert len(sched.base_lrs) == 2
    assert len(sched.warmup_start_lr) == 2
    assert len(sched.warmup_end_lr) == 2
    # Wrapped inner scheduler's base_lrs (the actual zip() crash site) too.
    assert len(inner.base_lrs) == 2

    # step() through warmup and into the main phase must not raise.
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        model(torch.randn(2, 8)).sum().backward()
        opt.step()
        sched.step()


def test_shared_scheduler_object_not_double_appended() -> None:
    """The same scheduler under two keys must be appended to only once."""
    _model, opt = _model_opt()
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
    extra = torch.nn.Parameter(torch.randn(3))
    opt.add_param_group({"params": [extra], "lr": 1e-3})
    _sync({"opt_g": sched, "main": sched}, 1e-3)
    assert len(sched.base_lrs) == 2  # not 3


def test_no_schedulers_is_noop() -> None:
    _sync({}, 1e-3)  # must not raise
    _sync(None, 1e-3)  # must not raise
