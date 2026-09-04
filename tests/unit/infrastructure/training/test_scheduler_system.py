"""Scheduler factories and the warmup wrapper.

Covers the factories added for issue #533 (``linear``, ``constant`` — both named
in the corpus' ``lr_scheduler_strategy`` but previously registered nowhere) and
the ``WarmupScheduler`` fix that made the warmup LR apply at construction instead
of one optimizer step later.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR

from spectramr.infrastructure.training.scheduler_system import (
    SCHEDULER_REGISTRY,
    WarmupScheduler,
)


@pytest.fixture
def optimizer() -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)


class TestRegistryCoverage:
    @pytest.mark.parametrize(
        "name",
        [
            "cosine_annealing",
            "cosine_annealing_warm_restarts",
            "step_lr",
            "exponential",
            "linear",
            "constant",
            "reduce_on_plateau",
        ],
    )
    def test_family_is_registered(self, name):
        assert name in SCHEDULER_REGISTRY

    def test_linear_and_constant_were_the_missing_two(self, optimizer):
        """``linear_decay`` / ``constant`` appear in the corpus; both now build."""
        assert isinstance(SCHEDULER_REGISTRY["linear"](optimizer, {}), LinearLR)
        assert isinstance(SCHEDULER_REGISTRY["constant"](optimizer, {}), ConstantLR)

    def test_constant_default_really_holds_the_lr(self, optimizer):
        """torch's ConstantLR default is 1/3 for 5 epochs; 'constant' must not be."""
        sched = SCHEDULER_REGISTRY["constant"](optimizer, {})
        start = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            sched.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(start)

    def test_linear_decays(self, optimizer):
        sched = SCHEDULER_REGISTRY["linear"](
            optimizer, {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 10}
        )
        start = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            sched.step()
        assert optimizer.param_groups[0]["lr"] < start


class TestWarmupScheduler:
    def test_warmup_lr_is_applied_at_construction(self, optimizer):
        """Without this the FIRST optimizer step ran at the full base LR.

        ``step()`` only runs after an optimizer step, so the update warmup exists
        to protect was the one update that never saw it (issue #533).
        """
        main = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        WarmupScheduler(optimizer, main, warmup_steps=100, warmup_start_lr=1e-7)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-7)

    def test_zero_warmup_leaves_the_lr_untouched(self, optimizer):
        main = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        before = optimizer.param_groups[0]["lr"]
        WarmupScheduler(optimizer, main, warmup_steps=0, warmup_start_lr=1e-7)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(before)

    def test_warmup_ramps_then_hands_over_to_the_main_scheduler(self, optimizer):
        main = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        sched = WarmupScheduler(optimizer, main, warmup_steps=10, warmup_start_lr=1e-7)
        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            sched.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        # monotone ramp through warmup, ending at the base LR
        assert all(b >= a for a, b in pairwise(lrs))
        assert lrs[-1] == pytest.approx(5e-5, rel=1e-3)
        # then the main scheduler drives it back down
        for _ in range(200):
            sched.step()
        assert optimizer.param_groups[0]["lr"] < 5e-5

    def test_state_dict_roundtrip(self, optimizer):
        main = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        sched = WarmupScheduler(optimizer, main, warmup_steps=5, warmup_start_lr=1e-7)
        for _ in range(3):
            sched.step()
        state = sched.state_dict()

        opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)
        main2 = CosineAnnealingLR(opt2, T_max=100, eta_min=1e-6)
        sched2 = WarmupScheduler(opt2, main2, warmup_steps=5, warmup_start_lr=1e-7)
        sched2.load_state_dict(state)
        assert sched2.current_step == sched.current_step
