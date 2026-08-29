"""Unit tests for PhaseScheduler.

Targets ``mriforge.infrastructure.optimization.schedulers.phase_scheduler.PhaseScheduler``.

Properties verified:
- ``get_phase`` returns correct string for each epoch range.
- ``get_loss_weights`` zeroes out specific keys during warmup.
- ``update_model_state`` toggles ``requires_grad`` based on phase.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyLaNSModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.siren_fc = nn.Linear(4, 4)
        self.biophysics_fc = nn.Linear(4, 4)
        self.velocity_fc = nn.Linear(4, 4)
        self.other_fc = nn.Linear(4, 4)

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_construct_default() -> None:
    from mriforge.infrastructure.optimization.schedulers.phase_scheduler import (
        PhaseScheduler,
    )

    sched = PhaseScheduler(warmup_epochs=5, dynamics_epochs=20, total_epochs=100)
    assert sched.get_phase(0) == "warmup"


# ---------------------------------------------------------------------------
# Parametrized: phase boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "warmup,dynamics,total,epoch,expected",
    [
        # warmup=5, dynamics=20 → warmup [0..4], dynamics [5..24], joint [25..]
        (5, 20, 100, 0, "warmup"),
        (5, 20, 100, 4, "warmup"),
        (5, 20, 100, 5, "dynamics"),
        (5, 20, 100, 24, "dynamics"),
        (5, 20, 100, 25, "joint"),
        (5, 20, 100, 99, "joint"),
        # warmup=10, dynamics=10 → warmup [0..9], dynamics [10..19], joint [20..]
        (10, 10, 50, 9, "warmup"),
        (10, 10, 50, 10, "dynamics"),
        (10, 10, 50, 20, "joint"),
    ],
)
def test_get_phase_boundaries(warmup, dynamics, total, epoch, expected) -> None:
    from mriforge.infrastructure.optimization.schedulers.phase_scheduler import (
        PhaseScheduler,
    )

    sched = PhaseScheduler(warmup_epochs=warmup, dynamics_epochs=dynamics, total_epochs=total)
    assert sched.get_phase(epoch) == expected


# ---------------------------------------------------------------------------
# get_loss_weights
# ---------------------------------------------------------------------------


def test_warmup_zeroes_smoothness_and_bloch() -> None:
    from mriforge.infrastructure.optimization.schedulers.phase_scheduler import (
        PhaseScheduler,
    )

    sched = PhaseScheduler(warmup_epochs=5, dynamics_epochs=20, total_epochs=100)
    base = {"lambda_smoothness": 1.0, "lambda_bloch": 0.5, "lambda_recon": 2.0}
    weights = sched.get_loss_weights(epoch=0, base_config=base)
    assert weights["lambda_smoothness"] == pytest.approx(0.0)
    assert weights["lambda_bloch"] == pytest.approx(0.0)
    # Other keys preserved
    assert weights["lambda_recon"] == pytest.approx(2.0)


def test_joint_phase_preserves_all_keys() -> None:
    from mriforge.infrastructure.optimization.schedulers.phase_scheduler import (
        PhaseScheduler,
    )

    sched = PhaseScheduler(warmup_epochs=5, dynamics_epochs=20, total_epochs=100)
    base = {"lambda_smoothness": 1.0, "lambda_bloch": 0.5}
    weights = sched.get_loss_weights(epoch=50, base_config=base)
    assert weights == base


# ---------------------------------------------------------------------------
# update_model_state
# ---------------------------------------------------------------------------


def test_update_model_state_warmup_freezes_siren() -> None:
    from mriforge.infrastructure.optimization.schedulers.phase_scheduler import (
        PhaseScheduler,
    )

    sched = PhaseScheduler(warmup_epochs=5, dynamics_epochs=20, total_epochs=100)
    model = _TinyLaNSModel()
    # First ensure all trainable
    for p in model.parameters():
        p.requires_grad = True
    sched.update_model_state(model, epoch=0)  # warmup → freeze siren/velocity
    for name, param in model.named_parameters():
        if "siren" in name or "velocity" in name:
            assert not param.requires_grad, f"{name} should be frozen in warmup"


def test_update_model_state_joint_unfreezes_all() -> None:
    from mriforge.infrastructure.optimization.schedulers.phase_scheduler import (
        PhaseScheduler,
    )

    sched = PhaseScheduler(warmup_epochs=5, dynamics_epochs=20, total_epochs=100)
    model = _TinyLaNSModel()
    for p in model.parameters():
        p.requires_grad = False
    sched.update_model_state(model, epoch=50)  # joint
    for name, param in model.named_parameters():
        assert param.requires_grad, f"{name} should be trainable in joint phase"


# ---------------------------------------------------------------------------
# Edge: epoch beyond total stays joint
# ---------------------------------------------------------------------------


def test_edge_epoch_beyond_total_is_joint() -> None:
    from mriforge.infrastructure.optimization.schedulers.phase_scheduler import (
        PhaseScheduler,
    )

    sched = PhaseScheduler(warmup_epochs=5, dynamics_epochs=10, total_epochs=20)
    assert sched.get_phase(9999) == "joint"
