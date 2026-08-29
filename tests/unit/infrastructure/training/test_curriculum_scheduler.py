"""Unit tests for CurriculumScheduler and MotionFreezeCallback.

Targets ``mriforge.infrastructure.training.curriculum_scheduler``.

Properties verified:
- Phase epoch boundaries are computed correctly.
- ``get_phase`` returns the correct :class:`TrainingPhase` for each epoch.
- ``configure_model`` sets ``requires_grad`` consistent with the current phase.
- ``MotionFreezeCallback`` freezes/unfreezes by pattern match.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model with volume and motion parameters for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.inr_layer = nn.Linear(4, 4)        # volume param
        self.deform_layer = nn.Linear(4, 4)     # motion param
        self.other_layer = nn.Linear(4, 4)      # neither

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_construct_and_phase_count() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import CurriculumScheduler

    sched = CurriculumScheduler(total_epochs=100)
    assert len(sched.phases) == 3


# ---------------------------------------------------------------------------
# Parametrized: phase lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total_epochs,fractions,epoch,expected_phase",
    [
        # Default fractions (0.4, 0.3, 0.3): boundaries at 40 and 70
        (100, (0.4, 0.3, 0.3), 0, "volume_only"),
        (100, (0.4, 0.3, 0.3), 39, "volume_only"),
        (100, (0.4, 0.3, 0.3), 40, "motion_only"),
        (100, (0.4, 0.3, 0.3), 69, "motion_only"),
        (100, (0.4, 0.3, 0.3), 70, "joint"),
        (100, (0.4, 0.3, 0.3), 99, "joint"),
    ],
)
def test_get_phase_epoch_boundaries(
    total_epochs, fractions, epoch, expected_phase
) -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import CurriculumScheduler, TrainingPhase

    sched = CurriculumScheduler(total_epochs=total_epochs, phase_fractions=fractions)
    pc = sched.get_phase(epoch)
    assert pc.phase.value == expected_phase


# ---------------------------------------------------------------------------
# configure_model: VOLUME_ONLY freezes deform, unfreezes inr
# ---------------------------------------------------------------------------


def test_configure_model_volume_only_phase() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import CurriculumScheduler

    sched = CurriculumScheduler(total_epochs=100)
    model = _TinyModel()
    # Epoch 0 → VOLUME_ONLY
    _, lr = sched.configure_model(model, epoch=0)
    # inr_layer params should be trainable
    for name, param in model.named_parameters():
        if "inr_layer" in name:
            assert param.requires_grad, f"{name} should be trainable in VOLUME_ONLY"
    # deform_layer params should be frozen
    for name, param in model.named_parameters():
        if "deform_layer" in name:
            assert not param.requires_grad, f"{name} should be frozen in VOLUME_ONLY"


def test_configure_model_joint_phase_all_trainable() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import CurriculumScheduler

    sched = CurriculumScheduler(total_epochs=100)
    model = _TinyModel()
    # Epoch 70 → JOINT
    _, lr = sched.configure_model(model, epoch=70)
    for name, param in model.named_parameters():
        assert param.requires_grad, f"{name} must be trainable in JOINT phase"


def test_configure_model_returns_lr_float() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import CurriculumScheduler

    sched = CurriculumScheduler(total_epochs=100, base_lr=1e-4)
    model = _TinyModel()
    _, lr = sched.configure_model(model, epoch=0)
    assert isinstance(lr, float)
    assert lr > 0


# ---------------------------------------------------------------------------
# MotionFreezeCallback
# ---------------------------------------------------------------------------


def test_freeze_motion_sets_requires_grad_false() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import MotionFreezeCallback

    cb = MotionFreezeCallback(motion_patterns=["deform"])
    model = _TinyModel()
    cb.freeze_motion(model)
    for name, param in model.named_parameters():
        if "deform" in name:
            assert not param.requires_grad


def test_unfreeze_motion_restores_grad() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import MotionFreezeCallback

    cb = MotionFreezeCallback(motion_patterns=["deform"])
    model = _TinyModel()
    cb.freeze_motion(model)
    cb.unfreeze_motion(model)
    for name, param in model.named_parameters():
        if "deform" in name:
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Edge: fractions must sum to 1.0
# ---------------------------------------------------------------------------


def test_raises_if_fractions_do_not_sum_to_one() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import CurriculumScheduler

    with pytest.raises(AssertionError):
        CurriculumScheduler(total_epochs=100, phase_fractions=(0.4, 0.4, 0.4))


# ---------------------------------------------------------------------------
# Edge: beyond last epoch returns last phase
# ---------------------------------------------------------------------------


def test_epoch_beyond_total_returns_joint() -> None:
    from mriforge.infrastructure.training.curriculum_scheduler import CurriculumScheduler

    sched = CurriculumScheduler(total_epochs=50)
    pc = sched.get_phase(1000)  # way beyond total
    assert pc.phase.value == "joint"
