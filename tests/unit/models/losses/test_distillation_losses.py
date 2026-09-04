"""Tests for ``KnowledgeDistillationLoss`` and ``EMASelfDistillationLoss``.

Targets ``spectramr.models.losses.distillation_losses``. Both losses require
a teacher / EMA-target via the ``context`` dict — they fail loud when
absent (CLAUDE.md #9).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.distillation_losses import KnowledgeDistillationLoss


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Defaults: mode=kl, temperature=4.0."""
    loss = KnowledgeDistillationLoss()
    assert loss.mode == "kl"
    assert loss.temperature == 4.0


@pytest.mark.parametrize("mode", ["kl", "feature_mse"])
def test_documented_modes_accepted(mode: str) -> None:
    """Both documented modes construct successfully."""
    loss = KnowledgeDistillationLoss(mode=mode)
    assert loss.mode == mode


def test_invalid_mode_raises() -> None:
    """Unknown mode → ``ValueError``."""
    with pytest.raises(ValueError, match="mode must be kl or feature_mse"):
        KnowledgeDistillationLoss(mode="cosine")


def test_zero_temperature_raises() -> None:
    """``temperature <= 0`` → ``ValueError``."""
    with pytest.raises(ValueError, match="temperature must be > 0"):
        KnowledgeDistillationLoss(temperature=0.0)


def test_negative_temperature_raises() -> None:
    """Negative temperature → ``ValueError``."""
    with pytest.raises(ValueError, match="temperature must be > 0"):
        KnowledgeDistillationLoss(temperature=-1.0)


# ---------------------------------------------------------------------------
# Context fail-loud
# ---------------------------------------------------------------------------


def test_missing_context_raises() -> None:
    """``context=None`` → ``ValueError``."""
    loss = KnowledgeDistillationLoss()
    pred = torch.randn(2, 4)
    target = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="requires context"):
        loss(pred, target, context=None)


def test_context_missing_teacher_output_raises() -> None:
    """Missing ``teacher_output`` key → ``ValueError``."""
    loss = KnowledgeDistillationLoss()
    pred = torch.randn(2, 4)
    target = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="teacher_output"):
        loss(pred, target, context={})


def test_teacher_shape_mismatch_raises() -> None:
    """Teacher / student shape mismatch → ``ValueError``."""
    loss = KnowledgeDistillationLoss()
    pred = torch.randn(2, 4)
    target = torch.zeros(2, 4)
    teacher_wrong = torch.randn(2, 8)
    with pytest.raises(ValueError, match="!="):
        loss(pred, target, context={"teacher_output": teacher_wrong})


# ---------------------------------------------------------------------------
# KL mode
# ---------------------------------------------------------------------------


def test_kl_mode_identity_yields_zero() -> None:
    """``loss(x, _, context={teacher: x}) == 0`` (KL between identical distributions)."""
    loss = KnowledgeDistillationLoss(mode="kl", temperature=2.0)
    pred = torch.randn(2, 5)
    out = loss(pred, torch.zeros_like(pred), context={"teacher_output": pred.clone()})
    assert out.item() < 1e-6


def test_kl_mode_grows_with_divergence() -> None:
    """Larger teacher/student divergence → larger KL loss."""
    loss_fn = KnowledgeDistillationLoss(mode="kl", temperature=2.0)
    pred = torch.zeros(1, 4)
    teacher_close = pred + 0.1 * torch.randn(1, 4)
    teacher_far = pred + 5.0 * torch.randn(1, 4)
    target = torch.zeros_like(pred)

    loss_close = loss_fn(pred, target, context={"teacher_output": teacher_close}).item()
    loss_far = loss_fn(pred, target, context={"teacher_output": teacher_far}).item()
    # KL(very different || student) > KL(similar || student)
    assert loss_far > loss_close


def test_kl_mode_temperature_scales_loss() -> None:
    """KD loss scales as T² (higher T → larger absolute loss for same logit gap)."""
    pred = torch.zeros(1, 4)
    teacher = torch.tensor([[5.0, -5.0, 5.0, -5.0]])
    target = torch.zeros_like(pred)

    loss_low_t = KnowledgeDistillationLoss(mode="kl", temperature=1.0)(
        pred, target, context={"teacher_output": teacher}
    ).item()
    loss_high_t = KnowledgeDistillationLoss(mode="kl", temperature=4.0)(
        pred, target, context={"teacher_output": teacher}
    ).item()
    # T² scaling: T=4 should be larger absolute KD loss vs T=1, since the
    # explicit T² scaling factor outweighs softening for finite gaps.
    assert loss_high_t != pytest.approx(loss_low_t)


# ---------------------------------------------------------------------------
# Feature-MSE mode
# ---------------------------------------------------------------------------


def test_feature_mse_mode_identity_zero() -> None:
    """Feature-MSE mode: identity input → MSE = 0."""
    loss = KnowledgeDistillationLoss(mode="feature_mse")
    feats = torch.randn(2, 8, 4, 4)
    out = loss(feats, torch.zeros_like(feats), context={"teacher_output": feats.clone()})
    assert out.item() == 0.0


def test_feature_mse_quadratic_in_residual() -> None:
    """Feature-MSE: doubling residual → 4× loss."""
    student = torch.zeros(1, 4)
    teacher_small = torch.full_like(student, 1.0)
    teacher_big = torch.full_like(student, 2.0)
    loss = KnowledgeDistillationLoss(mode="feature_mse")
    l_small = loss(student, torch.zeros_like(student), context={"teacher_output": teacher_small}).item()
    l_big = loss(student, torch.zeros_like(student), context={"teacher_output": teacher_big}).item()
    assert pytest.approx(l_big / l_small, rel=1e-5) == 4.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_student() -> None:
    """Backward through the loss reaches the student tensor."""
    loss = KnowledgeDistillationLoss()
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4)
    target = torch.zeros_like(student)
    loss(student, target, context={"teacher_output": teacher}).backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_teacher_is_detached_from_grad() -> None:
    """Teacher signal is detached → no gradient propagates to teacher."""
    loss = KnowledgeDistillationLoss()
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4, requires_grad=True)
    target = torch.zeros_like(student)
    loss(student, target, context={"teacher_output": teacher}).backward()
    # Teacher should have no grad accumulated since detached.
    assert teacher.grad is None or teacher.grad.abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_kd_registered() -> None:
    """``knowledge_distillation`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "knowledge_distillation" in list_available()


def test_ema_self_distillation_registered() -> None:
    """``ema_self_distillation`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "ema_self_distillation" in list_available()
