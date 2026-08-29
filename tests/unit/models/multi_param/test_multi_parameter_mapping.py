r"""Tests for ``MultiParameterHead`` and ``UncertaintyWeightedMultiTaskLoss``.

Targets ``mriforge.models.multi_param.multi_parameter_mapping`` — idea 10.

Plan acceptance criteria (§10.6):

- *"Forward returns 4 outputs with correct shapes; gradient flows
  through all heads."* — covered by ``test_forward_shapes`` and
  ``test_gradient_flows_through_all_heads``.

Other contract tests:

- Construction: invalid in_channels / scale / num_seg_classes rejected
- Heads start zero-initialised → ``T1`` ≈ ``0.5 ln(2) · 1000`` ms
- Disabling segmentation head omits the ``segmentation`` key
- Uncertainty-weighted loss matches Kendall closed form
- Loss-of-zero-tasks rejected; missing-task case ignored
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.multi_param.multi_parameter_mapping import (
    MultiParameterHead,
    UncertaintyWeightedMultiTaskLoss,
)


# ---------------------------------------------------------------------------
# MultiParameterHead — construction
# ---------------------------------------------------------------------------


def test_invalid_in_channels_rejected() -> None:
    """``in_channels ≤ 0`` raises."""
    with pytest.raises(ValueError, match="in_channels"):
        MultiParameterHead(in_channels=0)


def test_negative_seg_classes_rejected() -> None:
    """``num_seg_classes < 0`` raises."""
    with pytest.raises(ValueError, match="num_seg_classes"):
        MultiParameterHead(in_channels=8, num_seg_classes=-1)


def test_non_positive_scales_rejected() -> None:
    """T1/T2 scales must be > 0."""
    with pytest.raises(ValueError, match="Scale"):
        MultiParameterHead(in_channels=8, t1_scale_ms=0.0)


# ---------------------------------------------------------------------------
# MultiParameterHead — forward shapes
# ---------------------------------------------------------------------------


def test_forward_shapes() -> None:
    """All four heads return the expected shapes."""
    head = MultiParameterHead(in_channels=16, num_seg_classes=3)
    feats = torch.randn(2, 16, 32, 32)
    out = head(feats)
    assert out["t1_ms"].shape == (2, 1, 32, 32)
    assert out["t2_ms"].shape == (2, 1, 32, 32)
    assert out["pd"].shape == (2, 1, 32, 32)
    assert out["segmentation"].shape == (2, 3, 32, 32)


def test_seg_head_disabled_when_zero_classes() -> None:
    """``num_seg_classes=0`` removes the segmentation head."""
    head = MultiParameterHead(in_channels=16, num_seg_classes=0)
    out = head(torch.randn(1, 16, 8, 8))
    assert "segmentation" not in out


def test_wrong_input_channels_rejected() -> None:
    """Calling with a non-matching channel count raises."""
    head = MultiParameterHead(in_channels=8)
    with pytest.raises(ValueError, match="features"):
        head(torch.randn(1, 7, 4, 4))


# ---------------------------------------------------------------------------
# MultiParameterHead — head outputs
# ---------------------------------------------------------------------------


def test_t1_head_outputs_positive() -> None:
    """T1 output is strictly positive (softplus)."""
    head = MultiParameterHead(in_channels=4, t1_scale_ms=1000.0)
    feats = torch.randn(1, 4, 8, 8)
    out = head(feats)
    assert (out["t1_ms"] > 0).all()


def test_t1_initial_output_at_softplus_zero() -> None:
    """Zero-init weights → softplus(0) ≈ 0.693, scaled by t1_scale_ms."""
    head = MultiParameterHead(in_channels=4, t1_scale_ms=1000.0)
    feats = torch.zeros(1, 4, 8, 8)
    out = head(feats)
    expected = math.log(2.0) * 1000.0
    assert torch.allclose(
        out["t1_ms"], torch.full_like(out["t1_ms"], expected), atol=1e-3
    )


def test_pd_head_outputs_positive() -> None:
    """PD output is strictly positive (softplus, no scale)."""
    head = MultiParameterHead(in_channels=4)
    out = head(torch.randn(1, 4, 8, 8))
    assert (out["pd"] > 0).all()


def test_segmentation_head_returns_logits() -> None:
    """Segmentation output is raw (signed) logits — softmax is the caller's job."""
    head = MultiParameterHead(in_channels=4, num_seg_classes=2)
    out = head(torch.randn(1, 4, 8, 8))
    seg = out["segmentation"]
    # Init is zero → logits are zero everywhere.
    assert torch.allclose(seg, torch.zeros_like(seg), atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow through all heads
# ---------------------------------------------------------------------------


def test_gradient_flows_through_all_heads() -> None:
    """Backward through each head reaches the input features."""
    for head_name in ("t1_ms", "t2_ms", "pd", "segmentation"):
        head = MultiParameterHead(in_channels=4, num_seg_classes=2)
        feats = torch.randn(1, 4, 8, 8, requires_grad=True)
        out = head(feats)
        out[head_name].sum().backward()
        assert feats.grad is not None, f"{head_name} produced no gradient"
        assert torch.isfinite(feats.grad).all()


# ---------------------------------------------------------------------------
# UncertaintyWeightedMultiTaskLoss
# ---------------------------------------------------------------------------


def test_loss_empty_task_names_rejected() -> None:
    """Zero tasks rejected at construction."""
    with pytest.raises(ValueError, match="task_names"):
        UncertaintyWeightedMultiTaskLoss(task_names=[])


def test_loss_empty_losses_dict_rejected() -> None:
    """Empty per-task loss dict rejected."""
    loss = UncertaintyWeightedMultiTaskLoss(task_names=["t1"])
    with pytest.raises(ValueError, match="empty"):
        loss({})


def test_loss_kendall_closed_form_at_zero_log_var() -> None:
    r"""When σ = 0, the loss is ``0.5 · L_1 + 0.5 · L_2 + ... = 0.5 · sum``."""
    loss = UncertaintyWeightedMultiTaskLoss(
        task_names=["t1", "t2"], init_log_var=0.0
    )
    per_task = {
        "t1": torch.tensor(2.0),
        "t2": torch.tensor(4.0),
    }
    total = loss(per_task).item()
    # 0.5 · e^0 · 2 + 0.5 · 0 + 0.5 · e^0 · 4 + 0.5 · 0 = 1 + 2 = 3
    assert pytest.approx(total) == 3.0


def test_loss_grows_with_per_task_loss() -> None:
    """Doubling a task's loss increases the total."""
    loss = UncertaintyWeightedMultiTaskLoss(task_names=["t1"])
    small = loss({"t1": torch.tensor(1.0)}).item()
    big = loss({"t1": torch.tensor(10.0)}).item()
    assert big > small


def test_loss_log_var_parameter_learnable() -> None:
    """The per-task ``log_var`` parameters are real ``nn.Parameter`` objects."""
    loss = UncertaintyWeightedMultiTaskLoss(task_names=["t1", "t2"])
    for name in ("t1", "t2"):
        assert isinstance(loss.log_vars[name], torch.nn.Parameter)
        assert loss.log_vars[name].requires_grad


def test_loss_unknown_task_silently_skipped() -> None:
    """A loss key not in ``task_names`` is ignored without raising."""
    loss = UncertaintyWeightedMultiTaskLoss(task_names=["t1"])
    out = loss({"t1": torch.tensor(1.0), "unrelated": torch.tensor(99.0)}).item()
    expected = 0.5 * 1.0 + 0.0  # only t1 counted
    assert pytest.approx(out) == expected
