r"""Tests for ``SensorVFAnchorLoss`` (idea 9 §9.3).

Targets ``spectramr.models.losses.sensor_vf_anchor_loss``.

Categories:

- Construction: weight / eps validation
- Identity input → loss = 0
- Shape mismatches raise
- Mahalanobis path: covariance ``Σ = I`` reduces to ``mean(diff²) · 6``
  (sum across components / batch); a tighter variance increases the loss
- Without covariance, the loss collapses to plain MSE
- VF target is detached — no gradient flows back into ``vf_pose``
- Gradient flows back to ``sensor_pose``
- Registry: ``sensor_vf_anchor`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.sensor_vf_anchor_loss import SensorVFAnchorLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_weight_rejected() -> None:
    """``weight < 0`` raises."""
    with pytest.raises(ValueError, match="weight"):
        SensorVFAnchorLoss(weight=-1.0)


def test_non_positive_eps_rejected() -> None:
    """``eps ≤ 0`` raises."""
    with pytest.raises(ValueError, match="eps"):
        SensorVFAnchorLoss(eps=0.0)


# ---------------------------------------------------------------------------
# MSE branch (no covariance)
# ---------------------------------------------------------------------------


def test_zero_loss_for_identity_inputs_mse_branch() -> None:
    """``L(p, p) = 0`` without covariance."""
    pose = torch.randn(2, 6)
    loss = SensorVFAnchorLoss(weight=1.0)
    assert pytest.approx(loss(pose, pose).item(), abs=1e-6) == 0.0


def test_mse_branch_matches_mean_squared_diff() -> None:
    """Without covariance, the loss is ``λ · mean((sensor - vf)²)``."""
    sensor = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    vf = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    expected = (1.0 ** 2 + 0.0 * 5) / 6.0  # = 1/6
    loss = SensorVFAnchorLoss(weight=1.0)
    assert pytest.approx(loss(sensor, vf).item()) == expected


def test_weight_scales_loss_linearly() -> None:
    """Doubling ``weight`` doubles the output."""
    sensor = torch.randn(2, 6)
    vf = torch.zeros_like(sensor)
    val_1 = SensorVFAnchorLoss(weight=1.0)(sensor, vf).item()
    val_3 = SensorVFAnchorLoss(weight=3.0)(sensor, vf).item()
    assert pytest.approx(val_3) == 3.0 * val_1


# ---------------------------------------------------------------------------
# Mahalanobis branch
# ---------------------------------------------------------------------------


def test_mahalanobis_branch_with_identity_covariance() -> None:
    r"""``Σ = I`` → Mahalanobis = ``mean(Σ_b sum_i (diff_b_i)²)``."""
    sensor = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    vf = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    cov = torch.eye(6).unsqueeze(0)
    loss = SensorVFAnchorLoss(weight=1.0)
    val = loss(sensor, vf, cov).item()
    # diff² = [1,1,0,0,0,0]; sum = 2; per-sample mean of that sum = 2.
    assert pytest.approx(val) == 2.0


def test_tighter_variance_raises_loss() -> None:
    """Halving the variance doubles the Mahalanobis distance."""
    sensor = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    vf = torch.zeros(1, 6)
    cov_loose = torch.eye(6).unsqueeze(0) * 1.0
    cov_tight = torch.eye(6).unsqueeze(0) * 0.5
    loss = SensorVFAnchorLoss(weight=1.0)
    val_loose = loss(sensor, vf, cov_loose).item()
    val_tight = loss(sensor, vf, cov_tight).item()
    # eps-regularised inv(0.5·I + 1e-6·I) drifts ~1e-6 from a clean 2.0;
    # default approx tolerance (rel=1e-6) is too tight for that.
    assert val_tight == pytest.approx(2.0 * val_loose, rel=1e-5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_sensor_pose_wrong_shape_rejected() -> None:
    """``sensor_pose`` must be ``[B, 6]``."""
    with pytest.raises(ValueError, match="sensor_pose"):
        SensorVFAnchorLoss()(torch.zeros(2, 7), torch.zeros(2, 7))


def test_pose_shape_mismatch_rejected() -> None:
    """Mismatched batch sizes raise."""
    with pytest.raises(ValueError, match="vf_pose"):
        SensorVFAnchorLoss()(torch.zeros(2, 6), torch.zeros(3, 6))


def test_covariance_wrong_shape_rejected() -> None:
    """Covariance must be ``[B, 6, 6]``."""
    with pytest.raises(ValueError, match="covariance"):
        SensorVFAnchorLoss()(
            torch.zeros(2, 6),
            torch.zeros(2, 6),
            covariance=torch.eye(4).unsqueeze(0).expand(2, 4, 4),
        )


def test_covariance_batch_size_mismatch_rejected() -> None:
    """Covariance batch size must match pose batch size."""
    with pytest.raises(ValueError, match="Batch-size"):
        SensorVFAnchorLoss()(
            torch.zeros(3, 6),
            torch.zeros(3, 6),
            covariance=torch.eye(6).unsqueeze(0).expand(2, 6, 6),
        )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradient_flows_to_sensor_pose() -> None:
    """Backward through the loss reaches the sensor-pose tensor."""
    sensor = torch.randn(2, 6, requires_grad=True)
    vf = torch.zeros_like(sensor)
    SensorVFAnchorLoss()(sensor, vf).backward()
    assert sensor.grad is not None
    assert torch.isfinite(sensor.grad).all()


def test_no_gradient_flows_to_vf_pose() -> None:
    """``vf_pose`` is detached — backward never reaches it."""
    # ``sensor`` must require grad too, otherwise the loss has no grad path
    # at all and ``.backward()`` raises before we can check ``vf.grad``.
    sensor = torch.randn(2, 6, requires_grad=True)
    vf = torch.randn(2, 6, requires_grad=True)
    SensorVFAnchorLoss()(sensor, vf).backward()
    # vf had requires_grad=True but the loss detaches it inside.
    assert vf.grad is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``sensor_vf_anchor`` resolvable in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "sensor_vf_anchor" in list_available()
