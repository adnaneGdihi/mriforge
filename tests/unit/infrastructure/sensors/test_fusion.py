r"""Tests for ``cramer_rao_optimal_fusion``.

Targets ``spectramr.infrastructure.sensors.fusion`` — idea 9 §9.3.

Plan acceptance criterion (§9.7):

- *"Fused pose variance ≤ minimum of any single-modality variance
  (Cramér-Rao optimality)"* — the headline test below.

Other contract tests:

- ``PoseEstimate`` rejects malformed pose / covariance shapes
- ``inverse_variance_weights`` sums to 1
- Single-modality input returns the input pose (idempotent)
- Identical estimates with identical covariances → same pose, smaller variance
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.sensors.fusion import (
    FusedPose,
    PoseEstimate,
    cramer_rao_optimal_fusion,
    inverse_variance_weights,
)


# ---------------------------------------------------------------------------
# PoseEstimate validation
# ---------------------------------------------------------------------------


def test_pose_estimate_rejects_wrong_pose_shape() -> None:
    """Pose must be a 6-vector."""
    with pytest.raises(ValueError, match="pose"):
        PoseEstimate(
            pose=torch.zeros(7),
            covariance=torch.eye(6),
            modality="rgbd",
        )


def test_pose_estimate_rejects_wrong_covariance_shape() -> None:
    """Covariance must be 6×6."""
    with pytest.raises(ValueError, match="covariance"):
        PoseEstimate(
            pose=torch.zeros(6),
            covariance=torch.eye(4),
            modality="rgbd",
        )


# ---------------------------------------------------------------------------
# inverse_variance_weights
# ---------------------------------------------------------------------------


def test_inverse_variance_weights_sum_to_one() -> None:
    """Diagnostic weights are normalised to sum to 1."""
    estimates = [
        PoseEstimate(torch.zeros(6), torch.eye(6) * 0.5, "rgbd"),
        PoseEstimate(torch.zeros(6), torch.eye(6) * 2.0, "imu"),
    ]
    w = inverse_variance_weights(estimates)
    assert pytest.approx(w.sum().item()) == 1.0


def test_inverse_variance_weights_higher_for_lower_variance() -> None:
    """The lower-variance modality gets the bigger weight."""
    estimates = [
        PoseEstimate(torch.zeros(6), torch.eye(6) * 0.5, "rgbd"),  # low var
        PoseEstimate(torch.zeros(6), torch.eye(6) * 2.0, "imu"),   # high var
    ]
    w = inverse_variance_weights(estimates)
    assert w[0].item() > w[1].item()


def test_inverse_variance_weights_empty_list_raises() -> None:
    """Empty input rejected."""
    with pytest.raises(ValueError, match="empty"):
        inverse_variance_weights([])


def test_inverse_variance_weights_singular_covariance_does_not_crash() -> None:
    """A zero-variance (singular) sensor must not crash the weights path.

    Regression for the audit fix: ``inverse_variance_weights`` used to invert
    the raw covariance with no Tikhonov regulariser, so a perfectly-confident
    sensor (singular covariance) raised ``LinAlgError`` even though
    :func:`cramer_rao_optimal_fusion` (which *does* regularise) handled it.
    Both paths now share the same ``eps`` regularisation.
    """
    estimates = [
        PoseEstimate(torch.zeros(6), torch.zeros(6, 6), "rgbd"),  # singular
        PoseEstimate(torch.zeros(6), torch.eye(6), "imu"),
    ]
    w = inverse_variance_weights(estimates)
    assert pytest.approx(w.sum().item()) == 1.0
    # The infinitely-confident sensor dominates the (information) weight.
    assert w[0].item() > w[1].item()


def test_fusion_weights_match_standalone_diagnostic() -> None:
    """Weights returned by fusion equal the standalone diagnostic weights.

    Regression for the audit fix: fusion now reuses the regularised
    information matrices it already builds rather than re-calling (and
    re-inverting via) ``inverse_variance_weights`` — the two must still agree.
    """
    estimates = [
        PoseEstimate(torch.zeros(6), torch.eye(6) * 0.5, "rgbd"),
        PoseEstimate(torch.zeros(6), torch.eye(6) * 2.0, "imu"),
    ]
    fused = cramer_rao_optimal_fusion(estimates)
    standalone = inverse_variance_weights(estimates)
    assert torch.allclose(fused.weights, standalone, atol=1e-6)


def test_fusion_tolerates_singular_covariance() -> None:
    """Fusion with a singular covariance succeeds end-to-end (no crash)."""
    estimates = [
        PoseEstimate(torch.tensor([5.0, 0, 0, 0, 0, 0]), torch.zeros(6, 6), "rgbd"),
        PoseEstimate(torch.zeros(6), torch.eye(6), "imu"),
    ]
    out = cramer_rao_optimal_fusion(estimates)
    assert out.pose.shape == (6,)
    assert pytest.approx(out.weights.sum().item()) == 1.0


# ---------------------------------------------------------------------------
# Single-modality fusion is the identity
# ---------------------------------------------------------------------------


def test_single_modality_returns_input_pose() -> None:
    """One sensor in → fused pose ≈ input pose."""
    pose = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    cov = torch.eye(6) * 0.01
    out = cramer_rao_optimal_fusion(
        [PoseEstimate(pose, cov, "rgbd")]
    )
    assert torch.allclose(out.pose, pose, atol=1e-4)


# ---------------------------------------------------------------------------
# Cramér-Rao optimality (the headline acceptance criterion)
# ---------------------------------------------------------------------------


def test_fused_variance_no_larger_than_minimum_individual() -> None:
    """Fused diagonal variance ≤ min per-modality diagonal variance.

    This is the formal Cramér-Rao optimality the proposal cites.
    """
    var_a = 0.5
    var_b = 2.0
    estimates = [
        PoseEstimate(torch.zeros(6), torch.eye(6) * var_a, "rgbd"),
        PoseEstimate(torch.zeros(6), torch.eye(6) * var_b, "imu"),
    ]
    out = cramer_rao_optimal_fusion(estimates)
    fused_diag = out.covariance.diagonal()
    min_individual_var = min(var_a, var_b)
    assert (fused_diag <= min_individual_var + 1e-6).all(), (
        f"Fused diag {fused_diag} exceeds min individual var {min_individual_var}"
    )


def test_fused_variance_strictly_smaller_for_independent_sensors() -> None:
    """When two independent sensors disagree on no axis, fusion shrinks variance."""
    estimates = [
        PoseEstimate(torch.zeros(6), torch.eye(6), "rgbd"),
        PoseEstimate(torch.zeros(6), torch.eye(6), "imu"),
    ]
    out = cramer_rao_optimal_fusion(estimates)
    # Two sensors with var=1 → fused var=0.5 (closed form).
    assert torch.allclose(out.covariance.diagonal(), torch.full((6,), 0.5))


def test_fused_pose_lies_between_inputs_for_equal_variance() -> None:
    """Equal-variance fusion is the simple mean of the per-sensor poses."""
    pose_a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_b = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    estimates = [
        PoseEstimate(pose_a, torch.eye(6), "rgbd"),
        PoseEstimate(pose_b, torch.eye(6), "imu"),
    ]
    out = cramer_rao_optimal_fusion(estimates)
    assert pytest.approx(out.pose[0].item(), abs=1e-5) == 2.0


def test_fused_pose_skews_toward_low_variance_sensor() -> None:
    """Lower-variance sensor pulls the fused estimate."""
    pose_low = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_high = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    estimates = [
        PoseEstimate(pose_low, torch.eye(6) * 0.01, "rgbd"),
        PoseEstimate(pose_high, torch.eye(6) * 1.00, "imu"),
    ]
    out = cramer_rao_optimal_fusion(estimates)
    # Should be much closer to the low-variance estimate.
    assert out.pose[0].item() > 9.0


# ---------------------------------------------------------------------------
# Empty / degenerate
# ---------------------------------------------------------------------------


def test_empty_estimates_rejected() -> None:
    """Fusion with no estimates raises."""
    with pytest.raises(ValueError, match="empty"):
        cramer_rao_optimal_fusion([])


def test_returns_fused_pose_dataclass() -> None:
    """Return type is :class:`FusedPose`."""
    out = cramer_rao_optimal_fusion(
        [PoseEstimate(torch.zeros(6), torch.eye(6), "rgbd")]
    )
    assert isinstance(out, FusedPose)
    assert out.pose.shape == (6,)
    assert out.covariance.shape == (6, 6)
    assert out.weights.shape == (1,)
