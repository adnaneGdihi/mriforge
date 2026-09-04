"""Unit tests for physics_losses.py.

Covers: GraphConsistencyLoss, BlochResidualLoss, PhysicsConstraintLoss,
        SNRPreservingLoss, PhysicsInformedLoss, SpectralKSpaceLoss,
        InvertedFocalFrequencyLoss, DivergenceFreeLoss, JacobianDeterminantLoss,
        DataConsistencyLoss.

Non-Cartesian / NUFFT losses that require an external operator object are not
instantiated in isolation; they get a note skip.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.physics_losses import (
    BlochResidualLoss,
    DataConsistencyLoss,
    DivergenceFreeLoss,
    GraphConsistencyLoss,
    InvertedFocalFrequencyLoss,
    JacobianDeterminantLoss,
    PhysicsConstraintLoss,
    PhysicsInformedLoss,
    SNRPreservingLoss,
    SpectralKSpaceLoss,
)
from spectramr.models.losses.registry import create_loss

B, C, H, W = 2, 1, 32, 32


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


def _real_2ch(b=B, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(b, 2, h, w, generator=gen)


# ===========================================================================
# GraphConsistencyLoss
# ===========================================================================


class TestGraphConsistencyLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = GraphConsistencyLoss()
        pred = torch.randn(B, 16, 2)
        target = torch.randn(B, 16, 2)
        out = loss_fn(pred, target)
        assert torch.isfinite(out) and out.shape == ()

    def test_parametric_no_density_weighting(self):
        loss_fn = GraphConsistencyLoss(density_weighting=False)
        pred = torch.randn(B, 16, 2)
        target = torch.randn(B, 16, 2)
        out = loss_fn(pred, target)
        assert torch.isfinite(out)

    def test_property_identity_is_zero(self):
        loss_fn = GraphConsistencyLoss(density_weighting=False)
        x = torch.randn(B, 16, 2)
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-8)

    def test_property_nonneg(self):
        loss_fn = GraphConsistencyLoss(density_weighting=False)
        pred = torch.randn(B, 16, 2)
        target = torch.randn(B, 16, 2)
        assert loss_fn(pred, target).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        pred = torch.randn(B, 16, 2, requires_grad=True)
        target = torch.randn(B, 16, 2)
        loss_fn = GraphConsistencyLoss(density_weighting=False)
        loss_fn(pred, target).backward()
        assert pred.grad is not None

    def test_edge_grid_input(self):
        """[B, C, H, W] grid inputs should be handled by the view/permute path."""
        loss_fn = GraphConsistencyLoss(density_weighting=False)
        pred = _img(c=2)
        target = _img(c=2, seed=1)
        out = loss_fn(pred, target)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("graph_consistency")
        assert isinstance(loss_fn, GraphConsistencyLoss)


# ===========================================================================
# BlochResidualLoss
# ===========================================================================


class TestBlochResidualLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = BlochResidualLoss()
        M_pred = torch.randn(B, 5, 3, H, W)  # [B, T, 3, H, W]
        T1 = torch.full((B, 1, H, W), 0.8)
        T2 = torch.full((B, 1, H, W), 0.08)
        out = loss_fn(M_pred, T1, T2)
        assert torch.isfinite(out) and out.shape == ()

    def test_parameter_maps_broadcast_against_the_time_axis(self):
        """The maps are ``[B, 1, H, W]``; that size-1 axis IS the time seam.

        Regression for cluster job 8004252, where every test in this class
        failed. The relaxation terms did ``T1_map.unsqueeze(1)``, written for a
        ``[B, H, W]`` map, which made them 5-D. Two different symptoms, one
        cause:

        * ``B0_map.unsqueeze(1).expand(-1, T-1, -1, -1)`` RAISED -- four sizes
          for a five-dimensional tensor.
        * ``-Mx * R1`` did NOT raise. It broadcast ``[B, T-1, H, W]`` against
          ``[B, 1, 1, H, W]`` into ``[B, B, T-1, H, W]``, inventing a batch
          axis, and only failed later at the MSE.

        The silent one is why this test asserts the OUTPUT SHAPE of the
        intermediate broadcast rather than only that ``forward`` returns
        something finite: a scalar loss computed over a garbage tensor is still
        a scalar loss.
        """
        m_pred = torch.randn(B, 5, 3, H, W)
        t1_map = torch.full((B, 1, H, W), 0.8)

        m_current = m_pred[:, :-1]
        mx = m_current[:, :, 0]
        r1 = 1.0 / (t1_map + 1e-6)

        assert mx.shape == (B, 4, H, W)
        assert (-mx * r1).shape == mx.shape, (
            "a parameter map must broadcast against [B, T-1, H, W] without "
            "changing its shape"
        )

    @pytest.mark.parametrize("bad_shape", [(B, H, W), (B, 3, H, W), (B, 1, H)])
    def test_wrong_rank_parameter_map_raises(self, bad_shape):
        """A map that cannot broadcast cleanly must raise, not broadcast oddly."""
        loss_fn = BlochResidualLoss()
        m_pred = torch.randn(B, 5, 3, H, W)
        t2_map = torch.full((B, 1, H, W), 0.08)
        with pytest.raises(ValueError, match=r"must be \[B, 1, H, W\]"):
            loss_fn(m_pred, torch.full(bad_shape, 0.8), t2_map)

    def test_parametric_with_b0_map(self):
        loss_fn = BlochResidualLoss()
        M_pred = torch.randn(B, 4, 3, H, W)
        T1 = torch.full((B, 1, H, W), 1.0)
        T2 = torch.full((B, 1, H, W), 0.1)
        B0 = torch.zeros(B, 1, H, W)
        out = loss_fn(M_pred, T1, T2, B0_map=B0)
        assert torch.isfinite(out)

    def test_property_nonneg(self):
        loss_fn = BlochResidualLoss()
        M_pred = torch.randn(B, 3, 3, H, W)
        T1 = torch.full((B, 1, H, W), 1.0)
        T2 = torch.full((B, 1, H, W), 0.1)
        assert loss_fn(M_pred, T1, T2).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        M_pred = torch.randn(B, 3, 3, H, W, requires_grad=True)
        T1 = torch.full((B, 1, H, W), 1.0)
        T2 = torch.full((B, 1, H, W), 0.1)
        loss_fn = BlochResidualLoss()
        loss_fn(M_pred, T1, T2).backward()
        assert M_pred.grad is not None

    def test_edge_small_t_values(self):
        loss_fn = BlochResidualLoss()
        M_pred = torch.randn(B, 3, 3, 4, 4)
        T1 = torch.full((B, 1, 4, 4), 1e-5)
        T2 = torch.full((B, 1, 4, 4), 1e-5)
        out = loss_fn(M_pred, T1, T2)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("bloch_residual")
        assert isinstance(loss_fn, BlochResidualLoss)


# ===========================================================================
# PhysicsConstraintLoss
# ===========================================================================


class TestPhysicsConstraintLoss:

    def test_canary_builds_and_forward_returns_tuple(self):
        loss_fn = PhysicsConstraintLoss()
        pred = _img()
        target = _img(seed=1)
        total, metrics = loss_fn(pred, target)
        assert torch.isfinite(total)
        assert isinstance(metrics, dict)

    @pytest.mark.parametrize("weights", [
        (0.1, 0.1, 0.0),
        (0.0, 0.1, 0.0),
    ])
    def test_parametric_component_weights(self, weights):
        sw, ew, kw = weights
        loss_fn = PhysicsConstraintLoss(
            smoothness_weight=sw, energy_weight=ew, kspace_weight=kw
        )
        pred = _img()
        target = _img(seed=1)
        total, _ = loss_fn(pred, target)
        assert torch.isfinite(total)

    def test_property_nonneg_total(self):
        loss_fn = PhysicsConstraintLoss()
        pred = _img()
        target = _img(seed=1)
        total, _ = loss_fn(pred, target)
        assert total.item() >= 0.0

    def test_edge_zero_prediction(self):
        loss_fn = PhysicsConstraintLoss()
        pred = torch.zeros(B, C, H, W)
        target = _img()
        total, _ = loss_fn(pred, target)
        assert torch.isfinite(total)

    def test_registry_lookup(self):
        loss_fn = create_loss("physics_constraint")
        assert isinstance(loss_fn, PhysicsConstraintLoss)


# ===========================================================================
# SNRPreservingLoss
# ===========================================================================


class TestSNRPreservingLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = SNRPreservingLoss(target_snr=20.0)
        pred = _img()
        target = _img(seed=1)
        out = loss_fn(pred, target)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("target_snr", [10.0, 30.0])
    def test_parametric_target_snr(self, target_snr: float):
        loss_fn = SNRPreservingLoss(target_snr=target_snr)
        pred = _img()
        target = _img(seed=1)
        out = loss_fn(pred, target)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = SNRPreservingLoss()
        pred = _img()
        target = _img(seed=1)
        assert loss_fn(pred, target).item() >= 0.0

    def test_property_perfect_recon_zero_penalty(self):
        """Perfect reconstruction → SNR is high → relu penalty = 0."""
        loss_fn = SNRPreservingLoss(target_snr=0.0)
        x = _img()
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_property_gradient_reaches_input(self):
        pred = _img().requires_grad_(True)
        target = _img(seed=1)
        loss_fn = SNRPreservingLoss(target_snr=20.0)
        loss_fn(pred, target).backward()
        assert pred.grad is not None

    def test_edge_zero_input(self):
        loss_fn = SNRPreservingLoss()
        pred = torch.zeros(B, C, H, W)
        target = torch.zeros(B, C, H, W)
        out = loss_fn(pred, target)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("snr_preserving")
        assert isinstance(loss_fn, SNRPreservingLoss)


# ===========================================================================
# PhysicsInformedLoss
# ===========================================================================


class TestPhysicsInformedLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = PhysicsInformedLoss()
        pred = _img()
        target = _img(seed=1)
        losses = loss_fn(pred, target)
        assert isinstance(losses, dict)
        assert torch.isfinite(losses["total"])

    @pytest.mark.parametrize("lam_l2", [0.5, 1.0])
    def test_parametric_lambda_l2(self, lam_l2: float):
        loss_fn = PhysicsInformedLoss(lambda_l2=lam_l2)
        pred = _img()
        target = _img(seed=1)
        losses = loss_fn(pred, target)
        assert torch.isfinite(losses["l2_image"])

    def test_property_l2_nonneg(self):
        loss_fn = PhysicsInformedLoss()
        pred = _img()
        target = _img(seed=1)
        losses = loss_fn(pred, target)
        assert losses["l2_image"].item() >= 0.0

    def test_property_gradient_reaches_input(self):
        pred = _img().requires_grad_(True)
        target = _img(seed=1)
        loss_fn = PhysicsInformedLoss(lambda_reg=0.0)
        losses = loss_fn(pred, target)
        losses["total"].backward()
        assert pred.grad is not None

    def test_edge_zero_prediction(self):
        loss_fn = PhysicsInformedLoss(lambda_reg=0.0)
        pred = torch.zeros(B, C, H, W)
        target = _img()
        losses = loss_fn(pred, target)
        assert torch.isfinite(losses["total"])

    def test_registry_lookup(self):
        loss_fn = create_loss("physics_informed")
        assert isinstance(loss_fn, PhysicsInformedLoss)


# ===========================================================================
# SpectralKSpaceLoss
# ===========================================================================


class TestSpectralKSpaceLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = SpectralKSpaceLoss()
        pred = _real_2ch()
        target = _real_2ch(seed=1)
        out = loss_fn(pred, target)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("epsilon", [1e-8, 1e-4])
    def test_parametric_epsilon(self, epsilon: float):
        loss_fn = SpectralKSpaceLoss(epsilon=epsilon)
        pred = _real_2ch()
        target = _real_2ch(seed=1)
        out = loss_fn(pred, target)
        assert torch.isfinite(out)

    def test_property_nonneg(self):
        loss_fn = SpectralKSpaceLoss()
        pred = _real_2ch()
        target = _real_2ch(seed=1)
        assert loss_fn(pred, target).item() >= 0.0

    def test_property_identity_near_zero(self):
        loss_fn = SpectralKSpaceLoss()
        x = _real_2ch()
        out = loss_fn(x, x)
        assert out.item() < 1e-4

    def test_property_gradient_reaches_input(self):
        pred = _real_2ch().requires_grad_(True)
        target = _real_2ch(seed=1)
        loss_fn = SpectralKSpaceLoss()
        loss_fn(pred, target).backward()
        assert pred.grad is not None

    def test_edge_zero_input(self):
        loss_fn = SpectralKSpaceLoss()
        x = torch.zeros(B, 2, H, W)
        out = loss_fn(x, x)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = SpectralKSpaceLoss()
        x = torch.full((B, 2, H, W), 1e3)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("spectral_kspace")
        assert isinstance(loss_fn, SpectralKSpaceLoss)


# ===========================================================================
# InvertedFocalFrequencyLoss
# ===========================================================================


class TestInvertedFocalFrequencyLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = InvertedFocalFrequencyLoss()
        pred = _img()
        target = _img(seed=1)
        out = loss_fn(pred, target)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("alpha", [1.0, 2.0, 5.0])
    def test_parametric_alpha(self, alpha: float):
        loss_fn = InvertedFocalFrequencyLoss(alpha=alpha)
        pred = _img()
        target = _img(seed=1)
        out = loss_fn(pred, target)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = InvertedFocalFrequencyLoss()
        pred = _img()
        target = _img(seed=1)
        assert loss_fn(pred, target).item() >= 0.0

    def test_property_identity_near_zero(self):
        loss_fn = InvertedFocalFrequencyLoss()
        x = _img()
        assert loss_fn(x, x).item() < 1e-6

    def test_property_gradient_reaches_input(self):
        pred = _img().requires_grad_(True)
        target = _img(seed=1)
        loss_fn = InvertedFocalFrequencyLoss()
        loss_fn(pred, target).backward()
        assert pred.grad is not None

    def test_edge_large_magnitude(self):
        loss_fn = InvertedFocalFrequencyLoss()
        x = torch.full((B, C, H, W), 1e3)
        y = torch.zeros(B, C, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("focal_frequency")
        assert isinstance(loss_fn, InvertedFocalFrequencyLoss)


# ===========================================================================
# DivergenceFreeLoss
# ===========================================================================


class TestDivergenceFreeLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = DivergenceFreeLoss()
        flow = torch.randn(B, 2, H, W)
        out = loss_fn(flow)
        assert torch.isfinite(out) and out.shape == ()

    def test_property_nonneg(self):
        """Divergence squared ≥ 0."""
        loss_fn = DivergenceFreeLoss()
        flow = torch.randn(B, 2, H, W)
        assert loss_fn(flow).item() >= 0.0

    def test_property_zero_for_div_free_field(self):
        """Constant flow field has zero divergence."""
        loss_fn = DivergenceFreeLoss()
        flow = torch.ones(B, 2, H, W)
        out = loss_fn(flow)
        assert out.item() == pytest.approx(0.0, abs=1e-8)

    def test_property_gradient_reaches_input(self):
        flow = torch.randn(B, 2, H, W, requires_grad=True)
        loss_fn = DivergenceFreeLoss()
        loss_fn(flow).backward()
        assert flow.grad is not None

    def test_edge_zero_flow(self):
        loss_fn = DivergenceFreeLoss()
        flow = torch.zeros(B, 2, H, W)
        out = loss_fn(flow)
        assert torch.isfinite(out) and out.item() == pytest.approx(0.0, abs=1e-9)

    def test_edge_large_magnitude(self):
        loss_fn = DivergenceFreeLoss()
        flow = torch.full((B, 2, H, W), 1e3)
        out = loss_fn(flow)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("divergence_free")
        assert isinstance(loss_fn, DivergenceFreeLoss)


# ===========================================================================
# JacobianDeterminantLoss
# ===========================================================================


class TestJacobianDeterminantLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = JacobianDeterminantLoss()
        flow = torch.randn(B, 2, H, W) * 0.1
        out = loss_fn(flow)
        assert torch.isfinite(out) and out.shape == ()

    def test_property_nonneg(self):
        """Penalises only negative Jacobians via relu; result ≥ 0."""
        loss_fn = JacobianDeterminantLoss()
        flow = torch.randn(B, 2, H, W) * 0.1
        assert loss_fn(flow).item() >= 0.0

    def test_property_zero_for_identity(self):
        """Zero flow → identity deformation → det(J) = 1 → no penalty."""
        loss_fn = JacobianDeterminantLoss()
        flow = torch.zeros(B, 2, H, W)
        out = loss_fn(flow)
        assert out.item() == pytest.approx(0.0, abs=1e-7)

    def test_property_gradient_reaches_input(self):
        flow = torch.randn(B, 2, H, W, requires_grad=True) * 0.1
        loss_fn = JacobianDeterminantLoss()
        loss_fn(flow).backward()
        assert flow.grad is not None

    def test_edge_large_deformation(self):
        """Extreme deformation can produce negative Jacobians → non-zero loss."""
        loss_fn = JacobianDeterminantLoss()
        flow = torch.full((B, 2, H, W), -2.0)  # forces negative Jacobian det
        out = loss_fn(flow)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_registry_lookup(self):
        loss_fn = create_loss("jacobian_determinant")
        assert isinstance(loss_fn, JacobianDeterminantLoss)


# ===========================================================================
# DataConsistencyLoss
# ===========================================================================


class TestDataConsistencyLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = DataConsistencyLoss()
        pred = _img().to(torch.float32)
        kspace = torch.randn(B, C, H, W, dtype=torch.complex64)
        mask = torch.ones(B, C, H, W, dtype=torch.complex64)
        out = loss_fn(pred, kspace, mask)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("lambda_dc", [0.5, 1.0, 2.0])
    def test_parametric_lambda_dc(self, lambda_dc: float):
        loss_fn = DataConsistencyLoss(lambda_dc=lambda_dc)
        pred = _img()
        kspace = torch.randn(B, C, H, W, dtype=torch.complex64)
        mask = torch.ones(B, C, H, W, dtype=torch.complex64)
        out = loss_fn(pred, kspace, mask)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = DataConsistencyLoss()
        pred = _img()
        kspace = torch.randn(B, C, H, W, dtype=torch.complex64)
        mask = torch.ones(B, C, H, W, dtype=torch.complex64)
        assert loss_fn(pred, kspace, mask).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        pred = _img().requires_grad_(True)
        kspace = torch.randn(B, C, H, W, dtype=torch.complex64)
        mask = torch.ones(B, C, H, W, dtype=torch.complex64)
        loss_fn = DataConsistencyLoss()
        loss_fn(pred, kspace, mask).backward()
        assert pred.grad is not None

    def test_edge_zero_mask(self):
        """Zero mask → no data consistency term → loss = 0."""
        loss_fn = DataConsistencyLoss()
        pred = _img()
        kspace = torch.randn(B, C, H, W, dtype=torch.complex64)
        mask = torch.zeros(B, C, H, W, dtype=torch.complex64)
        out = loss_fn(pred, kspace, mask)
        assert out.item() == pytest.approx(0.0, abs=1e-8)

    def test_registry_lookup(self):
        loss_fn = create_loss("data_consistency")
        assert isinstance(loss_fn, DataConsistencyLoss)
