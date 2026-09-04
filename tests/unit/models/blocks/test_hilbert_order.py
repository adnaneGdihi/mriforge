"""Unit tests for HilbertOrder API + DTN2S J-invariance mask + SFC-windowed
attention + the four marker-research utilities.

Pin the contracts the IMPLEMENTATION_SPEC.md depends on:
* Bijection of the SFC permutation
* Round-trip linearize → unflatten = identity
* Locality preservation (Hilbert step length ≤ √2 in 2-D)
* J-invariance mask is deterministic and budget-correct
* SFCWindowedAttention preserves shape and is masking-aware
* Fisher info matches autograd Jacobian on a linear forward
* Conformal coverage rate matches nominal level on synthetic data
* Through-plane motion recovery returns the planted Δz within tolerance
"""
from __future__ import annotations

import numpy as np
import torch

from spectramr.models.blocks.dtn2s_mask import build_dtn2s_mask, dual_traversal_pair
from spectramr.models.blocks.hilbert_order import HilbertOrder
from spectramr.models.blocks.sfc_windowed_attention import (
    CrossDomainBlock,
    SFCWindowedAttention,
)
from spectramr.infrastructure.physics.marker_estimators import (
    crlb_from_fisher,
    conformal_coverage_check,
    fisher_information_motion,
    greedy_marker_placement,
    privileged_learning_loss,
    profile_likelihood_motion,
    split_conformal_marker_uq,
    through_plane_motion_recovery,
)


# ─────────────────────────────────────────────────────────────────────
# HilbertOrder
# ─────────────────────────────────────────────────────────────────────


class TestHilbertOrder:
    def test_2d_bijection(self) -> None:
        h = HilbertOrder(order=4, n_dims=2)   # 16x16
        assert h.permutation.numel() == 256
        # bijection
        assert torch.equal(
            torch.sort(h.permutation).values, torch.arange(256)
        )
        # inverse is the inverse
        assert torch.equal(
            h.permutation[h.inverse], torch.arange(256)
        )

    def test_3d_bijection(self) -> None:
        h = HilbertOrder(order=3, n_dims=3)   # 8x8x8 = 512
        assert h.permutation.numel() == 512
        assert torch.equal(h.permutation[h.inverse], torch.arange(512))

    def test_linearize_unflatten_identity(self) -> None:
        h = HilbertOrder(order=4, n_dims=2)
        x = torch.randn(2, 3, 16, 16)
        seq = h.linearize(x)
        assert seq.shape == (2, 3, 256)
        x_back = h.unflatten(seq, (16, 16))
        assert torch.allclose(x, x_back)

    def test_locality_step_length(self) -> None:
        # Hilbert: every step in 2-D moves between neighbouring voxels
        # → Manhattan distance == 1 between consecutive permutation entries
        h = HilbertOrder(order=5, n_dims=2)   # 32x32
        side = h.side
        coords = torch.stack(
            [h.permutation // side, h.permutation % side], dim=-1
        )
        diffs = (coords[1:] - coords[:-1]).abs().sum(-1)
        assert (diffs == 1).all(), (
            f"Hilbert step-length max should be 1, got {diffs.max().item()}"
        )

    def test_modes_dispatch(self) -> None:
        for mode in ("hilbert", "morton", "snake", "zigzag", "raster"):
            h = HilbertOrder(order=3, n_dims=2, mode=mode)
            assert h.permutation.numel() == 64

    def test_explicit_shape_non_power_of_two(self) -> None:
        # Allow non-cubic shapes (Hilbert pads internally)
        h = HilbertOrder(shape=(12, 16), mode="raster")
        assert h.permutation.numel() == 12 * 16

    def test_hilbert_non_pow2_square_builds(self) -> None:
        # mode="hilbert" on a non-power-of-2 SQUARE extent must build a valid
        # permutation (was: AssertionError from the strict builder — the crash
        # that hit xdta's default image_size=320 at construction).
        h = HilbertOrder(shape=(320, 320), mode="hilbert")
        assert h.permutation.numel() == 320 * 320
        assert torch.equal(
            torch.sort(h.permutation).values, torch.arange(320 * 320)
        )

    def test_hilbert_non_square_builds_and_roundtrips(self) -> None:
        h = HilbertOrder(shape=(120, 160), mode="hilbert")
        assert h.permutation.numel() == 120 * 160
        x = torch.randn(1, 2, 120, 160)
        assert torch.allclose(h.unflatten(h.linearize(x), (120, 160)), x)

    def test_hilbert_square_pow2_byte_identical_to_strict(self) -> None:
        # On a square power-of-2 grid the rect-routed builder must reduce
        # byte-identically to the strict one (zero behaviour change).
        from spectramr.models.blocks.topology_linearizer import _hilbert_2d_indices

        h = HilbertOrder(order=4, n_dims=2, mode="hilbert")  # 16x16
        assert np.array_equal(h.permutation.numpy(), _hilbert_2d_indices(16, 16))

    def test_hilbert_3d_non_cubic_builds(self) -> None:
        h = HilbertOrder(shape=(5, 6, 7), mode="hilbert")
        assert h.permutation.numel() == 5 * 6 * 7
        assert torch.equal(
            torch.sort(h.permutation).values, torch.arange(5 * 6 * 7)
        )


# ─────────────────────────────────────────────────────────────────────
# DTN2S mask
# ─────────────────────────────────────────────────────────────────────


class TestDTN2SMask:
    def test_mask_shape_and_bool(self) -> None:
        phi_a, phi_b = dual_traversal_pair((16, 16))
        mask = build_dtn2s_mask(phi_a, phi_b, receptive_window=2)
        assert mask.dtype == torch.bool
        assert mask.shape == (256,)

    def test_mask_budget_proportional_to_window(self) -> None:
        # A larger receptive window should mask at least as many voxels
        phi_a, phi_b = dual_traversal_pair((16, 16))
        m_small = build_dtn2s_mask(phi_a, phi_b, receptive_window=1)
        m_large = build_dtn2s_mask(phi_a, phi_b, receptive_window=4)
        assert int(m_large.sum()) >= int(m_small.sum())

    def test_mask_deterministic(self) -> None:
        phi_a, phi_b = dual_traversal_pair((8, 8))
        a = build_dtn2s_mask(phi_a, phi_b, 2)
        b = build_dtn2s_mask(phi_a, phi_b, 2)
        assert torch.equal(a, b)


# ─────────────────────────────────────────────────────────────────────
# SFCWindowedAttention + CrossDomainBlock
# ─────────────────────────────────────────────────────────────────────


class TestSFCWindowedAttention:
    def test_shape_preserved(self) -> None:
        attn = SFCWindowedAttention(d_model=64, n_heads=4, window=16, n_global=4)
        x = torch.randn(2, 100, 64)
        y = attn(x)
        assert y.shape == x.shape

    def test_window_pads_to_multiple(self) -> None:
        # T=100, window=16 → pad to 112; output sliced back to 100
        attn = SFCWindowedAttention(d_model=32, n_heads=2, window=16, n_global=0)
        x = torch.randn(1, 100, 32)
        y = attn(x)
        assert y.shape == (1, 100, 32)

    def test_global_tokens_optional(self) -> None:
        attn0 = SFCWindowedAttention(d_model=32, n_heads=2, window=8, n_global=0)
        attn4 = SFCWindowedAttention(d_model=32, n_heads=2, window=8, n_global=4)
        x = torch.randn(1, 32, 32)
        y0 = attn0(x); y4 = attn4(x)
        assert y0.shape == y4.shape == x.shape


class TestCrossDomainBlock:
    def test_self_only(self) -> None:
        blk = CrossDomainBlock(d_model=32, n_heads=2, win_attn=8, n_global=2, cross=False)
        z_k = torch.randn(1, 16, 32)
        z_x = torch.randn(1, 16, 32)
        zk2, zx2 = blk(z_k, z_x)
        assert zk2.shape == z_k.shape and zx2.shape == z_x.shape

    def test_cross(self) -> None:
        blk = CrossDomainBlock(d_model=32, n_heads=2, win_attn=8, n_global=2, cross=True)
        z_k = torch.randn(1, 16, 32)
        z_x = torch.randn(1, 16, 32)
        zk2, zx2 = blk(z_k, z_x)
        assert zk2.shape == z_k.shape and zx2.shape == z_x.shape


# ─────────────────────────────────────────────────────────────────────
# Fisher info / CRLB / submodular placement
# ─────────────────────────────────────────────────────────────────────


class TestFisherCRLB:
    def test_linear_forward_fisher_matches_AtA(self) -> None:
        # y = A @ theta; Fisher = (A^T A) / σ²
        torch.manual_seed(0)
        M, p = 32, 4
        A = torch.randn(M, p)
        sigma2 = 0.5
        theta = torch.zeros(p, requires_grad=True)
        F = fisher_information_motion(
            lambda t: A @ t, theta, noise_cov=sigma2,
        )
        expected = (A.t() @ A) / sigma2
        assert torch.allclose(F, expected, atol=1e-5)

    def test_crlb_inverse(self) -> None:
        F = torch.eye(3) * 4.0
        crlb = crlb_from_fisher(F)
        assert torch.allclose(crlb, torch.eye(3) * 0.25, atol=1e-6)

    def test_greedy_placement_decreases_trace(self) -> None:
        # 8 candidate marker Jacobians, picking 3 should monotonically
        # decrease tr(I⁻¹).
        torch.manual_seed(0)
        K, M, p = 8, 16, 4
        jac = torch.randn(K, M, p)
        res = greedy_marker_placement(jac, n_select=3, noise_var=1.0)
        traces = res.crlb_trace_history
        for i in range(1, len(traces)):
            assert traces[i] <= traces[i - 1] + 1e-6


# ─────────────────────────────────────────────────────────────────────
# Conformal UQ
# ─────────────────────────────────────────────────────────────────────


class TestConformalUQ:
    def test_coverage_close_to_nominal(self) -> None:
        # Synthetic: residuals ~ N(0, 0.5²); test 90% coverage at α=0.1.
        torch.manual_seed(0)
        n_cal = 500
        n_test = 1000
        sigma = 0.5
        cal = torch.randn(n_cal) * sigma
        test_truth = torch.randn(n_test) * sigma
        # Calibrate
        result = split_conformal_marker_uq(
            torch.zeros(n_test), cal, alpha=0.1,
        )
        cov = conformal_coverage_check(
            test_truth, result.coverage_lower, result.coverage_upper,
        )
        # Expect ≥ 0.85 (nominal 0.9 minus finite-sample tolerance)
        assert cov >= 0.85, f"Coverage too low: {cov:.3f}"

    def test_quantile_monotone_in_alpha(self) -> None:
        torch.manual_seed(0)
        cal = torch.randn(200).abs()
        q1 = split_conformal_marker_uq(torch.zeros(1), cal, alpha=0.20).quantile
        q2 = split_conformal_marker_uq(torch.zeros(1), cal, alpha=0.05).quantile
        assert q2 >= q1, f"Tighter alpha should give larger quantile: {q1} → {q2}"


# ─────────────────────────────────────────────────────────────────────
# Through-plane motion
# ─────────────────────────────────────────────────────────────────────


class TestThroughPlaneMotion:
    def test_recovers_planted_delta_z(self) -> None:
        # Generate forward-model intensities for known Δz=2.0 mm,
        # then ensure recovery returns ≈ 2.0
        from spectramr.infrastructure.physics.marker_estimators import _gaussian_profile

        slice_thickness = 5.0
        slice_centers = torch.tensor([-7.5, -2.5, 2.5, 7.5])
        rod_extent = (-3.0, 3.0)

        # Build forward predictor from the same code path
        n_grid = 256
        z_grid = torch.linspace(
            rod_extent[0] - slice_thickness - 10,
            rod_extent[1] + slice_thickness + 10,
            n_grid,
        )
        dz_grid = z_grid[1] - z_grid[0]
        profile_z = torch.linspace(-slice_thickness, slice_thickness, n_grid)
        sigma_z = slice_thickness / (2 * np.sqrt(2 * np.log(2)))
        profile_vals = torch.exp(-(profile_z ** 2) / (2 * sigma_z ** 2))
        profile_vals = profile_vals / profile_vals.sum()

        true_dz = 2.0
        intensities = []
        for zs in slice_centers:
            # I(s) = ∫ p(z - z_s) * 1_{[z1+Δz, z2+Δz]}(z) dz
            offsets = (z_grid - zs)
            u = (offsets + slice_thickness) / (2 * slice_thickness) * (n_grid - 1)
            u = u.clamp(0, n_grid - 1)
            lo = u.floor().long().clamp(max=n_grid - 2); hi = lo + 1
            frac = (u - lo.float())
            p_eval = (1 - frac) * profile_vals[lo] + frac * profile_vals[hi]
            ind = ((z_grid >= rod_extent[0] + true_dz) & (z_grid <= rod_extent[1] + true_dz)).float()
            intensities.append(float((p_eval * ind).sum() * dz_grid))
        meas = torch.tensor([intensities])    # (1, S)

        result = through_plane_motion_recovery(
            meas, rod_extent=rod_extent, slice_centers=slice_centers,
            slice_thickness=slice_thickness, rho_marker=1.0,
            search_range_mm=(-5.0, 5.0), n_search=51,
        )
        recovered = float(result.delta_z[0])
        # Search step is 10/50 = 0.2 mm, so accept |err| < 0.6 mm
        assert abs(recovered - true_dz) < 0.6, (
            f"Recovered Δz={recovered:.3f} too far from planted {true_dz}"
        )


# ─────────────────────────────────────────────────────────────────────
# Profile-likelihood motion estimator (Method II)
# ─────────────────────────────────────────────────────────────────────


class TestProfileLikelihoodMotion:
    def test_recovers_known_translation(self) -> None:
        # A simple 1-D shift model: forward(theta) = e^{-i k * theta} on a
        # known marker spectrum. The estimator should converge close to
        # the true shift.
        torch.manual_seed(0)
        M = 32
        k = torch.linspace(-1, 1, M)
        marker_amp = torch.exp(-(k ** 2) / 0.1)        # bell-shaped marker

        true_theta = torch.tensor([0.7])

        def fwd(theta: torch.Tensor) -> torch.Tensor:
            phase = torch.exp(-1j * 2 * 3.14159265 * k * theta[0])
            return marker_amp.to(torch.complex64) * phase

        y = fwd(true_theta) + 0.01 * torch.randn(M, dtype=torch.complex64)
        result = profile_likelihood_motion(
            fwd, y, theta_init=torch.zeros(1),
            anatomy_var=1.0, noise_var=0.01,
            n_iters=200, lr=5e-2,
        )
        assert abs(float(result.theta_hat[0]) - 0.7) < 0.1, (
            f"Recovered θ={float(result.theta_hat[0]):.3f} too far from 0.7"
        )
        # NLL should monotonically decrease (within numerical tolerance)
        assert result.nll_history[-1] < result.nll_history[0]


# ─────────────────────────────────────────────────────────────────────
# Privileged-learning loss (Method III)
# ─────────────────────────────────────────────────────────────────────


class TestPrivilegedLearning:
    def test_loss_components(self) -> None:
        torch.manual_seed(0)
        student = torch.randn(2, 1, 16, 16)
        teacher = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        out = privileged_learning_loss(
            student, teacher, target, alpha=1.0, beta=0.5, gamma=0.0,
        )
        assert "g_total_loss" in out
        assert "g_priv_recon" in out and "g_priv_distill" in out
        # Total = α·recon + β·distill (γ·hint = 0 here)
        expected = out["g_priv_recon"] + 0.5 * out["g_priv_distill"]
        assert torch.allclose(out["g_total_loss"], expected, atol=1e-6)

    def test_hint_loss_with_features(self) -> None:
        student_feat = [torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4)]
        teacher_feat = [torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4)]
        out = privileged_learning_loss(
            torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8),
            torch.zeros(2, 1, 8, 8),
            student_features=student_feat, teacher_features=teacher_feat,
            alpha=0.0, beta=0.0, gamma=1.0,
        )
        # γ-only with zero pred/target/teacher → loss == hint
        assert torch.allclose(out["g_total_loss"], out["g_priv_hint"])

    def test_mismatched_feature_lists_raise(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="match length"):
            privileged_learning_loss(
                torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8),
                torch.zeros(2, 1, 8, 8),
                student_features=[torch.zeros(2, 4, 8, 8)],
                teacher_features=[torch.zeros(2, 4, 8, 8), torch.zeros(2, 8, 4, 4)],
            )


# ─────────────────────────────────────────────────────────────────────
# XDTA backbone
# ─────────────────────────────────────────────────────────────────────


class TestXDTA:
    def test_forward_shape(self) -> None:
        from spectramr.models.generators.xdta import XDTA, COND_VOCAB

        # Tiny config — keeps wallclock under 5 s on CPU
        m = XDTA(
            image_size=32, win_tok=8, d_model=32, n_blocks=2,
            n_heads=2, win_attn=8, n_global=2, cross_every=2,
        )
        B = 1
        k_obs = torch.complex(torch.randn(B, 1, 32, 32), torch.randn(B, 1, 32, 32))
        mask = torch.zeros(B, 1, 32, 32, dtype=torch.bool)
        mask[..., ::4, :] = True
        x_init = torch.randn(B, 1, 32, 32)
        cond = torch.tensor([[COND_VOCAB["T1"], COND_VOCAB["3T"], COND_VOCAB["task=recon"]]])
        k_hat, x_hat = m(k_obs, mask, x_init, cond=cond)
        assert k_hat.shape == k_obs.shape
        assert x_hat.shape == x_init.shape
        assert torch.is_complex(k_hat)
        assert not torch.is_complex(x_hat)

    def test_unconditional_works(self) -> None:
        from spectramr.models.generators.xdta import XDTA
        m = XDTA(image_size=32, win_tok=8, d_model=32, n_blocks=1,
                 n_heads=2, win_attn=8, n_global=0, cross_every=1)
        B = 1
        k_obs = torch.complex(torch.randn(B, 1, 32, 32), torch.zeros(B, 1, 32, 32))
        mask = torch.ones(B, 1, 32, 32, dtype=torch.bool)
        x_init = torch.randn(B, 1, 32, 32)
        k_hat, x_hat = m(k_obs, mask, x_init, cond=None)
        assert k_hat.shape == k_obs.shape and x_hat.shape == x_init.shape
