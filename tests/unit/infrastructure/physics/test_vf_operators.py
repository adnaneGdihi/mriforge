"""Unit tests for Virtual Fiducial physics operators (vf_operators.py)."""

import pytest
import torch
from torch import nn

from spectramr.infrastructure.physics.vf_operators import (
    ADMMSolver,
    B0PhaseOperator,
    GeometricWarpOperator,
    MarkerPriorProjection,
    MasterForwardChain,
    NoisePreWhitening,
    RigidKinematicOperator,
    ToeplitzPSFOperator,
)

# ────────────────────────────────────────────────────────────────────
# 1. MarkerPriorProjection
# ────────────────────────────────────────────────────────────────────


class TestMarkerPriorProjection:
    """Tests for MarkerPriorProjection operator."""

    def test_shape_preserved(self) -> None:
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1.0
        prior = torch.ones(1, 1, 16, 16) * 0.5
        proj = MarkerPriorProjection(mask, prior)
        x = torch.randn(2, 1, 16, 16)
        assert proj(x).shape == x.shape

    def test_marker_pixels_anchored(self) -> None:
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1.0
        prior = torch.ones(1, 1, 16, 16) * 0.5
        proj = MarkerPriorProjection(mask, prior)
        x = torch.randn(2, 1, 16, 16)
        out = proj(x)
        assert torch.allclose(
            out[:, :, 0:4, 0:4],
            prior[:, :, 0:4, 0:4].expand(2, -1, -1, -1),
        )

    def test_non_marker_pixels_unchanged(self) -> None:
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1.0
        prior = torch.ones(1, 1, 16, 16) * 0.5
        proj = MarkerPriorProjection(mask, prior)
        x = torch.randn(2, 1, 16, 16)
        out = proj(x)
        assert torch.allclose(out[:, :, 4:, 4:], x[:, :, 4:, 4:])

    def test_gradient_flows(self) -> None:
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 0:2, 0:2] = 1.0
        prior = torch.ones(1, 1, 8, 8) * 0.5
        proj = MarkerPriorProjection(mask, prior)
        x = torch.randn(1, 1, 8, 8, requires_grad=True)
        loss = proj(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_multicoil_broadcast_single_marker_over_coils(self) -> None:
        """Regression (exp_vf_05 crash, smoke 2026-06-15): a single-marker
        real-stacked mask/prior (2ch = r/i) must project a multi-coil prediction
        (8ch = 4 coils x r/i) by tiling across coils — not raise
        'tensor a (2) must match tensor b (8)'. The marker is a data-independent
        reference grid, identical per coil, so tiling is the correct anchor."""
        mask = torch.zeros(1, 2, 16, 16)
        mask[:, :, 0:4, 0:4] = 1.0
        prior = torch.full((1, 2, 16, 16), 0.5)
        proj = MarkerPriorProjection(mask, prior)
        x = torch.randn(2, 8, 16, 16)
        out = proj(x)
        assert out.shape == x.shape
        # every coil's marker region anchored to the tiled prior (0.5)
        assert torch.allclose(out[:, :, 0:4, 0:4], torch.full((2, 8, 4, 4), 0.5))
        # non-marker pixels untouched
        assert torch.allclose(out[:, :, 4:, 4:], x[:, :, 4:, 4:])

    def test_raises_on_indivisible_channel_count(self) -> None:
        """No silent wrong-broadcast (pitfall #9): 3ch mask into 8ch x cannot be
        tiled cleanly, so it must raise rather than mis-anchor."""
        mask = torch.zeros(1, 3, 8, 8)
        prior = torch.zeros(1, 3, 8, 8)
        proj = MarkerPriorProjection(mask, prior)
        with pytest.raises((ValueError, RuntimeError)):
            proj(torch.randn(1, 8, 8, 8))


# ────────────────────────────────────────────────────────────────────
# 2. RigidKinematicOperator
# ────────────────────────────────────────────────────────────────────


class TestRigidKinematicOperator:
    """Tests for RigidKinematicOperator."""

    def test_zero_motion_is_identity(self) -> None:
        op = RigidKinematicOperator(im_size=(16, 16))
        kspace = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        params = torch.zeros(2, 3, 16)  # dx, dy, dθ = 0
        out = op(kspace, params)
        assert torch.allclose(out, kspace, atol=1e-6)

    def test_shape_preserved(self) -> None:
        op = RigidKinematicOperator(im_size=(32, 32))
        kspace = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
        params = torch.randn(2, 3, 32)
        params[:, 2, :] = 0.0  # translation-only (rotation requires regridding)
        assert op(kspace, params).shape == kspace.shape

    def test_nonzero_rotation_raises(self) -> None:
        """Non-zero dθ must raise, not silently no-op (pitfall #16)."""
        op = RigidKinematicOperator(im_size=(16, 16))
        kspace = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        params = torch.zeros(1, 3, 16)
        params[:, 2, :] = 0.2  # a real per-line rotation
        with pytest.raises(NotImplementedError, match="rotation"):
            op(kspace, params)

    def test_translation_applies_phase_ramp(self) -> None:
        """A pure translation must actually change k-space (not a no-op)."""
        op = RigidKinematicOperator(im_size=(16, 16))
        kspace = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        params = torch.zeros(1, 3, 16)
        params[:, 0, :] = 2.0  # dx = 2 px
        out = op(kspace, params)
        assert not torch.allclose(out, kspace, atol=1e-3)

    def test_translation_is_exact_integer_pixel_shift(self) -> None:
        """A constant integer shift must be an exact circular roll.

        Regression for the k-grid convention: with linspace(-0.5, 0.5, W) the
        commanded shift was applied as dx*W/(W-1) and off-center (roll-match
        error ~O(1)); the fftshift(fftfreq) grid makes it exact.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        op = RigidKinematicOperator(im_size=(16, 16))
        torch.manual_seed(0)
        img = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        params = torch.zeros(1, 3, 16)
        params[:, 0, :] = 2.0  # dx = 2 px, constant over PE lines
        shifted = ifft2c(op(fft2c(img), params))
        err = min(
            (shifted - torch.roll(img, -2, dims=-1)).abs().max().item(),
            (shifted - torch.roll(img, 2, dims=-1)).abs().max().item(),
        )
        assert err < 1e-4


# ────────────────────────────────────────────────────────────────────
# 3. B0PhaseOperator
# ────────────────────────────────────────────────────────────────────


class TestB0PhaseOperator:
    """Tests for B0PhaseOperator."""

    def test_no_b0_is_identity(self) -> None:
        op = B0PhaseOperator(b0_map=None, te=0.01)
        x = torch.randn(2, 1, 16, 16)
        assert torch.allclose(op(x), x)

    def test_zero_b0_is_identity(self) -> None:
        b0 = torch.zeros(1, 1, 16, 16)
        op = B0PhaseOperator(b0_map=b0, te=0.01)
        x_c = torch.complex(torch.randn(2, 1, 16, 16), torch.zeros(2, 1, 16, 16))
        out = op(x_c)
        assert torch.allclose(out.abs(), x_c.abs(), atol=1e-6)

    def test_shape_preserved(self) -> None:
        b0 = torch.ones(1, 1, 16, 16) * 50.0
        op = B0PhaseOperator(b0_map=b0, te=0.01)
        x = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        assert op(x).shape == x.shape


# ────────────────────────────────────────────────────────────────────
# 4. GeometricWarpOperator
# ────────────────────────────────────────────────────────────────────


class TestGeometricWarpOperator:
    """Tests for GeometricWarpOperator."""

    def test_zero_flow_is_identity(self) -> None:
        op = GeometricWarpOperator(im_size=(16, 16))
        x = torch.randn(2, 1, 16, 16)
        flow = torch.zeros(2, 2, 16, 16)
        out = op(x, flow)
        # Not exact identity due to bilinear interpolation at boundaries
        inner = slice(2, 14)
        assert torch.allclose(out[:, :, inner, inner], x[:, :, inner, inner], atol=1e-4)

    def test_shape_preserved(self) -> None:
        op = GeometricWarpOperator(im_size=(32, 32))
        assert op(torch.randn(2, 1, 32, 32), torch.zeros(2, 2, 32, 32)).shape == (
            2,
            1,
            32,
            32,
        )


# ────────────────────────────────────────────────────────────────────
# 5. ToeplitzPSFOperator
# ────────────────────────────────────────────────────────────────────


class TestToeplitzPSFOperator:
    """Tests for ToeplitzPSFOperator."""

    def test_identity_psf(self) -> None:
        """Default initialisation should be close to identity (delta)."""
        op = ToeplitzPSFOperator(kernel_size=5)
        x = torch.randn(2, 1, 16, 16)
        out = op(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_adjoint_shape(self) -> None:
        op = ToeplitzPSFOperator(kernel_size=7)
        x = torch.randn(2, 1, 16, 16)
        assert op.adjoint(op(x)).shape == x.shape

    def test_set_psf(self) -> None:
        op = ToeplitzPSFOperator(kernel_size=5)
        new_psf = torch.ones(1, 1, 5, 5) / 25.0
        op.set_psf(new_psf)
        assert torch.allclose(op.psf, new_psf)


# ────────────────────────────────────────────────────────────────────
# 6. NoisePreWhitening
# ────────────────────────────────────────────────────────────────────


class TestNoisePreWhitening:
    """Tests for NoisePreWhitening."""

    def test_single_coil_passthrough(self) -> None:
        op = NoisePreWhitening(num_coils=1)
        kspace = torch.randn(2, 1, 16, 16)
        assert torch.allclose(op(kspace), kspace)

    def test_shape_preserved_multicoil(self) -> None:
        op = NoisePreWhitening(num_coils=4)
        kspace = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
        assert op(kspace).shape == kspace.shape


# ────────────────────────────────────────────────────────────────────
# 7. MasterForwardChain
# ────────────────────────────────────────────────────────────────────


class TestMasterForwardChain:
    """Tests for MasterForwardChain."""

    def test_empty_chain_is_identity(self) -> None:
        chain = MasterForwardChain([])
        x = torch.randn(2, 1, 16, 16)
        assert torch.allclose(chain(x), x)

    def test_chaining(self) -> None:
        op1 = ToeplitzPSFOperator(kernel_size=3)
        op2 = ToeplitzPSFOperator(kernel_size=3)
        chain = MasterForwardChain([op1, op2])
        x = torch.randn(2, 1, 16, 16)
        assert chain(x).shape == x.shape

    def test_multi_arg_operator_in_chain_raises(self) -> None:
        """Pin the documented single-arg-only contract: kwargs are NOT forwarded,
        so a multi-argument operator (here RigidKinematicOperator, which needs
        motion_params) raises TypeError when driven through the chain."""
        import pytest

        chain = MasterForwardChain([RigidKinematicOperator(im_size=(16, 16))])
        with pytest.raises(TypeError):
            chain(
                torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)),
                motion_params=torch.zeros(2, 3, 16),
            )


# ────────────────────────────────────────────────────────────────────
# 8. ADMMSolver
# ────────────────────────────────────────────────────────────────────


class TestADMMSolver:
    """Tests for ADMMSolver."""

    def test_shape_preserved(self) -> None:
        fwd = ToeplitzPSFOperator(kernel_size=3)
        reg = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1)
        )
        solver = ADMMSolver(forward_op=fwd, regulariser=reg, num_iters=2)
        y = torch.randn(1, 1, 16, 16)
        out = solver(y)
        assert out.shape == (1, 1, 16, 16)

    def test_with_prior_projection(self) -> None:
        fwd = ToeplitzPSFOperator(kernel_size=3)
        reg = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1))
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:2, 0:2] = 1.0
        prior = torch.ones(1, 1, 16, 16) * 0.5
        proj = MarkerPriorProjection(mask, prior)
        solver = ADMMSolver(forward_op=fwd, regulariser=reg, prior_projection=proj, num_iters=2)
        y = torch.randn(1, 1, 16, 16)
        out = solver(y)
        assert out.shape == (1, 1, 16, 16)


class TestNUFFTTrajectoryCalibratorFrame:
    """The marker image must live in the frame the marker mask is drawn in."""

    @staticmethod
    def _delta_kspace(h: int = 16, w: int = 16):
        import torch

        from spectramr.infrastructure.physics.fft_ops import fft2c

        img = torch.zeros(1, 1, h, w, dtype=torch.complex64)
        img[0, 0, h // 2, w // 2] = 1.0  # a marker at the centre of the FOV
        return img, fft2c(img)

    def test_a_centred_marker_stays_at_the_centre(self):
        """Planted violation: a raw ``torch.fft.ifft2`` puts it at the corner."""
        import torch

        from spectramr.infrastructure.physics.vf_operators_extended import NUFFTTrajectoryCalibrator

        img, kspace = self._delta_kspace()
        cal = NUFFTTrajectoryCalibrator(im_size=(16, 16))
        mag = cal.marker_image(kspace, torch.zeros(2))
        assert divmod(int(mag[0, 0].flatten().argmax()), 16) == (8, 8)
        assert mag.max().item() == pytest.approx(1.0)
        raw = torch.fft.ifft2(kspace, dim=(-2, -1)).abs()
        assert divmod(int(raw[0, 0].flatten().argmax()), 16) == (0, 0)

    def test_calibrate_sees_the_marker_through_its_mask(self):
        """With zero delays the objective on a centred marker is already at its floor,
        so the optimiser must not move the delays away from zero."""
        import torch

        from spectramr.infrastructure.physics.vf_operators_extended import NUFFTTrajectoryCalibrator

        img, kspace = self._delta_kspace()
        mask = torch.zeros(1, 1, 16, 16)
        mask[0, 0, 6:11, 6:11] = 1.0
        cal = NUFFTTrajectoryCalibrator(im_size=(16, 16))
        delays = cal.calibrate(kspace, mask, img.abs(), num_iters=5, lr=0.05)
        assert torch.all(delays.abs() < 1e-3)
