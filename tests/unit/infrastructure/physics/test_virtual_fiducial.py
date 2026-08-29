"""Unit tests for VirtualFiducial and MotionTrajectory physics modules."""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.virtual_fiducial import (
    MotionTrajectory,
    VirtualFiducial,
)


class TestVirtualFiducial:
    """Tests for VirtualFiducial Gaussian grid probe."""

    def test_init_default(self) -> None:
        """Default init creates a VirtualFiducial with a grid buffer."""
        vf = VirtualFiducial(im_size=(64, 64))
        assert vf.im_size == (64, 64)
        assert hasattr(vf, "grid")

    def test_init_learnable(self) -> None:
        """Learnable mode creates grid as nn.Parameter."""
        vf = VirtualFiducial(im_size=(64, 64), learnable=True)
        assert isinstance(vf.grid, torch.nn.Parameter)
        assert vf.grid.requires_grad

    def test_forward_output_shape(self) -> None:
        """Output shape is [B, 1, H, W] complex."""
        vf = VirtualFiducial(im_size=(64, 64))
        out = vf(batch_size=2)
        assert out.shape == (2, 1, 64, 64)

    def test_forward_single_batch(self) -> None:
        """Default forward produces [1, 1, H, W]."""
        vf = VirtualFiducial(im_size=(64, 64))
        out = vf()
        assert out.shape == (1, 1, 64, 64)

    def test_forward_dtype_complex(self) -> None:
        """Output should be complex64."""
        vf = VirtualFiducial(im_size=(64, 64))
        out = vf()
        assert torch.is_complex(out), f"Expected complex dtype, got {out.dtype}"

    def test_nonzero_output(self) -> None:
        """Gaussian markers should produce non-zero signal."""
        vf = VirtualFiducial(im_size=(64, 64), grid_spacing=16)
        out = vf()
        assert out.abs().sum() > 0, "VirtualFiducial output is all zeros"

    def test_grid_spacing_affects_pattern(self) -> None:
        """Different grid spacing produces different patterns."""
        vf_dense = VirtualFiducial(im_size=(64, 64), grid_spacing=8)
        vf_sparse = VirtualFiducial(im_size=(64, 64), grid_spacing=32)
        out_dense = vf_dense()
        out_sparse = vf_sparse()
        # Dense grid should have more non-zero area
        area_dense = (out_dense.abs() > 0.01).sum()
        area_sparse = (out_sparse.abs() > 0.01).sum()
        assert area_dense > area_sparse

    def test_device_transfer(self) -> None:
        """VirtualFiducial should work after moving to CPU."""
        vf = VirtualFiducial(im_size=(32, 32))
        vf = vf.to("cpu")
        out = vf()
        assert out.device.type == "cpu"


class TestMotionTrajectory:
    """Tests for MotionTrajectory optimizable motion parameters."""

    def test_init_shape(self) -> None:
        """MotionTrajectory creates [1, 3, num_readout_lines] parameters."""
        mt = MotionTrajectory(num_readout_lines=256)
        assert mt.theta.shape == (1, 3, 256)

    def test_init_zeros(self) -> None:
        """Zero mode initializes to zeros."""
        mt = MotionTrajectory(num_readout_lines=128, init_mode="zero")
        assert torch.allclose(mt.theta.data, torch.zeros(1, 3, 128))

    def test_init_random(self) -> None:
        """Random mode initializes with small random values."""
        mt = MotionTrajectory(num_readout_lines=128, init_mode="random")
        assert mt.theta.abs().max() > 0  # Non-zero

    def test_init_invalid_mode(self) -> None:
        """Invalid init_mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown init_mode"):
            MotionTrajectory(num_readout_lines=64, init_mode="bad")

    def test_requires_grad(self) -> None:
        """Theta should require gradients for optimization."""
        mt = MotionTrajectory(num_readout_lines=64)
        assert mt.theta.requires_grad

    def test_forward_returns_theta(self) -> None:
        """Forward should return the theta parameters."""
        mt = MotionTrajectory(num_readout_lines=64)
        out = mt()
        assert out.shape == (1, 3, 64)

    def test_forward_batch_expansion(self) -> None:
        """Forward with batch_size > 1 expands correctly."""
        mt = MotionTrajectory(num_readout_lines=32)
        out = mt(batch_size=4)
        assert out.shape == (4, 3, 32)

    def test_decomposition(self) -> None:
        """Theta decomposes into dx, dy, dtheta."""
        mt = MotionTrajectory(num_readout_lines=32)
        theta = mt()
        dx, dy, dtheta = theta[:, 0], theta[:, 1], theta[:, 2]
        assert dx.shape == (1, 32)
        assert dy.shape == (1, 32)
        assert dtheta.shape == (1, 32)


class TestFiducialJitter:
    """A periodic fiducial cannot be registered; jitter is what makes it a probe.

    Measured on a 256x256 grid at spacing 16, sigma 2, recovering a known
    sub-pixel shift through the same 2x pooling the subvoxel_sr arm applies:
    jitter 0.0 leaves 0.27% of spectral bins populated and misses the shift by
    0.55 px; jitter 0.35 leaves 23% populated and lands within 0.003 px.
    """

    @staticmethod
    def _occupancy(field: torch.Tensor) -> float:
        """Fraction of spectral bins above 1% of peak, measured AFTER the 2x
        pooling — the resolution the estimator actually sees, not full-res."""
        import torch.nn.functional as F

        from mriforge.infrastructure.physics.fft_ops import fft2c

        k = fft2c(F.avg_pool2d(field, 2)).abs()[0, 0]
        return float((k > 0.01 * k.max()).float().mean())

    def test_periodic_lattice_has_a_comb_spectrum(self) -> None:
        field = VirtualFiducial(im_size=(256, 256), grid_spacing=16, jitter=0.0)(1).real
        assert self._occupancy(field) < 0.02

    def test_jitter_makes_the_probe_broadband(self) -> None:
        field = VirtualFiducial(im_size=(256, 256), grid_spacing=16, jitter=0.35)(1).real
        assert self._occupancy(field) > 0.15

    def test_jitter_is_what_makes_the_shift_recoverable(self) -> None:
        import torch.nn.functional as F

        from mriforge.infrastructure.physics.subpixel_registration import (
            estimate_subpixel_shifts,
            fourier_shift,
        )

        true = torch.tensor([[[0.7, -0.4], [-0.9, 0.55]]])

        def recover(jitter: float) -> float:
            field = VirtualFiducial(
                im_size=(256, 256), grid_spacing=16, sigma=2.0, jitter=jitter, seed=0
            )(1).real
            moved = fourier_shift(field.expand(1, 2, 256, 256).contiguous(), true)
            est = estimate_subpixel_shifts(F.avg_pool2d(field, 2), F.avg_pool2d(moved, 2))
            return float((est * 2.0 - true).abs().max())

        assert recover(0.0) > 0.1
        assert recover(0.35) < 0.02

    def test_jitter_draw_is_deterministic(self) -> None:
        """The fiducial is an ABSOLUTE registration reference, so the same seed
        must give a byte-identical pattern across instances and processes."""
        a = VirtualFiducial(im_size=(64, 64), jitter=0.35, seed=3)(1)
        b = VirtualFiducial(im_size=(64, 64), jitter=0.35, seed=3)(1)
        assert torch.equal(a, b)

    def test_different_seeds_give_different_patterns(self) -> None:
        a = VirtualFiducial(im_size=(64, 64), jitter=0.35, seed=3)(1)
        b = VirtualFiducial(im_size=(64, 64), jitter=0.35, seed=4)(1)
        assert not torch.equal(a, b)

    def test_rejects_negative_jitter(self) -> None:
        with pytest.raises(ValueError, match="jitter must be >= 0"):
            VirtualFiducial(im_size=(32, 32), jitter=-0.1)


class TestPhysicalUnits:
    """The marker must be sized in millimetres against the resolution it has to
    survive at, not in pixels against the grid it happens to be stored on.

    `preprocess_ulf_paired.py` resamples the 64mT volume onto the 3T grid, so a
    preprocessed ULF volume is stored at 0.22-0.49 mm while the scanner resolved
    1.6-1.7 mm in-plane. Sizing to the grid makes the marker 3-7x too fine to
    exist in the ULF channel.
    """

    def test_pixel_mode_is_the_default_and_unchanged(self) -> None:
        """exp_vf_01 declares no voxel size and must be untouched."""
        f = VirtualFiducial(im_size=(64, 64), grid_spacing=16, sigma=2.0)
        assert f.voxel_mm is None
        assert f.sigma_px == (2.0, 2.0)
        assert f(1).shape == (1, 1, 64, 64)

    def test_sigma_is_sized_against_the_effective_resolution(self) -> None:
        """1.6 mm marker on a 0.49 mm grid is 3.27 grid-pixels, not 1."""
        f = VirtualFiducial(
            im_size=(128, 128), voxel_mm=(0.49, 0.49), effective_voxel_mm=(1.6, 1.6)
        )
        assert f.sigma_mm == (1.6, 1.6)
        assert f.sigma_px[0] == pytest.approx(1.6 / 0.49, rel=1e-6)

    def test_anisotropic_3d_marker(self) -> None:
        """Measured 64mT protocol: 1.6 x 1.6 x 5.0 mm. The slab must be deep
        enough to hold a peak at the default 8*sigma spacing (40 mm through
        plane), hence 64 slices at 1 mm rather than 16."""
        f = VirtualFiducial(
            im_size=(64, 64, 64),
            voxel_mm=(0.49, 0.49, 1.0),
            effective_voxel_mm=(1.6, 1.6, 5.0),
            jitter=0.35,
        )
        assert f(1).shape == (1, 1, 64, 64, 64)
        assert f.sigma_mm[2] / f.sigma_mm[0] == pytest.approx(3.125)

    def test_axis_too_thin_for_a_peak_raises(self) -> None:
        """A 16-slice slab at 1 mm cannot hold a peak spaced 8*5.0 = 40 mm apart.
        Silently producing a marker with zero peaks on that axis would leave the
        through-plane shift unregistrable with nothing to say so."""
        with pytest.raises(ValueError, match="no marker peak fits"):
            VirtualFiducial(
                im_size=(64, 64, 16),
                voxel_mm=(0.49, 0.49, 1.0),
                effective_voxel_mm=(1.6, 1.6, 5.0),
            )

    def test_kappa_scales_the_width(self) -> None:
        f = VirtualFiducial(
            im_size=(64, 64), voxel_mm=(1.0, 1.0), effective_voxel_mm=(1.0, 1.0),
            kappa=2.0,
        )
        assert f.sigma_mm == (2.0, 2.0)

    def test_subresolution_sigma_raises(self) -> None:
        """The aliasing failure: a marker narrower than the voxel it must survive
        at is not a translate of a fixed template once sampled, so phase
        correlation acquires a bias no averaging removes."""
        with pytest.raises(ValueError, match="below the effective voxel size"):
            VirtualFiducial(
                im_size=(64, 64),
                voxel_mm=(0.49, 0.49),
                effective_voxel_mm=(1.6, 1.6),
                sigma_mm=(0.5, 0.5),
            )

    def test_effective_finer_than_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="finer than the sampling grid"):
            VirtualFiducial(
                im_size=(64, 64),
                voxel_mm=(1.6, 1.6),
                effective_voxel_mm=(0.49, 0.49),
            )

    def test_spacing_under_two_sigma_raises(self) -> None:
        """Peaks would merge into a smooth field with no localisable structure."""
        with pytest.raises(ValueError, match=r"under 2\*sigma_mm"):
            VirtualFiducial(
                im_size=(64, 64),
                voxel_mm=(1.0, 1.0),
                effective_voxel_mm=(1.0, 1.0),
                spacing_mm=(1.5, 1.5),
            )

    def test_axis_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="im_size is 2-D"):
            VirtualFiducial(im_size=(64, 64), voxel_mm=(1.0, 1.0, 1.0))

    def test_physical_marker_is_still_registrable(self) -> None:
        """End to end: the physically-sized marker must recover a known shift,
        otherwise the sizing rule would be correct in theory and useless."""
        import torch.nn.functional as F

        from mriforge.infrastructure.physics.subpixel_registration import (
            estimate_subpixel_shifts,
            fourier_shift,
        )

        f = VirtualFiducial(
            im_size=(256, 256),
            voxel_mm=(0.49, 0.49),
            effective_voxel_mm=(1.6, 1.6),
            jitter=0.35,
            seed=0,
        )(1).real
        true = torch.tensor([[[1.2, -0.8], [-1.5, 0.6]]])
        moved = fourier_shift(f.expand(1, 2, 256, 256).contiguous(), true)
        est = estimate_subpixel_shifts(F.avg_pool2d(f, 2), F.avg_pool2d(moved, 2)) * 2.0
        assert (est - true).abs().max().item() < 0.05

