"""Unit tests for differentiable sub-pixel shift and marker-anchored recovery.

These pin the two properties the ``subvoxel_sr`` arm depends on: the dither
operator is an exact band-limited translation (not a blurring resample), and the
recovery inverts it to sub-pixel accuracy while staying differentiable.
"""

from __future__ import annotations

import os

os.environ.setdefault("MRIFORGE_SUPPRESS_CLINICAL_WARNING", "1")

import math

import pytest
import torch

from mriforge.infrastructure.physics.fft_ops import fft2c
from mriforge.infrastructure.physics.subpixel_registration import (
    estimate_subpixel_shifts,
    fourier_shift,
)


def _smooth_image(b: int = 2, h: int = 64, w: int = 64, seed: int = 0) -> torch.Tensor:
    """Band-limited random image: a white-noise field has energy at Nyquist where
    the linear-phase model degenerates, which is not the regime MRI operates in."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(b, 1, 8, 8, generator=g)
    return torch.nn.functional.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)


class TestFourierShift:
    def test_integer_shift_moves_content_exactly(self) -> None:
        """(dy, dx) is (rows, cols) and positive moves toward higher indices."""
        x = torch.zeros(1, 1, 32, 32)
        x[0, 0, 10, 7] = 1.0
        y = fourier_shift(x, torch.tensor([[[3.0, -2.0]]]))
        peak = divmod(int(y.flatten().argmax()), 32)
        assert peak == (13, 5)

    def test_zero_shift_is_identity(self) -> None:
        x = _smooth_image()
        out = fourier_shift(x, torch.zeros(x.shape[0], x.shape[1], 2))
        assert torch.allclose(out, x, atol=1e-5)

    def test_integer_shift_equals_torch_roll(self) -> None:
        """Exact, image-independent identity. For integer shifts the k-space ramp
        is real at the Nyquist bin, so nothing is lost and the operator reduces
        to a circular shift."""
        x = _smooth_image(b=1)
        got = fourier_shift(x, torch.tensor([[[4.0, -3.0]]]))
        want = torch.roll(x, shifts=(4, -3), dims=(-2, -1))
        assert torch.allclose(got, want, atol=1e-4)

    def test_fractional_round_trip_returns_the_image(self) -> None:
        """Shift then unshift. The residual is the unpaired Nyquist bin of an
        even-length DFT, not resample blur."""
        x = _smooth_image()
        d = torch.tensor([[[0.37, -0.81]], [[-0.44, 0.29]]])
        back = fourier_shift(fourier_shift(x, d), -d)
        rms_rel = ((back - x) ** 2).mean().sqrt() / (x.amax() - x.amin())
        assert rms_rel.item() < 5e-3

    def test_preserves_high_frequencies_that_resampling_destroys(self) -> None:
        """The reason this operator exists.

        A translation is a pure phase ramp, so it must leave the MAGNITUDE
        spectrum untouched. Bilinear ``grid_sample`` instead multiplies the
        spectrum by an interpolation kernel whose rolloff depends on the
        fractional part of the shift, gutting exactly the high-frequency band
        that multi-frame SR exists to recover. At a half-pixel dither this is
        the difference between a few percent and a third of the band.
        """
        g = torch.Generator().manual_seed(7)
        x = torch.nn.functional.interpolate(
            torch.rand(1, 1, 32, 32, generator=g),
            size=(64, 64),
            mode="bicubic",
            align_corners=False,
        )
        half = torch.tensor([[[0.5, 0.5]]])

        b, _, h, w = x.shape
        theta = (
            torch.tensor([[1.0, 0.0, -1.0 / w], [0.0, 1.0, -1.0 / h]]).unsqueeze(0).expand(b, 2, 3)
        )
        resampled = torch.nn.functional.grid_sample(
            x,
            torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False),
            align_corners=False,
            padding_mode="reflection",
        )

        def spectrum(t: torch.Tensor) -> torch.Tensor:
            return fft2c(t).abs()[0, 0]

        base = spectrum(x)
        fy = (torch.arange(h) - h // 2).view(-1, 1).float()
        fx = (torch.arange(w) - w // 2).view(1, -1).float()
        high = (fy**2 + fx**2).sqrt() > h // 4

        def deviation(t: torch.Tensor) -> float:
            dev = (spectrum(t)[high] - base[high]).abs() / (base[high] + 1e-8)
            return float(dev.mean())

        fourier_dev = deviation(fourier_shift(x, half))
        resample_dev = deviation(resampled)
        assert fourier_dev < 0.10
        assert resample_dev > 5 * fourier_dev

    def test_per_channel_shifts_are_independent(self) -> None:
        """A frame stack must take a different shift per frame in one call."""
        x = torch.zeros(1, 2, 32, 32)
        x[0, :, 16, 16] = 1.0
        y = fourier_shift(x, torch.tensor([[[2.0, 0.0], [0.0, 5.0]]]))
        assert divmod(int(y[0, 0].flatten().argmax()), 32) == (18, 16)
        assert divmod(int(y[0, 1].flatten().argmax()), 32) == (16, 21)

    def test_preserves_energy(self) -> None:
        """A translation is unitary; a resampling kernel would shed energy."""
        x = _smooth_image()
        y = fourier_shift(x, torch.tensor([[[0.5]], [[0.5]]]).expand(2, 1, 2))
        assert y.pow(2).sum().item() == pytest.approx(x.pow(2).sum().item(), rel=1e-4)

    def test_rejects_mismatched_shift_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shifts must be \[B, C, 2\]"):
            fourier_shift(torch.zeros(1, 3, 8, 8), torch.zeros(1, 2, 2))

    def test_rejects_unbatched_input(self) -> None:
        """3-D `[B, C, H, W, D]` is valid since PR-B; a bare `[C, H, W]` is not."""
        with pytest.raises(ValueError, match=r"expected \[B, C, H, W\]"):
            fourier_shift(torch.zeros(3, 8, 8), torch.zeros(3, 8, 2))


class TestEstimateSubpixelShifts:
    def test_recovers_known_subpixel_shifts(self) -> None:
        ref = _smooth_image(b=2)
        true = (torch.rand(2, 5, 2, generator=torch.Generator().manual_seed(1)) - 0.5) * 2
        moving = fourier_shift(ref.expand(2, 5, 64, 64).contiguous(), true)
        est = estimate_subpixel_shifts(ref, moving)
        assert (est - true).abs().max().item() < 0.02

    def test_zero_shift_recovers_zero(self) -> None:
        ref = _smooth_image()
        est = estimate_subpixel_shifts(ref, ref.clone())
        assert est.abs().max().item() < 1e-3

    def test_wrap_free_beyond_one_pixel(self) -> None:
        """Reading the cross-power phase directly would alias here; the
        adjacent-bin difference does not."""
        ref = _smooth_image(b=1)
        true = torch.tensor([[[7.5, -6.25]]])
        moving = fourier_shift(ref, true)
        est = estimate_subpixel_shifts(ref, moving)
        assert (est - true).abs().max().item() < 0.05

    def test_is_differentiable_in_the_shift(self) -> None:
        """The shift-supervision loss must reach the frames that produced it."""
        ref = _smooth_image(b=1)
        d = torch.zeros(1, 3, 2, requires_grad=True)
        moving = fourier_shift(ref.expand(1, 3, 64, 64).contiguous(), d)
        estimate_subpixel_shifts(ref, moving).sum().backward()
        assert d.grad is not None and torch.isfinite(d.grad).all()

    def test_survives_the_pooling_the_simulator_applies(self) -> None:
        """Frames reach the estimator downsampled; shifts then read in LR pixels."""
        ref = _smooth_image(b=1, h=128, w=128, seed=3)
        true_hr = torch.tensor([[[1.0, -0.6], [-0.8, 0.4]]])
        moving_hr = fourier_shift(ref.expand(1, 2, 128, 128).contiguous(), true_hr)
        pool = torch.nn.functional.avg_pool2d
        est = estimate_subpixel_shifts(pool(ref, 2), pool(moving_hr, 2))
        assert (est - true_hr / 2).abs().max().item() < 0.05

    def test_rejects_multichannel_reference(self) -> None:
        with pytest.raises(ValueError, match="exactly 1 channel"):
            estimate_subpixel_shifts(torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))

    def test_rejects_spatial_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must share batch and spatial dims"):
            estimate_subpixel_shifts(torch.zeros(1, 1, 8, 8), torch.zeros(1, 2, 16, 16))


def test_recovery_needs_broadband_content() -> None:
    """A single-frequency image cannot be registered by this estimator, and must
    not silently pretend otherwise.

    The shift is read from the phase difference between ADJACENT frequency bins.
    A pure cosine occupies two isolated bins, so every adjacent pair has one
    empty member, the weights vanish, and the answer is numerical noise. Band-
    limited anatomy (and the Gaussian fiducial lattice) is broadband in this
    sense; this test documents the boundary of validity rather than asserting a
    number the method cannot deliver.
    """
    h = 32
    y = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1).expand(1, 1, h, h)
    pure = torch.cos(2 * math.pi * y / h)
    est_pure = estimate_subpixel_shifts(pure, fourier_shift(pure, torch.tensor([[[1.0, 0.0]]])))
    assert abs(est_pure[0, 0, 0].item() - 1.0) > 1e-3

    broadband = pure + 0.5 * torch.cos(4 * math.pi * y / h) + torch.rand(1, 1, h, h)
    est_bb = estimate_subpixel_shifts(
        broadband, fourier_shift(broadband, torch.tensor([[[1.0, 0.0]]]))
    )
    assert est_bb[0, 0, 0].item() == pytest.approx(1.0, abs=0.05)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_shift_runs_on_cuda_and_matches_cpu() -> None:
    """Regression: the phase ramp was built with a CPU scalar magnitude, which
    raised 'Expected all tensors to be on the same device' the moment the
    simulator ran on an accelerator. CPU-only testing could not see it."""
    x = _smooth_image(b=1)
    d = torch.tensor([[[0.37, -0.81]]])
    cpu = fourier_shift(x, d)
    gpu = fourier_shift(x.cuda(), d.cuda()).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4)


# ---------------------------------------------------------------------------
# 3-D and physical units (PR-B). ULF/HF volumes are anisotropic and the marker
# must be sized and measured in millimetres, not pixels.
# ---------------------------------------------------------------------------
def _smooth_volume(b=1, shape=(32, 32, 16), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.interpolate(
        torch.rand(b, 1, 8, 8, 8, generator=g),
        size=shape,
        mode="trilinear",
        align_corners=False,
    )


class TestNDimAndPhysicalUnits:
    def test_2d_transform_matches_the_fft_ops_convention(self) -> None:
        """`fft_ops` has no centred 3-D transform, so the convention is extended
        inside this module. That is only safe if the 2-D case is IDENTICAL to the
        SSOT, otherwise the codebase would carry two centring conventions."""
        from mriforge.infrastructure.physics.subpixel_registration import (
            centred_fft,
        )

        x = _smooth_image(b=2)
        assert torch.allclose(centred_fft(x), fft2c(x), atol=1e-5)

    def test_recovers_shifts_in_3d(self) -> None:
        ref = _smooth_volume()
        true = torch.tensor([[[0.4, -0.7, 0.9], [-1.2, 0.3, -0.5]]])
        moving = fourier_shift(ref.expand(1, 2, 32, 32, 16).contiguous(), true)
        est = estimate_subpixel_shifts(ref, moving)
        assert est.shape == (1, 2, 3)
        assert (est - true).abs().max().item() < 1e-3

    def test_millimetre_round_trip_on_real_ulf_anisotropy(self) -> None:
        """1.6 x 1.6 x 5.0 mm is the measured 64mT protocol. A shift of 3 mm is
        0.6 voxel through-plane and 1.9 voxels in-plane; only millimetres make
        those the same statement."""
        vox = (1.6, 1.6, 5.0)
        ref = _smooth_volume()
        d_mm = torch.tensor([[[1.0, -1.0, 3.0]]])
        moved = fourier_shift(ref, d_mm, voxel_mm=vox)
        est = estimate_subpixel_shifts(ref, moved, voxel_mm=vox)
        assert (est - d_mm).abs().max().item() < 0.01

    def test_pixel_and_millimetre_paths_agree(self) -> None:
        vox = (1.6, 1.6, 5.0)
        ref = _smooth_volume()
        d_mm = torch.tensor([[[1.6, -3.2, 5.0]]])
        d_px = torch.tensor([[[1.0, -2.0, 1.0]]])
        assert torch.allclose(
            fourier_shift(ref, d_mm, voxel_mm=vox), fourier_shift(ref, d_px), atol=1e-4
        )

    def test_isotropic_voxel_is_a_pure_rescale(self) -> None:
        ref = _smooth_image(b=1)
        d = torch.tensor([[[0.5, -0.25]]])
        est_px = estimate_subpixel_shifts(ref, fourier_shift(ref, d))
        est_mm = estimate_subpixel_shifts(
            ref, fourier_shift(ref, d), voxel_mm=(2.0, 2.0)
        )
        assert torch.allclose(est_mm, est_px * 2.0, atol=1e-5)

    def test_rejects_wrong_voxel_dimensionality(self) -> None:
        ref = _smooth_image(b=1)
        with pytest.raises(ValueError, match="voxel_mm has 3 entries"):
            fourier_shift(ref, torch.zeros(1, 1, 2), voxel_mm=(1.0, 1.0, 1.0))

    def test_rejects_non_positive_voxel(self) -> None:
        ref = _smooth_image(b=1)
        with pytest.raises(ValueError, match="voxel_mm must be positive"):
            fourier_shift(ref, torch.zeros(1, 1, 2), voxel_mm=(1.0, 0.0))

    def test_rejects_unsupported_rank(self) -> None:
        with pytest.raises(ValueError, match=r"expected \[B, C, H, W\]"):
            fourier_shift(torch.zeros(1, 1, 4, 4, 4, 4), torch.zeros(1, 1, 4))

    def test_rejects_rank_mismatch_between_reference_and_moving(self) -> None:
        with pytest.raises(ValueError, match="same rank"):
            estimate_subpixel_shifts(_smooth_image(b=1), _smooth_volume())



# ── shared centred N-D transform (PR-1) ───────────────────────────────────────


def test_centred_transforms_are_public_and_round_trip() -> None:
    """``band_partition`` needs the same centring convention. Exporting these
    rather than re-deriving them is what keeps ONE convention in the tree: a
    second, subtly different centred FFT is how a codebase ends up with two."""
    from mriforge.infrastructure.physics.subpixel_registration import (
        centred_fft,
        centred_freqs,
        centred_ifft,
        spatial_dims,
    )

    for shape in ((2, 3, 16, 16), (1, 2, 8, 8, 8)):
        x = torch.randn(*shape)
        assert torch.allclose(centred_ifft(centred_fft(x)).real, x, atol=1e-5)
        assert spatial_dims(x) == tuple(range(2, x.ndim))

    f = centred_freqs(8, torch.device("cpu"), torch.float32)
    assert float(f[4]) == pytest.approx(0.0)  # DC sits at n // 2
    assert float(f[0]) == pytest.approx(-0.5)
