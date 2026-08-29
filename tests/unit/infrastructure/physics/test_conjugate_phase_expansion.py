"""Unit tests for Conjugate-Phase reconstruction (CPR).

Covers :class:`mriforge.infrastructure.physics.conjugate_phase.ConjugatePhaseReconstructor`
which implements the multifrequency-interpolation (MFI) variant of CPR
from Koolstra et al. (2021) §2.4.

Test strategy
-------------
With ``b0 = 0`` the CPR reconstructor must reduce to the centred adjoint
DFT — i.e. :func:`mriforge.infrastructure.physics.fft_ops.ifft2c`. With a
non-zero, spatially-varying B0 map the MFI bins must be used: the
reconstruction differs from a plain IFFT.

Roundtrip on a synthetic phase ramp: starting from a complex image
``x = magnitude · exp(i · phase)``, mapping to k-space (no B0), then
reconstructing with the CPR engine, must recover ``x`` up to FFT
scaling.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

cp_mod = pytest.importorskip("mriforge.infrastructure.physics.conjugate_phase")
fft_mod = pytest.importorskip("mriforge.infrastructure.physics.fft_ops")
ConjugatePhaseReconstructor = cp_mod.ConjugatePhaseReconstructor
fft2c = fft_mod.fft2c
ifft2c = fft_mod.ifft2c


def _make_reconstructor(n: int = 16, bw: float = 20_000.0) -> ConjugatePhaseReconstructor:
    """Small reconstructor with a narrow frequency range — keeps the test fast."""
    return ConjugatePhaseReconstructor(
        freq_range_hz=(-100.0, 100.0),
        freq_step_hz=10.0,
        bw_hz=bw,
        n_readout=n,
    )


class TestReconstructorFrequencyBins:
    """``n_bins`` is determined by the (range, step) pair."""

    def test_n_bins_count(self):
        r = ConjugatePhaseReconstructor(
            freq_range_hz=(-100.0, 100.0),
            freq_step_hz=10.0,
            bw_hz=20_000.0,
            n_readout=16,
        )
        # 21 bins: -100, -90, ..., 0, ..., 90, 100
        assert r.n_bins == 21

    def test_dwell_time_matches_bandwidth(self):
        r = ConjugatePhaseReconstructor(bw_hz=20_000.0)
        assert r.dwell == pytest.approx(1.0 / 20_000.0)


class TestZeroB0ReducesToIFFT:
    """With ``b0 = 0`` the MFI correction collapses to a centred IFFT."""

    def test_zero_b0_matches_ifft2c(self, deterministic_seed):
        n = 16
        recon = _make_reconstructor(n=n)

        # Synthetic complex image with non-trivial phase.
        magnitude = torch.rand(n, n)
        phase = torch.rand(n, n) * 0.5  # bounded so we are well below π
        x = (magnitude * torch.exp(1j * phase)).to(torch.complex64)
        x_bchw = x.unsqueeze(0).unsqueeze(0)

        kspace = fft2c(x_bchw)
        b0 = torch.zeros(1, 1, n, n)

        recon_image = recon.reconstruct(kspace, b0).squeeze()
        ifft_image = ifft2c(kspace).squeeze()

        # Both must agree because the MFI bin at f=0 is selected for every voxel.
        assert torch.allclose(recon_image, ifft_image, atol=1e-4)

    def test_phase_preserved_when_b0_zero(self, deterministic_seed):
        """With ``b0 = 0`` the output phase must match the IFFT phase exactly."""
        n = 16
        recon = _make_reconstructor(n=n)

        magnitude = torch.ones(n, n)
        phase = torch.linspace(-0.3, 0.3, n * n).reshape(n, n)
        x = (magnitude * torch.exp(1j * phase)).to(torch.complex64)
        x_bchw = x.unsqueeze(0).unsqueeze(0)

        kspace = fft2c(x_bchw)
        b0 = torch.zeros(1, 1, n, n)

        recon_image = recon.reconstruct(kspace, b0).squeeze()
        ifft_image = ifft2c(kspace).squeeze()

        # Compare wrapped phases on voxels with non-negligible magnitude.
        mask = recon_image.abs() > 1e-4
        assert mask.any()
        phase_recon = torch.angle(recon_image[mask])
        phase_ifft = torch.angle(ifft_image[mask])
        diff = ((phase_recon - phase_ifft + torch.pi) % (2 * torch.pi)) - torch.pi
        assert diff.abs().max() < 1e-4


class TestRoundtripOnPhaseRamp:
    """End-to-end roundtrip ``image → k-space → CPR → image``."""

    def test_roundtrip_recovers_input(self, deterministic_seed):
        n = 16
        recon = _make_reconstructor(n=n)

        # ``x = magnitude * exp(i * phase_ramp)`` is the canonical CPR test input.
        magnitude = torch.rand(n, n) + 0.1   # avoid exact zeros
        phase_ramp = torch.linspace(-0.5, 0.5, n * n).reshape(n, n)
        x = (magnitude * torch.exp(1j * phase_ramp)).to(torch.complex64)
        x_bchw = x.unsqueeze(0).unsqueeze(0)

        kspace = fft2c(x_bchw)
        b0 = torch.zeros(1, 1, n, n)

        recon_image = recon.reconstruct(kspace, b0)
        assert recon_image.shape == (1, 1, n, n)
        assert recon_image.dtype == torch.complex64
        assert torch.allclose(recon_image.squeeze(), x, atol=1e-4)


class TestNonZeroB0DiffersFromIFFT:
    """With non-zero B0 the MFI bins activate and the result diverges from IFFT."""

    def test_uniform_b0_changes_phase(self, deterministic_seed):
        n = 16
        recon = _make_reconstructor(n=n)

        x = torch.randn(1, 1, n, n, dtype=torch.complex64)
        kspace = fft2c(x)
        b0 = torch.full((1, 1, n, n), 50.0)   # uniform off-resonance

        recon_image = recon.reconstruct(kspace, b0).squeeze()
        ifft_image = ifft2c(kspace).squeeze()

        # Reconstruction with non-trivial B0 must differ from a plain IFFT.
        assert not torch.allclose(recon_image, ifft_image, atol=1e-3)


class TestReconstructorContract:
    """Shape / dtype contract of the public ``reconstruct`` method."""

    def test_output_shape_and_dtype(self, deterministic_seed):
        n = 8
        recon = _make_reconstructor(n=n)
        kspace = torch.randn(1, 1, n, n, dtype=torch.complex64)
        b0 = torch.zeros(1, 1, n, n)

        out = recon.reconstruct(kspace, b0)

        assert out.shape == (1, 1, n, n)
        assert out.dtype == torch.complex64
