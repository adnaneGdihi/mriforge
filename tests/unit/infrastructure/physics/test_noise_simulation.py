"""Property tests for ``NoiseSimulator``.

Targets ``mriforge.infrastructure.physics.noise_simulation``. Verifies the SNR
contract: doubling the linear SNR (i.e. +6 dB) halves the noise standard
deviation, within Monte-Carlo tolerance over a large constant signal.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.noise_simulation import NoiseSimulator


def _measured_noise_sigma(
    simulator: NoiseSimulator, signal_level: float, n_samples: int = 4096
) -> float:
    """Empirical residual std after applying the simulator to a constant signal."""
    torch.manual_seed(0)
    signal = torch.full(
        (n_samples, 1, 1, 1), signal_level, dtype=torch.complex64
    )
    out = simulator(signal)
    residual = (out - signal).flatten()
    if torch.is_complex(residual):
        residual = torch.view_as_real(residual).flatten()
    return residual.std().item()


def test_doubling_snr_halves_noise_sigma() -> None:
    """+6 dB → noise std should fall by ≈0.5x within 10% MC tolerance.

    With ``snr_range=(s, s)`` (degenerate), the simulator's per-batch SNR draw
    is deterministic. ``signal_level`` is read from the 95th-percentile of
    ``|signal|``, which for a constant signal is exactly that constant.
    Therefore ``sigma_noise = signal_level / 10**(snr_dB/20)``.
    """
    sig = 8.0
    sigma_low = _measured_noise_sigma(
        NoiseSimulator(snr_range=(20.0, 20.0), noise_type="gaussian"),
        signal_level=sig,
    )
    sigma_high = _measured_noise_sigma(
        NoiseSimulator(snr_range=(26.0, 26.0), noise_type="gaussian"),
        signal_level=sig,
    )
    ratio = sigma_high / sigma_low
    # 6 dB amplitude → factor of 10**(6/20) = 1.995 in SNR_linear → 0.501 in sigma.
    assert 0.45 < ratio < 0.56, (
        f"Expected sigma ratio ≈0.5 for +6 dB, got {ratio:.3f}"
    )


def test_sigma_matches_analytical_form() -> None:
    """Empirical σ should match ``signal / (SNR_linear · √2)`` analytically.

    The simulator scales complex Gaussian noise by ``sigma/√2`` per component
    (real and imaginary), so when we measure std on the interleaved real+imag
    residual we get ``signal_level / SNR_linear / √2``.
    """
    sig = 4.0
    snr_db = 30.0
    expected_sigma = sig / 10 ** (snr_db / 20.0) / math.sqrt(2)
    measured_sigma = _measured_noise_sigma(
        NoiseSimulator(snr_range=(snr_db, snr_db), noise_type="gaussian"),
        signal_level=sig,
    )
    rel_err = abs(measured_sigma - expected_sigma) / expected_sigma
    assert rel_err < 0.10, (
        f"Measured σ={measured_sigma:.4f}, expected≈{expected_sigma:.4f} "
        f"(rel_err={rel_err:.3f})"
    )


def test_rician_output_is_real_nonnegative() -> None:
    """Rician noise on complex input yields a real, non-negative magnitude."""
    torch.manual_seed(1)
    signal = torch.randn(4, 1, 16, 16, dtype=torch.complex64) * 3.0
    sim = NoiseSimulator(snr_range=(20.0, 20.0), noise_type="rician")
    out = sim(signal)
    assert not torch.is_complex(out), "Rician output must be real magnitude"
    assert (out >= 0).all(), "Rician magnitude must be non-negative"
    assert torch.isfinite(out).all()


def test_zero_snr_floor_does_not_crash() -> None:
    """The +1e-8 floor in sigma protects against div-by-zero at SNR≈0 dB."""
    signal = torch.ones(2, 1, 8, 8, dtype=torch.complex64)
    sim = NoiseSimulator(snr_range=(0.0, 0.0), noise_type="gaussian")
    out = sim(signal)
    assert torch.isfinite(out).all()
