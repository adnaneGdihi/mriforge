"""Unit tests for the Model-Based Image Reconstruction (MBIR) and markers."""

import math

import torch

from mriforge.infrastructure.physics.mbir import positive_contrast_demodulation


def test_positive_contrast_demodulation_scalar_te() -> None:
    """Verify demodulation works with a scalar float TE."""
    kspace = torch.ones(2, 4, 32, 32, dtype=torch.complex64)
    delta_omega_hz = 150.0
    te_sec = 0.010  # 10 ms

    result = positive_contrast_demodulation(kspace, delta_omega_hz, te_sec)

    expected_phase_shift = 2.0 * math.pi * delta_omega_hz * te_sec
    expected_factor = torch.exp(torch.tensor(1j * expected_phase_shift))

    assert torch.allclose(result, kspace * expected_factor)
    assert result.shape == kspace.shape


def test_positive_contrast_demodulation_tensor_te() -> None:
    """Verify demodulation works with a 1D tensor of readout times."""
    kspace = torch.ones(2, 4, 100, dtype=torch.complex64)  # e.g. [B, C, K]
    delta_omega_hz = 150.0
    te_sec = torch.linspace(0.001, 0.010, 100)

    result = positive_contrast_demodulation(kspace, delta_omega_hz, te_sec)

    phase_shift_first = 2.0 * math.pi * delta_omega_hz * 0.001
    factor_first = torch.exp(torch.tensor(1j * phase_shift_first))
    assert torch.allclose(result[0, 0, 0], factor_first)

    phase_shift_last = 2.0 * math.pi * delta_omega_hz * 0.010
    factor_last = torch.exp(torch.tensor(1j * phase_shift_last))
    assert torch.allclose(result[0, 0, -1], factor_last)
    assert result.shape == kspace.shape
