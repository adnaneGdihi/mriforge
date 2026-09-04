"""Regression: ``NoiseSimulator.add_rician_noise`` must use the same √2 split
as ``NoiseSimulator.forward``.

``forward`` builds complex noise then scales by ``sigma / sqrt(2)`` so each
quadrature has std ``sigma/√2`` and the total complex noise power is ``sigma²``
(the amplitude-SNR convention ``sigma = signal / 10**(SNR/20)``).
``add_rician_noise`` instead put ``sigma`` on EACH quadrature → total power
``2·sigma²``, i.e. √2 (≈3 dB) too much noise for the same nominal SNR, and
inconsistent with ``forward`` in the same class.

At high SNR ``|c + n| - c ≈ Re(n)``, so the realized magnitude-noise std should
match the per-quadrature std ``sigma/√2``.
"""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.noise_simulation import NoiseSimulator


def test_add_rician_noise_uses_sqrt2_split():
    torch.manual_seed(0)
    sim = NoiseSimulator()
    c = 1.0
    snr_db = 30.0
    img = torch.full((1, 1, 256, 256), c)

    out = sim.add_rician_noise(img, snr=snr_db)

    sigma = c / (10 ** (snr_db / 20.0))  # quantile of a constant image == c
    expected_std = sigma / math.sqrt(2)  # per-quadrature std
    realized_std = (out - c).std().item()

    # Pre-fix realized_std ≈ sigma (≈ 0.0316); post-fix ≈ sigma/√2 (≈ 0.0224).
    # 12% tolerance cleanly separates the two (they differ by ~29%).
    assert realized_std == 0 or abs(realized_std - expected_std) / expected_std < 0.12, (
        f"realized {realized_std:.5f} vs expected {expected_std:.5f} "
        f"(sigma={sigma:.5f}); add_rician_noise injects √2 too much noise"
    )
