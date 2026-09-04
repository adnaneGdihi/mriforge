"""Tests for the joint iterative B0 / image reconstructor.

Focus: the MB_CS undersampling mask must actually be applied. It was previously
accepted by ``_mb_recon`` but never used, so ``ReconMethod.MB_CS`` silently
solved the fully-sampled problem (collapsed to MB).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.iterative_recon import (
    IterativeReconstructor,
    ReconMethod,
    ReconParams,
)


@pytest.mark.physics
class TestMbReconSamplingMask:
    def _make(self) -> IterativeReconstructor:
        return IterativeReconstructor(
            method=ReconMethod.MB,
            fov_m=0.225,
            bw_hz=20_000.0,
            recon_params=ReconParams.for_simulation(),
        )

    def test_sampling_mask_changes_reconstruction(self) -> None:
        rec = self._make()
        torch.manual_seed(0)
        kspace = torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)
        b0 = torch.zeros(1, 1, 8, 8)

        full = rec._mb_recon(kspace, b0, t_shift=0.0, sampling_mask=None)
        half = torch.ones(8)
        half[1::2] = 0.0  # drop every other phase-encode line
        masked = rec._mb_recon(kspace, b0, t_shift=0.0, sampling_mask=half)

        # If the mask were inert, these would be identical (the old bug).
        assert not torch.allclose(full, masked, atol=1e-4)

    def test_all_ones_mask_equals_unmasked(self) -> None:
        rec = self._make()
        torch.manual_seed(1)
        kspace = torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)
        b0 = torch.zeros(1, 1, 8, 8)

        full = rec._mb_recon(kspace, b0, t_shift=0.0, sampling_mask=None)
        ones = rec._mb_recon(
            kspace, b0, t_shift=0.0, sampling_mask=torch.ones(8)
        )
        assert torch.allclose(full, ones, atol=1e-5)
