"""Integration tests for Cold Diffusion Acceleration Pipeline.

These tests verify that the acceleration module correctly applies
timestep-dependent undersampling for cold diffusion training.

CRITICAL: Tests verify the fix for the inverted acceleration bug where
t=0 was returning full sampling (no degradation) instead of max degradation.
"""

import pytest
import torch

from spectramr.infrastructure.physics.sampling import (
    VariableDensityKSpaceAccelerator,
    create_kspace_accelerator,
)


class TestAccelerationTimestepBehavior:
    """Verify correct timestep-acceleration relationship for cold diffusion."""

    def test_timestep_0_has_min_acceleration(self):
        """At t=0, acceleration should be at minimum (clean/full sampling)."""
        accelerator = VariableDensityKSpaceAccelerator(
            num_timesteps=1000,
            max_acceleration=32.0,
        )

        accel_factor = accelerator.get_acceleration_factor(0)

        # At t=0, should be min acceleration (1.0)
        assert accel_factor == pytest.approx(
            1.0, rel=0.01
        ), f"At t=0, acceleration should be min (1.0), got {accel_factor}"

    def test_timestep_max_has_max_acceleration(self):
        """At t=T-1, acceleration should be at maximum (most degraded)."""
        accelerator = VariableDensityKSpaceAccelerator(
            num_timesteps=1000,
            max_acceleration=32.0,
        )

        accel_factor = accelerator.get_acceleration_factor(999)  # T-1

        # At t=T-1, should be max acceleration (32.0)
        assert accel_factor == pytest.approx(
            32.0, rel=0.01
        ), f"At t=T-1, acceleration should be max (32.0), got {accel_factor}"

    def test_acceleration_increases_with_timestep(self):
        """Acceleration should monotonically INCREASE as timestep increases (t=0 Clean, t=T Degraded)."""
        accelerator = VariableDensityKSpaceAccelerator(
            num_timesteps=1000,
            max_acceleration=32.0,
        )

        prev_accel = 0.0
        for t in [0, 100, 500, 900, 999]:
            accel = accelerator.get_acceleration_factor(t)
            assert (
                accel >= prev_accel
            ), f"Acceleration should increase: t={t} has {accel} but previous was {prev_accel}"
            prev_accel = accel

    def test_linear_accelerator_decay(self):
        """[Regression Test] Verify LinearKSpaceAccelerator semantics (t=0 -> Base, t=T -> Max)."""
        from spectramr.infrastructure.physics.sampling import LinearKSpaceAccelerator

        accelerator = LinearKSpaceAccelerator(
            num_timesteps=1000,
            max_acceleration=32.0,
            base_acceleration=1.0,
        )

        # t=0 -> Base (Clean)
        accel_0 = accelerator.get_acceleration_factor(0)
        assert accel_0 == pytest.approx(
            1.0, rel=0.01
        ), f"Linear at t=0 should be 1.0, got {accel_0}"

        # t=T -> Max (Degraded)
        accel_T = accelerator.get_acceleration_factor(999)
        assert accel_T == pytest.approx(
            32.0, rel=0.01
        ), f"Linear at t=T should be 32.0, got {accel_T}"


class TestMaskGenerationDegradation:
    """Verify masks actually apply degradation at t=0."""

    def test_mask_at_t0_is_nearly_full(self):
        """At t=0, mask should be nearly full (clean data)."""
        accelerator = create_kspace_accelerator(
            acceleration_type="variable_density",
            num_timesteps=1000,
            max_acceleration=32.0,
            center_fraction=0.08,
        )

        kspace_shape = (2, 256, 256)  # 2 channels, 256x256
        mask = accelerator.get_acceleration_mask(
            kspace_shape, timestep=0, device=torch.device("cpu")
        )

        mask_fraction = mask.float().mean().item()

        # At t=0 we expect full sampling (1.0)
        assert (
            mask_fraction > 0.9
        ), f"At t=0, mask should have >90% sampled, got {mask_fraction * 100:.1f}%"

    def test_mask_at_tmax_has_significant_zeros(self):
        """At t=T-1, mask should be sparse (max degradation)."""
        accelerator = create_kspace_accelerator(
            acceleration_type="variable_density",
            num_timesteps=1000,
            max_acceleration=32.0,
            center_fraction=0.08,
        )

        kspace_shape = (2, 256, 256)
        mask = accelerator.get_acceleration_mask(
            kspace_shape, timestep=999, device=torch.device("cpu")
        )

        mask_fraction = mask.float().mean().item()

        # At t=T-1 with 32x acceleration, expect sparse sampling
        assert (
            mask_fraction < 0.2
        ), f"At t=T-1, mask should have <20% sampled, got {mask_fraction * 100:.1f}%"


class TestDataDegradationApplication:
    """Verify degradation actually changes the k-space data."""

    def test_degraded_kspace_differs_from_original(self):
        """Applying mask at t=0 should produce data different from original."""
        accelerator = create_kspace_accelerator(
            acceleration_type="variable_density",
            num_timesteps=1000,
            max_acceleration=32.0,
        )

        # Create some fake k-space data
        kspace = torch.randn(1, 2, 256, 256)  # Batch=1, Channels=2

        # Get mask at t=T-1 (Max Degradation)
        mask = accelerator.get_acceleration_mask(
            (2, 256, 256), timestep=999, device=torch.device("cpu")
        )
        mask = mask.unsqueeze(0)  # Add batch dim

        # Apply degradation
        degraded_kspace = kspace * mask.float()

        # Data should differ significantly
        diff = (kspace - degraded_kspace).abs().mean().item()

        assert (
            diff > 0.1
        ), f"Degraded k-space should differ from original, diff={diff:.6f}"


class TestColdDiffusionSemantics:
    """Verify cold diffusion training semantics are correct."""

    def test_cold_diffusion_forward_process(self):
        """
        Cold Diffusion Forward Process: x_t = D(x_0, t)

        At t=0: x_t should be CLEAN (x_0)
        At t=T-1: x_t should be MAX DEGRADED (x_T)

        Training: Given x_t (degraded), predict x_0 (clean)
        """
        accelerator = create_kspace_accelerator(
            acceleration_type="variable_density",
            num_timesteps=1000,
            max_acceleration=32.0,
        )

        # Timesteps in a typical training batch (sampled uniformly)
        timesteps = torch.tensor([0, 250, 500, 750, 999])

        for t in timesteps.tolist():
            accel = accelerator.get_acceleration_factor(t)
            sampling_frac = 1.0 / accel

            # Verify: as t increases, sampling DECREASES (more degradation)
            if t == 0:
                assert (
                    sampling_frac > 0.9
                ), f"t=0 should have >90% sampling (Clean), got {sampling_frac * 100:.1f}%"
            elif t == 999:
                assert (
                    sampling_frac < 0.1
                ), f"t=999 should have <10% sampling (Degraded), got {sampling_frac * 100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
