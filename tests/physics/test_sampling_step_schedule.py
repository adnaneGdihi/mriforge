import math

import pytest
import torch

from spectramr.infrastructure.physics.sampling import UniformCartesianKSpaceAccelerator


def test_linear_schedule_acceleration_factors():
    """Verify linear progression of acceleration factor from base to max."""
    timesteps = 1000
    base_accel = 1.0
    max_accel = 8.0

    accelerator = UniformCartesianKSpaceAccelerator(
        num_timesteps=timesteps,
        acceleration_schedule="linear",
        base_acceleration=base_accel,
        max_acceleration=max_accel,
    )

    # t=0 should be exactly base_accel
    assert accelerator.get_acceleration_factor(0) == pytest.approx(base_accel)

    # t=T-1 should be exactly max_accel
    assert accelerator.get_acceleration_factor(timesteps - 1) == pytest.approx(max_accel)

    # Midpoint
    mid_t = (timesteps - 1) // 2
    actual_ratio = mid_t / (timesteps - 1)
    span = max_accel - base_accel
    expected_mid = base_accel + span * actual_ratio
    assert accelerator.get_acceleration_factor(mid_t) == pytest.approx(expected_mid)


def test_power_law_schedule_acceleration_factors():
    """Verify polynomial/power_law progression."""
    timesteps = 1000
    base_accel = 1.0
    max_accel = 8.0
    power = 2.0

    accelerator = UniformCartesianKSpaceAccelerator(
        num_timesteps=timesteps,
        acceleration_schedule="power_law",
        schedule_power=power,
        base_acceleration=base_accel,
        max_acceleration=max_accel,
    )

    # Boundaries
    assert accelerator.get_acceleration_factor(0) == pytest.approx(base_accel)
    assert accelerator.get_acceleration_factor(timesteps - 1) == pytest.approx(max_accel)

    mid_t = (timesteps - 1) // 2
    actual_ratio = mid_t / (timesteps - 1)
    span = max_accel - base_accel
    expected_mid = base_accel + (actual_ratio**power) * span
    assert accelerator.get_acceleration_factor(mid_t) == pytest.approx(expected_mid)


def test_exponential_schedule_acceleration_factors():
    """Verify exponential progression mapping."""
    timesteps = 1000
    base_accel = 1.0
    max_accel = 8.0

    accelerator = UniformCartesianKSpaceAccelerator(
        num_timesteps=timesteps,
        acceleration_schedule="exponential",
        base_acceleration=base_accel,
        max_acceleration=max_accel,
    )

    # Boundaries
    assert accelerator.get_acceleration_factor(0) == pytest.approx(base_accel)
    assert accelerator.get_acceleration_factor(timesteps - 1) == pytest.approx(max_accel)

    # Exponential math check: ratio = (exp(t_ratio * 5.0) - 1.0) / (exp(5.0) - 1.0)
    mid_t = (timesteps - 1) // 2
    t_ratio = float(mid_t) / (timesteps - 1)

    exp_ratio = (math.exp(t_ratio * 5.0) - 1.0) / (math.exp(5.0) - 1.0)
    span = max_accel - base_accel
    expected_val = base_accel + exp_ratio * span

    assert accelerator.get_acceleration_factor(mid_t) == pytest.approx(expected_val)


def test_step_schedule_acceleration_factors():
    """Verify step-wise cascading schedule."""
    accel_range = [1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0]
    timesteps = 1000

    accelerator = UniformCartesianKSpaceAccelerator(
        num_timesteps=timesteps,
        acceleration_schedule="step",
        acceleration_range=accel_range,
        center_fraction=0.08,
    )

    steps = len(accel_range)

    for t in range(timesteps):
        ratio = t / (timesteps - 1)
        expected_idx = min(int(ratio * steps), steps - 1)
        expected_factor = accel_range[expected_idx]

        actual_factor = accelerator.get_acceleration_factor(t)
        assert actual_factor == pytest.approx(expected_factor), f"Timestep {t}: Expected {expected_factor}, got {actual_factor}"


def test_default_step_schedule_fallback():
    """Verify fallback behavior when range is not provided."""
    timesteps = 1000
    base_accel = 1.2
    max_accel = 6.4

    accelerator = UniformCartesianKSpaceAccelerator(
        num_timesteps=timesteps,
        acceleration_schedule="step",
        base_acceleration=base_accel,
        max_acceleration=max_accel,
        # Intentionally empty / no range provided to test fallback
        acceleration_range=[],
    )

    assert accelerator.get_acceleration_factor(0) == pytest.approx(base_accel)
    assert accelerator.get_acceleration_factor(499) == pytest.approx(base_accel)  # ratio < 0.5
    assert accelerator.get_acceleration_factor(500) == pytest.approx(max_accel)  # ratio >= 0.5
    assert accelerator.get_acceleration_factor(999) == pytest.approx(max_accel)


@pytest.mark.parametrize(
    "schedule, timesteps",
    [("linear", 100), ("power_law", 500), ("exponential", 250), ("step", 200)],
)
def test_mask_fractions_adhere_to_level(schedule, timesteps):
    """
    Generate actual masks across different points (early, mid, late)
    in the diffusion process and calculate exactly what percentage
    was sampled. Should align with the expected acceleration.
    """
    shape = (1, 128, 128)
    center_fraction = 0.08
    accel_range = [1.0, 2.0, 4.0, 8.0]

    accelerator = UniformCartesianKSpaceAccelerator(
        num_timesteps=timesteps,
        acceleration_schedule=schedule,
        acceleration_range=accel_range,
        base_acceleration=1.0,
        max_acceleration=8.0,
        center_fraction=center_fraction,
        seed=42,
    )

    device = torch.device("cpu")

    # Check at 3 critical junctions: early (5%), mid (50%), late (95%)
    test_points = [
        int(timesteps * 0.05),
        int(timesteps * 0.50),
        int(timesteps * 0.95),
    ]

    for t in test_points:
        target_accel = accelerator.get_acceleration_factor(t)

        mask = accelerator.get_acceleration_mask(shape, t, device=device)

        total_elements = mask.shape[1] * mask.shape[2]
        sampled_elements = mask.sum().item()

        actual_fraction = sampled_elements / total_elements
        expected_fraction = 1.0 / target_accel

        # Center fraction provides a hard floor to undersampling
        effective_expected = max(expected_fraction, center_fraction)

        # 10% allowed variance because:
        # a) specific lines are chosen, rounding errors happen
        # b) Autocalibration region forces non-uniform structure
        assert (
            abs(actual_fraction - effective_expected) < 0.10
        ), f"Schedule={schedule} at t={t}. Expected Area {effective_expected:.3f} but got {actual_fraction:.3f}"
