"""Tests for the ``cartesian_peripheral`` sampling mask.

Locks Phase A.3 of ``TODO/backlog_baseline_replication_experiment_11.md``:
the FDB (Frequency-Decomposed Bridge) baseline acquires peripheral
high-frequency lines first, so the mask generator must offer a
first-class peripheral pattern (not a silent string fallback).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.sampling import (
    MaskType,
    create_mask_generator,
)


def test_cartesian_peripheral_is_a_first_class_enum_value() -> None:
    """The mask type appears in ``MaskType`` (CLAUDE.md #9 SSOT)."""
    assert MaskType("cartesian_peripheral") is MaskType.CARTESIAN_PERIPHERAL


def test_peripheral_mask_skips_center_band() -> None:
    """The center band (low-frequency lines) is unsampled."""
    gen = create_mask_generator(seed=0)
    mask = gen.generate_mask(
        mask_type=MaskType.CARTESIAN_PERIPHERAL,
        shape=(64, 64),
        acceleration_factor=4.0,
        center_skip_fraction=0.16,
    )
    H = mask.shape[0]
    center_lines = max(1, int(round(0.16 * H)))
    center_start = (H - center_lines) // 2
    center_end = center_start + center_lines

    # All lines inside the center band have mask sum 0 (no sampling).
    center_band = mask[center_start:center_end]
    assert center_band.sum().item() == 0.0


def test_peripheral_mask_samples_outside_center() -> None:
    """Peripheral lines (high |ky|) carry non-zero sampling."""
    gen = create_mask_generator(seed=0)
    mask = gen.generate_mask(
        mask_type=MaskType.CARTESIAN_PERIPHERAL,
        shape=(64, 64),
        acceleration_factor=4.0,
    )
    sampled_lines = (mask.sum(dim=-1) > 0).int().sum().item()
    assert sampled_lines > 0


@pytest.mark.parametrize("acceleration", (2.0, 4.0, 8.0))
def test_peripheral_mask_acceleration_inverse_to_count(acceleration: float) -> None:
    """Higher acceleration → fewer sampled peripheral lines, monotonically."""
    gen = create_mask_generator(seed=0)
    mask = gen.generate_mask(
        mask_type=MaskType.CARTESIAN_PERIPHERAL,
        shape=(128, 128),
        acceleration_factor=acceleration,
    )
    sampled = (mask.sum(dim=-1) > 0).int().sum().item()
    # With H=128 and ~8% center skipped, peripheral count ~ 118 lines.
    # Step = round(R) → expected sampled ≈ 118 / R within rounding slack.
    expected = 118 / acceleration
    assert abs(sampled - expected) < 12, (
        f"R={acceleration}: sampled={sampled}, expected~{expected}"
    )


def test_peripheral_mask_shape_matches_request() -> None:
    """Output shape matches the requested ``(H, W)``."""
    gen = create_mask_generator(seed=0)
    mask = gen.generate_mask(
        mask_type=MaskType.CARTESIAN_PERIPHERAL,
        shape=(64, 96),
        acceleration_factor=4.0,
    )
    assert mask.shape == (64, 96)


def test_peripheral_mask_3d_shape_with_coil_dim() -> None:
    """``shape=(C, H, W)`` broadcasts mask across the coil dimension."""
    gen = create_mask_generator(seed=0)
    mask = gen.generate_mask(
        mask_type=MaskType.CARTESIAN_PERIPHERAL,
        shape=(8, 64, 64),
        acceleration_factor=4.0,
    )
    assert mask.shape == (8, 64, 64)
    # Same pattern on every coil.
    assert torch.allclose(mask[0], mask[7])
