"""Every family must mean the same thing by an acceleration config.

Issue #957: ``get_acceleration_factor`` reads the floor as
``getattr(self, "base_acceleration", 1.0)``, and only two of sixteen accelerators
ever stored it. For the other fourteen the configured value was swallowed by
``**kwargs`` and silently replaced by 1.0 — so ``base_acceleration: 4`` meant
"fully sampled at t=0", which is the degenerate identity task issue #535 exists
to prevent.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.physics.sampling import (
    SUPPORTED_ACCELERATION_TYPES,
    create_kspace_accelerator,
)

MATRIX = 128

#: Families that legitimately do not track ``base_acceleration``, each for a
#: reason that is a design choice rather than an oversight. Keep this list short
#: and justified; a family added here without a reason is a bug in disguise.
EXEMPT = {
    # base_acceleration is a read-only property aliasing min_acceleration.
    "fractional_variable_density",
    # A half-scan samples a fixed contiguous block; it is not an acceleration
    # ladder and cannot honour an arbitrary R.
    "partial_fourier",
}


def _realised_acceleration(family: str, base: float) -> float:
    accelerator = create_kspace_accelerator(
        acceleration_type=family,
        num_timesteps=8,
        base_acceleration=base,
        max_acceleration=32.0,
        center_fraction=0.02,
        min_center_fraction=0.02,
        seed=42,
        acceleration_schedule="linear",
    )
    mask = accelerator.get_acceleration_mask((1, MATRIX, MATRIX), 0)[0]
    fraction = float(mask.float().mean())
    return 1.0 / fraction if fraction else float("inf")


@pytest.mark.parametrize("family", sorted(set(SUPPORTED_ACCELERATION_TYPES) - EXEMPT))
def test_base_acceleration_is_honoured_at_t0(family: str) -> None:
    """``base_acceleration: 4`` must mean 4x at t=0, for every family.

    The tolerance is loose because trajectory families quantise to whole
    spokes/arms; it is tight enough to separate "4x" from "not undersampled".
    """
    realised = _realised_acceleration(family, 4.0)
    assert realised == pytest.approx(4.0, rel=0.35), (
        f"{family} realises {realised:.1f}x where base_acceleration asked for 4x"
    )


@pytest.mark.parametrize("family", sorted(set(SUPPORTED_ACCELERATION_TYPES) - EXEMPT))
def test_t0_is_not_the_identity_when_a_floor_is_declared(family: str) -> None:
    """The whole point of base_acceleration (#535): t=0 must not be a no-op."""
    assert _realised_acceleration(family, 4.0) > 1.5, (
        f"{family} is (near) fully sampled at t=0 despite base_acceleration=4"
    )


def test_exempt_families_are_exempt_for_a_reason() -> None:
    """Guard the exemption list against quiet growth."""
    assert {"fractional_variable_density", "partial_fourier"} == EXEMPT, (
        "adding a family here needs a documented design reason, not a failing test"
    )
