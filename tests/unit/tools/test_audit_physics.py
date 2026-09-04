"""Coverage for tools/audit_physics.py tier classification (WS-8).

Closes the audit_physics 0-test gap. Exercises the Tier-1/2/3 logic of
``check_physics_compliance`` including the documented ``dc_weight == 0``
correction.
"""

from __future__ import annotations

from typing import Any

from spectramr.tools.audit_physics import (
    TIER_1_GREEN,
    TIER_2_YELLOW,
    TIER_3_RED,
    check_physics_compliance,
)


def _config(
    *,
    dc_enabled: bool = False,
    dc_weight: float = 0.0,
    lambda_kspace: float = 0.0,
    lambda_frequency: float = 0.0,
    pinn: bool = False,
    divergence_free: bool = False,
    motion: bool = False,
) -> dict[str, Any]:
    return {
        "physics": {
            "data_consistency": {"enabled": dc_enabled, "weight": dc_weight},
            "pinn": {"enabled": pinn},
            "divergence_free": divergence_free,
            "motion_correction": {"enable_motion_correction": motion},
        },
        "objectives": {
            "reconstruction": {
                "lambda_kspace": lambda_kspace,
                "lambda_frequency": lambda_frequency,
            }
        },
    }


def test_tier1_dc_plus_kspace() -> None:
    tier, issues, _ = check_physics_compliance(
        _config(dc_enabled=True, dc_weight=1.0, lambda_kspace=0.5)
    )
    assert tier == TIER_1_GREEN
    assert issues == []


def test_tier2_dc_only_no_kspace() -> None:
    tier, issues, _ = check_physics_compliance(_config(dc_enabled=True, dc_weight=1.0))
    assert tier == TIER_2_YELLOW
    assert "No K-Space Loss" in issues


def test_tier3_no_physics() -> None:
    tier, issues, _ = check_physics_compliance(_config())
    assert tier == TIER_3_RED
    assert "No Physics Detected" in issues


def test_dc_weight_zero_is_inactive() -> None:
    # dc enabled but weight 0 must NOT count as active physics -> Tier 3.
    tier, _, details = check_physics_compliance(
        _config(dc_enabled=True, dc_weight=0.0)
    )
    assert tier == TIER_3_RED
    assert details["dc_weight"] == 0.0


def test_pinn_counts_as_special_physics() -> None:
    tier, _, details = check_physics_compliance(
        _config(dc_enabled=True, dc_weight=1.0, pinn=True)
    )
    assert tier == TIER_1_GREEN
    assert "PINN" in details["special_physics"]


def test_frequency_loss_satisfies_kspace_active() -> None:
    tier, _, _ = check_physics_compliance(
        _config(dc_enabled=True, dc_weight=1.0, lambda_frequency=0.3)
    )
    assert tier == TIER_1_GREEN
