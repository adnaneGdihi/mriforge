"""Unit tests for the Hamiltonian-acquisition Pydantic schema (B-4 / Bundle P7)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.hamiltonian_acquisition import (
    HamiltonianAcquisitionConfig,
)


def test_defaults_are_valid():
    cfg = HamiltonianAcquisitionConfig()
    assert cfg.potential_arch == "siren"
    assert cfg.integrator == "leapfrog"
    assert cfg.dt_seconds == 4e-6
    assert cfg.G_max_mT_per_m == 40.0
    assert cfg.S_max_T_per_m_per_s == 200.0


def test_frozen_and_extra_forbid():
    cfg = HamiltonianAcquisitionConfig()
    # frozen
    with pytest.raises(ValidationError):
        cfg.dt_seconds = 1e-5  # type: ignore[misc]
    # extra forbidden
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(unknown_field=1)


def test_hardware_bounds_clamped_to_clinical_envelope():
    """G_max > 80 mT/m or S_max > 500 T/m/s must be rejected (Tier-1 intent)."""
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(G_max_mT_per_m=100.0)
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(S_max_T_per_m_per_s=600.0)


def test_positive_constraints():
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(dt_seconds=0.0)
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(T_total_seconds=-1.0)
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(samples_per_shot=1)


def test_both_aux_weights_zero_rejected():
    """A trajectory arm with no physics penalty is a misconfiguration."""
    with pytest.raises(ValidationError, match="at least one physics penalty"):
        HamiltonianAcquisitionConfig(
            energy_conservation_weight=0.0, hardware_penalty_weight=0.0
        )


def test_one_aux_weight_active_ok():
    cfg = HamiltonianAcquisitionConfig(
        energy_conservation_weight=0.0, hardware_penalty_weight=0.1
    )
    assert cfg.hardware_penalty_weight == 0.1


def test_unknown_arch_or_integrator_rejected():
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(potential_arch="quantum")
    with pytest.raises(ValidationError):
        HamiltonianAcquisitionConfig(integrator="magic")
