"""SPECTRA unified-top routing contract (chip repo rtl/spectra/spectra_top.sv).

Asserts that every hardware-feature field of ``BackendAccelerationConfigSchema``
is routed to a backing MMIO register block in ``SPECTRA_REGISTER_MAP`` — the
guard that keeps the application-side config and the silicon's control surface
in lock-step. ``backend``/``runtime``/``cosim_bridge_addr`` are dispatch config,
not hardware-feature registers, so they are excluded.
"""

from __future__ import annotations

from spectramr.config.schemas.spectra import BackendAccelerationConfigSchema
from spectramr.infrastructure.accelerator.spectra.spectra_backend import (
    SPECTRA_CAPS_ADDR,
    SPECTRA_REGISTER_MAP,
)

# Fields that select/dispatch rather than configure a hardware feature.
_NON_FEATURE_FIELDS = {"backend", "runtime", "cosim_bridge_addr"}


def test_every_feature_field_has_a_register() -> None:
    fields = set(BackendAccelerationConfigSchema.model_fields) - _NON_FEATURE_FIELDS
    mapped = set(SPECTRA_REGISTER_MAP)
    missing = fields - mapped
    assert not missing, f"unrouted backend_acceleration fields: {sorted(missing)}"


def test_no_stale_register_entries() -> None:
    """Every map key is a real config field (catches renames)."""
    fields = set(BackendAccelerationConfigSchema.model_fields)
    stale = set(SPECTRA_REGISTER_MAP) - fields
    assert not stale, f"SPECTRA_REGISTER_MAP references non-existent fields: {sorted(stale)}"


def test_mesh_dims_are_caps_advertised() -> None:
    """Mesh dims are build-time fixed → discovered via SPECTRA_CAPS, not written."""
    for dim in ("mesh_rows", "mesh_cols", "lanes"):
        assert SPECTRA_REGISTER_MAP[dim] == "SPECTRA_CAPS"
    assert SPECTRA_CAPS_ADDR == 0x0FFC


def test_precision_policy_spans_path_a_and_spectra() -> None:
    """pe_mode_policy must touch Path-A PE_MODE and the spectra precision blocks."""
    target = SPECTRA_REGISTER_MAP["pe_mode_policy"]
    assert isinstance(target, list)
    assert "PE_MODE" in target and "SCALE_CTRL" in target and "CPLX_CTRL" in target
