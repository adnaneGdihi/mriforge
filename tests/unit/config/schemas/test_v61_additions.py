"""Tests for the v6.1 schema bump.

Targets ``spectramr.config.schemas.physics`` (new sub-schemas) and
``spectramr.config.schemas.training.base`` (config_version validator).

v6.1 is a strict additive superset of v6.0 — every v6.0 YAML loads
unchanged. See ``TODO/integration_plan_ulf_cheap_fast_mri.md §0.1``.

Categories:

- ``ConcomitantFieldConfig`` defaults and constraints
- ``RelaxationPriorsConfig`` defaults
- ``PhysicsConfigSchema`` exposes the new sub-blocks
- ``BaseTrainingConfigSchema.validate_config_version`` accepts ``6.0``
  and ``6.1``, rejects ``5.0``, ``7.0``, etc.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.physics import (
    ConcomitantFieldConfig,
    PhysicsConfigSchema,
    RelaxationPriorsConfig,
)


# ---------------------------------------------------------------------------
# ConcomitantFieldConfig
# ---------------------------------------------------------------------------


def test_concomitant_field_defaults_disabled() -> None:
    """Default state is ``enabled=False`` (v6.0 parity)."""
    cfg = ConcomitantFieldConfig()
    assert cfg.enabled is False
    assert cfg.max_gradient_T_per_m == 0.040
    assert cfg.waveform_uri is None


def test_concomitant_field_zero_gradient_rejected() -> None:
    """``max_gradient_T_per_m`` must be > 0."""
    with pytest.raises(ValidationError):
        ConcomitantFieldConfig(max_gradient_T_per_m=0.0)


def test_concomitant_field_frozen() -> None:
    """Schema is frozen — mutation raises."""
    cfg = ConcomitantFieldConfig()
    with pytest.raises(ValidationError):
        cfg.enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RelaxationPriorsConfig
# ---------------------------------------------------------------------------


def test_relaxation_priors_defaults_disabled() -> None:
    """Default state is fully disabled."""
    cfg = RelaxationPriorsConfig()
    assert cfg.enabled is False
    assert cfg.t1_table_uri is None
    assert cfg.t2_table_uri is None


def test_relaxation_priors_round_trip() -> None:
    """Custom URIs persist through construction."""
    cfg = RelaxationPriorsConfig(
        enabled=True,
        t1_table_uri="data/priors/t1.json",
        t2_table_uri="data/priors/t2.json",
    )
    assert cfg.enabled is True
    assert cfg.t1_table_uri == "data/priors/t1.json"


# ---------------------------------------------------------------------------
# PhysicsConfigSchema integration
# ---------------------------------------------------------------------------


def test_physics_config_exposes_v61_subblocks() -> None:
    """``PhysicsConfigSchema`` now has the v6.1 sub-blocks."""
    physics = PhysicsConfigSchema()
    assert isinstance(physics.concomitant, ConcomitantFieldConfig)
    assert isinstance(physics.relaxation_priors, RelaxationPriorsConfig)
    assert physics.gradient_waveform_uri is None


def test_physics_config_v60_yaml_still_loads() -> None:
    """A v6.0-shaped physics dict still validates (backward-compat)."""
    # Just the v6.0 sub-set; the v6.1 fields take their defaults.
    physics = PhysicsConfigSchema(field_strength=0.064)
    assert physics.field_strength == 0.064
    assert physics.concomitant.enabled is False  # default-disabled


# ---------------------------------------------------------------------------
# config_version validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["6.0", "6.1"])
def test_the_6x_versions_are_now_rejected(version: str) -> None:
    """These were accepted until 2026-08-08, then RETIRED.

    The additive-superset claims in this module's docstring are still true as
    history — v6.1 was a strict superset of v6.0, and the sub-blocks it added
    are still here and still tested above. What changed is the version gate:
    `LEGACY_CONFIG_VERSIONS` is empty, so the only accepted spelling is the
    canonical one and both 6.x versions now raise like any other unsupported
    value.
    """
    from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

    with pytest.raises(ValueError, match="Unsupported config_version"):
        BaseTrainingConfigSchema.validate_config_version(version)


def test_the_canonical_version_is_accepted() -> None:
    """Anti-vacuity for the test above: the validator is not rejecting
    everything."""
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

    assert (
        BaseTrainingConfigSchema.validate_config_version(CANONICAL_CONFIG_VERSION)
        == CANONICAL_CONFIG_VERSION
    )


def test_v50_rejected() -> None:
    """``config_version: '5.0'`` raises."""
    from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

    with pytest.raises(ValueError, match="Unsupported config_version"):
        BaseTrainingConfigSchema.validate_config_version("5.0")


def test_v70_rejected() -> None:
    """Future versions also rejected (forces an explicit bump)."""
    from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

    with pytest.raises(ValueError, match="Unsupported config_version"):
        BaseTrainingConfigSchema.validate_config_version("7.0")
