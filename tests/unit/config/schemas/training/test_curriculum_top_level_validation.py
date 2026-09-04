"""Tests for the diffusion-curriculum field placement on
``TrainingStrategyConfigSchema``.

History (Phase 8, ``TODO/backlog_ssot_and_layering_cleanup.md``):
Real experiment YAMLs set ``curriculum_ramp_rate`` and
``curriculum_start_timestep`` at the ``training:`` top level (e.g.
``experiment_11_kspace_cold_diffusion.yaml``), but they were only
*declared* on the diffusion sub-schema. They survived only because
``TrainingStrategyConfigSchema.model_config["extra"] == "allow"``, so
the values were accepted without validation -- a typo like
``curriculum_ramp_rate: "oops"`` would be silently swallowed.

Phase 8 promoted both fields to ``TrainingStrategyConfigSchema`` so they
get type + range validation at the same level the YAMLs use. The
strategy reads them via ``self.config.training.curriculum_*`` -- no
strategy-code change was needed.

This test pins the validation contract so a future contributor who
removes the declarations regresses the silent-acceptance behaviour.
"""
from __future__ import annotations

import pytest

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema


def test_curriculum_fields_validate_at_top_level() -> None:
    """Valid curriculum values must instantiate without error."""
    cfg = TrainingStrategyConfigSchema(
        curriculum_start_timestep=50,
        curriculum_ramp_rate=0.01,
    )
    assert cfg.curriculum_start_timestep == 50
    assert cfg.curriculum_ramp_rate == 0.01


def test_curriculum_ramp_rate_rejects_wrong_type() -> None:
    """A string ramp-rate must raise ValidationError, not survive via extra='allow'."""
    with pytest.raises(Exception) as excinfo:
        TrainingStrategyConfigSchema(curriculum_ramp_rate="oops")
    assert "curriculum_ramp_rate" in str(excinfo.value).lower() or \
        "ramp" in str(excinfo.value).lower()


def test_curriculum_ramp_rate_rejects_negative() -> None:
    """Negative ramp-rate is non-physical and must be rejected (ge=0)."""
    with pytest.raises(Exception):
        TrainingStrategyConfigSchema(curriculum_ramp_rate=-1.0)


def test_curriculum_start_timestep_rejects_zero() -> None:
    """Zero start-timestep is invalid (ge=1)."""
    with pytest.raises(Exception):
        TrainingStrategyConfigSchema(curriculum_start_timestep=0)


def test_curriculum_fields_default_to_none() -> None:
    """No-curriculum is the default; omitting both fields must not raise."""
    cfg = TrainingStrategyConfigSchema()
    assert cfg.curriculum_start_timestep is None
    assert cfg.curriculum_ramp_rate is None
