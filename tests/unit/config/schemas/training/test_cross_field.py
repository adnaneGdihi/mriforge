"""Tests for CrossFieldConfig (MICCAI MRIxFields2026)."""

from __future__ import annotations

import pytest

from mriforge.config.schemas.training.cross_field import CrossFieldConfig


def test_defaults() -> None:
    c = CrossFieldConfig()
    assert c.lambda_latent_cycle >= 0.0
    assert c.n_render_fields >= 1
    assert c.contrast_conditioning in (True, False)


def test_frozen() -> None:
    c = CrossFieldConfig()
    with pytest.raises(Exception):
        c.lambda_latent_cycle = 5.0


def test_forbids_unknown_field() -> None:
    with pytest.raises(Exception):
        CrossFieldConfig(not_a_field=1)


def test_mounted_on_training_schema() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "cross_field" in TrainingStrategyConfigSchema.model_fields
    assert "field_flow" in TrainingStrategyConfigSchema.model_fields


def test_field_flow_config_defaults() -> None:
    from mriforge.config.schemas.training.cross_field import FieldFlowConfig

    c = FieldFlowConfig()
    assert c.n_steps >= 1
    assert c.solver in ("euler", "heun")
    assert c.lambda_straightness >= 0.0


def test_field_flow_config_frozen_and_validates() -> None:
    import pytest

    from mriforge.config.schemas.training.cross_field import FieldFlowConfig

    c = FieldFlowConfig()
    with pytest.raises(Exception):
        c.n_steps = 5
    with pytest.raises(Exception):
        FieldFlowConfig(solver="rk4")


def test_field_flow_contrast_conditioning_default_false() -> None:
    from mriforge.config.schemas.training.cross_field import FieldFlowConfig

    assert FieldFlowConfig().contrast_conditioning is False


def test_field_flow_rejects_unwired_contrast_conditioning() -> None:
    # Anti-#15: FieldFlowStrategy never reads contrast_conditioning, so True would be a silent
    # no-op. The schema must raise at config-load rather than degrade to a do-nothing knob.
    from mriforge.config.schemas.training.cross_field import FieldFlowConfig

    with pytest.raises(ValueError):
        FieldFlowConfig(contrast_conditioning=True)


# --- FieldCocycleConfig (idea 4.2) -------------------------------------------


def test_field_cocycle_defaults() -> None:
    from mriforge.config.schemas.training.cross_field import FieldCocycleConfig

    c = FieldCocycleConfig()
    assert c.reference_field_tesla == 3.0
    assert c.field_min_tesla < c.field_max_tesla
    assert c.cocycle_weight >= 0.0 and c.adversarial_weight >= 0.0
    assert c.triple_sampling == "uniform_distinct"


def test_field_cocycle_rejects_reference_out_of_range() -> None:
    from mriforge.config.schemas.training.cross_field import FieldCocycleConfig

    with pytest.raises(Exception):
        FieldCocycleConfig(reference_field_tesla=10.0)  # above field_max_tesla


def test_field_cocycle_rejects_inverted_range() -> None:
    from mriforge.config.schemas.training.cross_field import FieldCocycleConfig

    with pytest.raises(Exception):
        FieldCocycleConfig(field_min_tesla=7.0, field_max_tesla=0.1)


def test_field_cocycle_forbids_unknown_field() -> None:
    from mriforge.config.schemas.training.cross_field import FieldCocycleConfig

    with pytest.raises(Exception):
        FieldCocycleConfig(not_a_field=1)


def test_field_cocycle_mounted_on_training_schema() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "field_cocycle" in TrainingStrategyConfigSchema.model_fields
