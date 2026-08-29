"""Tests for ConditioningConfig — the YAML-facing adaptive-conditioning knob.

(2026-05-26) Declares which condition sources the model adapts to. ``extra`` is
forbidden so a typo'd source/key is rejected at load (CLAUDE.md pitfall #9),
and the Literal source set is kept in sync with the conditioner registry.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.conditioning import ConditioningConfig


def test_defaults_disabled_and_empty() -> None:
    c = ConditioningConfig()
    assert c.enabled is False
    assert c.sources == []
    assert c.embed_dim == 128
    assert c.fusion == "concat"
    assert c.injection == "film"


def test_accepts_valid_sources() -> None:
    c = ConditioningConfig(enabled=True, sources=["diffusion_t", "severity_vec"])
    assert c.sources == ["diffusion_t", "severity_vec"]


def test_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        ConditioningConfig(enabled=True, sources=["banana"])


def test_rejects_unknown_fusion() -> None:
    with pytest.raises(ValidationError):
        ConditioningConfig(fusion="quux")


def test_enabled_requires_at_least_one_source() -> None:
    with pytest.raises(ValidationError, match="requires at least one source"):
        ConditioningConfig(enabled=True, sources=[])


def test_extra_keys_forbidden() -> None:
    with pytest.raises(ValidationError):
        ConditioningConfig(bogus=1)


def test_frozen() -> None:
    c = ConditioningConfig()
    with pytest.raises(ValidationError):
        c.enabled = True  # type: ignore[misc]


def test_literal_sources_match_conditioner_registry() -> None:
    """The schema's allowed sources must equal the registered conditioner keys."""
    import mriforge.models.conditioning  # noqa: F401  (populates registry)
    from mriforge.models.conditioning.encoders import _CONDITIONER_REGISTRY

    assert set(ConditioningConfig.allowed_sources()) == set(_CONDITIONER_REGISTRY)


def test_model_schema_has_conditioning_default() -> None:
    from mriforge.config.schemas.model import ModelConfigSchema

    assert ModelConfigSchema().conditioning.enabled is False


def test_model_schema_accepts_conditioning_block() -> None:
    from mriforge.config.schemas.model import ModelConfigSchema

    m = ModelConfigSchema(
        conditioning={"enabled": True, "sources": ["severity_vec"]}
    )
    assert m.conditioning.sources == ["severity_vec"]
