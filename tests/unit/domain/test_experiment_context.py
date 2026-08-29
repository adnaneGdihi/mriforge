"""Unit tests for the ``ResolvedExperimentContext`` IR (cached-cascade WS-X)."""

from __future__ import annotations

import dataclasses

import pytest

from mriforge.domain.entities.experiment_context import (
    DataProfile,
    PhysicsProfile,
    ResolvedExperimentContext,
)

pytestmark = pytest.mark.unit


def test_defaults_are_empty_profiles() -> None:
    ctx = ResolvedExperimentContext(config_version="6.1", model_type="unet", strategy_name="GAN")
    assert ctx.loss_names == ()
    assert ctx.loss_profile == ()
    assert ctx.adapter_chain == ()
    assert ctx.unresolved_required_fields == ()  # WS-X Phase-B field default
    assert isinstance(ctx.data_profile, DataProfile)
    assert isinstance(ctx.physics_profile, PhysicsProfile)


def test_is_frozen() -> None:
    ctx = ResolvedExperimentContext(config_version="6.1", model_type="unet", strategy_name="GAN")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.model_type = "x"  # type: ignore[misc]


def test_render_is_human_readable_and_carries_provenance() -> None:
    ctx = ResolvedExperimentContext(
        config_version="6.1",
        model_type="srcnn",
        strategy_name="ReconStrategy",
        loss_names=("l1", "ssim"),
    )
    txt = ctx.render()
    assert "srcnn" in txt and "ReconStrategy" in txt
    assert "l1" in txt and "ssim" in txt
    assert "config v6.1" in txt


def test_unresolved_required_fields_roundtrips() -> None:
    ctx = ResolvedExperimentContext(
        config_version="6.1",
        model_type="m",
        strategy_name="s",
        unresolved_required_fields=("training.diffusion.num_timesteps",),
    )
    assert ctx.unresolved_required_fields == ("training.diffusion.num_timesteps",)
