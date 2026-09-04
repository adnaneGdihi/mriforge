"""Unit tests for the ResolvedExperimentContext IR (cached-cascade WS-X)."""

from __future__ import annotations

import dataclasses

import pytest

from spectramr.domain.entities import (
    DataProfile,
    PhysicsProfile,
    ResolvedExperimentContext,
)
from spectramr.models.capabilities import (
    LossCapabilities,
    ModelCapabilities,
    StrategyCapabilities,
)


def test_minimal_construction_uses_defaults() -> None:
    ctx = ResolvedExperimentContext(
        config_version="6.1", model_type="unet", strategy_name="GAN"
    )
    assert ctx.loss_names == ()
    assert isinstance(ctx.data_profile, DataProfile)
    assert isinstance(ctx.model_profile, ModelCapabilities)
    assert isinstance(ctx.strategy_profile, StrategyCapabilities)
    assert ctx.loss_profile == ()
    assert isinstance(ctx.physics_profile, PhysicsProfile)
    assert ctx.adapter_chain == ()


def test_context_is_frozen() -> None:
    ctx = ResolvedExperimentContext(
        config_version="6.1", model_type="unet", strategy_name="GAN"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.model_type = "vae"  # type: ignore[misc]


def test_profiles_reuse_capability_dataclasses() -> None:
    # The IR must aggregate the SAME capability dataclasses (one SSOT), not
    # re-declare its own model/strategy/loss capability types.
    ctx = ResolvedExperimentContext(
        config_version="6.1",
        model_type="unet",
        strategy_name="Recon",
        loss_names=("l1", "ssim"),
        model_profile=ModelCapabilities(input_domain="kspace", output_domain="image"),
        strategy_profile=StrategyCapabilities(supported_paradigms=frozenset({"reconstruction"})),
        loss_profile=(LossCapabilities(domain="image"), LossCapabilities(domain="image")),
    )
    assert ctx.model_profile.input_domain == "kspace"
    assert "reconstruction" in ctx.strategy_profile.supported_paradigms
    assert all(isinstance(c, LossCapabilities) for c in ctx.loss_profile)


def test_render_mentions_key_fields() -> None:
    ctx = ResolvedExperimentContext(
        config_version="6.1",
        model_type="cross_attention_oracle_unet",
        strategy_name="Diffusion",
        loss_names=("l1",),
        data_profile=DataProfile(domain="image", spatial_dims=(2,), multi_contrast=False),
        physics_profile=PhysicsProfile(forward_model="sense", data_consistency="soft"),
        strategy_profile=StrategyCapabilities(supported_paradigms=frozenset({"diffusion"})),
    )
    text = ctx.render()
    assert "cross_attention_oracle_unet" in text
    assert "Diffusion" in text
    assert "sense" in text
    assert "diffusion" in text  # paradigm line
    assert "6.1" in text
