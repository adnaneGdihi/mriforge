"""Tests for ``check_phase3_model_constraints`` (per-model Tier-1 gates)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.validation.config_health_checker import (  # noqa: E402
    ConfigHealthChecker,
)


def _cfg(model_type: str, kwargs: dict, patch=None) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(model_type=model_type, model_kwargs=kwargs),
        data=SimpleNamespace(patch_size=patch),
    )


def _sev(results):
    return {r.severity for r in results}


@pytest.mark.registry_contract
def test_glow_indivisible_patch_errors() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("glow", {"num_scales": 4}, patch=[200, 200]))
    assert "error" in _sev(res)  # 200 % 16 != 0


@pytest.mark.registry_contract
def test_glow_valid_patch_passes() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("glow", {"num_scales": 4}, patch=[256, 256]))
    assert _sev(res) == {"info"}


@pytest.mark.registry_contract
def test_octave_alpha_out_of_range_errors() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("octave_conv", {"alpha": 1.5}))
    assert "error" in _sev(res)


@pytest.mark.registry_contract
def test_octave_alpha_edge_warns() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("octave_conv", {"alpha": 0.0}))
    assert "warning" in _sev(res)


@pytest.mark.registry_contract
def test_progressive_gan_non_power_of_two_errors() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("progressive_gan", {"max_resolution": 200}))
    assert "error" in _sev(res)


@pytest.mark.registry_contract
def test_moe_vae_too_few_experts_errors() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("moe_vae", {"num_experts": 1}))
    assert "error" in _sev(res)


@pytest.mark.registry_contract
def test_gated_gnn_many_steps_warns() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("gated_gnn", {"num_steps": 20}))
    assert "warning" in _sev(res)


@pytest.mark.registry_contract
def test_blurring_diffusion_sigma_order_errors() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(
        _cfg("blurring_diffusion", {"sigma_min": 0.5, "sigma_max": 0.1})
    )
    assert "error" in _sev(res)


@pytest.mark.registry_contract
def test_non_phase3_model_is_clean_pass() -> None:
    c = ConfigHealthChecker()
    res = c.check_phase3_model_constraints(_cfg("unet", {}))
    assert _sev(res) == {"info"}
