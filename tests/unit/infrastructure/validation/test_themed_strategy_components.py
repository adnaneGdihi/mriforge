"""Tests for the themed-strategy component guardrail (PR-0).

A strategy key that routes to a generic base class (`reconstruction`,
`diffusion`, `cycle_bloch`) but advertises paradigm-specific mathematics
must wire at least one *registered* loss or model that actually encodes
that mathematics — otherwise the run silently degrades to a vanilla
baseline (CLAUDE.md pitfall #9). This guardrail turns that silent
fallback into a loud audit error.

The pure decision function is tested here; the ConfigHealthChecker
integration is tested in test_config_health_checker-adjacent suites.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.themed_strategy_components import (
    THEMED_GENERIC_COMPONENTS,
    themed_component_status,
)


def test_honest_generic_key_is_not_applicable():
    """`reconstruction` is an honest generic key, not a themed alias."""
    status = themed_component_status("reconstruction", declared_losses=set(), model_type="unet")
    assert status.applicable is False


def test_themed_key_without_component_fails():
    """A themed key whose YAML wires none of its required components fails."""
    status = themed_component_status(
        "tropical_mrf", declared_losses={"l1", "ssim"}, model_type="hermitian_fno"
    )
    assert status.applicable is True
    assert status.satisfied is False
    assert status.required  # non-empty required-component description


def test_themed_key_with_required_loss_passes():
    """Wiring any one required loss satisfies the guardrail."""
    req = THEMED_GENERIC_COMPONENTS["tropical_mrf"]
    a_loss = next(iter(req["losses"]))
    status = themed_component_status(
        "tropical_mrf", declared_losses={"l1", a_loss}, model_type="hermitian_fno"
    )
    assert status.applicable is True
    assert status.satisfied is True


def test_themed_key_with_required_model_passes():
    """A model-carried paradigm passes when the YAML wires the themed model."""
    # heisenberg_phase's math lives in the heisenberg_recon_unet model.
    req = THEMED_GENERIC_COMPONENTS.get("heisenberg_phase", {})
    if not req.get("models"):
        return  # only assert if this key is model-carried in the SSOT map
    a_model = next(iter(req["models"]))
    status = themed_component_status(
        "heisenberg_phase", declared_losses={"l1"}, model_type=a_model
    )
    assert status.satisfied is True


def test_every_mapped_key_routes_to_a_generic_base():
    """The SSOT map must only contain keys that actually route to a generic
    base — otherwise the guardrail is policing the wrong keys."""
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    generic_bases = (
        "reconstruction.ReconstructionTrainingStrategy",
        "diffusion.DiffusionTrainingStrategy",
        "cycle_bloch_strategy.CycleBlochStrategy",
    )
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for key in THEMED_GENERIC_COMPONENTS:
        assert key in paths, f"{key} not a real strategy key"
        assert paths[key].endswith(generic_bases), (
            f"{key} maps to {paths[key]} which is not a generic base — "
            "it should not be in the themed-generic map"
        )


def test_status_unknown_key_not_applicable():
    status = themed_component_status("gan", declared_losses=set(), model_type="x")
    assert status.applicable is False
