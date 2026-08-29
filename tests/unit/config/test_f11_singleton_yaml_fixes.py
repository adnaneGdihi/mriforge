"""Regression tests for F11 (smoke_audit_20260521 round 6) — singleton
YAML config bugs that crashed individual experiments at runtime.

Four singleton fixes:

1. ``experiment_pinn_csm_estimation.yaml`` — PINN strategy reads
   ``config.losses.pinn`` at __init__; absence raised
   ``[PINN Strategy] config.losses.pinn is None``. Added a
   ``losses.pinn`` block with all 4 lambdas + enables.

2. ``experiment_42_bloch_cycles.yaml`` — CycleBlochStrategy mounts a
   discriminator (declared via ``discriminator_component``) but the
   pre-fix had no ``losses.gan`` block, so the discriminator step
   had ``optimizer=None``. Added a ``losses.gan`` block.

3. ``exp_promoted_coord_kspace_gen.yaml`` and
4. ``exp_promoted_stochastic_interpolants.yaml`` — both inherit from
   ``DiffusionTrainingStrategy`` which requires
   ``training.diffusion.{timesteps, noise_schedule}`` at __init__.
   Added the block to both.

These tests pin each YAML's required block so a future bulk YAML edit
(or migration) can't silently strip the fix.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]


def _load(relpath: str) -> dict:
    p = REPO / relpath
    with p.open() as f:
        return yaml.safe_load(f)


def test_pinn_csm_estimation_has_losses_pinn_block() -> None:
    """Pin the ``losses.pinn`` block on the PINN CSM-estimation YAML.

    Without this block the PINN strategy raises at __init__ before
    any training step. See ``pinn_strategy.py:69`` for the loud-fail.
    """
    cfg = _load(
        "experiments/inprogress/physics_driven/experiment_pinn_csm_estimation.yaml"
    )
    pinn = cfg["losses"].get("pinn")
    assert pinn is not None, (
        "losses.pinn must be present so PINNStrategy.__init__ "
        "doesn't crash with 'config.losses.pinn is None'."
    )
    # Each of the four lambdas the strategy reads must be set.
    for key in (
        "lambda_pde",
        "lambda_unit_norm_coil",
        "lambda_magnitude_tv",
        "lambda_pinn_dc",
    ):
        assert key in pinn, (
            f"losses.pinn.{key} required by PINNStrategy.__init__ "
            f"(see pinn_strategy.py:78-81)."
        )


def test_experiment_42_bloch_cycles_has_losses_gan_block() -> None:
    """Pin the ``losses.gan`` block so CycleBlochStrategy's
    discriminator step gets a real optimizer instead of None."""
    cfg = _load(
        "experiments/inprogress/bloch_cycle/experiment_42_bloch_cycles.yaml"
    )
    gan = cfg["losses"].get("gan")
    assert gan is not None, (
        "losses.gan must be present so the discriminator step is "
        "built with a real optimizer. Without it, Trainer.execute_step "
        "raises ``step_configs[1] (name='discriminator') has "
        "optimizer=None``."
    )
    assert gan.get("enable_adversarial") is True, (
        "losses.gan.enable_adversarial must be True for the GAN composite "
        "to actually build the discriminator optimizer."
    )


@pytest.mark.parametrize(
    "yaml_path",
    [
        "experiments/inprogress/promoted/exp_promoted_coord_kspace_gen.yaml",
        "experiments/inprogress/promoted/exp_promoted_stochastic_interpolants.yaml",
    ],
)
def test_diffusion_dependent_yamls_have_training_diffusion_block(
    yaml_path: str,
) -> None:
    """Both ``coord_kspace_gen`` and ``stochastic_interpolants`` strategies
    inherit from DiffusionTrainingStrategy which requires
    ``training.diffusion.{timesteps, noise_schedule}`` at __init__. Pin
    both YAMLs so a strategy refactor can't silently re-introduce the
    crash."""
    cfg = _load(yaml_path)
    diffusion = cfg["training"].get("diffusion")
    assert diffusion is not None, (
        f"{yaml_path}: training.diffusion block required (strategy "
        "inherits from DiffusionTrainingStrategy)."
    )
    assert "timesteps" in diffusion, (
        f"{yaml_path}: training.diffusion.timesteps required (strategy "
        "reads it at __init__)."
    )
    assert "noise_schedule" in diffusion, (
        f"{yaml_path}: training.diffusion.noise_schedule required."
    )
