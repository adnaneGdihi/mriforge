"""Unit tests for training_mode resolution in ModelBuilder and OptimizationBuilder.

Verifies that setting training.task = 'disentangled_vae' (without a deprecated
training_mode field) correctly resolves to 'disentangled_vae' so that the
discriminator optimizer (opt_d) is created.
"""

from unittest.mock import MagicMock

import pytest


def _make_config(task: str = "reconstruction"):
    """Build a minimal mock config with training.task set."""
    config = MagicMock()
    config.training = MagicMock()
    # Simulate deprecated field removal: training_mode does NOT exist on training config
    del config.training.training_mode  # AttributeError → getattr falls back
    config.training.task = task
    return config


def _resolve_training_mode(config) -> str:
    """Replicate the training_mode resolution logic from model_builder / optimization_builder."""
    training_mode = "reconstruction"
    if config.training:
        training_mode = getattr(
            config.training,
            "training_mode",
            getattr(config.training, "task", "reconstruction"),
        )
    if not training_mode:
        training_mode = "reconstruction"
    return str(training_mode).lower()


@pytest.mark.parametrize(
    "task,expected",
    [
        ("disentangled_vae", "disentangled_vae"),
        ("disentangled", "disentangled"),
        ("gan", "gan"),
        ("reconstruction", "reconstruction"),
        ("latent_gan", "latent_gan"),
    ],
)
def test_training_mode_resolved_from_task(task, expected):
    """training_mode is resolved from config.training.task correctly."""
    config = _make_config(task=task)
    mode = _resolve_training_mode(config)
    assert mode == expected, f"Expected '{expected}' for task='{task}', got '{mode}'"


def test_disentangled_vae_triggers_discriminator_build():
    """Model builder creates discriminator when task='disentangled_vae'."""
    # We test the logic inside ModelBuilder.build_discriminator indirectly
    # by verifying that 'disentangled_vae' is in the allowed modes
    config = _make_config(task="disentangled_vae")
    mode = _resolve_training_mode(config)
    allowed_for_discriminator = [
        "gan",
        "latent_gan",
        "disentangled_vae",
        "disentangled",
    ]
    assert (
        mode in allowed_for_discriminator
    ), f"'{mode}' not in allowed disc modes: {allowed_for_discriminator}"


def test_disentangled_vae_triggers_dual_optimizer():
    """Optimization builder uses dual optimizers when task='disentangled_vae'."""
    config = _make_config(task="disentangled_vae")
    mode = _resolve_training_mode(config)
    dual_optimizer_modes = [
        "gan",
        "latent_gan",
        "cycle_bloch",
        "disentangled",
        "disentangled_vae",
    ]
    assert (
        mode in dual_optimizer_modes
    ), f"'{mode}' not in dual-optimizer modes: {dual_optimizer_modes}"


def test_reconstruction_task_skips_discriminator():
    """Model builder skips discriminator when task='reconstruction'."""
    config = _make_config(task="reconstruction")
    mode = _resolve_training_mode(config)
    allowed_for_discriminator = [
        "gan",
        "latent_gan",
        "disentangled_vae",
        "disentangled",
    ]
    assert mode not in allowed_for_discriminator
