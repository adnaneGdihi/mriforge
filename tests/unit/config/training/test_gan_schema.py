"""Tests for ``TrainingConfigGAN`` (config/schemas/training/gan.py).

Regression for WS-1 finding WS1-st-04: ``default_disc_lr`` tried to fall back to
``info.data["learning_rate"]`` if ``discriminator_learning_rate`` was None, but
there is no flat top-level ``learning_rate`` field (the canonical knob is nested
``optimization.optimizer.learning_rate``), so the fallback was permanently dead — the field
just stayed None. The validator is now an explicit pass-through documenting that
the strategy reads ``optimization.optimizer.learning_rate`` at runtime.
"""

from __future__ import annotations

from spectramr.config.schemas.training.gan import TrainingConfigGAN


def test_discriminator_lr_defaults_to_none() -> None:
    cfg = TrainingConfigGAN()
    assert cfg.discriminator_learning_rate is None


def test_discriminator_lr_explicit_respected() -> None:
    cfg = TrainingConfigGAN(discriminator_learning_rate=1e-4)
    assert cfg.discriminator_learning_rate == 1e-4


def test_config_is_frozen() -> None:
    assert TrainingConfigGAN.model_config.get("frozen") is True
