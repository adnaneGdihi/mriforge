"""Tests for ``TrainingConfigSSL`` (config/schemas/training/ssl.py).

Regression for WS-1 finding WS1-st-03: ``validate_ssl_objectives`` dereferenced
``obj.reconstruction.lambda_l1``, but in v6.0 ``ObjectiveConfigSchema.reconstruction``
defaults to ``None`` and ``ReconstructionObjectiveConfig`` no longer carries a
``lambda_l1`` field — so a bare ``TrainingConfigSSL()`` raised ``AttributeError``.
SSL loss weights live under ``losses.ssl`` now; the validator is a pass-through.
"""

from __future__ import annotations

from mriforge.config.schemas.training.ssl import TrainingConfigSSL


def test_default_construction_does_not_raise() -> None:
    """A default SSL config must construct (no objectives required)."""
    cfg = TrainingConfigSSL()
    assert isinstance(cfg, TrainingConfigSSL)


def test_config_is_frozen() -> None:
    """SSL config remains immutable (NN#1)."""
    assert TrainingConfigSSL.model_config.get("frozen") is True
