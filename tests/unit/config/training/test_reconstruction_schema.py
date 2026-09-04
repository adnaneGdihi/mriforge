"""Tests for ``TrainingConfigReconstruction`` (config/schemas/training/reconstruction.py).

Regression for WS-1 finding WS1-st-07: ``validate_recon_objectives`` had a dead
local rebind (``if obj is None: obj = ...``) and an always-firing
``if obj.reconstruction is None: raise`` — and since
``ObjectiveConfigSchema.reconstruction`` defaults to ``None`` in v6.0, a bare
``TrainingConfigReconstruction()`` was unconstructable. Loss weights moved to
``losses.reconstruction``; the validator is now a pass-through.
"""

from __future__ import annotations

from spectramr.config.schemas.training.reconstruction import TrainingConfigReconstruction


def test_default_construction_does_not_raise() -> None:
    """A default reconstruction config must construct (no objectives required)."""
    cfg = TrainingConfigReconstruction()
    assert isinstance(cfg, TrainingConfigReconstruction)


def test_config_is_frozen() -> None:
    assert TrainingConfigReconstruction.model_config.get("frozen") is True
