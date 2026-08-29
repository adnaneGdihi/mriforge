"""CS-MNO schema parse + integration with TrainingSettings."""

from __future__ import annotations

import pytest

from mriforge.config.schemas.training.cs_mno import (
    ArchitectureConfig,
    CSMNOTrainingConfigSchema,
    ScanConfig,
    SpectralConfig,
)


def test_defaults_parse() -> None:
    c = CSMNOTrainingConfigSchema()
    assert isinstance(c.spectral, SpectralConfig)
    assert isinstance(c.scan, ScanConfig)
    assert isinstance(c.architecture, ArchitectureConfig)
    assert c.scan.mode == "hilbert_2d"
    assert c.scan.use_physical_arc is True
    assert c.spectral.modes == 16
    assert c.architecture.n_layers == 4


def test_unknown_scan_mode_rejected() -> None:
    """No silent fallback on unrecognised scan modes."""
    with pytest.raises(Exception, match=r"scan|mode"):
        CSMNOTrainingConfigSchema(scan={"mode": "spiral_2d"})


def test_extra_fields_forbidden() -> None:
    """Schema is frozen — typos must raise."""
    with pytest.raises(Exception, match=r"extra|forbidden|spectral_extra"):
        CSMNOTrainingConfigSchema(spectral={"modes": 8, "spectral_extra": 1})


def test_can_attach_to_training_strategy_section() -> None:
    """The cs_mno field is reachable from TrainingStrategyConfigSchema."""
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    assert "cs_mno" in TrainingStrategyConfigSchema.model_fields
