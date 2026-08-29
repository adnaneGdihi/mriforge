"""Consistency test for the PR-4/5/7/9 strategy training-mode registration.

A new strategy key must be registered in THREE places or it fails the audit
in subtle ways: STRATEGY_CLASS_PATHS (dispatch), TRAINING_MODE_CONSTRAINTS
(else ``_validate_training_mode_compatibility`` reports "Unknown training
mode"), and VALID_TRAINING_MODES (reference list). This guards that the
four frontier strategies stay consistent across all three.
"""

from __future__ import annotations

import pytest

from mriforge.config.validation_constants import (
    TRAINING_MODE_CONSTRAINTS,
    VALID_TRAINING_MODES,
)
from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

_FRONTIER_MODES = [
    "bloch_schrodinger_bridge",
    "i2sb",
    "vf_consistency_distillation",
    "universal_reconstruction",
    "flow_matching_pfode",
]


@pytest.mark.parametrize("mode", _FRONTIER_MODES)
def test_mode_in_strategy_class_paths(mode: str) -> None:
    assert mode in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


@pytest.mark.parametrize("mode", _FRONTIER_MODES)
def test_mode_in_training_mode_constraints(mode: str) -> None:
    # Absence here makes the audit emit "Unknown training mode".
    assert mode in TRAINING_MODE_CONSTRAINTS


@pytest.mark.parametrize("mode", _FRONTIER_MODES)
def test_mode_in_valid_training_modes(mode: str) -> None:
    assert mode in VALID_TRAINING_MODES


@pytest.mark.parametrize("mode", _FRONTIER_MODES)
def test_frontier_modes_require_no_named_objective(mode: str) -> None:
    """They compute their objective internally, so they must not force a
    named loss sub-block (which would mis-fire on the declarative
    image_losses format)."""
    assert TRAINING_MODE_CONSTRAINTS[mode]["required_objectives"] == []
