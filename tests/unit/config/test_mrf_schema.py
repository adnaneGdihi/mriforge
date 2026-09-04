"""Tests for ``MRFConfigSchema`` (config/schemas/mrf.py).

Regression for WS-1 finding WS1-physics-01: the MRF block declared
``extra="forbid"`` but omitted ``frozen=True``, so a held reference to
``settings.mrf`` could be mutated post-load — a silent breach of the
immutable-config invariant (NN#1). Pydantic v2 does NOT propagate the parent
``TrainingSettings`` frozen-ness to child models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.mrf import MRFConfigSchema


def test_defaults_construct() -> None:
    cfg = MRFConfigSchema()
    assert cfg.spiral_rotation_schedule is None


def test_extra_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        MRFConfigSchema(not_a_real_mrf_key=1)  # type: ignore[call-arg]


def test_frozen_blocks_mutation() -> None:
    """A constructed MRF block must reject attribute assignment (NN#1)."""
    cfg = MRFConfigSchema()
    with pytest.raises(ValidationError):
        cfg.spiral_rotation_schedule = "linear"  # type: ignore[misc]


def test_frozen_flag_present() -> None:
    assert MRFConfigSchema.model_config.get("frozen") is True
