"""Tests for BlochSynthConfig (MICCAI MRIxFields2026, idea 2.1)."""

from __future__ import annotations

import pytest

from mriforge.config.schemas.training.bloch_field import BlochSynthConfig


def test_defaults() -> None:
    c = BlochSynthConfig()
    assert c.source_contrasts == ["T1w", "T2w", "FLAIR"]
    assert c.target_field_tesla == 7.0
    assert c.segmenter_backend == "label_dice"
    assert c.dispersion_beta_bounds == (0.3, 0.4)


def test_rejects_target_field_above_7t() -> None:
    with pytest.raises(Exception):
        BlochSynthConfig(target_field_tesla=9.0)


def test_rejects_segmenter_none_with_seg_weight() -> None:
    # Anti-facade (#16): advertising seg-consistency with no segmenter to compute it.
    with pytest.raises(Exception):
        BlochSynthConfig(segmenter_backend="none", seg_consistency_weight=0.5)


def test_allows_segmenter_none_with_zero_seg_weight() -> None:
    c = BlochSynthConfig(segmenter_backend="none", seg_consistency_weight=0.0)
    assert c.segmenter_backend == "none"


def test_rejects_inverted_dispersion_bounds() -> None:
    with pytest.raises(Exception):
        BlochSynthConfig(dispersion_beta_bounds=(0.4, 0.3))


def test_forbids_unknown_field() -> None:
    with pytest.raises(Exception):
        BlochSynthConfig(not_a_field=1)


def test_mounted_on_training_schema() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "bloch_synth" in TrainingStrategyConfigSchema.model_fields
