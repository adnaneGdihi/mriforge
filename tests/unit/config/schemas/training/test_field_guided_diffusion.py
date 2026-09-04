"""Tests for FieldGuidedDiffusionConfig (B-3.5) — incl. the CFG cross-validator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.field_guided_diffusion import (
    FieldGuidedDiffusionConfig,
)


def test_defaults_are_valid() -> None:
    c = FieldGuidedDiffusionConfig()
    assert c.guidance_prob == 0.1
    assert c.guidance_scale == 3.0
    assert c.beta_schedule == "cosine"


def test_guidance_without_dropout_is_rejected() -> None:
    # ANTI-DEGENERATE (#20): CFG (guidance_scale != 1) with guidance_prob=0 trains no
    # null path -> the config must fail at LOAD time, not silently certify nonsense.
    with pytest.raises(ValidationError, match="guidance_prob=0"):
        FieldGuidedDiffusionConfig(guidance_prob=0.0, guidance_scale=3.0)


def test_zero_prob_with_no_guidance_is_allowed() -> None:
    # guidance_scale=1 never invokes the null forward, so prob=0 is harmless there.
    c = FieldGuidedDiffusionConfig(guidance_prob=0.0, guidance_scale=1.0)
    assert c.guidance_scale == 1.0


def test_guidance_with_dropout_is_allowed() -> None:
    c = FieldGuidedDiffusionConfig(guidance_prob=0.1, guidance_scale=5.0)
    assert c.guidance_scale == 5.0
