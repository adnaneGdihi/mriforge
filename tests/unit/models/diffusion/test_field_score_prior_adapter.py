"""Tests for FieldScorePriorAdapter (A-5.2 — PLE/DPS consumability of the FieldScore prior)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.field_score_prior_adapter import FieldScorePriorAdapter
from spectramr.models.diffusion.posterior_langevin import PosteriorLangevinEnsemble
from spectramr.models.generators.field_physics_score_unet import FieldPhysicsEmbeddedScoreUNet


def _prior() -> FieldPhysicsEmbeddedScoreUNet:
    return FieldPhysicsEmbeddedScoreUNet(width=16, time_dim=8)


def test_adapter_exposes_ple_score_model_interface() -> None:
    # The adapter turns eps(x_t, timesteps=, field_strength=) into score_model(x_t, y, t) — exactly
    # the call PosteriorLangevinEnsemble makes (posterior_langevin.py: score_model(x_t, y, t_batch)).
    adapter = FieldScorePriorAdapter(_prior().eval(), field_strength=3.0)
    x_t = torch.rand(2, 1, 16, 16)
    t = torch.full((2,), 0.5)
    out = adapter(x_t, None, t)  # (x_t, y, t) positional — y ignored (sampler's likelihood input)
    assert out.shape == x_t.shape


def test_adapter_is_consumed_by_posterior_langevin_ensemble() -> None:
    # SHIPPED consumability (not a test-local closure): PLE holds the adapter as its score_model and
    # the score call it makes internally succeeds.
    adapter = FieldScorePriorAdapter(_prior().eval(), field_strength=1.5)
    ple = PosteriorLangevinEnsemble(score_model=adapter, num_samples=2, num_steps=3)
    x_t = torch.rand(2, 1, 16, 16)
    t_batch = torch.full((2,), 0.5)
    score = ple.score_model(x_t, None, t_batch)  # the exact internal call in PLE.forward
    assert score.shape == x_t.shape


def test_adapter_rejects_nonpositive_field() -> None:
    with pytest.raises(ValueError, match="field_strength"):
        FieldScorePriorAdapter(_prior(), field_strength=0.0)
