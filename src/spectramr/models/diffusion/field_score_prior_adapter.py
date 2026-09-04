"""Adapter exposing a field-conditioned score prior to the posterior-sampling interface (A-5.2).

:class:`~spectramr.models.generators.field_physics_score_unet.FieldPhysicsEmbeddedScoreUNet` is a
reusable prior ``eps_theta(x_t, timesteps, field_strength)`` (keyword args), but
:class:`~spectramr.models.diffusion.posterior_langevin.PosteriorLangevinEnsemble` (and DPS samplers)
call ``score_model(x_t, y, t)`` positionally. This thin adapter bridges the two: it fixes the field
at construction and forwards ``(x_t, y, t) -> eps_theta(x_t, timesteps=t, field_strength=b)``. The
measurement ``y`` is consumed by the sampler's likelihood term, NOT by the (unconditional) prior, so
it is intentionally unused here. This makes the "pluggable into PosteriorLangevinEnsemble / DPS"
claim of the FieldScore prior concrete and shippable, not a test-local closure.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FieldScorePriorAdapter(nn.Module):
    """Wrap a field-conditioned score prior as a ``score_model(x_t, y, t)`` callable (A-5.2)."""

    def __init__(self, score_prior: nn.Module, field_strength: float) -> None:
        super().__init__()
        if float(field_strength) <= 0.0:
            raise ValueError(f"field_strength must be > 0 Tesla; got {field_strength}.")
        self.score_prior = score_prior
        self.register_buffer("field_strength", torch.tensor(float(field_strength)))

    def forward(self, x_t: torch.Tensor, y: Any, t: torch.Tensor) -> torch.Tensor:
        """``y`` (the measurement) is the sampler's likelihood input, not the prior's — ignored."""
        del y
        b = self.field_strength.to(x_t.device).expand(x_t.shape[0])
        return self.score_prior(x_t, timesteps=t.reshape(-1), field_strength=b)
