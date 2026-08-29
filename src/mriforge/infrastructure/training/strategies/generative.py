"""Generative (likelihood-based) training strategy.

Added 2026-05-22 for the Phase-3 normalising-flow models (``glow``,
``equivariant_flow``) re-implemented from the deletion ledger
(TODO/deleted_model_types_reimplementation_plan.md). Those models declare
``training_mode="generative"`` and are trained by maximum likelihood, a
paradigm the framework previously lacked.

The strategy is deliberately model-agnostic: it asks the model for a
likelihood in priority order
  1. ``model.log_prob(x)``        — exact-likelihood flows (Glow, equivariant)
  2. ``model.training_loss(x)``   — models that expose their own objective
  3. ``model(x) -> (z, log_det)`` — generic flow contract; standard-normal NLL
and only falls back to a reconstruction MSE if none of those are available.
There is no silent default to an unrelated objective (CLAUDE.md pitfall #9).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class GenerativeTrainingStrategy(BaseTrainingStrategy):
    """Maximum-likelihood training for generative density models."""

    def __init__(self, env: Any = None, device: Any = None, **kwargs: Any) -> None:
        super().__init__(env=env, device=device, **kwargs)
        self._verify_strategy_config(expected_modes=("generative",))

    @staticmethod
    def _standard_normal_nll(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under a standard-normal prior on ``z``.

        ``log p(x) = log N(z; 0, I) + log|det J|``; the loss is its negation,
        averaged over the batch.
        """
        z_flat = z.reshape(z.shape[0], -1)
        d = z_flat.shape[1]
        log_pz = -0.5 * (z_flat**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)
        log_px = log_pz + log_det.reshape(z.shape[0])
        return -log_px.mean()

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Lazy fallback: only reach for ``self.generator_model`` (→
        # ``self.env.generator``) when the caller did NOT pass an explicit
        # model. ``dict.get`` evaluates its default eagerly, so the old form
        # touched ``self.env`` even on the explicit-model path — crashing any
        # caller that supplied ``model=`` without a full env.
        model = kwargs["model"] if "model" in kwargs else self.generator_model
        x = target_batch if target_batch is not None else input_batch

        log_prob = getattr(model, "log_prob", None)
        if callable(log_prob):
            nll = -log_prob(x).mean()
            return {"g_total_loss": nll, "loss": nll, "loss_nll": nll}

        training_loss = getattr(model, "training_loss", None)
        if callable(training_loss):
            loss = training_loss(x)
            return {"g_total_loss": loss, "loss": loss, "loss_objective": loss}

        out = model(x)
        if isinstance(out, tuple) and len(out) == 2:
            z, log_det = out
            nll = self._standard_normal_nll(z, log_det)
            return {"g_total_loss": nll, "loss": nll, "loss_nll": nll}

        # Last resort: an autoencoder-style reconstruction objective. This is
        # explicit (not a silent fallback) and only reached when the model
        # exposes none of the likelihood contracts above.
        pred = out[0] if isinstance(out, tuple) else out
        mse = F.mse_loss(pred, x)
        logger.warning(
            "GenerativeTrainingStrategy: model %s exposes no log_prob / "
            "training_loss / (z, log_det) contract; falling back to "
            "reconstruction MSE.",
            type(model).__name__,
        )
        return {"g_total_loss": mse, "loss": mse, "loss_recon": mse}
