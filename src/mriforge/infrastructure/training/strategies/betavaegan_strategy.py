"""β-TC-VAE-GAN training strategy.

Drives the cross-paradigm composition the ``beta_vae_gan`` model exists to
validate (TODO/phase3_model_implementations_detailed.md §12): the standard
GAN two-player loop (adversarial + reconstruction, via
:class:`GANTrainingStrategy`) plus the β-TC-VAE disentanglement term added
to the generator loss.

This is the "GANTrainingStrategy with a ``beta_tc_vae_aux`` objective"
described in the plan. We reuse the full, proven GAN machinery (n_critic
discriminator updates, AMP, gradient penalty, metrics) unchanged and only
augment the generator-side loss with the model-provided β-TC scalar, so the
composition is exercised in training without duplicating the GAN loop.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy

logger = logging.getLogger(__name__)


class BetaVAEGANStrategy(GANTrainingStrategy):
    """GAN strategy that adds the β-TC-VAE disentanglement term to G's loss."""

    def _beta_tc_weight(self) -> float:
        """Auxiliary weight for the β-TC term (default 1.0).

        The total-correlation strength is already controlled by the model's
        ``beta`` (inside ``BetaTCVAELoss``); this is the outer mixing weight
        against the adversarial + reconstruction terms.
        """
        gan_cfg = (
            getattr(self.config.training, "gan", None)
            if self.config and self.config.training
            else None
        )
        w = getattr(gan_cfg, "beta_tc_weight", None) if gan_cfg is not None else None
        return float(w) if w is not None else 1.0

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Standard GAN generator loss (adversarial + reconstruction).
        losses = super()._compute_losses_impl(input_batch, target_batch, epoch, **kwargs)

        model = kwargs.get("model", self.generator_model)
        beta_tc_fn = getattr(model, "beta_tc_term", None)
        if callable(beta_tc_fn):
            beta_tc = beta_tc_fn(input_batch)
            weight = self._beta_tc_weight()
            losses["loss_beta_tc"] = beta_tc
            if "g_total_loss" in losses and losses["g_total_loss"] is not None:
                losses["g_total_loss"] = losses["g_total_loss"] + weight * beta_tc
            else:  # pragma: no cover - defensive
                losses["g_total_loss"] = weight * beta_tc
        else:
            logger.warning(
                "BetaVAEGANStrategy: generator %s exposes no beta_tc_term(); "
                "running as a plain GAN (no disentanglement objective).",
                type(model).__name__,
            )
        return losses
