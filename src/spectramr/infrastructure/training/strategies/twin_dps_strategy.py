r"""Twin-Likelihood Diffusion Posterior Sampling strategy (Phase 3 / Idea 3).

Reverse-SDE sampling guided by *two* additive likelihood terms:

* Acquired-k-space likelihood: :math:`\nabla \log p(y \mid x_t)`
  (the standard DPS gradient, scaled by ``config.training.twin_dps.lambda_y``).
  This term is **not** computed by this class — it is supplied by the
  inherited / caller-side DPS sampler, which owns the measured ``y`` and the
  forward operator; this strategy contributes only the marker-side term.
* Phase-stego marker likelihood: :math:`\nabla \log p(M \mid x_t)`
  (new in this strategy, supplied by :class:`PhaseStegoScoreLoss`).

The score model :math:`s_\phi(x_t, t)` is the existing pretrained
diffusion generator — no fresh training is performed. This strategy
supplies the *marker-side* guidance-gradient helper
(:meth:`compute_guidance_gradient`); the parent
:class:`DiffusionTrainingStrategy` exposes no per-step
``_sampling_step`` hook, so the marker gradient is folded into the
reverse-process drift by a caller-side sampler loop rather than by an
override on this class.

At λ_M=0 the strategy recovers single-likelihood DPS (Chung et al. 2023).
The Tier-1 audit ``twin_dps_guidance_scales_finite_and_positive``
catches λ_M outside the safe range (0, 1e3].
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)
from spectramr.models.losses.phase_stego_score import PhaseStegoScoreLoss

logger = logging.getLogger(__name__)


class TwinLikelihoodDPSStrategy(DiffusionTrainingStrategy):
    """Diffusion sampling with k-space + phase-stego marker guidance.

    Subclassing the diffusion strategy reuses the noise scheduler,
    Tweedie estimator, and reverse-sampler scaffolding. The override
    is confined to the marker-side likelihood contribution.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        td = getattr(self.config.training, "twin_dps", None)
        if td is None:
            raise ValueError(
                "TwinLikelihoodDPSStrategy requires training.twin_dps block. "
                "Switch to DiffusionTrainingStrategy if you don't want the "
                "twin-likelihood guidance."
            )
        # NOTE: ``lambda_y`` (the acquired-k-space DPS likelihood scale) is
        # validated by the schema + Tier-1 audit but is *not* consumed here:
        # this class only computes the marker-side gradient. The k-space term
        # is owned by the inherited / caller-side DPS sampler (it needs the
        # measured ``y`` + forward operator). We therefore do NOT mirror it
        # onto ``self`` as a dead, never-read attribute (CLAUDE.md pitfall #15).
        self.lambda_M = float(td.lambda_M)
        self.sigma_M = float(td.sigma_M)
        self.phase_stego_basis = td.phase_stego_basis
        self.guidance_callback_interval = int(td.guidance_callback_interval)
        # Lazy marker template load (deferred to first sampling call so
        # construction is cheap during config validation).
        self._marker_template_path = str(td.marker_template_path)
        self._marker_template: torch.Tensor | None = None
        self._marker_score_loss = PhaseStegoScoreLoss(
            sigma_M=self.sigma_M, basis=self.phase_stego_basis
        )
        logger.info(
            "TwinLikelihoodDPSStrategy initialised: lambda_M=%.3f, "
            "sigma_M=%.4f, basis=%s, callback_interval=%d (lambda_y is owned "
            "by the inherited DPS sampler, not this class).",
            self.lambda_M,
            self.sigma_M,
            self.phase_stego_basis,
            self.guidance_callback_interval,
        )

    def _load_marker_template(self, device: torch.device) -> torch.Tensor:
        if self._marker_template is None:
            self._marker_template = torch.load(
                self._marker_template_path, map_location=device, weights_only=True
            )
        return self._marker_template.to(device)

    def marker_score(self, x_0_hat: torch.Tensor) -> torch.Tensor:
        r"""Compute the marker-likelihood loss for a Tweedie estimate.

        Returns a scalar tensor equal to
        :math:`\| M(\arg(\hat x_0)) - M \|_2^2 / (2 \sigma_M^2)`.
        """
        marker = self._load_marker_template(x_0_hat.device)
        return self._marker_score_loss(x_0_hat, marker)

    def compute_guidance_gradient(
        self,
        x_t: torch.Tensor,
        x_0_hat_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        r"""Return the marker score :math:`\nabla_{x_t} \log p(M \mid x_t)`.

        The marker likelihood is :math:`\log p(M \mid x_t) =
        -\| M(\arg(\hat x_0(x_t))) - M \|_2^2 / (2 \sigma_M^2)` with
        :math:`\hat x_0` the Tweedie posterior-mean estimate; the
        gradient is taken with autograd through the
        ``x_0_hat_fn`` callback. The returned tensor is
        :math:`\lambda_M \cdot \nabla_{x_t} \log p(M \mid x_t)`
        so it can be added directly to the reverse-SDE drift.

        Args:
            x_t: Current diffusion latent (complex tensor). Detached and
                re-leaf'd internally; the caller's tensor is not modified.
            x_0_hat_fn: A callable that maps ``x_t`` to its Tweedie
                estimate :math:`\hat x_0`. Typically this is a thin wrapper
                around :meth:`get_noise_prediction` plus the closed-form
                Tweedie inversion appropriate for the noise schedule.

        Returns:
            Complex tensor of the same shape as ``x_t``, suitable for
            adding to the standard reverse-SDE drift.

        Notes:
            This helper provides the mathematical core of Twin-DPS
            guidance --- the gradient itself. Folding it into the
            sampler requires a hook in :class:`DiffusionTrainingStrategy`
            that ties together the noise predictor, the Tweedie inversion,
            and the reverse-process update; that hook is not yet in
            place (the framework's diffusion strategy uses an
            integrated ``_generate_validation_prediction`` rather than a
            per-step sampler interface). Until then, downstream users
            invoking Twin-DPS at inference time can call this method
            from a custom sampler loop.
        """
        x_leaf = x_t.detach().clone().requires_grad_(True)
        x_0 = x_0_hat_fn(x_leaf)
        loss = self.marker_score(x_0)
        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=x_leaf,
            create_graph=False,
            retain_graph=False,
        )
        # ∇_{x_t} log p = -∇_{x_t} L. The lambda_M scaling lets the
        # caller add the result directly to the standard drift.
        return -self.lambda_M * grad


__all__ = ["TwinLikelihoodDPSStrategy"]
