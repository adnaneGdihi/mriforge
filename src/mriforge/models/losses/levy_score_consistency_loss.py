r"""alpha-stable Lévy score-consistency loss for heavy-tailed MRI residuals.

Standard score-based diffusion assumes a Gaussian (Brownian) forward
process, so the denoising-score-matching residual is implicitly fit to a
*light-tailed* model. For magnitude MRI at low SNR (the ULF regime) the
residuals follow heavy-tailed Rician / non-central-chi statistics whose
tails a Gaussian objective under-weights, producing over-confident
reconstructions.

This loss replaces the Gaussian assumption with the :math:`\alpha`-stable
Lévy model :math:`L_t^{(\alpha)}`, :math:`\alpha\in(0,2]`
(:math:`\alpha=2` recovers Brownian motion / the usual VP-SDE). It wraps
:func:`mriforge.models.diffusion.alpha_stable_levy.fractional_score_matching_loss`
as the training objective: the residual between ``prediction`` and
``target`` is treated as the network's *score estimate* and is matched
against the fractional (alpha-stable) score target
:math:`-\,\mathrm{noise}/\sigma^{\alpha}` rather than the Gaussian
:math:`-\,\mathrm{noise}/\sigma^{2}`.

Consistency check: at :math:`\alpha=2` the target reduces to the standard
Gaussian DSM score, so the loss degrades gracefully to the familiar
VP-SDE objective.

See :mod:`mriforge.models.diffusion.alpha_stable_levy` for the underlying
Chambers-Mallows-Stuck simulator and the fractional score primitive.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.diffusion.alpha_stable_levy import (
    fractional_score_matching_loss,
)
from mriforge.models.losses.registry import register_loss


@register_loss(
    name="levy_score_consistency",
    aliases=["alpha_stable_score", "fractional_levy_score"],
    domain="image",
)
class LevyScoreConsistencyLoss(nn.Module):
    r"""Heavy-tailed alpha-stable Lévy score-matching objective.

    The loss interprets the residual :math:`r = \text{prediction} -
    \text{target}` as the network's denoising score estimate
    :math:`s_\theta` and drives it toward the alpha-stable (fractional) score
    target via
    :func:`~mriforge.models.diffusion.alpha_stable_levy.fractional_score_matching_loss`.
    Because that primitive perturbs the *clean* signal (``target``) with an
    alpha-stable Lévy sample and matches against :math:`-\text{noise}/\sigma^\alpha`,
    minimising this loss forces the reconstruction residual to follow the
    heavy-tailed model instead of a Gaussian one.

    Args:
        alpha: Stability index :math:`\alpha\in(0,2]`. ``alpha=2`` is the
            Gaussian / Brownian limit; smaller values are heavier-tailed.
        sigma: Lévy noise dispersion :math:`\sigma > 0`.
        weight: Scalar multiplier applied to the returned loss.
        generator: Optional :class:`torch.Generator` for reproducible
            alpha-stable noise draws (useful in tests).
    """

    def __init__(
        self,
        alpha: float = 1.7,
        sigma: float = 1.0,
        weight: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < alpha <= 2.0):
            raise ValueError(f"alpha must lie in (0, 2]; got {alpha}.")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive; got {sigma}.")
        if weight < 0.0:
            raise ValueError(f"weight must be >= 0; got {weight}.")
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.weight = float(weight)
        self.generator = generator

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute the alpha-stable Lévy score-matching loss.

        Args:
            prediction: Reconstructed signal ``(B, ...)``.
            target: Ground-truth / clean signal, same shape as ``prediction``.
            **kwargs: Ignored (kept for the registry loss contract).

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If ``prediction`` and ``target`` shapes differ.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have identical shapes; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )

        # The network's score estimate is the residual it produces against
        # the clean target. ``fractional_score_matching_loss`` re-noises the
        # clean signal with an alpha-stable Lévy draw ``noise`` (giving x_t) and
        # matches ``score_model(x_t)`` to the fractional target
        # ``-noise / sigma**alpha``. We expose the reconstruction residual as
        # that score estimate, so minimising the loss drives the residual to
        # the heavy-tailed alpha-stable score (whose distribution and the
        # ``sigma**alpha`` scaling both depend on ``alpha``) rather than the
        # Gaussian one. At ``alpha=2`` the target collapses to the standard
        # Gaussian DSM score, recovering the usual VP-SDE objective.
        residual = prediction - target

        def score_model(x_t: torch.Tensor) -> torch.Tensor:
            # x_t is unused: the residual is a per-pixel score estimate that
            # must align with the alpha-stable target the primitive supplies.
            del x_t
            return residual

        loss = fractional_score_matching_loss(
            score_model,
            target,
            alpha=self.alpha,
            sigma=self.sigma,
            generator=self.generator,
        )
        return self.weight * loss
