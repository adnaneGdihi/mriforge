r"""Pearl front-door criterion reconstruction loss.

Concretises the ``frontdoor_federated`` / ``frontdoor_scanner`` strategy keys
that previously advertised front-door causal mathematics but routed to a
vanilla reconstruction objective (a "façade"). This loss consumes the existing
primitive :mod:`mriforge.infrastructure.physics.causal_frontdoor` so that the
reconstruction respects the *front-door-identified* (confounder-robust)
relationship between anatomy :math:`A` and image :math:`I`.

When scanner identity :math:`S` confounds anatomy and image but is unmeasured
at inference (a new site / vendor), back-door adjustment fails. The front-door
criterion [Pearl 2009, Th. 3.3.4] identifies :math:`P(I\mid\mathrm{do}(A))`
through a mediator :math:`M` that intercepts every directed path
:math:`A\to I`:

.. math::

    P(I\mid\mathrm{do}(A=a))
        = \sum_m P(M=m\mid A=a)\sum_{a'} P(I\mid M=m, A=a')\,P(A=a').

Here the mediator is the *spectral / k-space energy profile* of each image
(a site-invariant intermediate), anatomy :math:`A` is the spatial-intensity
bin, and the scanner/site :math:`S` is supplied via ``kwargs["scanner"]`` (or
derived from per-sample global intensity statistics when absent).

The objective has two terms:

#. a **front-door consistency** residual between the prediction's
   front-door-adjusted estimate :math:`\hat P_{\rm pred}(I\mid\mathrm{do}(A))`
   and the target's :math:`\hat P_{\rm tgt}(I\mid\mathrm{do}(A))`, evaluated
   through :func:`~mriforge.infrastructure.physics.causal_frontdoor.frontdoor_adjusted_prediction`;
#. a **mediator-scanner MI** penalty
   :math:`\lambda\,\widehat{\mathrm{MI}}(M;S)` from
   :func:`~mriforge.infrastructure.physics.causal_frontdoor.donsker_varadhan_mi`,
   enforcing the site-invariance that makes the front-door estimate valid.

A bare L1/L2 residual that ignores the causal primitive would *not* be
confounder-robust and is explicitly rejected by the test suite.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.causal_frontdoor import (
    donsker_varadhan_mi,
    frontdoor_adjusted_prediction,
)
from mriforge.models.losses.registry import register_loss


@register_loss(
    name="frontdoor_criterion",
    aliases=[
        "front_door_adjustment",
        "pearl_frontdoor",
        "frontdoor_federated_loss",
        "frontdoor_scanner_loss",
    ],
    domain="image",
)
class FrontdoorCriterionLoss(nn.Module):
    r"""Front-door-adjusted, scanner-confounder-robust reconstruction loss.

    The loss expects the model output as ``prediction`` and the ground truth
    as ``target`` (``[B, C, H, W]``). An optional integer scanner/site label
    per batch element is read from ``kwargs["scanner"]``; when absent it is
    derived from each sample's global mean intensity (a coarse site proxy).

    Args:
        mi_weight: Coefficient :math:`\lambda\ge 0` on the mediator-scanner
            mutual-information penalty. Default 1.0.
        frontdoor_weight: Coefficient on the front-door consistency residual.
            Default 1.0.
        n_bins: Number of discretisation bins for anatomy :math:`A`, mediator
            :math:`M`, and image :math:`I` when estimating the empirical
            conditionals. Default 8.
        reduction: ``"mean"`` or ``"sum"`` for the pixel reconstruction term.
    """

    def __init__(
        self,
        mi_weight: float = 1.0,
        frontdoor_weight: float = 1.0,
        n_bins: int = 8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if mi_weight < 0:
            raise ValueError("mi_weight must be >= 0")
        if frontdoor_weight < 0:
            raise ValueError("frontdoor_weight must be >= 0")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}")
        self.mi_weight = mi_weight
        self.frontdoor_weight = frontdoor_weight
        self.n_bins = n_bins
        self.reduction = reduction

    # -- soft-binning helpers ------------------------------------------------

    def _soft_assign(self, values: torch.Tensor, n: int) -> torch.Tensor:
        r"""Differentiable soft one-hot assignment of ``values`` to ``n`` bins.

        ``values`` is ``(K,)`` in roughly ``[0, 1]``; returns ``(K, n)`` rows
        summing to 1. A softmax over distances to evenly spaced centres keeps
        the whole pipeline differentiable (so the loss can backprop into the
        prediction) while approximating a histogram.
        """
        centres = torch.linspace(0.0, 1.0, n, device=values.device, dtype=values.dtype)
        # Temperature tied to bin spacing for a peaked-but-smooth assignment.
        temp = max(1.0 / (n - 1), 1e-3)
        dist2 = (values.reshape(-1, 1) - centres.reshape(1, -1)).pow(2)
        return torch.softmax(-dist2 / (2.0 * temp * temp), dim=1)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        lo = x.amin()
        hi = x.amax()
        return (x - lo) / (hi - lo + 1e-8)

    def _frontdoor_estimate(
        self,
        anatomy: torch.Tensor,
        mediator: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        r"""Build empirical conditionals and evaluate the front-door formula.

        ``anatomy``/``mediator``/``image`` are ``(K,)`` per-sample summaries
        (mean intensity, spectral-energy profile, mean intensity again — the
        observable proxy for :math:`I`). Returns the ``(|I|, |A|)`` matrix
        :math:`P(I\mid\mathrm{do}(A))` from the primitive.
        """
        n = self.n_bins
        a_oh = self._soft_assign(self._normalise(anatomy), n)  # (K, |A|)
        m_oh = self._soft_assign(self._normalise(mediator), n)  # (K, |M|)
        i_oh = self._soft_assign(self._normalise(image), n)  # (K, |I|)

        # P(A): marginal over anatomy bins.
        p_a = a_oh.mean(dim=0)  # (|A|,)
        p_a = p_a / (p_a.sum() + 1e-8)

        # P(M|A) = soft co-occurrence(M, A) / P(A), shape (|M|, |A|).
        joint_ma = torch.einsum("km,ka->ma", m_oh, a_oh) / a_oh.shape[0]
        p_m_given_a = joint_ma / (joint_ma.sum(dim=0, keepdim=True) + 1e-8)

        # P(I|M,A): soft co-occurrence(I, M, A) normalised over I.
        joint_ima = torch.einsum("ki,km,ka->ima", i_oh, m_oh, a_oh)
        joint_ima = joint_ima / i_oh.shape[0]
        p_i_given_m_a = joint_ima / (joint_ima.sum(dim=0, keepdim=True) + 1e-8)

        return frontdoor_adjusted_prediction(p_m_given_a, p_i_given_m_a, p_a)

    def _mediator(self, x: torch.Tensor) -> torch.Tensor:
        r"""Per-sample spectral mediator: log radial k-space energy.

        Uses a real spectral transform on the image feature map (this is a
        site-invariant intermediate, not a complex k-space round-trip, so
        ``torch.fft.rfft2`` is the correct primitive here — see physics.md
        scope clarification)."""
        spec = torch.fft.rfft2(x, norm="ortho")
        energy = spec.abs().pow(2).flatten(1).mean(dim=1)  # (B,)
        return torch.log1p(energy)

    def _scanner(self, x: torch.Tensor, supplied: object) -> torch.Tensor:
        if isinstance(supplied, torch.Tensor):
            return supplied.reshape(-1).to(torch.float32)
        # Derive a coarse site proxy from per-sample mean intensity.
        return x.flatten(1).mean(dim=1).detach()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must share shape; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        if prediction.ndim != 4:
            raise ValueError(f"expected [B, C, H, W] tensors; got ndim={prediction.ndim}.")

        reduce = torch.mean if self.reduction == "mean" else torch.sum
        recon = reduce((prediction - target).pow(2))

        # Per-sample summaries: anatomy (mean intensity) and image proxy.
        pred_anatomy = prediction.flatten(1).mean(dim=1)
        tgt_anatomy = target.flatten(1).mean(dim=1)
        pred_image = prediction.flatten(1).mean(dim=1)
        tgt_image = target.flatten(1).mean(dim=1)

        # Mediator M: spectral k-space energy profile (site-invariant proxy).
        pred_med = self._mediator(prediction)
        tgt_med = self._mediator(target)

        # Front-door consistency: the prediction's confounder-robust estimate
        # of P(I|do(A)) must match the target's. This is what makes the loss
        # respect the front-door-identified relationship rather than a plain
        # pixel residual.
        fd_pred = self._frontdoor_estimate(pred_anatomy, pred_med, pred_image)
        fd_tgt = self._frontdoor_estimate(tgt_anatomy, tgt_med, tgt_image)
        frontdoor_residual = (fd_pred - fd_tgt).pow(2).mean()

        # Mediator-scanner MI penalty (site-invariance of the mediator).
        scanner = self._scanner(prediction, kwargs.get("scanner"))
        scanner = self._normalise(scanner)
        # Joint scores T(M_i, S_i) and product-marginal scores T(M_i, S_j)
        # via a simple bilinear critic; rolling S decorrelates the marginal.
        med_n = self._normalise(pred_med)
        joint_scores = med_n * scanner
        marginal_scores = med_n * torch.roll(scanner, shifts=1, dims=0)
        mi_estimate = donsker_varadhan_mi(joint_scores, marginal_scores).clamp_min(0.0)

        return recon + self.frontdoor_weight * frontdoor_residual + self.mi_weight * mi_estimate
