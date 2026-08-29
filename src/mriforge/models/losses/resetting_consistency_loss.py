r"""Stochastic-resetting consistency loss for resetting-diffusion training.

The ``resetting_diffusion`` paradigm advertises *stochastic-resetting*
diffusion mathematics (Evans & Majumdar, 2011; Evans, Majumdar & Schehr,
2020): the reverse-SDE sample is intermittently reset toward a
data-consistent anchor, which strictly accelerates first-passage to the
data manifold whenever the coefficient of variation of the unreset
passage-time distribution exceeds 1. At the optimal resetting rate the
passage-time distribution sits at *criticality*, where

.. math::

    \mathrm{CoV}(\tau) = \frac{\sqrt{\mathrm{Var}(\tau)}}{\mathbb{E}[\tau]} = 1 .

This loss turns that physics into a trainable objective rather than a
façade. It treats the residual magnitude process
:math:`|\mathrm{prediction}-\mathrm{target}|` as an empirical sample of
the first-passage / residual distribution and combines three terms, all
of which route through the genuine primitives in
:mod:`mriforge.models.diffusion.stochastic_resetting`:

1. **Criticality term** — :math:`(\mathrm{CoV}(\tau)-1)^2`, driving the
   residual distribution toward the optimal-resetting condition.
2. **Reset-consistency term** — applies one Bernoulli
   :func:`~mriforge.models.diffusion.stochastic_resetting.apply_resetting`
   step to the prediction (anchor = target) and penalises the remaining
   discrepancy to the target. It is exactly zero when the prediction
   already coincides with the anchor.
3. **MFPT term** — penalises the mean-first-passage-time
   :func:`~mriforge.models.diffusion.stochastic_resetting.mfpt_with_resetting`
   of the residual passage proxy at a fixed resetting rate, so that
   training shortens the expected passage to the data manifold.

The primitive module is imported as a module (``import ... as sr``) and
its functions are accessed as attributes so that callers / tests can spy
on or substitute them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import mriforge.models.diffusion.stochastic_resetting as sr
from mriforge.models.losses.registry import register_loss


@register_loss(
    name="resetting_consistency",
    aliases=["stochastic_resetting_consistency", "mfpt_consistency"],
    domain="image",
)
class ResettingConsistencyLoss(nn.Module):
    r"""Stochastic-resetting consistency / criticality loss.

    ``prediction`` is the network output (the diffusion iterate) and
    ``target`` is the data-consistent anchor :math:`\mathcal{R}(\hat y)`.
    The residual magnitude :math:`|\mathrm{prediction}-\mathrm{target}|`
    is treated as the passage / residual process.

    Args:
        criticality_weight: Weight on :math:`(\mathrm{CoV}(\tau)-1)^2`.
        reset_weight: Weight on the post-``apply_resetting`` discrepancy.
        mfpt_weight: Weight on the mean-first-passage-time term.
        reset_rate: Resetting rate :math:`r` used by ``apply_resetting``
            and ``mfpt_with_resetting``.
        reset_dt: Time step :math:`dt` used by ``apply_resetting``.
        reduction: ``"mean"`` or ``"sum"`` for the reset-consistency
            discrepancy.
    """

    def __init__(
        self,
        criticality_weight: float = 1.0,
        reset_weight: float = 1.0,
        mfpt_weight: float = 0.1,
        reset_rate: float = 1.0,
        reset_dt: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        for nm, val in (
            ("criticality_weight", criticality_weight),
            ("reset_weight", reset_weight),
            ("mfpt_weight", mfpt_weight),
        ):
            if val < 0:
                raise ValueError(f"{nm} must be >= 0; got {val}.")
        if reset_rate < 0:
            raise ValueError("reset_rate must be >= 0.")
        if reset_dt <= 0:
            raise ValueError("reset_dt must be > 0.")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}.")
        self.criticality_weight = criticality_weight
        self.reset_weight = reset_weight
        self.mfpt_weight = mfpt_weight
        self.reset_rate = reset_rate
        self.reset_dt = reset_dt
        self.reduction = reduction

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

        reduce = torch.mean if self.reduction == "mean" else torch.sum

        # Residual magnitude as the empirical passage / residual sample.
        residual = (prediction - target).abs()
        passage = residual.flatten()

        # (1) Criticality: |CoV - 1|^2. Optimal resetting sits at CoV = 1.
        cov = sr.coefficient_of_variation(passage)
        criticality = (cov - 1.0).pow(2)

        # (2) Reset consistency: one Bernoulli reset toward the anchor, then
        # penalise the leftover discrepancy. apply_resetting is differentiable
        # in the non-reset branch (torch.where), so the gradient reaches the
        # prediction. Anchor = target (the data-consistent reset state).
        reset_iterate = sr.apply_resetting(
            prediction,
            target,
            rate=self.reset_rate,
            dt=self.reset_dt,
        )
        reset_discrepancy = reduce((reset_iterate - target).abs())

        # (3) Mean-first-passage-time at the configured rate. Lower MFPT means
        # faster convergence to the data manifold. The empirical Laplace
        # transform is differentiable in the passage samples.
        mfpt = sr.mfpt_with_resetting(passage, self.reset_rate)

        return (
            self.criticality_weight * criticality
            + self.reset_weight * reset_discrepancy
            + self.mfpt_weight * mfpt
        )
