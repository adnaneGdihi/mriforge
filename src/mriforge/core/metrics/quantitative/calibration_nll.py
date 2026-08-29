"""Gaussian-NLL calibration metric for heteroscedastic predictions (B-2.9).

Scores a 2-channel ``[mean, log-variance]`` prediction against the target by the
Gaussian negative log-likelihood — the metric that actually consumes the predicted
variance, so a heteroscedastic arm's calibration claim (and its variance-prior
ablation) is measurable at validation (lower NLL = better-calibrated uncertainty).
A strictly proper scoring rule: minimised by the true predictive mean AND variance.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.registry import register_metric


@register_metric("gaussian_nll", aliases=["het_nll"])
class GaussianNLLMetric:
    """Gaussian NLL over a 2-channel ``[mean, logvar]`` prediction (lower better)."""

    #: The one metric that consumes the whole distribution head, so it must NOT be
    #: narrowed to the target's channel count by
    #: ``ValidationMetricsComputer._align_prediction`` — it slices [mean, logvar] itself.
    REQUIRES_MATCHING_SHAPES: bool = False

    def __init__(self, device: object = None, **_: object) -> None:
        # The metrics computer constructs every metric as get(name, device=...);
        # accept (and ignore) device/extra kwargs (else a constructor error is
        # silently stored as NaN at validation).
        self._device = device

    @property
    def name(self) -> str:
        return "gaussian_nll"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> float:
        if prediction.shape[1] < 2:
            raise ValueError(
                "gaussian_nll needs a 2-channel [mean, logvar] prediction; got "
                f"{prediction.shape[1]} channel(s)."
            )
        # Lazy import: a module-scope import of models.losses from inside
        # core.metrics triggers the full losses-package init mid-way through
        # the metrics walk-discovery (and core → models is against the layer
        # direction). Deferring to call time keeps the package inits
        # independent.
        from mriforge.models.losses.heteroscedastic_losses import heteroscedastic_nll

        mean = prediction[:, 0:1]
        logvar = prediction[:, 1:2]
        return float(heteroscedastic_nll(mean, logvar, target).item())


__all__ = ["GaussianNLLMetric"]
