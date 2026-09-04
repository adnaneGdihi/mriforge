"""KSD metric — Idea 7 of audit_plan_novel.

Wraps the Phase 0 KSD utility (``src/core/metrics/stein/ksd.py``) as a
framework-registered metric. The score function defaults to the
"diffusion-prior" score evaluated on each sample, but any callable
``score_fn(samples) -> scores`` is accepted via the kwargs path.

This metric is generally invoked from the audit ladder rather than from
the training loop. See ``src/infrastructure/validation/ksd_defensibility.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from spectramr.core.metrics.registry import register_metric
from spectramr.core.metrics.stein.ksd import (
    kernelised_stein_discrepancy,
    ksd_p_value,
)


# needs=("prior_model",): KSD requires a score_fn ∇ₓ log p_θ(x), i.e. a trained
# density/score model. That is the un-providable prior_model, so declaring it here
# makes the metric classify as not_applicable (expected) rather than
# never_computed (a defect) on a sweep that has no such model.
@register_metric(
    "kernelised_stein_discrepancy",
    aliases=["ksd"],
    needs=("prior_model",),
)
class KernelisedSteinDiscrepancyMetric:
    """Goodness-of-fit metric: lower is better (KSD² → 0 ⇔ p_θ ≈ Q_n).

    Args (constructor):
        bandwidth: Optional kernel bandwidth (defaults to median heuristic).
        n_bootstrap: Wild-bootstrap replications for p-value (0 disables).
    """

    def __init__(
        self,
        *,
        bandwidth: float | None = None,
        n_bootstrap: int = 500,
    ) -> None:
        self.bandwidth = bandwidth
        self.n_bootstrap = n_bootstrap
        self._last_p_value: float | None = None

    @property
    def name(self) -> str:
        return "kernelised_stein_discrepancy"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def last_p_value(self) -> float | None:
        """Most recent wild-bootstrap p-value; ``None`` if not computed."""
        return self._last_p_value

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> float:
        """Compute KSD² between ``prediction`` and ``target`` distributions.

        Conventions:
            - ``prediction``: samples drawn from the *model* (shape [n, d]
              or [n, C, ...] flattened along non-batch axes).
            - ``target``: not used directly; placeholder kept for the
              ``IMetric`` protocol.
            - ``score_fn`` (kwargs): **required** callable mapping [n, d] →
              [n, d] giving ∇_x log p_θ(x). Omitting it raises ``ValueError``
              — there is no meaningful default (the Gaussian score ``-x``
              reduces KSD to a standardisation check and is almost never
              what a caller wants).
        """
        del target  # KSD is a one-sample test; target is a protocol stub.
        # No-silent-fallback (CLAUDE.md #9): a missing score_fn must raise
        # rather than degrade to the Gaussian score.
        score_fn: Callable[[torch.Tensor], torch.Tensor] | None = kwargs.get("score_fn")  # type: ignore[assignment]
        if score_fn is None:
            raise ValueError(
                "KernelisedSteinDiscrepancyMetric requires a 'score_fn' kwarg "
                "(callable [n, d] -> [n, d] giving the model score "
                "grad_x log p_theta(x)); no meaningful default exists."
            )
        samples = prediction.flatten(start_dim=1) if prediction.dim() > 2 else prediction
        ksd2, K = kernelised_stein_discrepancy(
            score_fn,
            samples,
            bandwidth=self.bandwidth,
            return_kernel_matrix=self.n_bootstrap > 0,
        )
        if self.n_bootstrap > 0 and K is not None:
            self._last_p_value = ksd_p_value(K, n_bootstrap=self.n_bootstrap)
        return float(ksd2)


__all__ = ["KernelisedSteinDiscrepancyMetric"]
