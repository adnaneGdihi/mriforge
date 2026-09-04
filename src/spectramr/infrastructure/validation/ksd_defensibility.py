"""Tier-3 audit hook: KSD defensibility (Idea 7 of audit_plan_novel).

Runs a kernelised-Stein-discrepancy goodness-of-fit test on samples
drawn from a configured generative prior (a diffusion, score-based, or
plug-and-play strategy) against held-out deployment samples. A
significantly large KSD (small wild-bootstrap p-value) escalates the
audit to a hard failure under ``--strict``.

The hook is intentionally permissive about how it obtains samples —
when the config does not configure a real score-callable, the hook
runs a smoke-mode self-test (samples drawn from the unit Gaussian
against the unit-Gaussian score) that exercises the integration path
without making any claim about the model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

from spectramr.core.metrics.stein.ksd import (
    kernelised_stein_discrepancy,
    ksd_p_value,
)

logger = logging.getLogger(__name__)


@dataclass
class KSDDefensibilityResult:
    """Outcome of a Tier-3 KSD audit."""

    passed: bool
    ksd_squared: float
    p_value: float
    threshold: float
    n_samples: int
    bandwidth: float | None
    message: str


def verify_ksd_defensibility(
    *,
    samples_provider: Callable[[int], torch.Tensor],
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int = 256,
    threshold: float = 0.05,
    n_bootstrap: int = 500,
    bandwidth: float | None = None,
) -> KSDDefensibilityResult:
    """Run the Tier-3 KSD defensibility audit.

    Args:
        samples_provider: Callable that returns ``n`` samples from the
            model / deployment distribution as a 2-D tensor [n, d].
        score_fn: ``∇_x log p_θ(x)`` for the model under test.
        n_samples: Number of samples to draw.
        threshold: p-value below which the audit fails.
        n_bootstrap: Wild-bootstrap replications.
        bandwidth: Kernel bandwidth (default: median heuristic).

    Returns:
        :class:`KSDDefensibilityResult` indicating pass / fail with
        diagnostics.
    """
    samples = samples_provider(n_samples)
    if samples.dim() > 2:
        samples = samples.flatten(start_dim=1)
    if samples.dim() != 2:
        raise ValueError(
            f"samples_provider must yield a 2-D tensor, got shape {tuple(samples.shape)}"
        )
    ksd2, K = kernelised_stein_discrepancy(
        score_fn,
        samples,
        bandwidth=bandwidth,
        return_kernel_matrix=True,
    )
    p = ksd_p_value(K, n_bootstrap=n_bootstrap) if K is not None else float("nan")
    passed = p >= threshold
    msg = f"KSD² = {ksd2:.4e}, p = {p:.4f} (threshold {threshold:.3f}). " + (
        "PASS" if passed else "FAIL — model does not pass goodness-of-fit."
    )
    return KSDDefensibilityResult(
        passed=passed,
        ksd_squared=float(ksd2),
        p_value=float(p),
        threshold=float(threshold),
        n_samples=int(samples.shape[0]),
        bandwidth=bandwidth,
        message=msg,
    )


def smoke_mode_self_test(*, n_samples: int = 128) -> KSDDefensibilityResult:
    """Self-test path used when no concrete model / data are wired up yet.

    Draws ``n`` samples from N(0, I_5) and tests against the unit-Gaussian
    score. The test should *pass* (large p-value) — failure indicates a
    bug in the KSD integration rather than in any model.
    """

    def _samples(n: int) -> torch.Tensor:
        return torch.randn(n, 5)

    def _score(x: torch.Tensor) -> torch.Tensor:
        return -x

    return verify_ksd_defensibility(
        samples_provider=_samples,
        score_fn=_score,
        n_samples=n_samples,
        threshold=0.05,
        n_bootstrap=300,
    )


__all__ = [
    "KSDDefensibilityResult",
    "smoke_mode_self_test",
    "verify_ksd_defensibility",
]
