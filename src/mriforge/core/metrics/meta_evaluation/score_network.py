"""Score-function approximation used by SFA.

The Score-Function Alignment ranker needs an estimate of
``s_phi(x_hat, sigma) ≈ ∇_x log p_sigma(x)``. The ideal source is a
DSM-trained network on clean MRI volumes; absent that we provide an
analytic *empirical-Parzen* score:

    p_sigma(x) ≈ (1 / N) sum_i N(x; x_i, sigma^2 I)
    ∇_x log p_sigma(x) = (1 / sigma^2) * E_{w_i}[ x_i - x ]

with ``w_i`` the softmax over per-reference Mahalanobis distances. This
estimator is consistent in N and matches the idealized DSM target as
sigma -> 0. It is also **closed-form**, so the ranker has zero training
cost and zero external-checkpoint dependency.

A user with a pretrained score model can pass it in via ``ScoreEstimator``
- the alignment ranker only requires the ``score(x, sigma)`` callable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass
class ScoreEstimatorConfig:
    sigma_levels: list[float]
    bandwidth_floor: float = 1e-2  # to prevent /0 for tiny sigma


def empirical_score(x: torch.Tensor, references: torch.Tensor, sigma: float) -> torch.Tensor:
    """Closed-form Parzen score against ``references``.

    Args:
        x: query tensor [..., H, W] or any shape; must broadcast with
            ``references`` along the leading reference axis.
        references: [M, ...] clean reference samples, same trailing shape
            as ``x``.
        sigma: noise scale.

    Returns:
        Score tensor with the same shape as ``x``.
    """
    if references.shape[0] == 0:
        return torch.zeros_like(x)
    if torch.is_complex(x) or torch.is_complex(references):
        x = x.abs() if torch.is_complex(x) else x
        references = references.abs() if torch.is_complex(references) else references
    sigma2 = max(float(sigma**2), 1e-9)
    # Flatten trailing dims for stable softmax.
    flat_x = x.reshape(1, -1).float()
    flat_ref = references.reshape(references.shape[0], -1).float()
    # Center-pad / crop refs to match query's flat dim.
    if flat_ref.shape[1] != flat_x.shape[1]:
        target = flat_x.shape[1]
        if flat_ref.shape[1] > target:
            flat_ref = flat_ref[:, :target]
        else:
            pad = target - flat_ref.shape[1]
            flat_ref = torch.cat([flat_ref, flat_ref.new_zeros(flat_ref.shape[0], pad)], dim=1)
    diff = flat_ref - flat_x  # [M, P]
    sq = (diff**2).sum(dim=1) / (2 * sigma2)
    log_w = -sq
    log_w = log_w - log_w.max()
    w = torch.softmax(log_w, dim=0).unsqueeze(1)  # [M, 1]
    weighted_diff = (w * diff).sum(dim=0)  # [P]
    score = (weighted_diff / sigma2).reshape(x.shape)
    return score.to(x.dtype)


class EmpiricalParzenScore:
    """Callable wrapping ``empirical_score`` with a fixed reference set."""

    def __init__(self, references: torch.Tensor) -> None:
        self.references = references

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return empirical_score(x, self.references, sigma)


ScoreFn = Callable[[torch.Tensor, float], torch.Tensor]


def build_default_score_fn(reference_pool: Sequence[torch.Tensor]) -> ScoreFn:
    """Build an empirical-Parzen score from a list of clean references."""
    flat = []
    for ref in reference_pool:
        if torch.is_complex(ref):
            ref = ref.abs()
        flat.append(ref.float().reshape(-1))
    if not flat:
        # Degenerate: returns zero score, SFA will simply produce 0.
        return lambda x, sigma: torch.zeros_like(x)
    target = max(t.numel() for t in flat)
    padded = []
    for t in flat:
        if t.numel() < target:
            t = torch.cat([t, t.new_zeros(target - t.numel())])
        padded.append(t)
    refs = torch.stack(padded, dim=0)
    return EmpiricalParzenScore(refs)
