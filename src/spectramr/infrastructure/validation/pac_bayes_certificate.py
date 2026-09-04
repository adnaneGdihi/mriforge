"""PAC-Bayes generalisation certificate (ULF-PR-24 / R5).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-24.

Implements the canonical McAllester PAC-Bayes bound for a learned
posterior ``Q`` over network weights and a fixed prior ``P``:

    L(Q) ≤ L̂(Q) + sqrt( (KL(Q ‖ P) + ln(2 √n / δ)) / (2 (n − 1)) )

We use a Gaussian variational posterior ``Q = N(θ, Σ)`` and a Gaussian
prior ``P = N(0, σ_P²)``.  KL has a closed form:

    KL(Q ‖ P) = ½ ( tr(Σ)/σ_P² + ‖θ‖²/σ_P² − k + ln(σ_P^{2k} / |Σ|) )

For practical use we treat each parameter as having an independent
diagonal posterior variance ``σ_q²`` (a single learned scalar per
parameter, or shared across the model), which collapses the bound to a
sum over scalars.

Public API
----------

- :func:`gaussian_kl_diagonal` — KL(Q ‖ P) for diagonal Gaussians.
- :func:`mcallester_bound`     — empirical-risk → upper bound.
- :class:`PACBayesCertifier`   — convenience wrapper that runs an
  empirical-risk pass on a held-out set and produces the certificate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn


def gaussian_kl_diagonal(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_sigma: float,
) -> torch.Tensor:
    """KL(N(μ, diag(σ²)) ‖ N(0, σ_p² I))  per element, summed."""
    var = posterior_logvar.exp()
    return 0.5 * (
        var.sum() / (prior_sigma**2)
        + posterior_mean.pow(2).sum() / (prior_sigma**2)
        - posterior_mean.numel()
        + 2 * math.log(prior_sigma) * posterior_mean.numel()
        - posterior_logvar.sum()
    )


def mcallester_bound(
    empirical_risk: float,
    kl: float,
    n: int,
    delta: float = 0.05,
) -> float:
    """McAllester's PAC-Bayes upper bound on the population risk.

    Returns ``empirical_risk + sqrt((KL + ln(2 √n / δ)) / (2 (n−1)))``.

    PRECONDITION: the loss must be bounded in ``[0, 1]``. The square-root slack
    derives from a Hoeffding-style change-of-measure step that assumes a
    ``[0, 1]``-valued loss; applying this to an unbounded loss (e.g. raw MSE)
    produces a number that is *not* a valid high-probability bound. Callers with
    a loss bounded by ``M ≠ 1`` must normalise to ``[0, 1]`` first and rescale
    the result by ``M`` — :meth:`PACBayesCertifier.certify` does this.
    """
    if n <= 1:
        return float("inf")
    slack = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * (n - 1))
    return float(empirical_risk + math.sqrt(max(slack, 0.0)))


@dataclass
class PACBayesCertificate:
    empirical_risk: float
    kl: float
    n: int
    delta: float
    bound: float

    def to_dict(self) -> dict[str, float]:
        return {
            "empirical_risk": self.empirical_risk,
            "kl_divergence": self.kl,
            "n_samples": float(self.n),
            "delta": self.delta,
            "pac_bayes_bound": self.bound,
        }


class PACBayesCertifier:
    """Compute a PAC-Bayes McAllester bound for a Gaussian variational posterior."""

    def __init__(self, prior_sigma: float = 0.1, delta: float = 0.05) -> None:
        self.prior_sigma = float(prior_sigma)
        self.delta = float(delta)

    def kl_for_model(self, model: nn.Module, posterior_logvar: torch.Tensor | None = None) -> float:
        """Sum KL across all trainable parameters using a single shared
        ``posterior_logvar`` scalar (default: log(1e-2²)).
        """
        if posterior_logvar is None:
            posterior_logvar = torch.log(torch.tensor(1e-4))
        kl = 0.0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            mean = p.detach()
            logvar = posterior_logvar.expand_as(mean).to(mean.device)
            kl += float(gaussian_kl_diagonal(mean, logvar, self.prior_sigma))
        return kl

    def certify(
        self,
        model: nn.Module,
        loss_fn: Callable[[nn.Module], float],
        n_samples: int,
        posterior_logvar: torch.Tensor | None = None,
        loss_bound: float = 1.0,
    ) -> PACBayesCertificate:
        """Produce a McAllester certificate from a held-out empirical risk.

        ``loss_fn`` MUST return a loss bounded in ``[0, loss_bound]`` — the
        McAllester bound is a statement about a bounded loss, so an unbounded
        risk (raw MSE, etc.) yields a number that is not a valid bound. Per
        pitfall #9 (no silent fallbacks) a risk outside ``[0, loss_bound]``
        RAISES rather than emitting a meaningless certificate. For a loss
        bounded by ``M`` the bound is computed on the normalised risk and
        rescaled by ``M``.
        """
        if loss_bound <= 0:
            raise ValueError(f"loss_bound must be > 0, got {loss_bound}.")
        emp_risk = float(loss_fn(model))
        if not (0.0 <= emp_risk <= loss_bound):
            raise ValueError(
                f"PAC-Bayes requires a loss in [0, {loss_bound}], but the "
                f"empirical risk was {emp_risk}. Clip/normalise the loss (e.g. "
                "clipped NMSE in [0, 1]) before certifying, or pass the correct "
                "loss_bound — the McAllester bound is invalid for an unbounded "
                "loss."
            )
        kl = self.kl_for_model(model, posterior_logvar)
        normalised_bound = mcallester_bound(emp_risk / loss_bound, kl, n_samples, self.delta)
        bound = loss_bound * normalised_bound
        return PACBayesCertificate(
            empirical_risk=emp_risk, kl=kl, n=n_samples, delta=self.delta, bound=bound
        )


__all__ = [
    "PACBayesCertificate",
    "PACBayesCertifier",
    "gaussian_kl_diagonal",
    "mcallester_bound",
]
