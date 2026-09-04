r"""Fisher-efficient injective neural fingerprinting certificates (A-1.1 *FEINF*).

MR-fingerprinting (MRF) inverts a tissue-parameter vector :math:`\theta` (T1, T2,
…) from a measured signal *fingerprint* :math:`s(\theta)`. Two properties decide
whether that inversion is trustworthy, and FEINF turns each into a **certificate**
(a measurable scalar), reusing the framework's Bloch simulator / MRF dictionary to
supply the fingerprints and their Jacobians:

* **Fisher efficiency** — is the estimator as precise as physics allows? The
  Cramér-Rao lower bound :math:`\mathrm{CRLB}=\sigma^2 (J^\top J)^{-1}` with
  :math:`J=\partial s/\partial\theta` is the smallest achievable estimator
  covariance; no unbiased estimator can beat it. The per-parameter efficiency is
  :math:`\eta_d=\mathrm{CRLB}_{dd}/\mathrm{Var}_d\in(0,1]` — ``1`` is statistically
  optimal, and :math:`\eta_d>1` is *impossible* (a self-check on any estimator).

* **Injectivity** — do distinct tissues produce distinct fingerprints, so the
  inverse is well-posed? The global bi-Lipschitz lower bound
  :math:`\mu=\min_{i\neq j}\lVert s_i-s_j\rVert/\lVert\theta_i-\theta_j\rVert>0`
  certifies the fingerprint map is injective on the sampled grid; the local
  version is :math:`\sigma_{\min}(J)>0` (a rank-deficient Jacobian means locally
  non-identifiable directions).

These are **certificate primitives** (operating on Jacobians / fingerprint sets
from either the Bloch dictionary or a learned encoder), not a training paradigm —
shipping them as functions + a cert object is the honest scope.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def cramer_rao_lower_bound(jacobian: Tensor, noise_var: float = 1.0) -> Tensor:
    r"""Per-parameter CRLB variances :math:`\mathrm{diag}(\sigma^2 (J^\top J)^{-1})`.

    Args:
        jacobian: ``[n_obs, n_params]`` signal Jacobian :math:`\partial s/\partial\theta`
            (complex Jacobians split into stacked real/imag rows).
        noise_var: measurement noise variance :math:`\sigma^2` (>0).

    Returns:
        ``[n_params]`` CRLB variance lower bounds. Non-identifiable parameters
        (singular Fisher matrix) yield ``+inf``.
    """
    if noise_var <= 0:
        raise ValueError(f"noise_var must be > 0, got {noise_var}")
    j = jacobian
    if j.is_complex():
        j = torch.cat([j.real, j.imag], dim=0)
    if j.dim() != 2:
        raise ValueError(f"jacobian must be 2-D [n_obs, n_params], got {tuple(j.shape)}")
    fisher = (j.transpose(-2, -1) @ j).double() / float(noise_var)  # [p, p]
    p = fisher.shape[0]
    eye = torch.eye(p, dtype=fisher.dtype, device=fisher.device)
    # Pseudo-inverse-safe: if singular, the parameter is non-identifiable -> inf CRLB.
    try:
        crlb = torch.linalg.solve(fisher, eye)
    except RuntimeError:
        crlb = torch.linalg.pinv(fisher)
        diag = torch.diagonal(crlb).clone()
        # Mark directions in the null space (zero Fisher) as +inf.
        null = torch.diagonal(fisher) <= 0
        diag[null] = float("inf")
        return diag.to(torch.float32)
    return torch.diagonal(crlb).clamp_min(0.0).to(torch.float32)


def crb_efficiency(
    jacobian: Tensor, estimator_variance: Tensor, noise_var: float = 1.0, eps: float = 1e-12
) -> Tensor:
    r"""Per-parameter Fisher efficiency :math:`\eta_d = \mathrm{CRLB}_d / \mathrm{Var}_d \in (0, 1]`.

    ``1`` is statistically optimal; a value ``> 1`` is impossible (the CRLB is a
    floor) and signals a buggy / biased estimator or an under-estimated variance.
    """
    crlb = cramer_rao_lower_bound(jacobian, noise_var)
    var = estimator_variance.detach().float().reshape(-1)
    if var.numel() != crlb.numel():
        raise ValueError(f"estimator_variance has {var.numel()} entries, expected {crlb.numel()}")
    return crlb / (var + eps)


def injectivity_margin(fingerprints: Tensor, params: Tensor, eps: float = 1e-12) -> float:
    r"""Global bi-Lipschitz lower bound :math:`\min_{i\neq j}\lVert s_i-s_j\rVert/\lVert\theta_i-\theta_j\rVert`.

    ``> 0`` certifies the fingerprint map is injective on the sampled grid (distinct
    parameters give distinct fingerprints). ``0`` means two distinct parameter
    points collide. ``fingerprints`` ``[N, d_s]``, ``params`` ``[N, d_theta]``.
    """
    s = fingerprints.detach().float().reshape(fingerprints.shape[0], -1)
    t = params.detach().float().reshape(params.shape[0], -1)
    if s.shape[0] != t.shape[0] or s.shape[0] < 2:
        raise ValueError("need >=2 matching fingerprint/param rows")
    ds = torch.cdist(s, s)  # [N, N] fingerprint distances
    dt = torch.cdist(t, t)  # [N, N] parameter distances
    n = s.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=s.device)
    valid = off & (dt > eps)  # ignore the diagonal and any coincident params
    if not bool(valid.any()):
        return float("nan")
    ratio = ds[valid] / dt[valid]
    return float(ratio.min())


def local_injectivity_margin(jacobian: Tensor) -> float:
    r"""Local injectivity certificate :math:`\sigma_{\min}(J)`.

    ``> 0`` ⇒ the Jacobian is full column rank (locally identifiable); ``~0`` ⇒ a
    locally non-identifiable parameter direction.
    """
    j = jacobian
    if j.is_complex():
        j = torch.cat([j.real, j.imag], dim=0)
    if j.dim() != 2:
        raise ValueError(f"jacobian must be 2-D [n_obs, n_params], got {tuple(j.shape)}")
    sv = torch.linalg.svdvals(j.double())
    return float(sv.min())


class FisherEfficientFingerprintCert:
    """Bundle the FEINF certificates for a fingerprint design (A-1.1).

    Reports Fisher efficiency (given an estimator's empirical variance), the global
    injectivity margin, and the local injectivity margin (min singular value of the
    Jacobian) — the three numbers that certify an MRF inversion is precise and
    well-posed.
    """

    def __init__(self, noise_var: float = 1.0) -> None:
        if noise_var <= 0:
            raise ValueError(f"noise_var must be > 0, got {noise_var}")
        self.noise_var = float(noise_var)

    def certify(
        self,
        jacobian: Tensor,
        fingerprints: Tensor,
        params: Tensor,
        estimator_variance: Tensor | None = None,
    ) -> dict[str, Any]:
        """Return ``{crlb, efficiency, injectivity_margin, local_injectivity}``."""
        crlb = cramer_rao_lower_bound(jacobian, self.noise_var)
        out: dict[str, Any] = {
            "crlb": crlb,
            "injectivity_margin": injectivity_margin(fingerprints, params),
            "local_injectivity": local_injectivity_margin(jacobian),
        }
        if estimator_variance is not None:
            out["efficiency"] = crb_efficiency(jacobian, estimator_variance, self.noise_var)
        return out


__all__ = [
    "FisherEfficientFingerprintCert",
    "cramer_rao_lower_bound",
    "crb_efficiency",
    "injectivity_margin",
    "local_injectivity_margin",
]
