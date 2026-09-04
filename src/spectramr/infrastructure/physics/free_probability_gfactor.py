r"""Marchenko-Pastur *surrogate* for the SENSE g-factor spectral bulk.

The SENSE g-factor

.. math::

    G^{2} \;=\; \bigl[(S^{*}\Sigma^{-1}S)^{-1}\bigr]\,
                  \mathrm{diag}\!\bigl(S^{*}\Sigma^{-1}S\bigr),

with :math:`S` a coil sensitivity matrix and :math:`\Sigma` the noise
covariance, has a bulk spectral distribution that — for a structured-random
encoding matrix with aspect ratio :math:`c=n_c/n_p\in(0,1)` — is **modelled
here by the Marchenko-Pastur law** on
:math:`[\lambda_-,\lambda_+]=[(1-\sqrt c)^2,(1+\sqrt c)^2]`.

.. warning::

   This is a **heuristic surrogate for the spectral bulk, not an exact
   result**. An earlier version claimed the g²-density followed from
   Voiculescu S-transform multiplicativity
   (:math:`S_A(z)S_B(z)=(1+cz)^{-1}(1+cz)=1/(1+cz)`). That derivation is
   **wrong**: the S-transform of a multiplicative *inverse* is
   :math:`S_{\mu^{-1}}(z)=1/S_\mu(-1-z)=1-c-cz`, not :math:`(1+cz)`, and the
   product does not collapse back to MP. More fundamentally, the g-factor is an
   *element-wise product of the diagonals* of :math:`A^{-1}` and :math:`A`
   (with :math:`A=S^{*}\Sigma^{-1}S`) — diagonals of a matrix product are not a
   free multiplicative convolution of the operands' spectra, so there is no
   exact closed form. The MP law is retained only as a tractable bulk proxy.

This module returns:

* :func:`mp_s_transform` — the (correct) MP S-transform :math:`1/(1+cz)`.
* :func:`marchenko_pastur_density` — the closed-form MP density.
* :func:`asymptotic_gfactor_density` — the MP **surrogate** for the g² bulk.
* :func:`gfactor_moments` — first two moments of the MP law (mean 1, var c).
* :func:`worst_case_gfactor` — the MP upper-edge :math:`\sqrt{\lambda_+}=1+\sqrt c`,
  used as a heuristic worst-case proxy.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "asymptotic_gfactor_density",
    "gfactor_moments",
    "marchenko_pastur_density",
    "mp_s_transform",
    "worst_case_gfactor",
]


def mp_s_transform(z: torch.Tensor, c: float) -> torch.Tensor:
    r"""Voiculescu S-transform of the Marchenko-Pastur law :math:`\mu_{\mathrm{MP},c}`.

    Closed form: :math:`S_{\mu_{\mathrm{MP},c}}(z)=1/(1+cz)` for
    :math:`c\in(0,1)`. See Mingo-Speicher (2017), §3.3.2.
    """
    if not 0.0 < c < 1.0:
        raise ValueError(f"MP parameter c must lie in (0, 1); got {c}.")
    return 1.0 / (1.0 + c * z)


def marchenko_pastur_density(
    t: torch.Tensor, c: float, *, support_eps: float = 1e-8
) -> torch.Tensor:
    r"""Marchenko-Pastur density :math:`\rho(t)=\sqrt{(t-\lambda_-)(\lambda_+-t)}/(2\pi c t)`.

    Returns zero outside the support :math:`[\lambda_-,\lambda_+]`.

    Args:
        t: Tensor of points at which to evaluate the density. Strictly
            positive entries only.
        c: Aspect-ratio parameter :math:`c=n_c/n_p\in(0,1)`.
        support_eps: Numerical floor below which support boundaries are
            treated as exact.

    Returns:
        Density values, same shape as ``t``.
    """
    if not 0.0 < c < 1.0:
        raise ValueError(f"MP parameter c must lie in (0, 1); got {c}.")
    lam_minus = (1.0 - math.sqrt(c)) ** 2
    lam_plus = (1.0 + math.sqrt(c)) ** 2
    inside = (t > lam_minus - support_eps) & (t < lam_plus + support_eps)
    inside_safe = (t - lam_minus).clamp_min(0.0) * (lam_plus - t).clamp_min(0.0)
    density = torch.sqrt(inside_safe) / (2.0 * math.pi * c * t.clamp_min(support_eps))
    return torch.where(inside, density, torch.zeros_like(density))


def asymptotic_gfactor_density(t: torch.Tensor, c: float) -> torch.Tensor:
    r"""Marchenko-Pastur **surrogate** for the g² spectral bulk.

    Returns the Marchenko-Pastur density on
    :math:`[\lambda_-,\lambda_+]` as a tractable proxy for the bulk of the
    SENSE g² distribution at aspect ratio ``c``. This is a heuristic model, not
    an exact free-probability result — see the module-level warning (the
    S-transform-multiplicativity derivation it replaces is incorrect). Use it
    for order-of-magnitude bulk shape only, not as a certified bound.
    """
    return marchenko_pastur_density(t, c)


def gfactor_moments(c: float) -> tuple[float, float]:
    r"""First two moments of the asymptotic g² density.

    Returns:
        ``(mean, variance)``. Closed forms for the Marchenko-Pastur law:
        :math:`\mathbb{E}[t]=1`, :math:`\mathrm{Var}(t)=c`.
    """
    if not 0.0 < c < 1.0:
        raise ValueError(f"MP parameter c must lie in (0, 1); got {c}.")
    return 1.0, c


def worst_case_gfactor(c: float) -> float:
    r"""Heuristic worst-case g-factor proxy :math:`\sqrt{\lambda_+}=1+\sqrt c`.

    The Marchenko-Pastur upper edge :math:`\sqrt{(1+\sqrt c)^{2}}=1+\sqrt c`.
    Under the MP **surrogate** model (see the module warning) this is taken as a
    worst-case proxy for ``max g``; it is not a certified upper bound on the true
    SENSE g-factor, whose bulk MP only approximates.
    """
    if not 0.0 < c < 1.0:
        raise ValueError(f"MP parameter c must lie in (0, 1); got {c}.")
    return 1.0 + math.sqrt(c)
