r"""Non-central-χ (Bessel-ratio) data-consistency primitive (SOTA plan T4).

For RSS magnitude MRI with ``L`` coils and noise σ, the measured magnitude ``m``
given the true signal ``ν`` is non-central-χ (2L DoF):

.. math::
    p(m\mid\nu) \propto \frac{m^{L}}{\nu^{L-1}}\,
    e^{-(m^2+\nu^2)/2\sigma^2}\, I_{L-1}\!\Big(\tfrac{m\nu}{\sigma^2}\Big).

The score with respect to ν collapses (the ``(L-1)/ν`` terms cancel) to

.. math::
    \frac{\partial(-\log p)}{\partial\nu}
      = \frac{1}{\sigma^2}\Big(\nu - m\,R_L(z)\Big),
    \quad z=\tfrac{m\nu}{\sigma^2},\;
    R_L(z)=\frac{I_L(z)}{I_{L-1}(z)},

so the data-consistency fixed point ``ν = m·R_L(mν/σ²)`` is the nc-χ ML
magnitude. This generalizes the moment estimate
``√max(m²−2Lσ²,0)`` (the high-SNR approximation) and the T1 GSURE k-space term
to the diffusion image/magnitude domain (Rician/EDM strategies).

**Corner cases.** ``L=1`` ⇒ Rician (``I_1/I_0``). High SNR (``z→∞``) ⇒
``R_L→1`` ⇒ residual ``ν−m`` (Gaussian data-consistency / EDM-DPS). Below the
noise floor ``√(2Lσ²)`` ⇒ ML → 0 (signal suppressed, the correct nc-χ behaviour).

``bessel_ratio_i`` evaluates ``R_L`` for any ``L`` via the stable backward
continued fraction ``R_ν = 1/(2ν/z + R_{ν+1})`` (pure-arithmetic, differentiable;
no arbitrary-order Bessel function required), validated against ``torch.special``
``i1e/i0e`` at ``L=1``.
"""

from __future__ import annotations

import torch

__all__ = ["bessel_ratio_i", "ncchi_dc_residual", "ncchi_ml_magnitude"]

_EPS = 1e-8


def bessel_ratio_i(z: torch.Tensor, order: int = 1, n_terms: int = 24) -> torch.Tensor:
    r"""Modified-Bessel ratio ``R_order(z) = I_order(z) / I_{order-1}(z)``.

    Uses the backward continued fraction ``R_ν = 1/(2ν/z + R_{ν+1})`` seeded at a
    high order (where the ratio → 0), which is numerically stable for all ``z ≥ 0``
    and differentiable. Returns values in ``[0, 1)``.

    Args:
        z: non-negative argument (``mν/σ²``).
        order: Bessel order ``L`` (≥ 1).
        n_terms: continued-fraction depth above ``order``.
    """
    if order < 1:
        raise ValueError(f"order (n_coils) must be >= 1, got {order}")
    z = z.clamp(min=_EPS)
    # Seed at the top order with the asymptotic ratio R_top ≈ z/(top+√(top²+z²)),
    # which is accurate at BOTH z→0 (→ z/2·top) and z→∞ (→ 1) — a plain r=0 seed
    # only holds when 2·top/z ≫ 1 and blows up for large z. Downward recurrence
    # R_ν = 1/(2ν/z + R_{ν+1}) then refines it to R_order.
    top = order + n_terms
    r = z / (top + torch.sqrt(top * top + z * z))
    for k in range(top, order - 1, -1):
        r = 1.0 / (2.0 * k / z + r)
    return r


def ncchi_dc_residual(
    nu: torch.Tensor,
    measured_magnitude: torch.Tensor,
    sigma: float,
    n_coils: int = 1,
) -> torch.Tensor:
    r"""nc-χ data-consistency residual ``ν − m·R_L(mν/σ²)`` (σ²·score).

    Its zero is the nc-χ ML magnitude. At high SNR it reduces to ``ν − m`` (the
    Gaussian residual); at ``L=1`` it is the Rician score.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    z = measured_magnitude * nu / (sigma * sigma)
    return nu - measured_magnitude * bessel_ratio_i(z, order=n_coils)


def ncchi_ml_magnitude(
    measured_magnitude: torch.Tensor,
    sigma: float,
    n_coils: int = 1,
    *,
    n_iter: int = 20,
) -> torch.Tensor:
    r"""nc-χ ML magnitude via the fixed-point ``ν ← m·R_L(mν/σ²)``.

    Seeded at the moment estimate ``√max(m²−2Lσ²,0)`` for fast, stable
    convergence. Returns the bias-corrected magnitude (0 below the noise floor,
    → ``m`` at high SNR).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    s2 = sigma * sigma
    floor = 2.0 * n_coils * s2
    nu = torch.sqrt((measured_magnitude**2 - floor).clamp(min=0.0) + _EPS)
    for _ in range(n_iter):
        z = measured_magnitude * nu / s2
        nu = measured_magnitude * bessel_ratio_i(z, order=n_coils)
    return nu
