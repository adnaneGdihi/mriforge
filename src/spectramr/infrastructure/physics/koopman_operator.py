r"""Koopman-operator linear posterior for fMRI k-space-time reconstruction.

Koopman theory [Mezić 2005] lifts nonlinear dynamics on :math:`\mathbb{R}^d`
to *linear* dynamics on an infinite-dimensional observable space:

.. math::

    \psi(x(t+\Delta t)) = K\,\psi(x(t)) + \varepsilon(t),

with :math:`K` a bounded linear operator on the observable space and
:math:`\psi` a (learned) observable lift. Extended Dynamic Mode
Decomposition (EDMD) [Williams-Kevrekidis-Rowley 2015] computes a
finite-dimensional approximation by least-squares on snapshot data:

.. math::

    K = X' X^+ \quad \text{(Moore-Penrose pseudoinverse)},

where :math:`X=[\psi(x(t_0))\,|\,\ldots\,|\,\psi(x(t_{N-1}))]` and
:math:`X'=[\psi(x(t_1))\,|\,\ldots\,|\,\psi(x(t_N))]`.

For fMRI reconstruction the lifted state is a small linear system that
admits an *extended Kalman smoother* whose runtime is :math:`O(r^3 T)`
instead of the :math:`O(N^2 T^2)` of nonlinear backprop through time.

Exports:

* :func:`learn_koopman_operator` — EDMD via least-squares
  :math:`K=X' X^+`.
* :func:`koopman_predict` — multi-step linear roll-out.
* :func:`closure_residual_norm` — operator-norm error
  :math:`\|\psi(x_{t+1})-K\psi(x_t)\|`.
* :func:`stable_koopman_eigvals` — diagnostic counting eigenvalues with
  :math:`|\lambda|<1` (operator stability).
"""

from __future__ import annotations

import torch

__all__ = [
    "closure_residual_norm",
    "koopman_continuous_propagate",
    "koopman_numerical_abscissa",
    "koopman_predict",
    "koopman_spectral_abscissa",
    "learn_koopman_operator",
    "stable_koopman_eigvals",
]


def learn_koopman_operator(observables: torch.Tensor, rcond: float | None = None) -> torch.Tensor:
    r"""Least-squares EDMD: :math:`K=X' X^+`.

    Args:
        observables: ``(T, r)`` time series of observable vectors. :math:`T`
            must be at least 2 for a single-step regression to be well-defined.
        rcond: Cutoff for pseudoinverse small singular values.

    Returns:
        ``(r, r)`` Koopman operator :math:`K`.
    """
    if observables.ndim != 2:
        raise ValueError(f"observables must be (T, r); got {tuple(observables.shape)}.")
    if observables.shape[0] < 2:
        raise ValueError("need at least 2 time steps.")
    x = observables[:-1].T  # (r, T-1)
    x_prime = observables[1:].T  # (r, T-1)
    x_pinv = torch.linalg.pinv(x) if rcond is None else torch.linalg.pinv(x, rcond=rcond)
    return x_prime @ x_pinv


def koopman_predict(
    initial_observable: torch.Tensor, koopman: torch.Tensor, n_steps: int
) -> torch.Tensor:
    r"""Linear roll-out :math:`\psi_n=K^n\,\psi_0`.

    Args:
        initial_observable: ``(r,)`` initial observable :math:`\psi_0`.
        koopman: ``(r, r)`` linear operator :math:`K`.
        n_steps: Number of forward steps.

    Returns:
        ``(n_steps + 1, r)`` predicted trajectory.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if initial_observable.shape != (koopman.shape[0],):
        raise ValueError("initial_observable shape must match koopman.")
    out = [initial_observable]
    current = initial_observable
    for _ in range(n_steps):
        current = koopman @ current
        out.append(current)
    return torch.stack(out)


def closure_residual_norm(observables: torch.Tensor, koopman: torch.Tensor) -> torch.Tensor:
    r"""Operator-norm closure error
    :math:`\|\psi_{t+1}-K\psi_t\|_F^2/((T-1)\,r)`.

    The elementwise mean is taken over the ``(T-1, r)`` tensor of single-step
    residuals, i.e. the Frobenius squared error normalized by ``(T-1) * r``.

    Zero (up to floating point) when :math:`\psi` spans a :math:`K`-invariant
    subspace. Used as an auxiliary training loss to encourage invariance.
    """
    if observables.shape[1] != koopman.shape[0] or koopman.shape[0] != koopman.shape[1]:
        raise ValueError("shape mismatch between observables and koopman.")
    predicted = observables[:-1] @ koopman.T  # (T-1, r)
    actual = observables[1:]
    return (predicted - actual).pow(2).mean()


def koopman_continuous_propagate(
    z0: torch.Tensor, generator: torch.Tensor, dtau: torch.Tensor
) -> torch.Tensor:
    r"""Continuous Koopman semigroup roll-out :math:`z(\tau)=\exp(\Delta\tau\,A)\,z_0`.

    Unlike :func:`koopman_predict` (discrete integer powers :math:`K^n`), this evolves a
    *continuous* coordinate by exponentiating a learned infinitesimal generator
    :math:`A` (:math:`A=\log K`). The one-parameter group
    :math:`\{\exp(\Delta\tau\,A)\}` obeys the EXACT semigroup law
    :math:`\exp((a+b)A)=\exp(aA)\exp(bA)`, so a single generator propagates ANY gap
    :math:`\Delta\tau` and composition is exact — the structural basis of the B-3.10
    cross-field propagator (:math:`\Delta\tau=\log_{10}B_t-\log_{10}B_s`).

    Operates per spatial location on the channel (observable) axis, batched over
    per-sample gaps.

    Args:
        z0: ``(B, d, ...)`` latent observable; ``d`` is the channel/observable dim
            and ``...`` any trailing spatial dims (or none).
        generator: ``(d, d)`` shared infinitesimal generator :math:`A`.
        dtau: ``(B,)`` per-sample continuous gap :math:`\Delta\tau`.

    Returns:
        ``z0``-shaped propagated latent :math:`\exp(\Delta\tau\,A)\,z_0`.
    """
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError(f"generator must be (d, d); got {tuple(generator.shape)}.")
    d = generator.shape[0]
    if z0.ndim < 2 or z0.shape[1] != d:
        raise ValueError(f"z0 must be (B, d, ...) with d={d} on axis 1; got {tuple(z0.shape)}.")
    b = z0.shape[0]
    if dtau.reshape(-1).shape[0] != b:
        raise ValueError(f"dtau must have B={b} entries; got {tuple(dtau.shape)}.")
    # M_i = exp(dtau_i * A): batched matrix exponential over the per-sample generators.
    m = torch.matrix_exp(dtau.reshape(b, 1, 1).to(generator) * generator.unsqueeze(0))  # (B,d,d)
    # Apply M_i across the channel axis at every spatial location: z'[b,i,s] = M[b,i,j] z[b,j,s].
    z_flat = z0.reshape(b, d, -1)  # (B, d, S)
    out = torch.bmm(m, z_flat)  # (B, d, S)
    return out.reshape(z0.shape)


def koopman_spectral_abscissa(generator: torch.Tensor) -> float:
    r"""Continuous-time spectral abscissa :math:`\max_i \mathrm{Re}\,\lambda_i(A)`.

    For the continuous semigroup :math:`\exp(\Delta\tau\,A)`, the largest real part of
    the generator's eigenvalues governs stability: ``< 0`` ⇒ every mode contracts as
    :math:`\Delta\tau\to+\infty` (a contractive propagator), ``> 0`` ⇒ a growth mode.
    Interpretable provenance / monitor for the learned operator (NOT a hard gate).
    """
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be a square matrix.")
    return float(torch.linalg.eigvals(generator).real.max())


def koopman_numerical_abscissa(generator: torch.Tensor) -> torch.Tensor:
    r"""Differentiable numerical abscissa :math:`\mu_2(A)=\lambda_{\max}\!\big(\tfrac12(A+A^\top)\big)`.

    The logarithmic 2-norm (numerical abscissa) is a differentiable UPPER BOUND on the
    spectral abscissa :func:`koopman_spectral_abscissa`:

    .. math::

        \max_i \mathrm{Re}\,\lambda_i(A) \;\le\; \mu_2(A)
            \;=\; \lambda_{\max}\!\Big(\tfrac12\big(A+A^\top\big)\Big),

    because the field of values of :math:`A` has real part bounded by the largest eigenvalue
    of its symmetric part. Unlike :func:`koopman_spectral_abscissa` (which calls the
    non-symmetric ``eigvals`` on a scalar and detaches), :math:`\mu_2` is computed from
    ``eigvalsh`` on the symmetric part and is smoothly differentiable — so ``relu(mu2)`` is a
    well-behaved training-time stability penalty that drives the continuous semigroup
    :math:`\exp(\Delta\tau\,A)` toward NON-EXPANSIVE (:math:`\mu_2\le 0` ⇒ every mode
    non-growing). Returns a scalar tensor carrying grad to ``generator``.
    """
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be a square matrix.")
    sym = 0.5 * (generator + generator.transpose(-1, -2))
    return torch.linalg.eigvalsh(sym).max()


def stable_koopman_eigvals(koopman: torch.Tensor) -> int:
    r"""Count of Koopman eigenvalues with :math:`|\lambda|<1`.

    A fully stable Koopman operator (all eigenvalues inside the unit disc)
    produces bounded forward roll-outs; mixed-stability operators model
    growth modes. Diagnostic only — does not raise on instability.
    """
    if koopman.ndim != 2 or koopman.shape[0] != koopman.shape[1]:
        raise ValueError("koopman must be a square matrix.")
    eigs = torch.linalg.eigvals(koopman)
    return int((eigs.abs() < 1.0).sum().item())
