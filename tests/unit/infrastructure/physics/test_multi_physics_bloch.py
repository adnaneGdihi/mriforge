"""Tests for ``MultiPhysicsBlochLayer`` signal equations.

Focus: the GRE susceptibility coherence term must be a *dimensionless*
function of the χ-induced phase (Gaussian static-dephasing envelope),
not the dimensionally-invalid ``exp(-|φ|·TE)`` it used to be.
"""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer


def _layer() -> MultiPhysicsBlochLayer:
    return MultiPhysicsBlochLayer(default_flip_angle_deg=10.0, learnable_flip_angle=False)


def _gre_args():
    shape = (1, 1, 4, 4)
    rho = torch.full(shape, 0.3)
    t1 = torch.full(shape, 1.0)
    t2star = torch.full(shape, 0.05)
    tr = torch.full(shape, 2.0)
    return rho, t1, t2star, tr


def test_zero_susceptibility_phase_is_identity() -> None:
    """φ=0 → coherence factor 1 → signal unchanged."""
    layer = _layer()
    rho, t1, t2star, tr = _gre_args()
    te = torch.full_like(rho, 0.01)
    base = layer._synthesize_gradient_echo(rho, t1, t2star, tr, te, susceptibility_phase=None)
    zero_phi = layer._synthesize_gradient_echo(
        rho, t1, t2star, tr, te, susceptibility_phase=torch.zeros_like(rho)
    )
    assert torch.allclose(base, zero_phi, atol=1e-6)


def test_susceptibility_attenuation_is_te_independent() -> None:
    """For fixed φ the susceptibility attenuation ratio is the same at any TE.

    Regression for the dimensional bug: the old ``exp(-|φ|·TE)`` made the
    attenuation grow with TE for fixed φ. The Gaussian envelope ``exp(-φ²/2)``
    depends on φ alone (φ already carries the TE-dependence).
    """
    layer = _layer()
    rho, t1, t2star, tr = _gre_args()
    phi = torch.full_like(rho, 1.0)

    def ratio(te_val: float) -> torch.Tensor:
        te = torch.full_like(rho, te_val)
        base = layer._synthesize_gradient_echo(rho, t1, t2star, tr, te, susceptibility_phase=None)
        with_susc = layer._synthesize_gradient_echo(
            rho, t1, t2star, tr, te, susceptibility_phase=phi
        )
        return with_susc / base

    expected = math.exp(-0.5)  # exp(-φ²/2) at φ=1
    r1, r2 = ratio(0.01), ratio(0.03)
    assert torch.allclose(r1, torch.full_like(r1, expected), atol=1e-5)
    assert torch.allclose(r2, torch.full_like(r2, expected), atol=1e-5)
    assert torch.allclose(r1, r2, atol=1e-6)


def test_susceptibility_attenuation_monotone_in_phase() -> None:
    """Larger phase dispersion → more signal loss (bounded in (0, 1])."""
    layer = _layer()
    rho, t1, t2star, tr = _gre_args()
    te = torch.full_like(rho, 0.02)
    base = layer._synthesize_gradient_echo(rho, t1, t2star, tr, te, susceptibility_phase=None)
    s_small = layer._synthesize_gradient_echo(
        rho, t1, t2star, tr, te, susceptibility_phase=torch.full_like(rho, 0.5)
    )
    s_large = layer._synthesize_gradient_echo(
        rho, t1, t2star, tr, te, susceptibility_phase=torch.full_like(rho, 2.0)
    )
    assert (s_large <= s_small).all()
    assert (s_small <= base).all()
    assert (s_large > 0).all()
