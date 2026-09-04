"""Tests for free-probability asymptotic g-factor."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.free_probability_gfactor import (
    asymptotic_gfactor_density,
    gfactor_moments,
    marchenko_pastur_density,
    mp_s_transform,
    worst_case_gfactor,
)


def test_mp_s_transform_closed_form() -> None:
    """S-transform identity :math:`S_{\\mathrm{MP},c}(z)=1/(1+cz)`."""
    z = torch.linspace(-0.4, 0.4, 11)
    for c in (0.1, 0.3, 0.5, 0.9):
        s = mp_s_transform(z, c)
        expected = 1.0 / (1.0 + c * z)
        assert torch.allclose(s, expected, atol=1e-6)


def test_inverse_s_transform_is_not_one_plus_cz() -> None:
    """Pin the corrected fact that retired the false g²=MP derivation.

    The old docstring claimed the multiplicative-inverse factor had S-transform
    ``(1 + cz)``. The correct inverse S-transform is
    ``S_{mu^{-1}}(z) = 1 / S_mu(-1 - z) = 1 - c - cz`` — which is NOT ``1 + cz``,
    so the product does not collapse back to the MP S-transform. This test
    guards against the incorrect derivation reappearing.
    """
    z = torch.linspace(-0.3, 0.3, 7)
    for c in (0.2, 0.5, 0.8):
        inv_s = 1.0 / mp_s_transform(-1.0 - z, c)  # S_{mu^{-1}}(z)
        assert torch.allclose(inv_s, 1.0 - c - c * z, atol=1e-6)
        assert not torch.allclose(inv_s, 1.0 + c * z, atol=1e-3)


def test_asymptotic_gfactor_is_mp_surrogate() -> None:
    """``asymptotic_gfactor_density`` is exactly the MP density (the surrogate)."""
    t = torch.linspace(0.05, 4.0, 50)
    for c in (0.25, 0.5):
        assert torch.allclose(
            asymptotic_gfactor_density(t, c),
            marchenko_pastur_density(t, c),
            atol=1e-8,
        )


def test_marchenko_pastur_density_integrates_to_one() -> None:
    """The MP density on its compact support integrates to 1."""
    c = 0.3
    lam_minus = (1.0 - math.sqrt(c)) ** 2
    lam_plus = (1.0 + math.sqrt(c)) ** 2
    t = torch.linspace(lam_minus, lam_plus, 4096)
    dt = (lam_plus - lam_minus) / (4096 - 1)
    rho = marchenko_pastur_density(t, c)
    mass = rho.sum().item() * dt
    assert mass == pytest.approx(1.0, abs=2e-2)


def test_density_zero_outside_support() -> None:
    c = 0.4
    lam_minus = (1.0 - math.sqrt(c)) ** 2
    lam_plus = (1.0 + math.sqrt(c)) ** 2
    t = torch.tensor([lam_minus - 0.1, lam_plus + 0.1, lam_plus + 1.0])
    rho = marchenko_pastur_density(t, c)
    assert torch.all(rho == 0.0)


def test_gfactor_density_equals_mp() -> None:
    """The asymptotic g²-density coincides with the MP density."""
    c = 0.25
    t = torch.linspace(0.05, 4.0, 64)
    rho_g = asymptotic_gfactor_density(t, c)
    rho_mp = marchenko_pastur_density(t, c)
    assert torch.allclose(rho_g, rho_mp)


def test_moments_match_closed_form() -> None:
    """First two moments match the MP closed form: mean=1, var=c."""
    for c in (0.1, 0.5, 0.9):
        mean, var = gfactor_moments(c)
        assert mean == pytest.approx(1.0)
        assert var == pytest.approx(c)


def test_worst_case_gfactor_is_upper_edge() -> None:
    """:math:`\\max g=1+\\sqrt c=\\sqrt{\\lambda_+}`."""
    for c in (0.1, 0.5, 0.9):
        w = worst_case_gfactor(c)
        assert w == pytest.approx(1.0 + math.sqrt(c))
        lam_plus = (1.0 + math.sqrt(c)) ** 2
        assert w == pytest.approx(math.sqrt(lam_plus))


def test_invalid_c_raises() -> None:
    with pytest.raises(ValueError):
        worst_case_gfactor(0.0)
    with pytest.raises(ValueError):
        worst_case_gfactor(1.0)
    with pytest.raises(ValueError):
        marchenko_pastur_density(torch.tensor([1.0]), 1.5)


def test_registered_metric_returns_worst_case() -> None:
    from spectramr.core.metrics import get_metric

    metric = get_metric("asymptotic_gfactor")
    metric.__class__.__init__(metric, c=0.5)  # type: ignore[arg-type]
    out = metric(torch.zeros(1), torch.zeros(1))
    assert out == pytest.approx(1.0 + math.sqrt(0.5))
