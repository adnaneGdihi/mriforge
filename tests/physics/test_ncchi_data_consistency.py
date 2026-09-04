r"""Tests for the Bessel-ratio nc-χ data-consistency primitive (SOTA plan T4).

The non-central-χ magnitude NLL score is ``(ν − m·R_L(mν/σ²))/σ²`` with
``R_L(z) = I_L(z)/I_{L−1}(z)`` — so the data-consistency fixed point
``ν = m·R_L(mν/σ²)`` is the nc-χ ML magnitude. ``bessel_ratio_i`` computes the
ratio for any coil count ``L`` via a stable backward continued fraction.
Corner cases: ``L=1`` ⇒ Rician (``I_1/I_0``); high SNR ⇒ ``R_L→1`` ⇒ Gaussian DC.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.ncchi_data_consistency import (  # noqa: E402
    bessel_ratio_i,
    ncchi_dc_residual,
    ncchi_ml_magnitude,
)


def test_bessel_ratio_order1_matches_torch_special():
    """R_1(z) = I_1(z)/I_0(z) must match torch's scaled Bessel i1e/i0e."""
    z = torch.linspace(0.05, 40.0, 50)
    got = bessel_ratio_i(z, order=1)
    ref = torch.special.i1e(z) / torch.special.i0e(z)  # exp factors cancel
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


def test_bessel_ratio_limits():
    z = torch.tensor([1e-3, 1e-2])
    # small z: R_L(z) ≈ z/(2L)
    assert torch.allclose(bessel_ratio_i(z, order=1), z / 2.0, atol=1e-3)
    assert torch.allclose(bessel_ratio_i(z, order=4), z / 8.0, atol=1e-3)
    # large z: R_L → 1. The approach is ~1 − (2L−1)/(2z), so high order needs
    # larger z (R_6(200) ≈ 0.97); use z big enough that even L=6 clears 0.99.
    big = torch.tensor([5000.0, 20000.0])
    assert torch.all(bessel_ratio_i(big, order=1) > 0.99)
    assert torch.all(bessel_ratio_i(big, order=6) > 0.99)
    # monotone: ratio increases toward 1 with z (fixed order).
    assert bessel_ratio_i(torch.tensor([2000.0]), 6) > bessel_ratio_i(torch.tensor([200.0]), 6)


@pytest.mark.parametrize("n_coils", [1, 4])
def test_dc_residual_gaussian_at_high_snr(n_coils):
    """High SNR ⇒ R_L→1 ⇒ residual → (ν − m): the Gaussian data-consistency."""
    m = torch.tensor([10.0, 8.0])
    nu = torch.tensor([9.5, 8.2])
    sigma = 0.05  # tiny noise → z huge
    res = ncchi_dc_residual(nu, m, sigma, n_coils=n_coils)
    assert torch.allclose(res, nu - m, atol=1e-2)


def test_dc_residual_zero_at_ml_fixed_point():
    """At ν = nc-χ ML magnitude, the DC residual vanishes (fixed point)."""
    m = torch.tensor([5.0, 3.0, 1.0])
    sigma, L = 1.0, 4
    nu_ml = ncchi_ml_magnitude(m, sigma, n_coils=L, n_iter=40)
    res = ncchi_dc_residual(nu_ml, m, sigma, n_coils=L)
    assert res.abs().max() < 1e-3


def test_ml_suppresses_below_noise_floor():
    """A measurement below the nc-χ noise floor sqrt(2Lσ²) ⇒ ML magnitude → ~0."""
    sigma, L = 1.0, 4
    floor = (2.0 * L) ** 0.5 * sigma
    m = torch.tensor([0.2 * floor])  # well below floor
    nu_ml = ncchi_ml_magnitude(m, sigma, n_coils=L, n_iter=40)
    assert float(nu_ml) < 0.3
    # well above floor ⇒ ML ≈ measurement
    m_hi = torch.tensor([20.0 * floor])
    assert torch.allclose(ncchi_ml_magnitude(m_hi, sigma, L, n_iter=40), m_hi, rtol=0.05)


def test_ml_matches_moment_correction_at_moderate_snr():
    """Bessel-ratio ML generalizes the moment estimate sqrt(max(m²−2Lσ²,0))
    (they agree closely well above the noise floor)."""
    sigma, L = 1.0, 1
    m = torch.tensor([6.0, 8.0, 10.0])
    moment = torch.sqrt((m**2 - 2.0 * L * sigma**2).clamp(min=0.0))
    ml = ncchi_ml_magnitude(m, sigma, L, n_iter=40)
    assert torch.allclose(ml, moment, atol=0.2)
