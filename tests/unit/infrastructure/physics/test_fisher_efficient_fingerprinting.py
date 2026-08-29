"""Oracle tests for Fisher-efficient injective fingerprinting certs (A-1.1 FEINF)."""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.fisher_efficient_fingerprinting import (
    FisherEfficientFingerprintCert,
    cramer_rao_lower_bound,
    crb_efficiency,
    injectivity_margin,
    local_injectivity_margin,
)

# --------------------------------------------------------------------------- #
# CRLB + Fisher efficiency
# --------------------------------------------------------------------------- #


def test_crlb_orthonormal_jacobian_is_noise_var() -> None:
    # J with orthonormal columns -> J^T J = I -> CRLB = noise_var per param.
    j = torch.eye(5, 3)  # 3 orthonormal columns
    crlb = cramer_rao_lower_bound(j, noise_var=2.0)
    assert torch.allclose(crlb, torch.full((3,), 2.0), atol=1e-5)


def test_efficiency_at_crlb_is_one() -> None:
    # An estimator whose variance equals the CRLB is exactly efficient (eta == 1).
    torch.manual_seed(0)
    j = torch.randn(40, 3)
    crlb = cramer_rao_lower_bound(j, noise_var=1.0)
    eta = crb_efficiency(j, crlb, noise_var=1.0)
    assert torch.allclose(eta, torch.ones(3), atol=1e-4)


def test_efficiency_below_one_for_worse_estimator() -> None:
    # A worse estimator (variance above the CRLB) has efficiency < 1.
    torch.manual_seed(1)
    j = torch.randn(40, 3)
    crlb = cramer_rao_lower_bound(j, noise_var=1.0)
    eta = crb_efficiency(j, 4.0 * crlb, noise_var=1.0)
    assert (eta < 1.0).all()
    assert torch.allclose(eta, torch.full((3,), 0.25), atol=1e-3)


def test_efficiency_never_exceeds_one_for_valid_estimator() -> None:
    # No unbiased estimator beats the CRLB: eta <= 1 + tol for var >= CRLB.
    torch.manual_seed(2)
    j = torch.randn(50, 4)
    crlb = cramer_rao_lower_bound(j, noise_var=1.0)
    eta = crb_efficiency(j, crlb * 1.5, noise_var=1.0)  # variance above the floor
    assert (eta <= 1.0 + 1e-4).all()


def test_non_identifiable_parameter_has_infinite_crlb() -> None:
    # A Jacobian with a zero column (a parameter the signal does not depend on)
    # is non-identifiable -> infinite CRLB for that parameter.
    j = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.5, 0.0]])  # col 1 is all-zero
    crlb = cramer_rao_lower_bound(j, noise_var=1.0)
    assert math.isinf(float(crlb[1]))
    assert math.isfinite(float(crlb[0]))


# --------------------------------------------------------------------------- #
# Injectivity certificates
# --------------------------------------------------------------------------- #


def test_injectivity_margin_positive_for_separated_fingerprints() -> None:
    # An identity-like fingerprint map (s == theta scaled) is injective -> margin > 0.
    params = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fps = 2.0 * params  # bi-Lipschitz with constant 2
    assert injectivity_margin(fps, params) == pytest.approx(2.0, rel=1e-4)


def test_injectivity_margin_zero_for_colliding_fingerprints() -> None:
    # Two distinct parameter points map to the SAME fingerprint -> margin 0 (non-injective).
    params = torch.tensor([[0.0], [1.0], [2.0]])
    fps = torch.tensor([[5.0], [5.0], [9.0]])  # params 0 and 1 collide
    assert injectivity_margin(fps, params) == pytest.approx(0.0, abs=1e-6)


def test_local_injectivity_full_rank_positive_rank_deficient_zero() -> None:
    full = torch.randn(10, 3)
    assert local_injectivity_margin(full) > 0.0
    rank_def = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # col2 = 2*col1
    assert local_injectivity_margin(rank_def) == pytest.approx(0.0, abs=1e-5)


def test_local_injectivity_matches_min_singular_value() -> None:
    torch.manual_seed(3)
    j = torch.randn(12, 4)
    expected = float(torch.linalg.svdvals(j.double()).min())
    assert local_injectivity_margin(j) == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# Cert bundle + complex Jacobian + guards
# --------------------------------------------------------------------------- #


def test_cert_bundle_reports_all_three() -> None:
    torch.manual_seed(0)
    j = torch.randn(30, 2)
    params = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    fps = torch.randn(3, 8)
    out = FisherEfficientFingerprintCert(noise_var=1.0).certify(
        j, fps, params, estimator_variance=cramer_rao_lower_bound(j)
    )
    assert {"crlb", "injectivity_margin", "local_injectivity", "efficiency"} <= set(out)
    assert torch.allclose(out["efficiency"], torch.ones(2), atol=1e-3)


def test_complex_jacobian_supported() -> None:
    j = torch.randn(15, 3, dtype=torch.cfloat)
    crlb = cramer_rao_lower_bound(j)
    assert crlb.shape == (3,) and torch.isfinite(crlb).all()


def test_noise_var_must_be_positive() -> None:
    with pytest.raises(ValueError, match="noise_var"):
        cramer_rao_lower_bound(torch.randn(5, 2), noise_var=0.0)
