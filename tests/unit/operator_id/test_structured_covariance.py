"""Unit tests for the structured noise covariance (Proposal 1).

Verifies ``apply`` / ``solve`` / ``logdet`` against a dense reference at small
``N`` and exercises the Woodbury / determinant-lemma paths (with and without
the low-rank term).
"""

# ruff: noqa: E402  (pytest.importorskip must precede torch-dependent imports)

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.structured_covariance import (
    StructuredCovariance,
    lowrank_from_basis,
    radial_to_grid,
)

DT = torch.complex128


def _dense(cov, n, c, h, w, b=0):
    rows = []
    for i in range(n):
        e = torch.zeros(cov.batch, c, h, w, dtype=DT)
        e.reshape(cov.batch, n)[:, i] = 1.0
        rows.append(cov.apply(e)[b].reshape(n))
    return torch.stack(rows, dim=1)


@pytest.fixture
def small_cov():
    torch.manual_seed(0)
    c, h, w = 1, 4, 4
    n = c * h * w
    rho_rad = torch.rand(2, 5).double() + 0.5
    rho_grid = radial_to_grid(rho_rad, h, w)
    basis = torch.randn(3, c, h, w, dtype=DT)
    gains = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).to(DT)
    u = lowrank_from_basis(basis, gains)
    cov = StructuredCovariance(rho_grid, u, channels=c, height=h, width=w, eps=0.0)
    return cov, n, c, h, w


def test_apply_matches_dense(small_cov):
    cov, n, c, h, w = small_cov
    s0 = _dense(cov, n, c, h, w, b=0)
    v = torch.randn(2, c, h, w, dtype=DT)
    err = (s0 @ v[0].reshape(n) - cov.apply(v)[0].reshape(n)).abs().max()
    assert err < 1e-4


def test_hermitian(small_cov):
    cov, n, c, h, w = small_cov
    s0 = _dense(cov, n, c, h, w, b=0)
    assert (s0 - s0.conj().t()).abs().max() < 1e-5


def test_solve_matches_dense(small_cov):
    cov, n, c, h, w = small_cov
    s0 = _dense(cov, n, c, h, w, b=0)
    v = torch.randn(2, c, h, w, dtype=DT)
    ref = torch.linalg.solve(s0, v[0].reshape(n))
    got = cov.solve(v)[0].reshape(n)
    assert (ref - got).abs().max() < 1e-4


def test_logdet_matches_dense(small_cov):
    cov, n, c, h, w = small_cov
    s0 = _dense(cov, n, c, h, w, b=0)
    ref = torch.linalg.slogdet(s0)[1]
    assert (cov.logdet()[0] - ref).abs() < 1e-4


def test_diagonal_only_covariance():
    """With no low-rank term, Sigma = F^{-1} diag(rho) F and solve == A^{-1}."""
    c, h, w = 1, 4, 4
    rho_grid = torch.rand(1, 1, h, w).double() + 0.5
    cov = StructuredCovariance(rho_grid, None, channels=c, height=h, width=w, eps=0.0)
    v = torch.randn(1, c, h, w, dtype=DT)
    round_trip = cov.solve(cov.apply(v))
    assert (round_trip - v).abs().max() < 1e-4


def test_quadratic_form_is_real_and_positive(small_cov):
    cov, _n, c, h, w = small_cov
    v = torch.randn(2, c, h, w, dtype=DT)
    q = cov.quadratic_form(v)
    assert q.shape == (2,)
    assert torch.all(q > 0)
