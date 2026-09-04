"""Tests for the matrix-free ``expm_multiply`` scaling-and-squaring path.

Targets ``spectramr.infrastructure.physics.magnus_exponential``. The BCH /
Magnus machinery reconstructs ``exp(op)·x`` from a Krylov projection; when
the operator norm is large it scales the operator by ``1/2^s`` and applies
the scaled exponential ``2^s`` times.

Regression (2026-07): the squaring loop applied ``exp(op/2^s)`` only ``s+1``
times instead of ``2^s`` times, so for ``s ≥ 2`` it computed
``exp((s+1)/2^s · op)·x`` — the wrong operator.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.magnus_exponential import (
    LinearOperator,
    expm_multiply,
)


def _diag_operator(diag: torch.Tensor) -> LinearOperator:
    """A diagonal (hence normal) matrix-free operator ``v ↦ diag ⊙ v``."""
    return LinearOperator(
        lambda v: diag * v, lambda v: diag.conj() * v, name="diag"
    )


def test_squaring_path_matches_exact_exponential() -> None:
    """Large-norm diagonal op: scaling_squaring reproduces exp(diag)·x.

    diag = 6 forces ``s = ceil(log2(6)) = 3`` (8 applications). The pre-fix
    loop applied only 4, giving exp(3)·x instead of exp(6)·x.
    """
    diag = torch.full((1, 4), 6.0, dtype=torch.complex64)
    op = _diag_operator(diag)
    x = torch.randn(1, 4, dtype=torch.complex64)
    y = expm_multiply(op, x, krylov_dim=4, scaling_squaring=True, norm_threshold=1.0)
    exact = torch.exp(diag) * x
    rel = (y - exact).abs().max() / exact.abs().max()
    assert rel < 1e-4


def test_no_squaring_path_matches_exact_exponential() -> None:
    """Small-norm op: the default (scaling_squaring=False) is already exact."""
    diag = torch.tensor([[0.3, -0.2, 0.1, -0.4]], dtype=torch.complex64)
    op = _diag_operator(diag)
    x = torch.randn(1, 4, dtype=torch.complex64)
    y = expm_multiply(op, x, krylov_dim=4, scaling_squaring=False)
    exact = torch.exp(diag) * x
    assert torch.allclose(y, exact, atol=1e-5)


def test_squaring_and_no_squaring_agree_for_moderate_norm() -> None:
    """Both paths converge to the same answer when both are accurate."""
    diag = torch.tensor([[2.5, -1.5, 2.0, -2.0]], dtype=torch.complex64)
    op = _diag_operator(diag)
    x = torch.randn(1, 4, dtype=torch.complex64)
    y_scaled = expm_multiply(
        op, x, krylov_dim=4, scaling_squaring=True, norm_threshold=0.5
    )
    exact = torch.exp(diag) * x
    assert (y_scaled - exact).abs().max() / exact.abs().max() < 1e-4
