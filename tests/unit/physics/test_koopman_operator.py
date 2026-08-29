"""Regression test for the Koopman closure_residual_norm normalization contract.

WS-4 doc-contract fix: ``closure_residual_norm`` takes ``.mean()`` over the
``(T-1, r)`` single-step residual tensor, i.e. it divides the Frobenius squared
error by ``(T-1) * r``. The docstring previously claimed ``/(T r)`` (off by one
in T). This test pins the *implemented* normalization to the now-corrected
docstring so the two cannot drift apart again.

CPU-only, tiny tensors, fixed seed.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.koopman_operator import closure_residual_norm


def test_closure_residual_normalization_matches_doc() -> None:
    """closure_residual_norm == sum|psi_{t+1} - K psi_t|^2 / ((T-1) * r)."""
    torch.manual_seed(0)
    T, r = 5, 3
    obs = torch.randn(T, r)
    K = torch.randn(r, r)
    predicted = obs[:-1] @ K.T  # (T-1, r)
    actual = obs[1:]
    expected = (predicted - actual).pow(2).sum() / ((T - 1) * r)
    got = closure_residual_norm(obs, K)
    assert torch.isclose(got, expected)


def test_closure_residual_denominator_is_T_minus_one_not_T() -> None:
    """Guard against a (wrong) /(T*r) normalization regression.

    With a non-trivial residual the (T-1)*r and T*r denominators differ, so the
    value must NOT match the /(T*r) form -- this locks the documented (T-1)*r.
    """
    torch.manual_seed(1)
    T, r = 6, 4
    obs = torch.randn(T, r)
    K = torch.randn(r, r)
    predicted = obs[:-1] @ K.T
    actual = obs[1:]
    sq_sum = (predicted - actual).pow(2).sum()
    got = closure_residual_norm(obs, K)
    assert torch.isclose(got, sq_sum / ((T - 1) * r))
    assert not torch.isclose(got, sq_sum / (T * r))
