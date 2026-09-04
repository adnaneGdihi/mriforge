"""Tests for Koopman-operator EDMD primitives."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.koopman_operator import (
    closure_residual_norm,
    koopman_continuous_propagate,
    koopman_numerical_abscissa,
    koopman_predict,
    koopman_spectral_abscissa,
    learn_koopman_operator,
    stable_koopman_eigvals,
)


def test_edmd_recovers_known_linear_dynamics() -> None:
    """For x_{t+1} = A x_t generated from a known A, EDMD recovers A."""
    torch.manual_seed(0)
    a = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    x0 = torch.randn(2)
    trajectory = [x0]
    for _ in range(64):
        x0 = a @ x0
        trajectory.append(x0)
    obs = torch.stack(trajectory)
    a_hat = learn_koopman_operator(obs)
    assert torch.allclose(a_hat, a, atol=1e-4)


def test_koopman_predict_rolls_out_correctly() -> None:
    """Multi-step prediction equals iterated matrix-vector product."""
    a = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])  # 90° rotation
    x0 = torch.tensor([1.0, 0.0])
    trajectory = koopman_predict(x0, a, n_steps=4)
    expected = torch.tensor([
        [1.0, 0.0],
        [0.0, -1.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    assert torch.allclose(trajectory, expected, atol=1e-5)


def test_closure_residual_zero_on_invariant_subspace() -> None:
    """If the data lies on a K-invariant subspace, residual is zero."""
    a = torch.tensor([[0.9, 0.0], [0.0, 0.5]])
    x0 = torch.tensor([1.0, 1.0])
    trajectory = torch.stack([torch.linalg.matrix_power(a, t) @ x0 for t in range(32)])
    residual = closure_residual_norm(trajectory, a).item()
    assert residual < 1e-8


def test_closure_residual_positive_off_subspace() -> None:
    """Mismatched K gives positive residual."""
    torch.manual_seed(0)
    obs = torch.randn(32, 4)
    fake_k = torch.eye(4)
    res = closure_residual_norm(obs, fake_k).item()
    assert res > 0.0


def test_stable_koopman_eigvals_counts_correctly() -> None:
    a_stable = torch.tensor([[0.5, 0.0], [0.0, 0.3]])
    a_mixed = torch.tensor([[0.5, 0.0], [0.0, 1.5]])
    a_unstable = torch.tensor([[2.0, 0.0], [0.0, 1.5]])
    assert stable_koopman_eigvals(a_stable) == 2
    assert stable_koopman_eigvals(a_mixed) == 1
    assert stable_koopman_eigvals(a_unstable) == 0


def test_continuous_propagate_obeys_exact_semigroup() -> None:
    """exp((a+b)A) z0 == exp(bA) exp(aA) z0 — the any-to-any composition law (s->m->t == s->t)."""
    torch.manual_seed(0)
    d = 6
    a_gen = torch.randn(d, d) * 0.3
    z0 = torch.randn(2, d, 4, 4)  # (B, d, H, W)
    dtau_a = torch.tensor([0.4, -0.7])
    dtau_b = torch.tensor([0.5, 0.9])
    # one-shot gap (a+b)
    direct = koopman_continuous_propagate(z0, a_gen, dtau_a + dtau_b)
    # composed: first a, then b
    composed = koopman_continuous_propagate(
        koopman_continuous_propagate(z0, a_gen, dtau_a), a_gen, dtau_b
    )
    assert torch.allclose(direct, composed, atol=1e-5)


def test_continuous_propagate_zero_gap_is_identity() -> None:
    """dtau=0 -> exp(0)=I -> the latent is unchanged (the autoencoder/identity branch)."""
    torch.manual_seed(1)
    d = 5
    a_gen = torch.randn(d, d)
    z0 = torch.randn(3, d, 2, 2)
    out = koopman_continuous_propagate(z0, a_gen, torch.zeros(3))
    assert torch.allclose(out, z0, atol=1e-6)


def test_spectral_abscissa_sign() -> None:
    contractive = torch.diag(torch.tensor([-0.5, -2.0]))
    growth = torch.diag(torch.tensor([-0.5, 1.5]))
    assert koopman_spectral_abscissa(contractive) < 0.0
    assert koopman_spectral_abscissa(growth) > 0.0
    assert abs(koopman_spectral_abscissa(growth) - 1.5) < 1e-5


def test_numerical_abscissa_upper_bounds_spectral_abscissa() -> None:
    """Bendixson: max Re(lambda(A)) <= lambda_max(0.5(A+A^T)) = mu2(A) for any square A."""
    torch.manual_seed(0)
    for _ in range(5):
        a = torch.randn(6, 6)
        mu2 = float(koopman_numerical_abscissa(a))
        alpha = koopman_spectral_abscissa(a)
        assert mu2 >= alpha - 1e-5
    # tight (equality) for a symmetric matrix
    s = torch.tensor([[-0.5, 0.2], [0.2, 1.5]])
    assert abs(float(koopman_numerical_abscissa(s)) - koopman_spectral_abscissa(s)) < 1e-5


def test_numerical_abscissa_skew_symmetric_is_zero() -> None:
    """A skew-symmetric generator (pure rotation) has symmetric part 0 -> mu2 = 0."""
    a = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    assert abs(float(koopman_numerical_abscissa(a))) < 1e-6


def test_numerical_abscissa_differentiable() -> None:
    """Unlike koopman_spectral_abscissa (eigvals + detach), mu2 carries grad (eigvalsh)."""
    a = torch.eye(4, requires_grad=True)  # symmetric part = I -> mu2 = 1
    loss = torch.relu(koopman_numerical_abscissa(a))
    loss.backward()
    assert float(loss) > 0.0
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        learn_koopman_operator(torch.zeros(8))
    with pytest.raises(ValueError):
        learn_koopman_operator(torch.zeros(1, 3))  # only one timestep
    with pytest.raises(ValueError):
        koopman_predict(torch.zeros(3), torch.zeros(2, 2), n_steps=2)
    with pytest.raises(ValueError):
        koopman_predict(torch.zeros(2), torch.zeros(2, 2), n_steps=-1)
    with pytest.raises(ValueError):
        closure_residual_norm(torch.zeros(8, 3), torch.zeros(4, 4))
    with pytest.raises(ValueError):
        stable_koopman_eigvals(torch.zeros(3, 4))
    with pytest.raises(ValueError):
        koopman_continuous_propagate(torch.zeros(2, 3, 4), torch.zeros(3, 4), torch.zeros(2))
    with pytest.raises(ValueError):  # z0 channel dim != generator dim
        koopman_continuous_propagate(torch.zeros(2, 5, 4), torch.zeros(3, 3), torch.zeros(2))
    with pytest.raises(ValueError):  # dtau batch mismatch
        koopman_continuous_propagate(torch.zeros(2, 3, 4), torch.zeros(3, 3), torch.zeros(5))
    with pytest.raises(ValueError):
        koopman_spectral_abscissa(torch.zeros(3, 4))
    with pytest.raises(ValueError):
        koopman_numerical_abscissa(torch.zeros(3, 4))
