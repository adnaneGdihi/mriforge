"""Unit tests for SinkhornSFCLinearizer.

Pin the doubly-stochastic invariant, the differentiable forward
pass, and the temperature-controlled sharpness of the soft
permutation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.sinkhorn_sfc import (  # noqa: E402
    SinkhornSFCLinearizer,
    sinkhorn,
)


def test_sinkhorn_output_is_doubly_stochastic() -> None:
    log_alpha = torch.randn(8, 8)
    P = sinkhorn(log_alpha, n_iters=30, tau=1.0)
    assert torch.allclose(P.sum(dim=-1), torch.ones(8), atol=1e-3)
    assert torch.allclose(P.sum(dim=-2), torch.ones(8), atol=1e-3)


def test_low_temperature_approaches_permutation() -> None:
    """As τ → 0 the Sinkhorn output concentrates on a permutation."""
    log_alpha = torch.randn(6, 6)
    P_hot = sinkhorn(log_alpha, n_iters=50, tau=10.0)
    P_cold = sinkhorn(log_alpha, n_iters=200, tau=0.01)
    # Cold matrix has near-binary entries (more zeros and ones).
    cold_extremes = ((P_cold < 0.05) | (P_cold > 0.95)).float().mean()
    hot_extremes = ((P_hot < 0.05) | (P_hot > 0.95)).float().mean()
    assert cold_extremes > hot_extremes


def test_linearizer_forward_shape() -> None:
    block = SinkhornSFCLinearizer(n_foreground=16)
    x = torch.randn(2, 16, 8)
    y = block(x)
    assert y.shape == x.shape


def test_forward_is_autograd_traceable() -> None:
    block = SinkhornSFCLinearizer(n_foreground=10, sinkhorn_iters=5)
    x = torch.randn(1, 10, 4, requires_grad=True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)
    assert block.cost.grad is not None and torch.any(block.cost.grad != 0)


def test_warm_start_with_init_cost() -> None:
    # init_cost is a *distance* matrix: high off-diagonal cost biases
    # the Sinkhorn projection toward the diagonal (identity) permutation.
    init_cost = (1.0 - torch.eye(6)) * 5.0
    block = SinkhornSFCLinearizer(n_foreground=6, init_cost=init_cost, tau=0.05)
    P = block.soft_permutation()
    # With low τ the permutation should concentrate near the diagonal.
    diag_mass = P.diag().mean()
    off_diag_mass = (P.sum() - P.diag().sum()) / (6 * 6 - 6)
    assert diag_mass > off_diag_mass


def test_hard_permutation_is_a_real_permutation() -> None:
    block = SinkhornSFCLinearizer(n_foreground=8)
    P = block.hard_permutation()
    assert P.shape == (8, 8)
    assert torch.all((P == 0) | (P == 1))
    assert torch.all(P.sum(dim=0) == 1)
    assert torch.all(P.sum(dim=1) == 1)


def test_rejects_bad_init_cost_shape() -> None:
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        SinkhornSFCLinearizer(n_foreground=4, init_cost=torch.zeros(3, 3))


def test_rejects_bad_input_seq_length() -> None:
    block = SinkhornSFCLinearizer(n_foreground=8)
    with pytest.raises(ValueError, match="length 8"):
        block(torch.randn(1, 7, 4))


def test_learnable_tau_receives_gradient() -> None:
    """``learnable_tau`` must actually be trainable: the gradient of a loss on
    the soft permutation must reach ``log_tau``. Previously ``soft_permutation``
    detached tau to a Python float (``.detach().exp().item()``), so ``log_tau``
    got no gradient and the knob was inert (pitfall #16).

    A non-trivial ``init_cost`` is required: with the default all-zero cost the
    Sinkhorn output is uniform and mathematically tau-invariant (zero gradient
    for a real reason, not a detach). We use a random symmetric distance matrix
    and a P-dependent loss (weighted sum) so the tau gradient is non-zero."""
    torch.manual_seed(0)
    d = torch.rand(6, 6)
    cost = (d + d.T) / 2.0  # symmetric non-degenerate distance matrix
    block = SinkhornSFCLinearizer(
        n_foreground=6, tau=1.0, learnable_tau=True, init_cost=cost
    )
    weights = torch.randn(6, 6)
    loss = (block.soft_permutation() * weights).sum()
    loss.backward()
    assert block.log_tau is not None
    assert block.log_tau.grad is not None
    assert torch.any(block.log_tau.grad != 0)


# ── Sanity-shape parametrization ────────────────────────────────────────


@pytest.mark.parametrize("B,N,D", [
    (1, 4, 2),
    (2, 8, 4),
    (1, 16, 8),
], ids=["B1N4D2", "B2N8D4", "B1N16D8"])
def test_sinkhorn_linearizer_shape_matrix(B: int, N: int, D: int) -> None:
    """forward returns exactly (B, N, D) for the shape matrix."""
    block = SinkhornSFCLinearizer(n_foreground=N, sinkhorn_iters=5)
    block.eval()
    x = torch.randn(B, N, D)
    out = block(x)
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()


def test_reverse_shape_matches_forward() -> None:
    """reverse() output has the same shape as forward() output."""
    block = SinkhornSFCLinearizer(n_foreground=8, sinkhorn_iters=5)
    x = torch.randn(2, 8, 4)
    y = block(x)
    y_rev = block.reverse(y)
    assert y_rev.shape == x.shape


def test_reverse_wrong_ndim_raises() -> None:
    """reverse() with wrong ndim raises ValueError."""
    block = SinkhornSFCLinearizer(n_foreground=4)
    with pytest.raises(ValueError):
        block.reverse(torch.randn(4, 4))  # 2-D, not 3-D


def test_sinkhorn_non_square_raises() -> None:
    """Non-square log_alpha raises ValueError."""
    with pytest.raises(ValueError, match="square"):
        from spectramr.models.blocks.sinkhorn_sfc import sinkhorn
        sinkhorn(torch.randn(3, 4))


def test_sinkhorn_zero_tau_raises() -> None:
    """tau <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="tau"):
        from spectramr.models.blocks.sinkhorn_sfc import sinkhorn
        sinkhorn(torch.randn(4, 4), tau=0.0)


def test_zero_n_foreground_raises() -> None:
    """n_foreground=0 raises ValueError."""
    with pytest.raises(ValueError):
        SinkhornSFCLinearizer(n_foreground=0)
