"""Unit tests for TALL learnable SFC + privileged-learning gradient-reversal.

Pin three contracts:

1. **Motif tables** are valid permutations of 4 cells.
2. **TALL forward** returns a valid permutation (bijection of N voxels)
   in eval mode and a differentiable sequence in train mode.
3. **Gradient reversal** flips the sign of upstream gradients.

Sanity-shape parametrized tests are in ``TestTALLSanityShape``.
"""
from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.tall import TALL, build_motif_permutations, locality_loss


class TestMotifPermutations:
    def test_2d_table_shape(self) -> None:
        m = build_motif_permutations(n_dims=2)
        assert m.shape == (4, 4)
        # Each row is a permutation of {0, 1, 2, 3}
        for k in range(4):
            assert torch.equal(torch.sort(m[k]).values, torch.arange(4))

    def test_3d_table_shape(self) -> None:
        m = build_motif_permutations(n_dims=3)
        assert m.shape == (24, 8)
        for k in range(24):
            assert torch.equal(torch.sort(m[k]).values, torch.arange(8))

    def test_invalid_ndims_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            build_motif_permutations(n_dims=4)


class TestTALL3D:
    def test_3d_forward_shape(self) -> None:
        m = TALL(order=2, feature_dim=4, n_dims=3)   # 4x4x4 = 64
        m.eval()
        x = torch.randn(1, 4, 4, 4, 4)
        seq, perm = m(x)
        assert seq.shape == (1, 4, 64) and perm.shape == (1, 64)

    def test_3d_permutation_bijection(self) -> None:
        m = TALL(order=2, feature_dim=4, n_dims=3)
        m.eval()
        x = torch.randn(1, 4, 4, 4, 4)
        _, perm = m(x)
        for b in range(perm.shape[0]):
            assert torch.equal(torch.sort(perm[b]).values, torch.arange(64))

    def test_3d_train_mode_grad_flow(self) -> None:
        m = TALL(order=2, feature_dim=4, n_dims=3)
        m.train()
        x = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        seq, _ = m(x, hard=False)
        seq.pow(2).mean().backward()
        assert x.grad is not None and x.grad.abs().max() > 0
        assert m.policy[0].weight.grad is not None


class TestTALL:
    def test_forward_shape(self) -> None:
        m = TALL(order=4, feature_dim=8)   # 16x16 grid
        x = torch.randn(2, 8, 16, 16)
        m.eval()
        seq, perm = m(x)
        assert seq.shape == (2, 8, 256)
        assert perm.shape == (2, 256)

    def test_permutation_is_bijection(self) -> None:
        m = TALL(order=3, feature_dim=4)   # 8x8 = 64
        x = torch.randn(1, 4, 8, 8)
        m.eval()
        _, perm = m(x)
        for b in range(perm.shape[0]):
            sorted_perm = torch.sort(perm[b]).values
            assert torch.equal(sorted_perm, torch.arange(64))

    def test_canonical_init_emits_motif_zero_at_step_0(self) -> None:
        # With canonical_init=True, the initial bias strongly favours
        # motif 0; argmax should pick motif 0 for every cell at step 0.
        m = TALL(order=3, feature_dim=4, canonical_init=True)
        m.eval()
        x = torch.zeros(1, 4, 8, 8)
        # Need to build the cell features manually to avoid randomness
        feat = m._cell_features(x)
        logits = m.policy(feat)
        argmax = logits.argmax(dim=-1)
        assert (argmax == 0).all(), (
            f"canonical_init should put argmax at motif 0; got {argmax}"
        )

    def test_train_mode_returns_differentiable_seq(self) -> None:
        m = TALL(order=3, feature_dim=4)
        m.train()
        x = torch.randn(1, 4, 8, 8, requires_grad=True)
        seq, _ = m(x, hard=False)
        loss = seq.pow(2).mean()
        loss.backward()
        # Gradient should flow back to x and to the policy
        assert x.grad is not None and x.grad.abs().max() > 0
        assert m.policy[0].weight.grad is not None

    def test_tau_anneals_after_steps(self) -> None:
        m = TALL(order=3, feature_dim=4, tau_init=1.0, tau_min=0.1, anneal_steps=10)
        m.train()
        # Tau is 1.0 at step 0
        assert abs(m.tau - 1.0) < 1e-6
        x = torch.randn(1, 4, 8, 8)
        for _ in range(10):
            m(x, hard=False)
        # After 10 steps tau should be at the minimum
        assert abs(m.tau - 0.1) < 0.05


class TestLocalityLoss:
    def test_canonical_hilbert_is_low(self) -> None:
        # An identity permutation in raster order has step length 1 mostly,
        # but jumps W at row boundaries → ~W on average.
        # A canonical Hilbert permutation has step length exactly 1.
        from spectramr.models.blocks.hilbert_order import HilbertOrder
        side = 16
        h = HilbertOrder(shape=(side, side), mode="hilbert")
        perm = h.permutation.unsqueeze(0)
        loss_h = float(locality_loss(perm, side))
        r = HilbertOrder(shape=(side, side), mode="raster")
        rperm = r.permutation.unsqueeze(0)
        loss_r = float(locality_loss(rperm, side))
        # Hilbert mean step = 1; raster has occasional row-jumps of ~side
        assert loss_h < loss_r


class TestGradientReversal:
    def test_forward_identity_backward_negate(self) -> None:
        from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
            gradient_reversal,
        )
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = gradient_reversal(x, lambda_=2.0)
        # Forward is identity
        assert torch.equal(y, x)
        # Backward inverts the sign and scales by lambda
        y.sum().backward()
        assert torch.equal(x.grad, torch.tensor([-2.0, -2.0]))


class TestDomainDiscriminator:
    def test_classifies_features(self) -> None:
        from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
            DomainDiscriminator,
        )
        d = DomainDiscriminator(feature_dim=16)
        x = torch.randn(4, 16, 8, 8)
        logits = d(x, lambda_=1.0)
        assert logits.shape == (4,)


class TestStrategyRegistration:
    def test_new_strategies_registered(self) -> None:
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
        factory = TrainingStrategyFactory()
        assert "privileged_learning" in factory.STRATEGY_CLASS_PATHS
        assert "privileged" in factory.STRATEGY_CLASS_PATHS
        assert "slice_to_volume" in factory.STRATEGY_CLASS_PATHS
        assert "2d_to_3d" in factory.STRATEGY_CLASS_PATHS

    def test_strategy_classes_importable(self) -> None:
        from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
            PrivilegedLearningStrategy,
        )
        from spectramr.infrastructure.training.strategies.slice_to_volume_strategy import (
            SliceToVolumeStrategy,
        )
        assert PrivilegedLearningStrategy is not None
        assert SliceToVolumeStrategy is not None


# ── Sanity-shape parametrization ────────────────────────────────────────


# (order, feature_dim, expected_spatial)  — 2^order × 2^order grid
_TALL_2D_SHAPES = [
    (2, 4, 4),    # 4×4 = 16 tokens
    (3, 8, 8),    # 8×8 = 64 tokens
    (4, 8, 16),   # 16×16 = 256 tokens
]


@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "order,feat,side",
    _TALL_2D_SHAPES,
    ids=[f"ord{o}feat{f}side{s}" for o, f, s in _TALL_2D_SHAPES],
)
def test_tall_2d_sanity_shape(order: int, feat: int, side: int) -> None:
    """TALL forward returns (B, F, side²) seq and (B, side²) perm in eval mode."""
    m = TALL(order=order, feature_dim=feat)
    m.eval()
    x = torch.randn(1, feat, side, side)
    seq, perm = m(x)
    assert seq.shape == (1, feat, side * side)
    assert perm.shape == (1, side * side)
    assert torch.isfinite(seq).all()
    # Permutation is a bijection of {0, …, N-1}
    assert torch.equal(torch.sort(perm[0]).values, torch.arange(side * side))


def test_tall_wrong_feature_dim_raises() -> None:
    """Passing x with wrong feature_dim produces mismatched shapes (no silent reshape)."""
    m = TALL(order=2, feature_dim=4)  # expects C=4
    m.eval()
    x = torch.randn(1, 8, 4, 4)  # C=8 — wrong
    # The cell-feature MLP will reject the wrong channel count.
    with pytest.raises((RuntimeError, Exception)):
        m(x)
