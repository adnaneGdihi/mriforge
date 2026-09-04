"""Tests for multi-contrast Hankel primary-decomposition primitives."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.primary_decomposition import (
    build_hankel_matrix,
    multi_contrast_hankel_rank_stratification,
    primary_projection_loss,
    recover_loraks_limit,
)


def test_build_hankel_matrix_shape_and_anti_diagonals() -> None:
    """Hankel matrices have constant anti-diagonals by construction."""
    signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    h = build_hankel_matrix(signal, window=3)
    assert h.shape == (3, 3)
    # Anti-diagonals: each anti-diagonal entry should be the same signal value.
    expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    assert torch.allclose(h, expected)


def test_hankel_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        build_hankel_matrix(torch.zeros(5), window=0)
    with pytest.raises(ValueError):
        build_hankel_matrix(torch.zeros(5), window=10)


def test_multi_contrast_stratification_shape() -> None:
    """SVD of the joint Hankel matrix returns the right-rank decomposition."""
    torch.manual_seed(0)
    signals = torch.randn(3, 16)  # 3 contrasts, length 16
    u, sigma, vh = multi_contrast_hankel_rank_stratification(signals, window=4)
    # joint shape: (3*4, 16-4+1) = (12, 13). SVD truncates to min(12, 13) = 12.
    assert sigma.shape == (12,)
    assert u.shape[1] == 12
    assert vh.shape[0] == 12


def test_truncated_rank_preserves_top_singular_values() -> None:
    """Rank truncation keeps the leading r singular values."""
    torch.manual_seed(0)
    signals = torch.randn(2, 16)
    _, sigma_full, _ = multi_contrast_hankel_rank_stratification(signals, window=4)
    _, sigma_trunc, _ = multi_contrast_hankel_rank_stratification(signals, window=4, rank=3)
    assert torch.allclose(sigma_trunc, sigma_full[:3])


def test_loraks_recovery_at_single_contrast() -> None:
    """C=1 → SVD of single-contrast Hankel = LORAKS regulariser kernel."""
    torch.manual_seed(0)
    signal = torch.randn(1, 16)
    u, sigma, vh = recover_loraks_limit(signal, window=4)
    # Cross-check with the direct Hankel SVD.
    h = build_hankel_matrix(signal[0], window=4)
    _, sigma_ref, _ = torch.linalg.svd(h, full_matrices=False)
    assert torch.allclose(sigma, sigma_ref)


def test_recover_loraks_invalid_c_raises() -> None:
    with pytest.raises(ValueError):
        recover_loraks_limit(torch.zeros(2, 16), window=4)


def test_primary_projection_loss_zero_on_aligned_candidate() -> None:
    """If the candidate lies in span(component_basis) the projection loss vanishes."""
    # Identity components: candidate is one of the basis vectors.
    basis = torch.eye(4)
    candidate = basis[:, 0]
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
    loss = primary_projection_loss(candidate, basis, weights)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_primary_projection_loss_positive_on_orthogonal_candidate() -> None:
    basis = torch.eye(4)[:, :2]  # Span of first two axes
    candidate = torch.tensor([0.0, 0.0, 1.0, 0.0])  # orthogonal to span
    weights = torch.tensor([1.0, 1.0])
    loss = primary_projection_loss(candidate, basis, weights)
    # ||candidate - 0||² = 1 per component, summed → 2.0 weighted sum.
    assert loss.item() == pytest.approx(2.0, abs=1e-5)


def test_invalid_primary_projection_shapes() -> None:
    with pytest.raises(ValueError):
        primary_projection_loss(torch.zeros(4, 1), torch.zeros(4, 2), torch.ones(2))
    with pytest.raises(ValueError):
        primary_projection_loss(torch.zeros(4), torch.zeros(3, 2), torch.ones(2))
    with pytest.raises(ValueError):
        primary_projection_loss(torch.zeros(4), torch.zeros(4, 2), torch.ones(3))
