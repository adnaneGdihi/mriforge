"""Unit tests for KSD utility — Phase 0 of audit_plan_novel."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.stein.ksd import (
    kernelised_stein_discrepancy,
    ksd_p_value,
)


def _gaussian_score(samples: torch.Tensor) -> torch.Tensor:
    """∇_x log N(0, I) = -x."""
    return -samples


def test_ksd_small_on_correct_model() -> None:
    """When samples are drawn from N(0,I) and we test against N(0,I), KSD² ≈ 0."""
    torch.manual_seed(0)
    samples = torch.randn(200, 5)
    ksd2, _ = kernelised_stein_discrepancy(_gaussian_score, samples)
    assert abs(ksd2) < 0.05


def test_ksd_large_on_wrong_model() -> None:
    """Samples from N(2,I) vs. score of N(0,I) — KSD² should be detectably positive."""
    torch.manual_seed(0)
    samples = torch.randn(200, 5) + 2.0
    ksd2, _ = kernelised_stein_discrepancy(_gaussian_score, samples)
    assert ksd2 > 0.5


def test_ksd_requires_2d_samples() -> None:
    with pytest.raises(ValueError):
        kernelised_stein_discrepancy(_gaussian_score, torch.zeros(10))


def test_ksd_p_value_in_unit_interval() -> None:
    torch.manual_seed(1)
    samples = torch.randn(50, 3)
    _, K = kernelised_stein_discrepancy(_gaussian_score, samples, return_kernel_matrix=True)
    assert K is not None
    p = ksd_p_value(K, n_bootstrap=200)
    assert 0.0 <= p <= 1.0


def test_ksd_p_value_rejects_clearly_wrong_model() -> None:
    """When the model is clearly wrong, the bootstrap should reject with low p."""
    torch.manual_seed(2)
    samples = torch.randn(80, 3) + 3.0
    _, K = kernelised_stein_discrepancy(_gaussian_score, samples, return_kernel_matrix=True)
    p = ksd_p_value(K, n_bootstrap=500)
    assert p < 0.1
