"""Device/shape regression tests for the empirical-Parzen score.

The padding branches in ``empirical_score`` and ``build_default_score_fn``
used to allocate zeros via ``torch.zeros(...)`` (CPU default), which crashed
with a cross-device ``RuntimeError`` when references lived on CUDA and a
shape mismatch forced padding. They now use ``Tensor.new_zeros`` so padding
inherits the source device and dtype.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.score_network import (
    build_default_score_fn,
    empirical_score,
)


def test_empirical_score_preserves_shape_and_device() -> None:
    x = torch.randn(1, 8, 8)
    refs = torch.randn(4, 1, 8, 8)
    score = empirical_score(x, refs, sigma=0.1)
    assert score.shape == x.shape
    assert score.device == x.device


def test_empirical_score_pads_when_refs_smaller() -> None:
    # refs have fewer elements than x -> padding branch must run cleanly.
    x = torch.randn(1, 8, 8)
    refs = torch.randn(3, 1, 6, 6)
    score = empirical_score(x, refs, sigma=0.2)
    assert score.shape == x.shape
    assert torch.isfinite(score).all()


def test_build_default_score_fn_mismatched_refs() -> None:
    pool = [torch.randn(1, 8, 8), torch.randn(1, 6, 6)]
    fn = build_default_score_fn(pool)
    score = fn(torch.randn(1, 8, 8), 0.1)
    assert torch.isfinite(score).all()


@pytest.mark.gpu
def test_empirical_score_padding_on_cuda() -> None:
    """Padding zeros must land on the reference device, not CPU."""
    x = torch.randn(1, 8, 8, device="cuda:0")
    refs = torch.randn(3, 1, 6, 6, device="cuda:0")
    score = empirical_score(x, refs, sigma=0.2)
    assert score.device.type == "cuda"
    assert score.shape == x.shape
    assert torch.isfinite(score).all()


@pytest.mark.gpu
def test_build_default_score_fn_cuda_mismatched_refs() -> None:
    pool = [
        torch.randn(1, 8, 8, device="cuda:0"),
        torch.randn(1, 6, 6, device="cuda:0"),
    ]
    fn = build_default_score_fn(pool)
    score = fn(torch.randn(1, 8, 8, device="cuda:0"), 0.1)
    assert score.device.type == "cuda"
    assert torch.isfinite(score).all()
