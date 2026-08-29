"""Tests for S4D + Hyena blocks (PR-22)."""

from __future__ import annotations

import torch

from mriforge.models.blocks.hyena_block import HyenaOperator
from mriforge.models.blocks.s4d_block import S4DBlock


def test_s4d_block_shape_preserved() -> None:
    block = S4DBlock(d_model=8, d_state=4)
    x = torch.randn(2, 32, 8)
    out = block(x)
    assert out.shape == (2, 32, 8)
    assert torch.isfinite(out).all()


def test_s4d_block_bidirectional() -> None:
    block = S4DBlock(d_model=8, d_state=4, bidirectional=True)
    x = torch.randn(2, 16, 8)
    out = block(x)
    assert out.shape == (2, 16, 8)


def test_hyena_operator_shape_preserved() -> None:
    op = HyenaOperator(d_model=8, order=2, max_seq_len=64)
    x = torch.randn(2, 32, 8)
    out = op(x)
    assert out.shape == (2, 32, 8)
    assert torch.isfinite(out).all()


def test_hyena_operator_grad_flows() -> None:
    op = HyenaOperator(d_model=8, order=1, max_seq_len=32)
    x = torch.randn(1, 16, 8, requires_grad=True)
    out = op(x).sum()
    out.backward()
    assert x.grad is not None


def test_hyena_max_seq_len_enforced() -> None:
    import pytest
    op = HyenaOperator(d_model=8, order=1, max_seq_len=16)
    too_long = torch.randn(1, 32, 8)
    with pytest.raises(ValueError, match="max_seq_len"):
        op(too_long)
