"""Tests for the P4 (C_4) spatially-rotation-equivariant block family (B-1.4)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.models.blocks.equivariant.steerable_p4 import (
    P4GroupConv2d,
    P4LiftingConv2d,
    p4_group_norm,
    p4_to_scalar,
    rotate_p4_field,
)


def _equivariant_net() -> tuple:
    torch.manual_seed(0)
    lift = P4LiftingConv2d(1, 6, 3).eval()
    n1 = p4_group_norm(6).eval()
    gc = P4GroupConv2d(6, 4, 3).eval()
    n2 = p4_group_norm(4).eval()
    proj = P4GroupConv2d(4, 1, 3).eval()
    act = nn.SiLU()

    def forward(x: torch.Tensor) -> torch.Tensor:
        h = act(n1(lift(x)))
        h = act(n2(gc(h)))
        return p4_to_scalar(proj(h), 1)

    return forward, (lift, gc, proj)


def test_lifting_output_shape_is_regular_field() -> None:
    out = P4LiftingConv2d(1, 5, 3)(torch.rand(2, 1, 16, 16))
    assert out.shape == (2, 5 * 4, 16, 16)


def test_group_conv_preserves_regular_field_shape() -> None:
    out = P4GroupConv2d(5, 7, 3)(torch.rand(2, 5 * 4, 16, 16))
    assert out.shape == (2, 7 * 4, 16, 16)


def test_to_scalar_averages_orientation_axis() -> None:
    field = torch.rand(2, 3 * 4, 8, 8)
    scal = p4_to_scalar(field, 3)
    assert scal.shape == (2, 3, 8, 8)
    # mean of the 4 consecutive orientation channels of base channel 0
    expect0 = field[:, 0:4].mean(dim=1)
    assert torch.allclose(scal[:, 0], expect0, atol=1e-6)


@pytest.mark.parametrize("s", [1, 2, 3])
def test_full_stack_is_exactly_rotation_equivariant(s: int) -> None:
    # THE anti-facade test (#16/#19): rotate the image by s*90deg and the output must
    # rotate by s*90deg, exactly (rot90 + square + zero-pad => no interpolation/boundary
    # error). This is genuine SPATIAL rotation equivariance, the B-1.4 claim.
    forward, _ = _equivariant_net()
    x = torch.randn(2, 1, 32, 32)
    lhs = forward(torch.rot90(x, s, dims=(-2, -1)))  # G(g.x)
    rhs = torch.rot90(forward(x), s, dims=(-2, -1))  # g.G(x)
    assert torch.allclose(lhs, rhs, atol=1e-4), f"max err {(lhs - rhs).abs().max():.2e}"


def test_lifting_satisfies_field_action_contract() -> None:
    # lift(rot90^s x) == rotate_p4_field(lift(x), s): rotating the image both rotates each
    # orientation map AND cyclically shifts the orientation fibre.
    torch.manual_seed(1)
    lift = P4LiftingConv2d(1, 4, 3).eval()
    x = torch.randn(1, 1, 24, 24)
    base = lift(x)
    for s in (1, 2, 3):
        lhs = lift(torch.rot90(x, s, dims=(-2, -1)))
        rhs = rotate_p4_field(base, s)
        assert torch.allclose(lhs, rhs, atol=1e-4)


def test_plain_conv_is_not_equivariant_control() -> None:
    # Confirms the equivariance test can FAIL — a vanilla conv breaks under rotation.
    c = nn.Conv2d(1, 1, 3, padding=1).eval()
    x = torch.randn(1, 1, 32, 32)
    err = (
        c(torch.rot90(x, 1, dims=(-2, -1))) - torch.rot90(c(x), 1, dims=(-2, -1))
    ).abs().max().detach()
    assert float(err) > 1e-2


def test_group_conv_rejects_misshaped_field() -> None:
    with pytest.raises(RuntimeError):
        # in_base=5 expects 20 channels; give 18 -> conv weight mismatch
        P4GroupConv2d(5, 4, 3)(torch.rand(1, 18, 8, 8))


def test_rotate_p4_field_rejects_non_multiple_of_four() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        rotate_p4_field(torch.rand(1, 6, 8, 8), 1)
