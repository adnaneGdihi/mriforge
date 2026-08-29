"""Tests for FieldConditionedLoRAConv2d (B-3.6 field-conditioned low-rank weight modulation)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.field_conditioned_lora import FieldConditionedLoRAConv2d


def test_forward_shape() -> None:
    out = FieldConditionedLoRAConv2d(8, 16, 3, rank=4)(torch.rand(2, 8, 12, 12), torch.rand(2, 2))
    assert out.shape == (2, 16, 12, 12)


def test_lora_delta_is_zero_at_init() -> None:
    # lora_up zero-init -> the weight delta is 0 at init -> the conv == the shared backbone.
    c = FieldConditionedLoRAConv2d(8, 8, 3, rank=4)
    x, fld = torch.rand(2, 8, 8, 8), torch.rand(2, 2)
    assert torch.allclose(c(x, fld), c.base(x), atol=1e-6)


def test_field_is_load_bearing_after_perturbation() -> None:
    # After escaping the zero-init delta (perturb lora_up + alpha_mlp), distinct field pairs
    # give distinct weight modulations -> distinct outputs.
    torch.manual_seed(0)
    c = FieldConditionedLoRAConv2d(8, 8, 3, rank=4)
    with torch.no_grad():
        c.lora_up.weight.normal_(0.0, 0.5)
        c.alpha_mlp.weight.normal_(0.0, 0.5)
    x = torch.rand(2, 8, 8, 8)
    o1 = c(x, torch.tensor([[-1.0, 0.8], [-1.0, 0.8]]))
    o2 = c(x, torch.tensor([[0.5, -1.0], [0.5, -1.0]]))
    assert not torch.allclose(o1.detach(), o2.detach())


def test_alpha_mlp_is_bias_free() -> None:
    # bias=False is load-bearing (#16): with a bias, W_alpha->0 leaves a NON-ZERO but
    # field-INVARIANT delta, so a "delta != 0" probe false-PASSes while the field is ignored.
    # Without the bias, zeroing the field-weight gives alpha=0 -> delta=0 -> the conv collapses
    # to the bare backbone (field-invariant coincides with the already-probed inert solution).
    c = FieldConditionedLoRAConv2d(8, 8, 3, rank=4)
    assert c.alpha_mlp.bias is None
    with torch.no_grad():
        c.lora_up.weight.normal_(0.0, 0.5)
        c.alpha_mlp.weight.zero_()
    x = torch.rand(2, 8, 8, 8)
    o1 = c(x, torch.tensor([[-1.0, 0.8], [-1.0, 0.8]]))
    o2 = c(x, torch.tensor([[0.5, -1.0], [0.5, -1.0]]))
    assert torch.allclose(o1.detach(), o2.detach())  # field-invariant
    assert torch.allclose(o1.detach(), c.base(x).detach(), atol=1e-6)  # delta == 0 (inert)


def test_rank_zero_is_field_blind() -> None:
    # rank=0 drops the adapter -> a plain conv -> the field is unused (the ablation control).
    c = FieldConditionedLoRAConv2d(8, 8, 3, rank=0)
    x = torch.rand(2, 8, 8, 8)
    o1 = c(x, torch.tensor([[-1.0, 0.8], [-1.0, 0.8]]))
    o2 = c(x, torch.tensor([[0.5, -1.0], [0.5, -1.0]]))
    assert torch.allclose(o1, o2)
    assert c(x).shape == (2, 8, 8, 8)  # field is optional at rank=0


def test_rank_positive_requires_field() -> None:
    with pytest.raises(ValueError, match="requires"):
        FieldConditionedLoRAConv2d(8, 8, 3, rank=4)(torch.rand(2, 8, 8, 8))


def test_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        FieldConditionedLoRAConv2d(8, 8, 3, rank=-1)
