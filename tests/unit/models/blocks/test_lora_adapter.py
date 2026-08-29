"""Unit tests for the real LoRA adapter block (PR-7).

Covers the foundation-reconstruction differentiator: a low-rank adapter that
starts as an identity (B=0) and only adds trainable low-rank deviations on top
of a frozen base layer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mriforge.models.blocks.adapters.lora import LoRAAdapter, inject_lora


class TestLoRAAdapterIdentityInit:
    """The adapter must be a no-op at initialization (B=0)."""

    def test_linear_starts_as_identity(self):
        base = nn.Linear(8, 5)
        adapter = LoRAAdapter(base, rank=2, alpha=4.0)
        x = torch.randn(3, 8)
        torch.testing.assert_close(adapter(x), base(x))

    def test_conv2d_starts_as_identity(self):
        base = nn.Conv2d(4, 6, kernel_size=3, padding=1)
        adapter = LoRAAdapter(base, rank=2, alpha=4.0)
        x = torch.randn(2, 4, 16, 16)
        torch.testing.assert_close(adapter(x), base(x))

    def test_becomes_non_identity_after_perturbing_b_factor(self):
        base = nn.Linear(8, 5)
        adapter = LoRAAdapter(base, rank=2, alpha=4.0)
        x = torch.randn(3, 8)
        with torch.no_grad():
            adapter.lora_B.add_(1.0)
        assert not torch.allclose(adapter(x), base(x))


class TestLoRAAdapterParameters:
    """Frozen base + trainable low-rank factors."""

    def test_base_frozen_adapter_trainable(self):
        base = nn.Linear(8, 5)
        adapter = LoRAAdapter(base, rank=2, alpha=4.0)
        # base params frozen
        assert all(not p.requires_grad for p in adapter.base.parameters())
        # lora params trainable
        assert adapter.lora_A.requires_grad
        assert adapter.lora_B.requires_grad

    def test_param_count_base_plus_lowrank(self):
        in_f, out_f, r = 8, 5, 2
        base = nn.Linear(in_f, out_f)
        base_params = sum(p.numel() for p in base.parameters())
        adapter = LoRAAdapter(base, rank=r, alpha=4.0)
        total = sum(p.numel() for p in adapter.parameters())
        lowrank = r * in_f + out_f * r
        assert total == base_params + lowrank

    def test_trainable_count_is_lowrank_only(self):
        in_f, out_f, r = 8, 5, 3
        adapter = LoRAAdapter(nn.Linear(in_f, out_f), rank=r, alpha=6.0)
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        assert trainable == r * in_f + out_f * r

    def test_gradient_flows_to_lora_only(self):
        adapter = LoRAAdapter(nn.Linear(8, 5), rank=2, alpha=4.0)
        with torch.no_grad():
            adapter.lora_B.add_(0.1)
        out = adapter(torch.randn(3, 8)).sum()
        out.backward()
        assert adapter.lora_A.grad is not None
        assert adapter.lora_B.grad is not None
        assert all(p.grad is None for p in adapter.base.parameters())

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError):
            LoRAAdapter(nn.Linear(8, 5), rank=0, alpha=4.0)

    def test_unsupported_base_raises(self):
        with pytest.raises(TypeError):
            LoRAAdapter(nn.ReLU(), rank=2, alpha=4.0)


class TestInjectLoRA:
    """inject_lora wraps the right submodules."""

    def test_wraps_conv2d_by_default(self):
        model = nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, 3, padding=1),
        )
        n = inject_lora(model, rank=2, alpha=4.0)
        assert n == 2
        assert isinstance(model[0], LoRAAdapter)
        assert isinstance(model[2], LoRAAdapter)
        assert isinstance(model[1], nn.ReLU)

    def test_target_types_linear(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Conv2d(2, 2, 3, padding=1))
        n = inject_lora(model, rank=2, alpha=4.0, target_types=(nn.Linear,))
        assert n == 1
        assert isinstance(model[0], LoRAAdapter)
        assert isinstance(model[1], nn.Conv2d)

    def test_inject_preserves_forward_at_init(self):
        model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1))
        x = torch.randn(1, 2, 8, 8)
        before = model(x)
        inject_lora(model, rank=2, alpha=4.0)
        after = model(x)
        torch.testing.assert_close(after, before)
