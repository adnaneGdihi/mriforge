"""Unit tests for inject_lora_adapters utility.

Tests that LoRA adapters are correctly injected into matching nn.Linear and
nn.Conv2d layers while preserving pre-trained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.lora import LoRAConv2d, LoRALinear, inject_lora_adapters


class SimpleModel(nn.Module):
    """Toy model for testing LoRA injection."""

    def __init__(self) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(64, 64)
        self.attn_out = nn.Linear(64, 64)
        self.mlp_fc = nn.Linear(64, 128)
        self.conv_proj = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv_other = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestInjectLoraAdapters:
    """Tests for the inject_lora_adapters function."""

    def test_inject_matching_linear(self) -> None:
        """Should replace matching nn.Linear with LoRALinear."""
        model = SimpleModel()
        count = inject_lora_adapters(model, target_modules=["attn"], rank=4, alpha=8)

        assert count == 2  # attn_proj and attn_out
        assert isinstance(model.attn_proj, LoRALinear)
        assert isinstance(model.attn_out, LoRALinear)
        # mlp_fc should NOT be replaced
        assert isinstance(model.mlp_fc, nn.Linear)
        assert not isinstance(model.mlp_fc, LoRALinear)

    def test_inject_matching_conv2d(self) -> None:
        """Should replace matching nn.Conv2d with LoRAConv2d."""
        model = SimpleModel()
        count = inject_lora_adapters(model, target_modules=["conv_proj"], rank=4)

        assert count == 1  # Only conv_proj, not conv_other
        assert isinstance(model.conv_proj, LoRAConv2d)
        assert isinstance(model.conv_other, nn.Conv2d)
        assert not isinstance(model.conv_other, LoRAConv2d)

    def test_no_match_returns_zero(self) -> None:
        """No matching modules should return 0 replacements."""
        model = SimpleModel()
        count = inject_lora_adapters(model, target_modules=["nonexistent"])
        assert count == 0

    def test_weights_preserved(self) -> None:
        """Pre-trained weights should be copied to LoRA module."""
        model = SimpleModel()
        original_weight = model.attn_proj.weight.data.clone()
        original_bias = model.attn_proj.bias.data.clone()

        inject_lora_adapters(model, target_modules=["attn_proj"], rank=4)

        assert isinstance(model.attn_proj, LoRALinear)
        assert torch.allclose(model.attn_proj.weight.data, original_weight)
        assert torch.allclose(model.attn_proj.bias.data, original_bias)

    def test_base_weights_frozen(self) -> None:
        """Base weights should be frozen after LoRA injection."""
        model = SimpleModel()
        inject_lora_adapters(model, target_modules=["attn_proj"], rank=4)

        assert isinstance(model.attn_proj, LoRALinear)
        assert model.attn_proj.weight.requires_grad is False
        # LoRA A and B should be trainable
        assert model.attn_proj.lora_A.requires_grad is True
        assert model.attn_proj.lora_B.requires_grad is True

    def test_lora_rank_and_alpha(self) -> None:
        """LoRA rank and alpha should be set correctly."""
        model = SimpleModel()
        inject_lora_adapters(model, target_modules=["attn_proj"], rank=16, alpha=32)

        assert isinstance(model.attn_proj, LoRALinear)
        assert model.attn_proj.r == 16
        assert model.attn_proj.lora_alpha == 32

    def test_forward_pass_unchanged_shape(self) -> None:
        """Output shape should be the same after LoRA injection."""
        model = SimpleModel()
        x = torch.randn(2, 64)
        original_shape = model.attn_proj(x).shape

        inject_lora_adapters(model, target_modules=["attn_proj"], rank=4)
        new_shape = model.attn_proj(x).shape

        assert original_shape == new_shape

    def test_default_target_modules(self) -> None:
        """Default target_modules should match 'attn' and 'proj'."""
        model = SimpleModel()
        count = inject_lora_adapters(model, rank=4)
        # attn_proj matches both, attn_out matches "attn", conv_proj matches "proj"
        assert count >= 2
