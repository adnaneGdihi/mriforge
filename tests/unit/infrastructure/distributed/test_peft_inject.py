"""Tests for the PEFT (LoRA) injector (PR-14)."""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.config.schemas.base import PEFTConfigSchema
from spectramr.infrastructure.distributed.peft_inject import LoRAAdapter, inject_peft


class _TwoLinears(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 8)
        self.unrelated = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.qkv(x))


def test_lora_adapter_identity_at_init() -> None:
    """B is initialised to zero → adapter == base at init."""
    base = nn.Linear(4, 4)
    adapter = LoRAAdapter(base, rank=2, alpha=4.0)
    x = torch.randn(2, 4)
    assert torch.allclose(adapter(x), base(x), atol=1e-6)


def test_lora_adapter_freezes_base() -> None:
    base = nn.Linear(4, 4)
    adapter = LoRAAdapter(base, rank=2, alpha=4.0)
    assert not adapter.base.weight.requires_grad
    assert adapter.lora_A.requires_grad
    assert adapter.lora_B.requires_grad


def test_inject_peft_replaces_target_modules() -> None:
    cfg = PEFTConfigSchema(enabled=True, method="lora", rank=2, alpha=4.0,
                           target_modules=["qkv", "proj"], bias_mode="none")
    model = _TwoLinears()
    inject_peft(model, cfg)
    # qkv and proj are LoRAAdapters; unrelated is still a plain Linear.
    assert isinstance(model.qkv, LoRAAdapter)
    assert isinstance(model.proj, LoRAAdapter)
    assert isinstance(model.unrelated, nn.Linear)


def test_inject_peft_disabled_is_noop() -> None:
    cfg = PEFTConfigSchema(enabled=False)
    model = _TwoLinears()
    inject_peft(model, cfg)
    assert isinstance(model.qkv, nn.Linear)


def test_inject_peft_bitfit_unfreezes_only_biases() -> None:
    cfg = PEFTConfigSchema(enabled=True, method="bitfit")
    model = _TwoLinears()
    inject_peft(model, cfg)
    for n, p in model.named_parameters():
        if "bias" in n:
            assert p.requires_grad, f"bias {n} should be trainable"
        else:
            assert not p.requires_grad, f"weight {n} should be frozen"


def test_lora_rejects_zero_rank() -> None:
    import pytest
    base = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="rank"):
        LoRAAdapter(base, rank=0)
