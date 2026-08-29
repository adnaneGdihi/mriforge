"""Unit tests for SpectraTestTimeAdaptationStrategy (plan §D).

Verifies the load-bearing TTA behaviours:
  * the backbone is frozen, leaving only norm/adapter params trainable;
  * the self-supervised k-space-consistency step returns a scalar ``g_total_loss``;
  * gradients flow ONLY through the adaptable params (zero through the backbone) —
    this is what bounds the adaptation memory footprint.

Construction follows the repo convention of patching the DI
``resolve_service`` so ``BaseTrainingStrategy.__init__`` does not need a live
container (see tests/strategies/test_strategy_data_leak.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

_DI = "mriforge.infrastructure.di.di_container.resolve_service"


class _TinyBackbone(nn.Module):
    """conv (backbone) → InstanceNorm (adaptable) → 1x1 'adapter' (adaptable)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.norm = nn.InstanceNorm2d(4, affine=True)
        self.adapter = nn.Conv2d(4, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.norm(self.conv(x)))


class _EnvStub:
    """Minimal duck-typed TrainingEnvironment: device / config / models + .generator."""

    def __init__(self, gen: nn.Module, config: object | None = None) -> None:
        self.device = torch.device("cpu")
        self.config = config
        self.models = {"generator": gen}
        self.discriminator = None  # TTA is generator-only

    @property
    def generator(self) -> nn.Module:
        return self.models["generator"]


def _make_config() -> MagicMock:
    """Minimal config the base __init__ reads (config.optimization.*)."""
    cfg = MagicMock()
    cfg.optimization.gradient.clip.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    cfg.optimization.precision.dtype = "float32"
    cfg.optimization.gradient.clip.value = 1.0
    # A real config always carries the schema default; without it the auto-mock
    # value fails StandardOptimizerStepper's raise-on-unknown validation.
    cfg.optimization.gradient.clip.method = "norm"
    cfg.training.spectra_tta = None  # use strategy defaults
    return cfg


def _make_strategy(gen: nn.Module, config: object | None = None):
    from mriforge.infrastructure.training.strategies.spectra_tta_strategy import (
        SpectraTestTimeAdaptationStrategy,
    )

    if config is None:
        config = _make_config()
    with patch(_DI, return_value=MagicMock()):
        return SpectraTestTimeAdaptationStrategy(env=_EnvStub(gen, config))


def test_backbone_is_frozen_adapter_trainable() -> None:
    gen = _TinyBackbone()
    strat = _make_strategy(gen)
    acct = strat.apply_backbone_freeze()
    # conv weight+bias frozen (2); norm + adapter weight+bias trainable (4)
    assert acct["frozen"] == 2
    assert acct["trainable"] == 4
    assert gen.conv.weight.requires_grad is False
    assert gen.norm.weight.requires_grad is True
    assert gen.adapter.weight.requires_grad is True


def test_compute_losses_returns_scalar_g_total() -> None:
    gen = _TinyBackbone()
    strat = _make_strategy(gen)
    x = torch.randn(1, 1, 16, 16)
    tgt = torch.randn(1, 1, 16, 16)
    losses = strat._compute_losses_impl(x, tgt, epoch=0)
    assert "g_total_loss" in losses
    assert "kspace_consistency_tta" in losses
    g = losses["g_total_loss"]
    assert g.ndim == 0
    assert torch.isfinite(g)


def test_gradient_flows_only_through_adapter() -> None:
    """The bounded-memory guarantee: no gradient on the frozen backbone."""
    gen = _TinyBackbone()
    strat = _make_strategy(gen)
    strat.apply_backbone_freeze()
    x = torch.randn(1, 1, 16, 16)
    tgt = torch.randn(1, 1, 16, 16)
    losses = strat._compute_losses_impl(x, tgt, epoch=0)
    losses["g_total_loss"].backward()
    assert gen.conv.weight.grad is None  # frozen backbone — no grad
    assert gen.norm.weight.grad is not None  # adaptable
    assert gen.adapter.weight.grad is not None  # adaptable
    assert gen.adapter.weight.grad.abs().sum() > 0


def test_adapter_parameters_are_the_trainable_set() -> None:
    gen = _TinyBackbone()
    strat = _make_strategy(gen)
    strat.apply_backbone_freeze()
    adapter_ids = {id(p) for p in strat.adapter_parameters()}
    trainable_ids = {id(p) for p in gen.parameters() if p.requires_grad}
    assert adapter_ids == trainable_ids
    assert len(adapter_ids) == 4  # norm w/b + adapter w/b
