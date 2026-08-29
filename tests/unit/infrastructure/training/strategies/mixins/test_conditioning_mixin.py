"""ConditioningMixin (2026-05-26): the shared adaptive-conditioning seam lifted
out of the VF strategy so ANY strategy can FiLM-condition on its declared
sources. Mounted on BaseTrainingStrategy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mriforge.config.schemas.conditioning import ConditioningConfig
from mriforge.infrastructure.training.strategies.mixins.conditioning_mixin import (
    ConditioningMixin,
)
from mriforge.models.conditioning import ConditioningContext


class _Host(ConditioningMixin):
    _SUPPORTED_CONDITION_SOURCES = ("field_strength", "scanner_id")

    def __init__(self, cfg, opt=None):
        self.config = SimpleNamespace(model=SimpleNamespace(conditioning=cfg))
        self.env = SimpleNamespace(opt_g=opt)
        self.device = torch.device("cpu")
        self._conditioner = None


def test_validate_rejects_unsupported_source():
    with pytest.raises(ValueError, match="site_id"):
        _Host._validate_condition_sources(["field_strength", "site_id"])


def test_ensure_conditioner_builds_and_registers_params():
    cfg = ConditioningConfig(enabled=True, sources=["field_strength"], embed_dim=8)
    dummy = nn.Linear(2, 2)
    opt = torch.optim.Adam(dummy.parameters(), lr=1e-3)
    host = _Host(cfg, opt)
    cond = host._ensure_conditioner(n_channels=4)
    assert cond is not None
    assert len(opt.param_groups) == 2  # conditioner params registered for training


def test_ensure_conditioner_disabled_returns_none():
    assert _Host(ConditioningConfig(), None)._ensure_conditioner(4) is None


def test_ensure_conditioner_rejects_unsupported_enabled_source():
    cfg = ConditioningConfig(enabled=True, sources=["site_id"], embed_dim=8)
    with pytest.raises(ValueError, match="site_id"):
        _Host(cfg, None)._ensure_conditioner(4)


def test_domain_context_from_batch_keys():
    batch = {
        "field_strength": torch.tensor([0.064, 3.0]),
        "scanner_id": torch.tensor([0, 1]),
        "other": object(),
    }
    ctx = ConditioningMixin._domain_conditioning_context(batch, torch.device("cpu"))
    assert ctx is not None and ctx.has("field_strength") and ctx.has("scanner_id")
    assert isinstance(ctx, ConditioningContext)


def test_domain_context_none_when_no_axes():
    assert ConditioningMixin._domain_conditioning_context({"x": 1}, torch.device("cpu")) is None


def test_base_strategy_inherits_mixin():
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

    assert issubclass(BaseTrainingStrategy, ConditioningMixin)


class _ModelCfg:
    def __init__(self, conditioning):
        self.conditioning = conditioning


class _Cfg:
    def __init__(self, conditioning):
        self.model = _ModelCfg(conditioning)


class _Env:
    def __init__(self, opt):
        self.opt_g = opt


class _Strat(ConditioningMixin):
    _SUPPORTED_CONDITION_SOURCES = ("contrast_id",)

    def __init__(self, conditioning):
        self.config = _Cfg(conditioning)
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.env = _Env(torch.optim.SGD([self._dummy], lr=0.0))
        self.device = torch.device("cpu")
        self._conditioner = None


def _enabled_cfg():
    return ConditioningConfig(
        enabled=True, sources=["contrast_id"], embed_dim=16, fusion="concat"
    )


def test_apply_input_conditioning_is_identity_at_init():
    strat = _Strat(_enabled_cfg())
    x = torch.randn(2, 3, 8, 8)
    batch = {"contrast_id": torch.tensor([0, 1])}
    out = strat._apply_input_conditioning(x, batch)
    assert out.shape == x.shape
    # Zero-init FiLM head ⇒ (1+0)*x + 0 == x at initialisation.
    torch.testing.assert_close(out, x)


def test_apply_input_conditioning_disabled_returns_same_object():
    strat = _Strat(ConditioningConfig(enabled=False))
    x = torch.randn(2, 3, 8, 8)
    batch = {"contrast_id": torch.tensor([0, 1])}
    assert strat._apply_input_conditioning(x, batch) is x


def test_apply_input_conditioning_enabled_but_no_axis_raises():
    """Mechanism-fires guard (pitfall #16): conditioning enabled with declared
    sources, but the batch carried none of them ⇒ the FiLM modulation would
    silently no-op. The seam must RAISE, not quietly pass the input through."""
    strat = _Strat(_enabled_cfg())  # enabled, sources=[contrast_id]
    x = torch.randn(2, 3, 8, 8)
    with pytest.raises(ValueError, match="conditioning is enabled"):
        strat._apply_input_conditioning(x, {"input": x})  # no contrast_id present


def test_domain_conditioning_context_aliases_contrast_idx_to_contrast_id():
    batch = {"contrast_idx": torch.tensor([0, 2])}
    ctx = ConditioningMixin._domain_conditioning_context(batch, torch.device("cpu"))
    assert ctx is not None
    assert ctx.contrast_id is not None
    torch.testing.assert_close(ctx.contrast_id, torch.tensor([0, 2]))


def test_domain_conditioning_context_prefers_contrast_id_over_idx():
    batch = {"contrast_id": torch.tensor([1, 1]), "contrast_idx": torch.tensor([0, 0])}
    ctx = ConditioningMixin._domain_conditioning_context(batch, torch.device("cpu"))
    torch.testing.assert_close(ctx.contrast_id, torch.tensor([1, 1]))
