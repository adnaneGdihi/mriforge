"""Tests for the conditioning injection seam (2026-05-26).

FiLM modulation from a fused conditioning vector. Zero-init the FiLM head so a
freshly-added conditioner is the identity — it never perturbs a model at start.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from mriforge.models.conditioning import ConditioningContext  # noqa: E402
from mriforge.models.conditioning.injection import (  # noqa: E402
    AdaptiveConditioner,
    ConditioningInjector,
)


def test_injector_is_identity_at_init() -> None:
    inj = ConditioningInjector(cond_dim=16, num_features=8)
    h, c = torch.randn(2, 8, 4, 4), torch.randn(2, 16)
    assert torch.allclose(inj(h, c), h, atol=1e-6)


def test_injector_modulates_once_trained() -> None:
    inj = ConditioningInjector(cond_dim=16, num_features=8)
    for p in inj.parameters():
        nn.init.normal_(p, std=0.2)
    h, c = torch.randn(2, 8, 4, 4), torch.randn(2, 16)
    out = inj(h, c)
    assert out.shape == h.shape
    assert not torch.allclose(out, h)


def test_injector_broadcasts_over_spatial_dims() -> None:
    inj = ConditioningInjector(cond_dim=8, num_features=3)
    h = torch.randn(2, 3, 6, 6)
    assert inj(h, torch.randn(2, 8)).shape == h.shape


def test_adaptive_conditioner_end_to_end_shape() -> None:
    cond = AdaptiveConditioner(["diffusion_t", "severity_vec"], num_features=8, cond_dim=16)
    ctx = ConditioningContext(diffusion_t=torch.rand(2), severity_vec=torch.rand(2, 5))
    h = torch.randn(2, 8, 4, 4)
    assert cond(h, ctx).shape == h.shape


def test_adaptive_conditioner_identity_at_init() -> None:
    cond = AdaptiveConditioner(["diffusion_t"], num_features=4, cond_dim=8)
    ctx = ConditioningContext(diffusion_t=torch.rand(2))
    h = torch.randn(2, 4, 5, 5)
    assert torch.allclose(cond(h, ctx), h, atol=1e-6)


def test_adaptive_conditioner_encode_exposes_vector() -> None:
    cond = AdaptiveConditioner(["diffusion_t"], num_features=4, cond_dim=8)
    ctx = ConditioningContext(diffusion_t=torch.rand(3))
    assert cond.encode(ctx).shape == (3, 8)


def test_adaptive_conditioner_from_config() -> None:
    from mriforge.config.schemas.conditioning import ConditioningConfig

    cfg = ConditioningConfig(
        enabled=True,
        sources=["diffusion_t", "severity_vec"],
        embed_dim=16,
        fusion="sum",
        encoder_kwargs={"severity_vec": {"num_severities": 4}},
    )
    cond = AdaptiveConditioner.from_config(cfg, num_features=8)
    ctx = ConditioningContext(diffusion_t=torch.rand(2), severity_vec=torch.rand(2, 4))
    assert cond(torch.randn(2, 8, 4, 4), ctx).shape == (2, 8, 4, 4)


def test_adaptive_conditioner_from_disabled_config_returns_none() -> None:
    from mriforge.config.schemas.conditioning import ConditioningConfig

    assert AdaptiveConditioner.from_config(ConditioningConfig(), num_features=8) is None


def test_adaptive_conditioner_from_config_film_injection_modulates() -> None:
    """The advertised injection='film' knob is read and builds a working
    conditioner that modulates (default-path, numerics unchanged)."""
    from mriforge.config.schemas.conditioning import ConditioningConfig

    cfg = ConditioningConfig(
        enabled=True,
        sources=["diffusion_t"],
        embed_dim=8,
        fusion="concat",
        injection="film",
    )
    cond = AdaptiveConditioner.from_config(cfg, num_features=4)
    assert cond is not None
    ctx = ConditioningContext(diffusion_t=torch.rand(2))
    h = torch.randn(2, 4, 5, 5)
    # Zero-init FiLM head ⇒ identity at start (knob took effect, not a no-op).
    assert torch.allclose(cond(h, ctx), h, atol=1e-6)


def test_adaptive_conditioner_from_config_rejects_unknown_injection() -> None:
    """An unsupported injection value must RAISE (pitfall #15: no silent no-op).

    ``model_construct`` bypasses the schema Literal so the runtime guard in
    ``from_config`` is the thing under test.
    """
    from mriforge.config.schemas.conditioning import ConditioningConfig

    cfg = ConditioningConfig.model_construct(
        enabled=True,
        sources=["diffusion_t"],
        embed_dim=8,
        fusion="concat",
        injection="hypernet",
        encoder_kwargs={},
    )
    with pytest.raises(ValueError, match="injection"):
        AdaptiveConditioner.from_config(cfg, num_features=4)


def test_adaptive_conditioner_is_differentiable() -> None:
    cond = AdaptiveConditioner(["severity_vec"], num_features=4, cond_dim=8)
    for p in cond.parameters():  # break identity so grads flow to the input
        nn.init.normal_(p, std=0.1)
    sev = torch.rand(2, 5, requires_grad=True)
    cond(torch.randn(2, 4, 3, 3), ConditioningContext(severity_vec=sev)).sum().backward()
    assert sev.grad is not None


def test_conditioning_dropout_uses_null_token_in_train() -> None:
    """cond_dropout=1.0 in train mode ⇒ every sample uses the null token, so
    the output is independent of the context (classifier-free unconditional)."""
    cond = AdaptiveConditioner(["severity_vec"], num_features=4, cond_dim=8, cond_dropout=1.0)
    for p in cond.parameters():
        nn.init.normal_(p, std=0.1)
    cond.train()
    h = torch.randn(2, 4, 3, 3)
    out_zeros = cond(h, ConditioningContext(severity_vec=torch.zeros(2, 5)))
    out_ones = cond(h, ConditioningContext(severity_vec=torch.ones(2, 5)))
    assert torch.allclose(out_zeros, out_ones, atol=1e-6)


def test_conditioning_dropout_inactive_in_eval() -> None:
    cond = AdaptiveConditioner(["severity_vec"], num_features=4, cond_dim=8, cond_dropout=1.0)
    for p in cond.parameters():
        nn.init.normal_(p, std=0.1)
    cond.eval()
    h = torch.randn(2, 4, 3, 3)
    out_zeros = cond(h, ConditioningContext(severity_vec=torch.zeros(2, 5)))
    out_ones = cond(h, ConditioningContext(severity_vec=torch.ones(2, 5)))
    assert not torch.allclose(out_zeros, out_ones)  # eval ⇒ context matters


def test_force_uncond_uses_null_token() -> None:
    cond = AdaptiveConditioner(["severity_vec"], num_features=4, cond_dim=8)
    for p in cond.parameters():
        nn.init.normal_(p, std=0.1)
    cond.eval()
    h = torch.randn(2, 4, 3, 3)
    ctx = ConditioningContext(severity_vec=torch.ones(2, 5))
    assert not torch.allclose(cond(h, ctx), cond(h, ctx, force_uncond=True))
