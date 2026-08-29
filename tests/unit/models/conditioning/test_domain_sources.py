"""Domain-axis conditioning sources (2026-05-26): field_strength, scanner_id,
site_id. These extend the conditioning subsystem so non-VF strategies can adapt
to paired field-strength (ULF↔HF), multi-scanner/vendor, and multi-site regimes.

Mirrors the convention in ``test_encoders.py`` / ``test_context.py``:
continuous axes use a sinusoidal encoder (like ``diffusion_t``), categorical axes
use an ``nn.Embedding`` (like ``contrast_id``).
"""

from __future__ import annotations

import torch

from mriforge.config.schemas.conditioning import ConditioningConfig
from mriforge.models.conditioning.context import ConditioningContext
from mriforge.models.conditioning.encoders import (
    ConditioningEncoderBank,
    get_conditioner,
)

_NEW_SOURCES = ("field_strength", "scanner_id", "site_id")


def test_schema_accepts_new_domain_sources():
    cfg = ConditioningConfig(enabled=True, sources=list(_NEW_SOURCES), embed_dim=16)
    assert tuple(cfg.sources) == _NEW_SOURCES
    # The frozen-schema allowlist must include them.
    assert set(_NEW_SOURCES).issubset(set(ConditioningConfig.allowed_sources()))


def test_context_carries_new_domain_sources():
    ctx = ConditioningContext(
        field_strength=torch.tensor([0.064, 3.0]),
        scanner_id=torch.tensor([0, 1]),
        site_id=torch.tensor([2, 0]),
    )
    for s in _NEW_SOURCES:
        assert ctx.has(s), f"context missing {s}"
        assert ctx.get(s) is not None
    # frozen-safe device move keeps them
    moved = ctx.to(torch.device("cpu"))
    assert moved.has("field_strength")


def test_each_new_encoder_emits_embed_dim():
    ctx = ConditioningContext(
        field_strength=torch.tensor([0.064, 1.5, 3.0]),
        scanner_id=torch.tensor([0, 1, 2]),
        site_id=torch.tensor([1, 0, 2]),
    )
    for s in _NEW_SOURCES:
        enc = get_conditioner(s, embed_dim=16)
        out = enc(ctx)
        assert out.shape == (3, 16), f"{s} encoder emitted {out.shape}"


def test_field_strength_is_continuous_discriminative():
    """field_strength is continuous: distinct Tesla values must give distinct
    embeddings (a categorical embedding would collapse near values)."""
    enc = get_conditioner("field_strength", embed_dim=32)
    a = enc(ConditioningContext(field_strength=torch.tensor([0.064])))
    b = enc(ConditioningContext(field_strength=torch.tensor([3.0])))
    assert not torch.allclose(a, b)


def test_bank_fuses_three_domain_sources():
    bank = ConditioningEncoderBank(list(_NEW_SOURCES), embed_dim=16, fusion="concat")
    ctx = ConditioningContext(
        field_strength=torch.tensor([0.3, 0.3]),
        scanner_id=torch.tensor([0, 1]),
        site_id=torch.tensor([0, 1]),
    )
    out = bank(ctx)
    assert out.shape == (2, 16)
