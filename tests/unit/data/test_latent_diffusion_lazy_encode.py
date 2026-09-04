"""Phase 4f Amendment H — LazyEncodeWrapper + schema."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torchio as tio
from pydantic import ValidationError
from torch.utils.data import Dataset

from spectramr.config.schemas.data import (
    DataConfigSchema,
    LatentDiffusionConfigSchema,
)
from spectramr.data.builders.lazy_encode import LazyEncodeWrapper


# ── Schema validation ───────────────────────────────────────────────────────


def test_latent_diffusion_defaults_are_disabled() -> None:
    cfg = LatentDiffusionConfigSchema()
    assert cfg.enabled is False
    assert cfg.stage1_checkpoint is None
    assert cfg.latent_key == "latent"
    assert cfg.cache_dir is None


def test_latent_diffusion_enabled_requires_checkpoint() -> None:
    with pytest.raises(ValidationError, match="stage1_checkpoint"):
        LatentDiffusionConfigSchema(enabled=True)


def test_data_config_threads_latent_diffusion_block() -> None:
    cfg = DataConfigSchema(
        latent_diffusion={
            "enabled": True,
            "stage1_checkpoint": "/tmp/dummy.pt",
            "latent_key": "z",
        }
    )
    assert cfg.latent_diffusion.enabled is True
    assert cfg.latent_diffusion.latent_key == "z"


# ── LazyEncodeWrapper behavior ──────────────────────────────────────────────


class _ToyInnerDataset(Dataset):
    """Yields tio.Subjects with an 'input' tensor."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tio.Subject:
        t = torch.full((1, 4, 4, 2), float(idx))
        return tio.Subject(
            input=tio.ScalarImage(tensor=t),
            file_id=f"toy-{idx}",
        )


def _identity_encoder(x: torch.Tensor) -> torch.Tensor:
    """Simplest possible encoder: pass-through."""
    return x


def test_lazy_encode_rejects_disabled_config() -> None:
    cfg = LatentDiffusionConfigSchema()  # disabled
    with pytest.raises(ValueError, match="enabled=True"):
        LazyEncodeWrapper(inner=_ToyInnerDataset(), config=cfg)


def test_lazy_encode_attaches_latent_to_subject() -> None:
    cfg = LatentDiffusionConfigSchema(
        enabled=True,
        stage1_checkpoint="/dummy",   # not loaded — we pass an explicit encoder
        latent_key="latent",
    )
    wrapper = LazyEncodeWrapper(
        inner=_ToyInnerDataset(),
        config=cfg,
        encoder=_identity_encoder,
    )
    s = wrapper[2]
    assert "latent" in s
    assert isinstance(s["latent"], tio.ScalarImage)
    # Identity encoder ⇒ latent equals input (modulo CPU move).
    assert torch.allclose(s["latent"].data, s["input"].data)


def test_lazy_encode_preserves_input_alongside_latent() -> None:
    """The wrapper must NOT replace ``input`` — downstream losses still need it."""
    cfg = LatentDiffusionConfigSchema(
        enabled=True,
        stage1_checkpoint="/dummy",
        latent_key="z",
    )
    wrapper = LazyEncodeWrapper(
        inner=_ToyInnerDataset(),
        config=cfg,
        encoder=_identity_encoder,
    )
    s = wrapper[0]
    assert "input" in s
    assert "z" in s


def test_lazy_encode_disk_cache_writes_and_reuses(tmp_path: Path) -> None:
    """First access writes the latent; second access reloads from disk."""
    cache_dir = tmp_path / "latent_cache"
    cfg = LatentDiffusionConfigSchema(
        enabled=True,
        stage1_checkpoint="/dummy",
        cache_dir=str(cache_dir),
    )

    call_count = {"n": 0}

    def counting_encoder(x: torch.Tensor) -> torch.Tensor:
        call_count["n"] += 1
        return x

    wrapper = LazyEncodeWrapper(
        inner=_ToyInnerDataset(),
        config=cfg,
        encoder=counting_encoder,
    )
    # First access ⇒ one encoder call, latent persisted.
    _ = wrapper[1]
    assert call_count["n"] == 1
    cached_files = list(cache_dir.iterdir())
    assert len(cached_files) == 1
    # Second access at the SAME index ⇒ cache hit, no extra encoder call.
    _ = wrapper[1]
    assert call_count["n"] == 1


def test_lazy_encode_len_passes_through() -> None:
    cfg = LatentDiffusionConfigSchema(
        enabled=True, stage1_checkpoint="/dummy"
    )
    inner = _ToyInnerDataset(n=7)
    wrapper = LazyEncodeWrapper(
        inner=inner, config=cfg, encoder=_identity_encoder
    )
    assert len(wrapper) == 7
