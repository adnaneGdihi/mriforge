"""Unit tests for src/data/transforms/acquisition_params_injector.py."""

from __future__ import annotations

import pytest
import torch

from mriforge.data.transforms.acquisition_params_injector import (
    attach_acquisition_params,
    infer_acquisition_params,
)


def test_existing_acquisition_params_passthrough() -> None:
    """If batch already has a [B, 5] tensor, it is returned unchanged."""
    batch = {
        "image": torch.randn(3, 1, 4, 4),
        "acquisition_params": torch.tensor(
            [[8.0, 4.0, 0.0, 0.2618, 3.0]] * 3
        ),
    }
    out = infer_acquisition_params(batch)
    assert out.shape == (3, 5)
    assert torch.allclose(out, batch["acquisition_params"])


def test_contrast_list_drives_per_sample_lookup() -> None:
    batch = {
        "image": torch.randn(2, 1, 8, 8),
        "contrast": ["T1w", "T2w"],
    }
    out = infer_acquisition_params(batch)
    assert out.shape == (2, 5)
    # T1w and T2w differ on TR and TE — distinct rows.
    assert not torch.allclose(out[0], out[1])


def test_fallback_uses_default_contrast() -> None:
    batch = {"image": torch.randn(4, 1, 4, 4)}
    out = infer_acquisition_params(batch)
    assert out.shape == (4, 5)
    # All rows identical under fallback.
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[0], out[3])


def test_no_image_key_falls_back_to_batch_1() -> None:
    batch: dict[str, object] = {}
    out = infer_acquisition_params(batch)
    assert out.shape == (1, 5)


def test_attach_acquisition_params_mutates_in_place() -> None:
    batch = {"image": torch.randn(2, 1, 4, 4)}
    returned = attach_acquisition_params(batch)
    assert returned is batch
    assert "acquisition_params" in batch
    assert batch["acquisition_params"].shape == (2, 5)


def test_data_config_schema_accepts_expose_flag() -> None:
    """The Pydantic schema must accept the new field."""
    from mriforge.config.schemas.data import DataConfigSchema

    cfg = DataConfigSchema(expose={"acquisition_params": True})
    assert cfg.expose.acquisition_params is True
    cfg2 = DataConfigSchema()  # default
    assert cfg2.expose.acquisition_params is False
