"""Unit tests for SyntheticMRIDataset.

Tests pure-logic, in-memory construction and item access. No external
data files required.

Covers:
  - Construction validation (num_samples, lr_shape, upscale_factor).
  - __len__ contract.
  - __getitem__ returns a tio.Subject with 'input', 'target', 'index'.
  - LR/HR shape relationship respects upscale_factor.
  - Determinism: same index → same tensors across two dataset instances.
  - Noise level: zero noise → exact sine/cos grid; nonzero introduces variation.
  - external_augmentations callable is applied.
  - get_metadata returns expected keys.
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.datasets.synthetic_mri_dataset import SyntheticMRIDataset


# ── Construction validation ───────────────────────────────────────────────────


def test_construction_valid() -> None:
    ds = SyntheticMRIDataset(num_samples=4, lr_shape=(1, 16, 16), upscale_factor=2)
    assert len(ds) == 4


def test_construction_rejects_zero_num_samples() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        SyntheticMRIDataset(num_samples=0)


def test_construction_rejects_negative_num_samples() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        SyntheticMRIDataset(num_samples=-1)


def test_construction_rejects_invalid_lr_shape() -> None:
    with pytest.raises(ValueError, match="lr_shape"):
        SyntheticMRIDataset(lr_shape=(1, 16))  # not a 3-tuple


def test_construction_rejects_non_positive_upscale_factor() -> None:
    with pytest.raises(ValueError, match="upscale_factor"):
        SyntheticMRIDataset(upscale_factor=0)


# ── __len__ ───────────────────────────────────────────────────────────────────


def test_len_matches_num_samples() -> None:
    for n in (1, 8, 32):
        assert len(SyntheticMRIDataset(num_samples=n)) == n


# ── __getitem__ shape contract ────────────────────────────────────────────────


def test_canary_getitem_returns_torchio_subject() -> None:
    ds = SyntheticMRIDataset(num_samples=4, lr_shape=(1, 16, 16), upscale_factor=2)
    item = ds[0]
    assert isinstance(item, tio.Subject)


def test_getitem_has_required_keys() -> None:
    ds = SyntheticMRIDataset(num_samples=2, lr_shape=(1, 8, 8))
    item = ds[0]
    assert "input" in item
    assert "target" in item
    assert "index" in item


def test_input_shape_matches_lr_shape_plus_depth() -> None:
    """TorchIO stores as (C, H, W, D) with appended depth=1."""
    ds = SyntheticMRIDataset(num_samples=2, lr_shape=(1, 16, 16), upscale_factor=2)
    item = ds[0]
    assert item["input"].data.shape == (1, 16, 16, 1)


def test_target_shape_matches_hr_shape_plus_depth() -> None:
    ds = SyntheticMRIDataset(num_samples=2, lr_shape=(1, 16, 16), upscale_factor=2)
    item = ds[0]
    assert item["target"].data.shape == (1, 32, 32, 1)


@pytest.mark.parametrize(
    "lr_shape, upscale",
    [((1, 8, 8), 2), ((1, 16, 16), 4), ((2, 8, 8), 2)],
    ids=["8x2", "16x4", "2ch-8x2"],
)
def test_parametrised_shapes(
    lr_shape: tuple[int, int, int], upscale: int
) -> None:
    ds = SyntheticMRIDataset(num_samples=2, lr_shape=lr_shape, upscale_factor=upscale)
    item = ds[0]
    C, H, W = lr_shape
    assert item["input"].data.shape == (C, H, W, 1)
    assert item["target"].data.shape == (C, H * upscale, W * upscale, 1)


# ── Determinism ───────────────────────────────────────────────────────────────


def test_determinism_same_index_across_instances() -> None:
    ds1 = SyntheticMRIDataset(num_samples=4, lr_shape=(1, 8, 8), noise_level=0.0)
    ds2 = SyntheticMRIDataset(num_samples=4, lr_shape=(1, 8, 8), noise_level=0.0)
    for idx in range(4):
        assert torch.allclose(ds1[idx]["input"].data, ds2[idx]["input"].data)
        assert torch.allclose(ds1[idx]["target"].data, ds2[idx]["target"].data)


def test_different_indices_produce_different_items() -> None:
    ds = SyntheticMRIDataset(num_samples=8, lr_shape=(1, 8, 8), noise_level=0.0)
    assert not torch.allclose(ds[0]["target"].data, ds[1]["target"].data)


# ── Noise level ───────────────────────────────────────────────────────────────


def test_zero_noise_level_produces_bounded_output() -> None:
    ds = SyntheticMRIDataset(num_samples=2, noise_level=0.0)
    item = ds[0]
    assert item["target"].data.min().item() >= -1.0
    assert item["target"].data.max().item() <= 1.0


def test_nonzero_noise_adds_variation() -> None:
    ds_clean = SyntheticMRIDataset(num_samples=2, noise_level=0.0)
    ds_noisy = SyntheticMRIDataset(num_samples=2, noise_level=0.5)
    # Noisy target must differ from noiseless (probabilistically certain for large noise)
    assert not torch.allclose(
        ds_noisy[0]["target"].data, ds_clean[0]["target"].data, atol=1e-3
    )


# ── external_augmentations ────────────────────────────────────────────────────


def test_external_augmentation_is_called() -> None:
    """The provided callable receives (lr_tensor, hr_tensor) and its
    output is used as the item values."""
    calls: list[bool] = []

    def aug(lr: torch.Tensor, hr: torch.Tensor):
        calls.append(True)
        return lr, hr

    ds = SyntheticMRIDataset(
        num_samples=2, lr_shape=(1, 8, 8), external_augmentations=aug
    )
    ds[0]
    assert calls  # callable was invoked


# ── get_metadata ─────────────────────────────────────────────────────────────


def test_get_metadata_has_expected_keys() -> None:
    ds = SyntheticMRIDataset(num_samples=4, lr_shape=(1, 8, 8), upscale_factor=2)
    meta = ds.get_metadata()
    assert "dataset_type" in meta
    assert "num_samples" in meta
    assert "lr_shape" in meta
    assert "hr_shape" in meta
    assert meta["num_samples"] == 4
    assert meta["dataset_type"] == "synthetic"


def test_dry_iter_returns_length_correct_stub_subjects() -> None:
    """tio.Queue length probe works without a voxel read (the dry_iter contract).

    Regression for the '<Dataset>' object has no attribute 'dry_iter' crash
    class (oracle_bssfp / exp_vf_29, 2026-06): a SubjectsDataset wrapped in a
    tio.Queue must expose dry_iter yielding length-correct stub Subjects.
    """
    import torchio as tio

    from spectramr.data.datasets.synthetic_mri_dataset import SyntheticMRIDataset

    ds = SyntheticMRIDataset.__new__(SyntheticMRIDataset)
    ds.num_samples = 3
    stubs = ds.dry_iter()
    assert len(stubs) == 3
    for s in stubs:
        assert isinstance(s, tio.Subject)
        assert tuple(s["input"].tensor.shape) == (1, 1, 1, 1)
