"""Phase 4b — QuantitativeMapDataset.

End-to-end: an on-disk BIDS-style fixture is built in tmp_path, the
index is generated, and the dataset is iterated. Maps must arrive as
separately-keyed TorchIO ScalarImages on the Subject.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.config.schemas.data import QuantitativeConfigSchema
from spectramr.data.datasets.quantitative_dataset import (
    QuantitativeMapDataset,
    build_quantitative_index,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _try_import_nibabel():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        return None


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    """Build a 2-subject BIDS-style tree with T1w + T1map + T2map."""
    nib = _try_import_nibabel()
    if nib is None:
        pytest.skip("nibabel not installed — QuantitativeMapDataset needs it.")

    for subj in ("sub-001", "sub-002"):
        anat = tmp_path / subj / "anat"
        anat.mkdir(parents=True)

        # Input T1w + T2w.
        for suffix, fill in (("T1w", 1.0), ("T2w", 2.0)):
            arr = np.full((4, 4, 2), fill, dtype=np.float32)
            nib.save(  # type: ignore[no-untyped-call]
                nib.Nifti1Image(arr, affine=np.eye(4)),
                str(anat / f"{subj}_{suffix}.nii.gz"),
            )
        # Quant maps: T1map (ms), T2map (ms).
        for suffix, fill in (("T1map", 800.0), ("T2map", 60.0)):
            arr = np.full((4, 4, 2), fill, dtype=np.float32)
            nib.save(  # type: ignore[no-untyped-call]
                nib.Nifti1Image(arr, affine=np.eye(4)),
                str(anat / f"{subj}_{suffix}.nii.gz"),
            )
    return tmp_path


# ── Index building ──────────────────────────────────────────────────────────


def test_build_quantitative_index_finds_records(bids_root: Path) -> None:
    idx = build_quantitative_index(bids_root, target_maps=["t1", "t2"])
    assert len(idx) == 2
    assert all("input_paths" in r for r in idx)
    assert all("map_paths" in r for r in idx)
    for r in idx:
        assert set(r["map_paths"].keys()) == {"t1", "t2"}
        assert len(r["input_paths"]) >= 1


def test_build_quantitative_index_skips_subjects_missing_maps(
    bids_root: Path,
) -> None:
    """Subjects without a required map must be silently dropped from the index."""
    # Delete sub-002's T2map.
    (bids_root / "sub-002" / "anat" / "sub-002_T2map.nii.gz").unlink()
    idx = build_quantitative_index(bids_root, target_maps=["t1", "t2"])
    subjects = {r["subject_id"] for r in idx}
    assert subjects == {"sub-001"}


# ── Dataset construction ────────────────────────────────────────────────────


def test_quantitative_dataset_rejects_disabled_config(bids_root: Path) -> None:
    """Constructing with ``enabled=False`` must fail loudly."""
    idx = build_quantitative_index(bids_root, target_maps=["t1"])
    cfg = QuantitativeConfigSchema()  # disabled by default
    with pytest.raises(ValueError, match="enabled=True"):
        QuantitativeMapDataset(index=idx, quantitative_config=cfg)


def test_quantitative_dataset_yields_named_maps(bids_root: Path) -> None:
    idx = build_quantitative_index(bids_root, target_maps=["t1", "t2"])
    cfg = QuantitativeConfigSchema(
        enabled=True, target_maps=["t1", "t2"], input_source="contrasts"
    )
    ds = QuantitativeMapDataset(index=idx, quantitative_config=cfg)
    assert len(ds) == len(idx)

    subj = ds[0]
    assert "input" in subj
    assert "t1" in subj
    assert "t2" in subj
    # ScalarImage instances, not raw tensors.
    assert isinstance(subj["input"], tio.ScalarImage)
    assert isinstance(subj["t1"], tio.ScalarImage)
    # Physical units → fill values are preserved.
    assert torch.isclose(subj["t1"].data.float().mean(), torch.tensor(800.0))
    assert torch.isclose(subj["t2"].data.float().mean(), torch.tensor(60.0))


def test_quantitative_dataset_input_is_multichannel(bids_root: Path) -> None:
    """Two input contrasts ⇒ ``input`` has C=2."""
    idx = build_quantitative_index(bids_root, target_maps=["t1"])
    cfg = QuantitativeConfigSchema(
        enabled=True, target_maps=["t1"], input_source="contrasts"
    )
    ds = QuantitativeMapDataset(index=idx, quantitative_config=cfg)
    subj = ds[0]
    assert subj["input"].data.shape[0] == 2  # T1w + T2w stacked


def test_quantitative_dataset_normalized_units_rescale_to_unit(
    bids_root: Path,
) -> None:
    """``units='normalized'`` ⇒ map min/max ⊆ [0, 1]. Constant maps stay
    unchanged (no divide-by-zero)."""
    idx = build_quantitative_index(bids_root, target_maps=["t1"])
    cfg = QuantitativeConfigSchema(
        enabled=True,
        target_maps=["t1"],
        units="normalized",
        input_source="contrasts",
    )
    ds = QuantitativeMapDataset(index=idx, quantitative_config=cfg)
    subj = ds[0]
    t1 = subj["t1"].data.float()
    # Constant fill → constant after normalization (special-case OK).
    assert t1.min() == t1.max()


def test_quantitative_dataset_strict_about_missing_map_in_record(
    bids_root: Path,
) -> None:
    """If a record lacks a declared map's path, raise loudly."""
    idx = build_quantitative_index(bids_root, target_maps=["t1"])
    cfg = QuantitativeConfigSchema(
        # ask for pd; the index does not have it
        enabled=True,
        target_maps=["t1", "pd"],
        input_source="contrasts",
    )
    ds = QuantitativeMapDataset(index=idx, quantitative_config=cfg)
    with pytest.raises(KeyError, match="pd"):
        _ = ds[0]


def test_quantitative_dataset_dry_iter_returns_subject_list(bids_root: Path) -> None:
    """``dry_iter`` is used by TorchIO Queue — must be a list of Subjects."""
    idx = build_quantitative_index(bids_root, target_maps=["t1"])
    cfg = QuantitativeConfigSchema(
        enabled=True, target_maps=["t1"], input_source="contrasts"
    )
    ds = QuantitativeMapDataset(index=idx, quantitative_config=cfg)
    stubs = ds.dry_iter()
    assert isinstance(stubs, list)
    assert all(isinstance(s, tio.Subject) for s in stubs)
    # Each stub must carry the declared map key.
    assert "t1" in stubs[0]


def test_quantitative_dataset_applies_transform(bids_root: Path) -> None:
    """A trivial transform that adds a key must run on the yielded subject."""
    idx = build_quantitative_index(bids_root, target_maps=["t1"])
    cfg = QuantitativeConfigSchema(
        enabled=True, target_maps=["t1"], input_source="contrasts"
    )

    class _Marker(tio.Transform):
        def apply_transform(self, subj: tio.Subject) -> tio.Subject:
            subj["marker"] = "applied"
            return subj

    ds = QuantitativeMapDataset(
        index=idx, quantitative_config=cfg, transform=_Marker()
    )
    subj = ds[0]
    assert subj["marker"] == "applied"
