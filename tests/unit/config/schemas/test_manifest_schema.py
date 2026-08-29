"""Tests for the dataset manifest dataclasses.

Targets ``mriforge.config.schemas.manifest_schema``. Frozen dataclasses for
file-level metadata, dataset variants, and the in-memory manifest
index. Includes legacy CSV row support.

Categories:

- Construction: required fields, defaults via ``field(default_factory=...)``
- Frozen: assignment raises ``FrozenInstanceError``
- ``DatasetManifest`` helpers: ``get_shapes`` / ``get_contrasts`` /
  ``get_file_count_by_contrast`` aggregate correctly over files
- ``DatasetManifestIndex.get_manifest`` / ``get_all_variants`` /
  ``get_dataset_keys`` lookups
- ``DatasetManifestCSV.from_dict`` / ``to_dict`` round-trip
"""

from __future__ import annotations

import dataclasses

import pytest

from mriforge.config.schemas.manifest_schema import (
    DatasetManifest,
    DatasetManifestCSV,
    DatasetManifestIndex,
    FileMetadata,
)


# ---------------------------------------------------------------------------
# FileMetadata
# ---------------------------------------------------------------------------


def test_file_metadata_required_field() -> None:
    """``file_path`` is the only required positional field."""
    fm = FileMetadata(file_path="/tmp/scan.nii")
    assert fm.file_path == "/tmp/scan.nii"
    assert fm.shape is None
    assert fm.preprocessing_status == "original"
    assert fm.metadata == {}


def test_file_metadata_frozen() -> None:
    """Assignment raises ``FrozenInstanceError``."""
    fm = FileMetadata(file_path="/tmp/scan.nii")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fm.file_path = "/tmp/other.nii"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DatasetManifest helpers
# ---------------------------------------------------------------------------


def _make_manifest() -> DatasetManifest:
    """Helper: a manifest with mixed contrasts + shapes."""
    files = [
        FileMetadata(file_path="/a", shape=(1, 256, 256), contrast="T1w"),
        FileMetadata(file_path="/b", shape=(1, 256, 256), contrast="T1w"),
        FileMetadata(file_path="/c", shape=(2, 128, 128), contrast="T2w"),
        FileMetadata(file_path="/d", shape=None),  # no shape, no contrast
    ]
    return DatasetManifest(
        dataset_key="test",
        variant="original",
        display_name="Test Set",
        dataset_type="2d",
        files=files,
    )


def test_get_shapes_dedupes_and_sorts() -> None:
    """``get_shapes`` returns a sorted, de-duplicated list."""
    manifest = _make_manifest()
    shapes = manifest.get_shapes()
    assert shapes == [(1, 256, 256), (2, 128, 128)] or shapes == [
        (1, 256, 256),
        (2, 128, 128),
    ] or shapes == sorted(shapes)
    assert (1, 256, 256) in shapes
    assert (2, 128, 128) in shapes
    # No duplicates: T1w files share a shape but it's listed once.
    assert len(shapes) == 2


def test_get_contrasts() -> None:
    """``get_contrasts`` returns the unique non-None set."""
    manifest = _make_manifest()
    contrasts = manifest.get_contrasts()
    assert contrasts == {"T1w", "T2w"}


def test_file_count_by_contrast() -> None:
    """``get_file_count_by_contrast`` tallies per-contrast counts."""
    manifest = _make_manifest()
    counts = manifest.get_file_count_by_contrast()
    assert counts == {"T1w": 2, "T2w": 1}


def test_dataset_manifest_frozen() -> None:
    """Manifest is frozen — mutation raises."""
    manifest = _make_manifest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.dataset_key = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DatasetManifestIndex
# ---------------------------------------------------------------------------


def test_index_get_manifest() -> None:
    """``get_manifest`` returns the registered manifest."""
    manifest = _make_manifest()
    index = DatasetManifestIndex(manifests={("test", "original"): manifest})
    found = index.get_manifest("test", "original")
    assert found is manifest


def test_index_get_manifest_missing_returns_none() -> None:
    """Unknown key → ``None``."""
    index = DatasetManifestIndex()
    assert index.get_manifest("missing") is None


def test_index_get_all_variants() -> None:
    """All variants for a key are returned."""
    m1 = _make_manifest()
    m2 = DatasetManifest(
        dataset_key="test",
        variant="kspace",
        display_name="Test K",
        dataset_type="kspace",
    )
    index = DatasetManifestIndex(
        manifests={("test", "original"): m1, ("test", "kspace"): m2}
    )
    variants = index.get_all_variants("test")
    assert len(variants) == 2


def test_index_get_dataset_keys() -> None:
    """``get_dataset_keys`` returns the unique key set."""
    m1 = _make_manifest()
    m2 = DatasetManifest(
        dataset_key="other",
        variant="original",
        display_name="Other",
        dataset_type="2d",
    )
    index = DatasetManifestIndex(
        manifests={("test", "original"): m1, ("other", "original"): m2}
    )
    assert index.get_dataset_keys() == {"test", "other"}


# ---------------------------------------------------------------------------
# DatasetManifestCSV
# ---------------------------------------------------------------------------


def test_csv_from_to_dict_round_trip() -> None:
    """CSV row round-trips through ``from_dict`` → ``to_dict``."""
    row = {
        "dataset_key": "fastmri",
        "display_name": "FastMRI",
        "dataset_type": "kspace",
        "description": "knee",
        "total_files": "100",
        "total_size_mb": "500",
        "scanned_at": "1700000000",
        "dataset_dir": "/data/fastmri",
        "variants_available": "original;normalized",
    }
    cm = DatasetManifestCSV.from_dict(row)
    assert cm.dataset_key == "fastmri"
    assert cm.total_files == 100
    assert cm.total_size_mb == 500
    assert cm.variants_available == ["original", "normalized"]

    out = cm.to_dict()
    assert out["dataset_key"] == "fastmri"
    assert out["variants_available"] == "original;normalized"


def test_csv_from_dict_handles_missing_optional() -> None:
    """Missing optional CSV columns degrade to default values."""
    row = {
        "dataset_key": "test",
        "display_name": "Test",
        "dataset_type": "2d",
        "dataset_dir": "/data",
    }
    cm = DatasetManifestCSV.from_dict(row)
    assert cm.total_files == 0
    assert cm.variants_available == []
    assert cm.description == ""
