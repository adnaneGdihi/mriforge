"""Tests for ``ManifestService``.

Targets ``mriforge.infrastructure.services.manifest_service``. JSON-backed
manifest creation, save/load, listing, and structural validation for
preprocessing-pipeline artifact tracking.

Categories:

- ``create_manifest`` populates required keys
- ``save_manifest`` writes valid JSON
- ``load_manifest`` round-trips a written manifest
- ``load_manifest`` missing file → ``FileNotFoundError``
- ``list_manifests`` returns empty list / all jsons / dataset-filtered
- ``validate_manifest``: missing key / wrong type / bad timestamp → False;
  full valid → True
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mriforge.infrastructure.services.manifest_service import ManifestService


# ---------------------------------------------------------------------------
# create_manifest
# ---------------------------------------------------------------------------


def test_create_manifest_populates_required_keys(tmp_path: Path) -> None:
    """A freshly-created manifest contains all required keys."""
    service = ManifestService(manifests_dir=str(tmp_path))
    manifest = service.create_manifest(
        dataset_name="test_set",
        processing_tasks=["bias_correction", "registration"],
        artifacts={"output": "/tmp/out.nii"},
    )
    for key in ["dataset_name", "processing_tasks", "artifacts", "timestamp"]:
        assert key in manifest


def test_create_manifest_preserves_optional_metadata(tmp_path: Path) -> None:
    """Optional metadata is forwarded into the manifest."""
    service = ManifestService(manifests_dir=str(tmp_path))
    manifest = service.create_manifest(
        dataset_name="set",
        processing_tasks=[],
        artifacts={},
        metadata={"author": "tester"},
    )
    assert manifest["metadata"] == {"author": "tester"}


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Manifest survives JSON write+read intact."""
    service = ManifestService(manifests_dir=str(tmp_path))
    manifest = service.create_manifest(
        dataset_name="set",
        processing_tasks=["a"],
        artifacts={"x": 1},
    )
    out_path = tmp_path / "manifest.json"
    service.save_manifest(manifest, str(out_path))
    loaded = service.load_manifest(str(out_path))
    assert loaded["dataset_name"] == "set"
    assert loaded["processing_tasks"] == ["a"]


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent file → ``FileNotFoundError``."""
    service = ManifestService(manifests_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        service.load_manifest(str(tmp_path / "missing.json"))


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    """Save creates intermediate directories as needed."""
    service = ManifestService(manifests_dir=str(tmp_path))
    deep = tmp_path / "a" / "b" / "c" / "manifest.json"
    service.save_manifest({"k": "v"}, str(deep))
    assert deep.exists()


# ---------------------------------------------------------------------------
# list_manifests
# ---------------------------------------------------------------------------


def test_list_empty_dir_returns_empty(tmp_path: Path) -> None:
    """Empty directory → empty manifest list."""
    service = ManifestService(manifests_dir=str(tmp_path))
    assert service.list_manifests() == []


def test_list_returns_all_jsons(tmp_path: Path) -> None:
    """Two saved manifests both appear in ``list_manifests``."""
    service = ManifestService(manifests_dir=str(tmp_path))
    m1 = service.create_manifest(dataset_name="A", processing_tasks=[], artifacts={})
    m2 = service.create_manifest(dataset_name="B", processing_tasks=[], artifacts={})
    service.save_manifest(m1, str(tmp_path / "a.json"))
    service.save_manifest(m2, str(tmp_path / "b.json"))
    listed = service.list_manifests()
    assert len(listed) == 2


def test_list_filters_by_dataset_name(tmp_path: Path) -> None:
    """Filter by ``dataset_name`` yields only matching manifests."""
    service = ManifestService(manifests_dir=str(tmp_path))
    m1 = service.create_manifest(dataset_name="A", processing_tasks=[], artifacts={})
    m2 = service.create_manifest(dataset_name="B", processing_tasks=[], artifacts={})
    service.save_manifest(m1, str(tmp_path / "a.json"))
    service.save_manifest(m2, str(tmp_path / "b.json"))
    listed = service.list_manifests(dataset_name="A")
    assert len(listed) == 1
    assert "a.json" in listed[0]


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------


def test_validate_full_manifest(tmp_path: Path) -> None:
    """A complete, well-typed manifest validates True."""
    service = ManifestService(manifests_dir=str(tmp_path))
    manifest = service.create_manifest(
        dataset_name="set", processing_tasks=["a"], artifacts={}
    )
    assert service.validate_manifest(manifest) is True


def test_validate_missing_key_fails(tmp_path: Path) -> None:
    """Manifest lacking ``timestamp`` fails validation."""
    service = ManifestService(manifests_dir=str(tmp_path))
    manifest = {"dataset_name": "set", "processing_tasks": [], "artifacts": {}}
    assert service.validate_manifest(manifest) is False


def test_validate_wrong_type_fails(tmp_path: Path) -> None:
    """Wrong type for ``processing_tasks`` fails validation."""
    service = ManifestService(manifests_dir=str(tmp_path))
    manifest = {
        "dataset_name": "set",
        "processing_tasks": "not-a-list",  # invalid
        "artifacts": {},
        "timestamp": datetime.now().isoformat(),
    }
    assert service.validate_manifest(manifest) is False


def test_validate_bad_timestamp_fails(tmp_path: Path) -> None:
    """Non-ISO timestamp fails validation."""
    service = ManifestService(manifests_dir=str(tmp_path))
    manifest = {
        "dataset_name": "set",
        "processing_tasks": [],
        "artifacts": {},
        "timestamp": "not-a-date",
    }
    assert service.validate_manifest(manifest) is False
