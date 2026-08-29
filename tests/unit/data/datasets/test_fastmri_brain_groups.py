"""Tests for the fastMRI-brain contrast-aware grid discovery (issue #308).

The grid must be balanced and interleaved subject-major / contrast-minor so that
``group_idx = s * n_contrasts + c`` maps ``c`` to a real contrast, and the
contrast label must come from ``attrs['acquisition']`` (preferred) or the ``AX*``
filename token — never a silently mislabelled full filename.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from mriforge.data.datasets.fastmri_brain_groups import (
    BrainContrastGrid,
    brain_contrast_of,
    discover_brain_contrast_groups,
)


def _make_h5(path: Path, acquisition: str | None = None) -> Path:
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=np.zeros((2, 2, 4, 4), dtype=np.complex64))
        if acquisition is not None:
            f.attrs["acquisition"] = acquisition
    return path


def _write_cohort(root: Path, counts: dict[str, int], with_attrs: bool = True) -> None:
    for contrast, n in counts.items():
        for i in range(n):
            name = f"file_brain_{contrast}_200_{contrast}{i:04d}.h5"
            _make_h5(root / name, acquisition=contrast if with_attrs else None)


class TestBrainContrastOf:
    def test_attrs_acquisition_is_preferred(self, tmp_path: Path) -> None:
        # Filename says AXT1 but the attr says AXT2 -> attr wins.
        p = _make_h5(tmp_path / "file_brain_AXT1_200_2000001.h5", acquisition="AXT2")
        assert brain_contrast_of(p) == "AXT2"

    def test_filename_token_when_no_attr(self, tmp_path: Path) -> None:
        p = _make_h5(tmp_path / "file_brain_AXT2FLAIR_201_6002670.h5", acquisition=None)
        assert brain_contrast_of(p) == "AXT2FLAIR"

    def test_bytes_attr_is_decoded(self, tmp_path: Path) -> None:
        p = tmp_path / "vol.h5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("kspace", data=np.zeros((2, 2, 4, 4), dtype=np.complex64))
            f.attrs["acquisition"] = np.bytes_(b"AXFLAIR")
        assert brain_contrast_of(p) == "AXFLAIR"

    def test_unrecognisable_contrast_raises(self, tmp_path: Path) -> None:
        p = _make_h5(tmp_path / "mystery_volume.h5", acquisition=None)
        with pytest.raises(ValueError, match="Cannot determine a contrast"):
            brain_contrast_of(p)


class TestDiscoverBrainContrastGroups:
    def test_grid_is_balanced_and_interleaved(self, tmp_path: Path) -> None:
        _write_cohort(tmp_path, {"AXT1": 2, "AXT2": 2, "AXT2FLAIR": 2})
        grid = discover_brain_contrast_groups(tmp_path, max_subjects=5, max_contrasts=3)
        assert isinstance(grid, BrainContrastGrid)
        assert grid.n_subjects == 2
        assert grid.n_contrasts == 3
        assert len(grid.groups) == 6
        # Each group is a singleton reference.
        assert all(len(g) == 1 for g in grid.groups)
        # Interleaved: group c of every subject-row is the SAME contrast.
        for s in range(grid.n_subjects):
            for c in range(grid.n_contrasts):
                idx = s * grid.n_contrasts + c
                assert brain_contrast_of(grid.groups[idx][0]) == grid.contrasts[c]

    def test_imbalanced_buckets_are_capped_to_the_min(
        self, tmp_path: Path, caplog
    ) -> None:
        _write_cohort(tmp_path, {"AXT1": 5, "AXT2": 3, "AXT2FLAIR": 2})
        with caplog.at_level("INFO"):
            grid = discover_brain_contrast_groups(
                tmp_path, max_subjects=10, max_contrasts=3
            )
        assert grid.n_subjects == 2  # min bucket
        assert len(grid.groups) == 6
        # The drop must be logged, not silent.
        assert any("dropped" in r.getMessage() for r in caplog.records)

    def test_max_subjects_caps_per_contrast(self, tmp_path: Path) -> None:
        _write_cohort(tmp_path, {"AXT1": 5, "AXT2": 5})
        grid = discover_brain_contrast_groups(tmp_path, max_subjects=3, max_contrasts=2)
        assert grid.n_subjects == 3
        assert grid.n_contrasts == 2
        assert len(grid.groups) == 6

    def test_max_contrasts_selects_most_populated(self, tmp_path: Path) -> None:
        _write_cohort(tmp_path, {"AXT1": 4, "AXT2": 3, "AXT2FLAIR": 2, "AXFLAIR": 1})
        grid = discover_brain_contrast_groups(
            tmp_path, max_subjects=10, max_contrasts=2
        )
        assert grid.n_contrasts == 2
        assert set(grid.contrasts) == {"AXT1", "AXT2"}  # top two by count

    def test_explicit_contrasts_are_honoured(self, tmp_path: Path) -> None:
        _write_cohort(tmp_path, {"AXT1": 3, "AXT2": 3, "AXT2FLAIR": 3})
        grid = discover_brain_contrast_groups(
            tmp_path, max_subjects=10, max_contrasts=3, contrasts=["AXT2FLAIR", "AXT1"]
        )
        assert grid.contrasts == ["AXT2FLAIR", "AXT1"]
        assert grid.n_contrasts == 2

    def test_requested_absent_contrast_raises(self, tmp_path: Path) -> None:
        _write_cohort(tmp_path, {"AXT1": 2, "AXT2": 2})
        with pytest.raises(ValueError, match="not found"):
            discover_brain_contrast_groups(tmp_path, contrasts=["AXT1", "AXDWI"])

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No H5 files"):
            discover_brain_contrast_groups(tmp_path)
