"""Unit tests for data/datasets/utils/image_volume_utils.py.

Covers: list_png_files, group_slices_by_prefix, find_common_png_stems.
PNG loading helpers (load_grayscale_png, load_grayscale_stack) require
real files on disk — they are covered via the edge tests with tmp files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from spectramr.data.datasets.utils.image_volume_utils import (
    find_common_png_stems,
    group_slices_by_prefix,
    list_png_files,
    load_grayscale_png,
    load_grayscale_stack,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pngs(directory: Path, names: list[str], size: tuple[int, int] = (8, 8)) -> list[Path]:
    """Write grayscale PNG files to a directory and return their paths."""
    paths = []
    rng = np.random.default_rng(0)
    for name in names:
        img = Image.fromarray((rng.integers(0, 256, size, dtype=np.uint8)))
        p = directory / name
        img.save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_png_files_canary():
    """list_png_files returns all and only PNG files, sorted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        _write_pngs(d, ["b.png", "a.png", "c.png"])
        (d / "readme.txt").write_text("nope")
        files = list_png_files(d)
    stems = [f.stem for f in files]
    assert stems == sorted(stems)
    assert len(files) == 3
    assert all(f.suffix == ".png" for f in files)


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "names,prefix_parts,min_depth,expected_groups",
    [
        (["vol01_sl00.png", "vol01_sl01.png", "vol02_sl00.png", "vol02_sl01.png"], 2, 2, 2),
        (["vol01_sl00.png"], 2, 2, 0),  # only 1 slice → filtered out
        (["a_b_c.png", "a_b_d.png", "x_y_z.png"], 2, 1, 2),
    ],
)
def test_group_slices_by_prefix(names, prefix_parts, min_depth, expected_groups):
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        _write_pngs(d, names)
        groups = group_slices_by_prefix(d, min_depth=min_depth, prefix_parts=prefix_parts)
    assert len(groups) == expected_groups


@pytest.mark.unit
@pytest.mark.parametrize("n_common", [0, 2, 4])
def test_find_common_png_stems(n_common):
    """find_common_png_stems returns exactly the intersection of stems."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d1 = Path(tmpdir) / "d1"
        d2 = Path(tmpdir) / "d2"
        d1.mkdir(); d2.mkdir()
        common_names = [f"common_{i}.png" for i in range(n_common)]
        _write_pngs(d1, common_names + ["only_d1.png"])
        _write_pngs(d2, common_names + ["only_d2.png"])
        common = find_common_png_stems([d1, d2])
    assert len(common) == n_common


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_grayscale_png_shape_and_range():
    """load_grayscale_png produces a (1, H, W) float32 tensor in [0, 1]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.png"
        _write_pngs(Path(tmpdir), ["test.png"])
        t = load_grayscale_png(p)
    assert t.dtype == torch.float32
    assert t.ndim == 3
    assert t.shape[0] == 1
    assert 0.0 <= float(t.min()) <= float(t.max()) <= 1.0


@pytest.mark.unit
def test_load_grayscale_stack_depth_dimension():
    """Stack of D slices → tensor with last dim == D."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        paths = _write_pngs(d, [f"s{i:02d}.png" for i in range(5)], size=(8, 8))
        stack = load_grayscale_stack(paths)
    assert stack.shape[-1] == 5


@pytest.mark.unit
def test_find_common_stems_empty_directories():
    """Empty iterable yields empty list."""
    result = find_common_png_stems([])
    assert result == []


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_grayscale_png_missing_file_raises():
    """Loading a nonexistent file must raise (FileNotFoundError or similar)."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_grayscale_png(Path("/nonexistent/path/does_not_exist.png"))
