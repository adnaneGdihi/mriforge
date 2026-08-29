"""Utilities for loading MRI slice stacks and shared dataset helpers.

These helpers centralize the repeated logic that was previously duplicated
across the TRELLIS dataset variants so that future datasets can reuse
consistent preprocessing behaviour.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_grayscale_png(path: Path) -> torch.Tensor:
    """Load a single-channel PNG image as a normalized float tensor."""

    image = Image.open(path).convert("L")
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def load_grayscale_stack(slice_paths: Sequence[Path]) -> torch.Tensor:
    """Stack a sequence of PNG slices into a (C, H, W, D) tensor."""

    slices = [load_grayscale_png(path) for path in slice_paths]
    return torch.stack(slices, dim=3)


def list_png_files(directory: Path) -> list[Path]:
    """Return all PNG files in ``directory`` sorted lexicographically."""

    return sorted(directory.glob("*.png"))


def group_slices_by_prefix(
    directory: Path,
    *,
    min_depth: int,
    prefix_parts: int = 2,
    delimiter: str = "_",
) -> list[list[Path]]:
    """Group PNG slices belonging to the same volume via filename prefix."""

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in list_png_files(directory):
        parts = path.stem.split(delimiter)
        key = delimiter.join(parts[:prefix_parts]) if parts else path.stem
        grouped[key].append(path)

    volumes = [sorted(paths) for paths in grouped.values() if len(paths) >= min_depth]
    return sorted(volumes, key=lambda paths: paths[0].name if paths else "")


def find_common_png_stems(directories: Iterable[Path]) -> list[str]:
    """Return file stems that are present across every provided directory."""

    stem_sets = []
    for directory in directories:
        stems = {path.name for path in list_png_files(directory)}
        stem_sets.append(stems)

    if not stem_sets:
        return []

    common = set.intersection(*stem_sets)
    return sorted(common)
