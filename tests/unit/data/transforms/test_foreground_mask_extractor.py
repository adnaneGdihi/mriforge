"""Tests for ``ForegroundMaskExtractor``.

Targets ``spectramr.data.transforms.foreground_mask_extractor``. Attaches a
brain foreground mask to a TorchIO Subject — either by reading a
sidecar file or by lazy in-memory thresholding.

Categories:

- Sidecar lookup helpers: ``_strip_nifti_suffix`` + ``_find_sidecar_mask``
- Lazy thresholding: ``_threshold_mask`` returns binary tensor at the
  documented thresholding rule
- Forward: source missing → no-op; mask already present → no-op; lazy
  fallback adds a ``LabelMap``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.transforms.foreground_mask_extractor import (
    ForegroundMaskExtractor,
    _find_sidecar_mask,
    _strip_nifti_suffix,
)


# ---------------------------------------------------------------------------
# Sidecar helpers
# ---------------------------------------------------------------------------


def test_strip_nifti_suffix_dual_suffix() -> None:
    """``foo.nii.gz`` strips both ``.gz`` and ``.nii``."""
    assert _strip_nifti_suffix(Path("/tmp/foo.nii.gz")) == "foo"


def test_strip_nifti_suffix_single_suffix() -> None:
    """``foo.nii`` strips just the single suffix."""
    assert _strip_nifti_suffix(Path("/tmp/foo.nii")) == "foo"


def test_strip_nifti_suffix_no_match_uses_stem() -> None:
    """Unknown suffix → falls back to ``Path.stem``."""
    assert _strip_nifti_suffix(Path("/tmp/foo.txt")) == "foo"


def test_find_sidecar_mask_returns_none_when_missing(tmp_path: Path) -> None:
    """If no sidecar is on disk, return None."""
    volume = tmp_path / "subject.nii.gz"
    volume.touch()
    assert _find_sidecar_mask(volume) is None


def test_find_sidecar_mask_finds_brain_mask(tmp_path: Path) -> None:
    """Canonical ``_brain_mask.nii.gz`` sidecar is discovered."""
    volume = tmp_path / "subject.nii.gz"
    volume.touch()
    mask = tmp_path / "subject_brain_mask.nii.gz"
    mask.touch()
    found = _find_sidecar_mask(volume)
    assert found == mask


# ---------------------------------------------------------------------------
# Threshold mask
# ---------------------------------------------------------------------------


def test_threshold_mask_binary_output() -> None:
    """Output of ``_threshold_mask`` contains only 0 and 1."""
    data = torch.cat(
        [
            torch.zeros(1, 4, 4, 4),
            torch.ones(1, 4, 4, 4) * 10.0,
        ],
        dim=2,
    )
    mask = ForegroundMaskExtractor._threshold_mask(data, threshold=0.05)
    unique = torch.unique(mask)
    assert torch.all((unique == 0) | (unique == 1))


def test_threshold_mask_high_signal_retained() -> None:
    """Pixels above ``threshold * percentile_99`` are kept."""
    intensity = torch.zeros(1, 8, 8, 8)
    intensity[0, 4, 4, 4] = 100.0  # single hot voxel
    intensity[0, 0, 0, 0] = 0.001  # below threshold
    mask = ForegroundMaskExtractor._threshold_mask(intensity, threshold=0.05)
    assert mask.shape == (1, 8, 8, 8)
    assert mask[0, 4, 4, 4].item() == 1.0
    assert mask[0, 0, 0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# Subject-level transform
# ---------------------------------------------------------------------------


def _make_subject(tensor: torch.Tensor) -> tio.Subject:
    """Build a single-image TorchIO subject."""
    affine = np.eye(4)
    return tio.Subject(input=tio.ScalarImage(tensor=tensor, affine=affine))


def test_apply_no_source_image_is_no_op() -> None:
    """If source image is missing, subject returns unchanged."""
    subj = tio.Subject(other=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 4)))
    transform = ForegroundMaskExtractor(source_image="input")
    out = transform.apply_transform(subj)
    assert "foreground_mask" not in out


def test_apply_mask_already_present_is_no_op() -> None:
    """Existing mask key is preserved (not overwritten)."""
    img = torch.ones(1, 4, 4, 4)
    affine = np.eye(4)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=img, affine=affine),
        foreground_mask=tio.LabelMap(tensor=torch.zeros_like(img), affine=affine),
    )
    transform = ForegroundMaskExtractor(allow_lazy=True)
    out = transform.apply_transform(subj)
    # Original mask preserved.
    assert torch.all(out["foreground_mask"].data == 0.0)


def test_apply_lazy_fallback_attaches_mask() -> None:
    """With no path on the image, lazy fallback produces a binary mask."""
    img = torch.cat(
        [torch.zeros(1, 4, 4, 4), torch.ones(1, 4, 4, 4) * 5.0], dim=2
    )
    subj = _make_subject(img)
    transform = ForegroundMaskExtractor(allow_lazy=True, threshold=0.1)
    out = transform.apply_transform(subj)
    assert "foreground_mask" in out
    mask_data = out["foreground_mask"].data
    assert mask_data.shape == img.shape
    # Some voxels should be foreground (1.0) given the synthetic intensity.
    assert mask_data.max().item() == 1.0


def test_apply_lazy_disabled_no_attach() -> None:
    """``allow_lazy=False`` and no sidecar → mask not attached."""
    img = torch.ones(1, 4, 4, 4)
    subj = _make_subject(img)
    transform = ForegroundMaskExtractor(allow_lazy=False)
    out = transform.apply_transform(subj)
    assert "foreground_mask" not in out


# --- registry wiring (D9) ---------------------------------------------------


def test_foreground_mask_is_registered_under_its_config_name() -> None:
    """The transform must be reachable from ``data.processing.transforms``.

    Before the registry it existed on disk with no way to be constructed from
    any config, so the consumer chain that reads its output was dead at link 0.
    """
    from spectramr.data.transforms.registry import get_transform

    entry = get_transform("foreground_mask")
    assert entry.cls is ForegroundMaskExtractor
    assert "foreground_mask" in entry.produces
