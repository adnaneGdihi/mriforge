"""Tests for the pluggable segmentation backends + Dice (B-1.7)."""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.quantitative.segmentation import (
    LabelDiceBackend,
    SynthSegBackend,
    dice_score,
    get_segmenter,
)


def test_label_backend_produces_multiregion_labels() -> None:
    seg = LabelDiceBackend(n_classes=5)
    labels = seg.segment(torch.rand(2, 1, 16, 16))
    assert labels.shape == (2, 16, 16)
    assert labels.dtype == torch.long
    assert labels.max().item() <= 4
    assert labels.unique().numel() >= 3


def test_no_collapse_on_mostly_background_brain() -> None:
    # Realistic regime (review finding): a large zero background must NOT collapse the
    # foreground classes — foreground-only quantiles keep all classes reachable.
    torch.manual_seed(0)
    img = torch.zeros(1, 1, 64, 64)
    flat = img.reshape(-1)
    idx = torch.randperm(flat.numel())[: int(0.30 * flat.numel())]  # 70% background
    flat[idx] = torch.rand(idx.numel()) * 0.9 + 0.1
    labels = LabelDiceBackend(n_classes=5).segment(img)
    assert labels.unique().numel() == 5  # 0 (bg) + 4 foreground classes, no collapse


def test_disjoint_hallucination_scores_low_dice() -> None:
    # A spatially-disjoint (wrong-location) structure must score low Dice — the
    # proxy must SURFACE hallucination, not mask it (present-class averaging).
    seg = LabelDiceBackend(n_classes=5)
    img = torch.zeros(1, 1, 64, 64)
    img[..., 8:24, 8:24] = torch.rand(16, 16) * 0.9 + 0.1
    hall = torch.zeros(1, 1, 64, 64)
    hall[..., 40:56, 40:56] = torch.rand(16, 16) * 0.9 + 0.1  # disjoint location
    assert float(dice_score(seg.segment(img), seg.segment(hall), 5)) < 0.3


def test_blank_prediction_not_free_dice() -> None:
    # A degenerate blank/constant prediction must NOT score Dice 1 (no free absent-
    # in-both credit).
    seg = LabelDiceBackend(n_classes=5)
    img = torch.zeros(1, 1, 32, 32)
    img[..., 8:24, 8:24] = torch.rand(16, 16) * 0.9 + 0.1
    blank = torch.zeros(1, 1, 32, 32)
    assert float(dice_score(seg.segment(img), seg.segment(blank), 5)) < 0.1


def test_dice_perfect_and_imperfect() -> None:
    seg = LabelDiceBackend(n_classes=5)
    x = torch.rand(2, 1, 16, 16)
    sx = seg.segment(x)
    assert torch.allclose(dice_score(sx, sx, 5), torch.ones(2), atol=1e-6)
    sy = seg.segment(torch.rand(2, 1, 16, 16))
    assert float(dice_score(sx, sy, 5).mean()) < 1.0


def test_get_segmenter_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_segmenter("not_a_backend")


def test_synthseg_requires_model() -> None:
    with pytest.raises(RuntimeError):
        SynthSegBackend(n_classes=14)


def test_complex_input_raises() -> None:
    seg = LabelDiceBackend(n_classes=4)
    with pytest.raises(ValueError):
        seg.segment(torch.rand(1, 1, 8, 8, dtype=torch.cfloat))
