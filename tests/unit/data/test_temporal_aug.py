"""Phase 4c — Temporal augmentations.

Pins the contract of phase-shift, temporal-flip, and frame-dropout
transforms — input/target alignment, parameter handling, and edge cases.
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.temporal_aug import (
    FrameDropoutTransform,
    PhaseShiftTransform,
    TemporalFlipTransform,
)


def _frame_indexed_subject(n_frames: int = 8) -> tio.Subject:
    """Subject where each frame is the constant tensor `i` — easy to track shifts."""
    base = torch.arange(n_frames, dtype=torch.float32).view(n_frames, 1, 1, 1)
    data = base.expand(n_frames, 3, 3, 1).clone()
    return tio.Subject(
        input=tio.ScalarImage(tensor=data),
        target=tio.ScalarImage(tensor=data.clone()),
    )


# ── PhaseShiftTransform ─────────────────────────────────────────────────────


def test_phase_shift_max_zero_is_identity() -> None:
    subj = _frame_indexed_subject(n_frames=6)
    transform = PhaseShiftTransform(max_shift=0)
    out = transform(subj)
    assert torch.equal(out["input"].data, subj["input"].data)


def test_phase_shift_keeps_input_target_aligned() -> None:
    """Same shift applied to both keys."""
    torch.manual_seed(42)
    subj = _frame_indexed_subject(n_frames=8)
    transform = PhaseShiftTransform(max_shift=3, include_zero=False)
    out = transform(subj)
    assert torch.equal(out["input"].data, out["target"].data)


def test_phase_shift_records_applied_shift_on_subject() -> None:
    torch.manual_seed(0)
    subj = _frame_indexed_subject(n_frames=6)
    out = PhaseShiftTransform(max_shift=3)(subj)
    assert "phase_shift_applied" in out
    assert -3 <= out["phase_shift_applied"] <= 3


def test_phase_shift_rejects_negative_max() -> None:
    with pytest.raises(ValueError, match="max_shift"):
        PhaseShiftTransform(max_shift=-1)


def test_phase_shift_preserves_total_frames() -> None:
    """Cyclic roll keeps n_frames unchanged."""
    subj = _frame_indexed_subject(n_frames=5)
    out = PhaseShiftTransform(max_shift=2)(subj)
    assert out["input"].data.shape[0] == 5


# ── TemporalFlipTransform ───────────────────────────────────────────────────


def test_temporal_flip_with_p1_always_flips() -> None:
    subj = _frame_indexed_subject(n_frames=4)
    out = TemporalFlipTransform(p=1.0)(subj)
    # First frame was 0; after flip should be 3.
    assert out["input"].data[0, 0, 0, 0] == 3.0
    assert out["input"].data[-1, 0, 0, 0] == 0.0
    assert out.get("temporal_flip_applied") is True


def test_temporal_flip_with_p0_never_flips() -> None:
    subj = _frame_indexed_subject(n_frames=4)
    out = TemporalFlipTransform(p=0.0)(subj)
    assert torch.equal(out["input"].data, subj["input"].data)


def test_temporal_flip_keeps_input_target_aligned() -> None:
    subj = _frame_indexed_subject(n_frames=5)
    out = TemporalFlipTransform(p=1.0)(subj)
    assert torch.equal(out["input"].data, out["target"].data)


def test_temporal_flip_rejects_out_of_range_p() -> None:
    with pytest.raises(ValueError, match="p"):
        TemporalFlipTransform(p=1.5)


# ── FrameDropoutTransform ───────────────────────────────────────────────────


def test_frame_dropout_zero_rate_is_identity() -> None:
    subj = _frame_indexed_subject(n_frames=6)
    out = FrameDropoutTransform(dropout_rate=0.0)(subj)
    assert torch.equal(out["input"].data, subj["input"].data)


def test_frame_dropout_with_high_rate_keeps_at_least_one_frame() -> None:
    """Guarantees the never-all-dropped invariant."""
    torch.manual_seed(7)
    subj = _frame_indexed_subject(n_frames=10)
    out = FrameDropoutTransform(dropout_rate=0.99)(subj)
    mask = out["frame_dropout_mask"]
    assert any(mask)  # at least one True


def test_frame_dropout_only_input_preserves_target() -> None:
    """only_input=True ⇒ target untouched."""
    torch.manual_seed(7)
    subj = _frame_indexed_subject(n_frames=8)
    out = FrameDropoutTransform(dropout_rate=0.5, only_input=True)(subj)
    assert torch.equal(out["target"].data, subj["target"].data)


def test_frame_dropout_only_input_false_drops_both() -> None:
    """only_input=False ⇒ both input and target zero the same frames."""
    torch.manual_seed(7)
    subj = _frame_indexed_subject(n_frames=8)
    out = FrameDropoutTransform(
        dropout_rate=0.5, only_input=False, image_keys=("input", "target")
    )(subj)
    # Frames where input is zeroed must also be zeroed in target.
    in_zero_mask = (out["input"].data.sum(dim=(1, 2, 3)) == 0)
    tg_zero_mask = (out["target"].data.sum(dim=(1, 2, 3)) == 0)
    assert torch.equal(in_zero_mask, tg_zero_mask)


def test_frame_dropout_records_mask_on_subject() -> None:
    torch.manual_seed(7)
    subj = _frame_indexed_subject(n_frames=6)
    out = FrameDropoutTransform(dropout_rate=0.3)(subj)
    mask = out["frame_dropout_mask"]
    assert isinstance(mask, list)
    assert len(mask) == 6


def test_frame_dropout_rejects_out_of_range_rate() -> None:
    with pytest.raises(ValueError, match="dropout_rate"):
        FrameDropoutTransform(dropout_rate=1.5)
