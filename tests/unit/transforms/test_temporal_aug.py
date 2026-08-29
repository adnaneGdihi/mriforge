"""Unit tests for temporal_aug transforms (Phase 4c).

Covers all three transforms:
  PhaseShiftTransform, TemporalFlipTransform, FrameDropoutTransform.

These tests are CPU-deterministic and use only tiny synthetic subjects.
All tests are marked ``unit``.
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from mriforge.data.transforms.temporal_aug import (
    FrameDropoutTransform,
    PhaseShiftTransform,
    TemporalFlipTransform,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _indexed_subject(n_frames: int = 8) -> tio.Subject:
    """Subject whose frame-i tensor equals float(i) everywhere — trackable."""
    base = torch.arange(n_frames, dtype=torch.float32).view(n_frames, 1, 1, 1)
    data = base.expand(n_frames, 3, 3, 1).clone()
    return tio.Subject(
        input=tio.ScalarImage(tensor=data),
        target=tio.ScalarImage(tensor=data.clone()),
    )


# ── PhaseShiftTransform ───────────────────────────────────────────────────────


class TestPhaseShiftTransform:
    def test_canary_applies_without_error(self) -> None:
        torch.manual_seed(0)
        subj = _indexed_subject(n_frames=6)
        out = PhaseShiftTransform(max_shift=2)(subj)
        assert out["input"].data.shape[0] == 6

    def test_identity_at_max_shift_zero(self) -> None:
        subj = _indexed_subject(n_frames=6)
        out = PhaseShiftTransform(max_shift=0)(subj)
        assert torch.equal(out["input"].data, subj["input"].data)

    def test_input_and_target_shifted_identically(self) -> None:
        """Same shift is applied to all listed image keys."""
        torch.manual_seed(42)
        subj = _indexed_subject(n_frames=8)
        out = PhaseShiftTransform(max_shift=3, include_zero=False)(subj)
        assert torch.equal(out["input"].data, out["target"].data)

    def test_cyclic_roll_preserves_frame_count(self) -> None:
        subj = _indexed_subject(n_frames=5)
        out = PhaseShiftTransform(max_shift=2)(subj)
        assert out["input"].data.shape[0] == 5

    def test_phase_shift_recorded_on_subject(self) -> None:
        torch.manual_seed(0)
        out = PhaseShiftTransform(max_shift=3)(_indexed_subject(n_frames=6))
        assert "phase_shift_applied" in out
        assert -3 <= out["phase_shift_applied"] <= 3

    def test_negative_max_shift_raises(self) -> None:
        with pytest.raises(ValueError, match="max_shift"):
            PhaseShiftTransform(max_shift=-1)

    @pytest.mark.parametrize("max_shift", [1, 3, 5])
    def test_parametrised_max_shift_shape_preserved(self, max_shift: int) -> None:
        torch.manual_seed(0)
        subj = _indexed_subject(n_frames=10)
        out = PhaseShiftTransform(max_shift=max_shift)(subj)
        assert out["input"].data.shape == subj["input"].data.shape


# ── TemporalFlipTransform ────────────────────────────────────────────────────


class TestTemporalFlipTransform:
    def test_canary_p1_always_flips(self) -> None:
        subj = _indexed_subject(n_frames=4)
        out = TemporalFlipTransform(p=1.0)(subj)
        assert out["input"].data[0, 0, 0, 0] == 3.0
        assert out["input"].data[-1, 0, 0, 0] == 0.0

    def test_identity_at_p_zero(self) -> None:
        subj = _indexed_subject(n_frames=4)
        out = TemporalFlipTransform(p=0.0)(subj)
        assert torch.equal(out["input"].data, subj["input"].data)

    def test_p1_flip_keeps_input_target_aligned(self) -> None:
        subj = _indexed_subject(n_frames=5)
        out = TemporalFlipTransform(p=1.0)(subj)
        assert torch.equal(out["input"].data, out["target"].data)

    def test_p1_marks_flip_applied(self) -> None:
        subj = _indexed_subject(n_frames=4)
        out = TemporalFlipTransform(p=1.0)(subj)
        assert out.get("temporal_flip_applied") is True

    def test_p0_does_not_mark_flip_applied(self) -> None:
        subj = _indexed_subject(n_frames=4)
        out = TemporalFlipTransform(p=0.0)(subj)
        assert out.get("temporal_flip_applied") is not True

    def test_out_of_range_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p"):
            TemporalFlipTransform(p=1.5)

    def test_shape_preserved(self) -> None:
        subj = _indexed_subject(n_frames=6)
        out = TemporalFlipTransform(p=1.0)(subj)
        assert out["input"].data.shape == subj["input"].data.shape


# ── FrameDropoutTransform ─────────────────────────────────────────────────────


class TestFrameDropoutTransform:
    def test_canary_dropout_applied_and_mask_recorded(self) -> None:
        torch.manual_seed(7)
        subj = _indexed_subject(n_frames=8)
        out = FrameDropoutTransform(dropout_rate=0.5)(subj)
        assert "frame_dropout_mask" in out
        assert len(out["frame_dropout_mask"]) == 8

    def test_identity_at_zero_dropout(self) -> None:
        subj = _indexed_subject(n_frames=6)
        out = FrameDropoutTransform(dropout_rate=0.0)(subj)
        assert torch.equal(out["input"].data, subj["input"].data)

    def test_at_least_one_frame_survives_extreme_dropout(self) -> None:
        torch.manual_seed(7)
        subj = _indexed_subject(n_frames=10)
        out = FrameDropoutTransform(dropout_rate=0.99)(subj)
        assert any(out["frame_dropout_mask"])

    def test_only_input_true_preserves_target(self) -> None:
        torch.manual_seed(7)
        subj = _indexed_subject(n_frames=8)
        out = FrameDropoutTransform(dropout_rate=0.5, only_input=True)(subj)
        assert torch.equal(out["target"].data, subj["target"].data)

    def test_only_input_false_drops_same_frames_from_both(self) -> None:
        torch.manual_seed(7)
        subj = _indexed_subject(n_frames=8)
        out = FrameDropoutTransform(
            dropout_rate=0.5, only_input=False, image_keys=("input", "target")
        )(subj)
        in_zero = out["input"].data.sum(dim=(1, 2, 3)) == 0
        tg_zero = out["target"].data.sum(dim=(1, 2, 3)) == 0
        assert torch.equal(in_zero, tg_zero)

    def test_out_of_range_dropout_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout_rate"):
            FrameDropoutTransform(dropout_rate=1.5)

    @pytest.mark.parametrize("rate", [0.1, 0.5, 0.9])
    def test_parametrised_rates_shape_preserved(self, rate: float) -> None:
        torch.manual_seed(0)
        subj = _indexed_subject(n_frames=8)
        out = FrameDropoutTransform(dropout_rate=rate)(subj)
        assert out["input"].data.shape == subj["input"].data.shape
