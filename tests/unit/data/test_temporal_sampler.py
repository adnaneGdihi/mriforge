"""Phase 4c — Temporal samplers + aggregator.

Pins the frame-window slicing semantics on the channel axis, the grid
sampler's deterministic enumeration, and the aggregator's frame-order
guarantees.
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.builders.temporal_sampler import (
    TemporalGridAggregator,
    TemporalGridSampler,
    TemporalUniformSampler,
)


def _make_cine_subject(n_frames: int = 12, w: int = 4, h: int = 4, d: int = 2) -> tio.Subject:
    """Synthesise a Subject with frame-distinct values per channel."""
    # Each frame gets a constant value equal to its index → easy to verify
    # that windowing pulled the right frames.
    base = torch.arange(n_frames, dtype=torch.float32).view(n_frames, 1, 1, 1)
    data = base.expand(n_frames, w, h, d).clone()
    return tio.Subject(
        input=tio.ScalarImage(tensor=data),
        target=tio.ScalarImage(tensor=data.clone()),
        n_frames=n_frames,
    )


# ── TemporalUniformSampler ──────────────────────────────────────────────────


def test_uniform_sampler_emits_window_with_correct_channel_size() -> None:
    subj = _make_cine_subject(n_frames=12)
    sampler = TemporalUniformSampler(frames_per_window=4)
    iterator = sampler(subj, num_patches=5)
    out = list(iterator)
    assert len(out) == 5
    for s in out:
        assert s["input"].data.shape[0] == 4
        # All values within a window are contiguous frame indices.
        vals = s["input"].data[:, 0, 0, 0].tolist()
        assert vals == sorted(vals)
        # Each frame index appears exactly w*h*d times across the channel —
        # we don't need to check that here; size + start_recorded suffice.
    assert all("temporal_window_start" in s for s in out)


def test_uniform_sampler_windows_align_input_and_target() -> None:
    subj = _make_cine_subject(n_frames=12)
    sampler = TemporalUniformSampler(frames_per_window=3, image_keys=("input", "target"))
    s = next(sampler(subj, num_patches=1))
    in_vals = s["input"].data[:, 0, 0, 0].tolist()
    tg_vals = s["target"].data[:, 0, 0, 0].tolist()
    assert in_vals == tg_vals


def test_uniform_sampler_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="frames_per_window"):
        TemporalUniformSampler(frames_per_window=0)


def test_uniform_sampler_falls_back_to_input_channel_count() -> None:
    """Subject without n_frames attribute ⇒ uses input.data.shape[0]."""
    data = torch.zeros(6, 2, 2, 1)
    subj = tio.Subject(input=tio.ScalarImage(tensor=data))
    sampler = TemporalUniformSampler(frames_per_window=2)
    s = next(sampler(subj, num_patches=1))
    assert s["input"].data.shape[0] == 2


# ── TemporalGridSampler ─────────────────────────────────────────────────────


def test_grid_sampler_starts_cover_full_sequence() -> None:
    sampler = TemporalGridSampler(frames_per_window=4, overlap=1)
    starts = sampler.starts(n_frames=12)
    # stride = 4 - 1 = 3 → 0, 3, 6, 9; last = 12 - 4 = 8 already in set? no — 8 not multiple of 3.
    # We expect 0, 3, 6, 9 and possibly an appended 8 to cover the tail.
    assert 0 in starts
    assert starts[-1] == 12 - 4  # last window ends exactly at n_frames


def test_grid_sampler_handles_n_frames_smaller_than_window() -> None:
    """Short sequence ⇒ single start at 0; sampler pads-by-repeat."""
    sampler = TemporalGridSampler(frames_per_window=8)
    starts = sampler.starts(n_frames=4)
    assert starts == [0]


def test_grid_sampler_starts_are_monotonic_and_unique() -> None:
    sampler = TemporalGridSampler(frames_per_window=4, overlap=2)
    starts = sampler.starts(n_frames=20)
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))


def test_grid_sampler_emits_subjects_with_temporal_metadata() -> None:
    subj = _make_cine_subject(n_frames=10)
    sampler = TemporalGridSampler(frames_per_window=4, overlap=1)
    out = list(sampler(subj))
    for s in out:
        assert s["temporal_window_n_total"] == 10
        assert s["temporal_window_end"] - s["temporal_window_start"] == 4


def test_grid_sampler_rejects_overlap_geq_window() -> None:
    with pytest.raises(ValueError, match="overlap"):
        TemporalGridSampler(frames_per_window=4, overlap=4)


# ── TemporalGridAggregator ──────────────────────────────────────────────────


def test_aggregator_identity_round_trip_no_overlap() -> None:
    """No overlap ⇒ aggregator returns the input frames unchanged."""
    n_frames = 8
    sampler = TemporalGridSampler(frames_per_window=4, overlap=0)
    subj = _make_cine_subject(n_frames=n_frames)
    aggregator = TemporalGridAggregator(
        n_frames=n_frames, spatial_shape=(4, 4, 2)
    )
    for windowed in sampler(subj):
        pred = windowed["input"].data
        aggregator.add(pred, start=windowed["temporal_window_start"])
    out = aggregator.get_output()
    assert out.shape == (n_frames, 4, 4, 2)
    # Values should equal the original frame indices.
    expected = torch.arange(n_frames, dtype=torch.float32).view(
        n_frames, 1, 1, 1
    ).expand(n_frames, 4, 4, 2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_aggregator_averages_overlapping_predictions() -> None:
    """Identical predictions over an overlap region ⇒ averaged unchanged."""
    n_frames = 10
    sampler = TemporalGridSampler(frames_per_window=4, overlap=2)
    subj = _make_cine_subject(n_frames=n_frames)
    aggregator = TemporalGridAggregator(
        n_frames=n_frames, spatial_shape=(4, 4, 2)
    )
    for windowed in sampler(subj):
        aggregator.add(
            windowed["input"].data,
            start=windowed["temporal_window_start"],
        )
    out = aggregator.get_output()
    expected = torch.arange(n_frames, dtype=torch.float32).view(
        n_frames, 1, 1, 1
    ).expand(n_frames, 4, 4, 2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_aggregator_raises_when_frame_uncovered() -> None:
    """Frame never written ⇒ get_output raises."""
    aggregator = TemporalGridAggregator(
        n_frames=8, spatial_shape=(2, 2, 1)
    )
    aggregator.add(torch.ones(4, 2, 2, 1), start=0)
    # Frames 4-7 never written.
    with pytest.raises(RuntimeError, match="never covered"):
        aggregator.get_output()


def test_aggregator_rejects_non_4d_prediction() -> None:
    aggregator = TemporalGridAggregator(
        n_frames=4, spatial_shape=(2, 2, 1)
    )
    with pytest.raises(ValueError):
        aggregator.add(torch.ones(4, 2), start=0)


def test_aggregator_clamps_overshoot_predictions() -> None:
    """Padded last window pred extends past n_frames ⇒ clamped, no crash."""
    aggregator = TemporalGridAggregator(
        n_frames=6, spatial_shape=(2, 2, 1)
    )
    # 8-frame prediction starting at frame 2 ⇒ would extend to frame 10.
    pred = torch.ones(8, 2, 2, 1)
    aggregator.add(pred, start=2)  # only frames 2..5 land
    aggregator.add(torch.zeros(2, 2, 2, 1), start=0)  # cover frames 0..1
    out = aggregator.get_output()
    assert out.shape == (6, 2, 2, 1)
