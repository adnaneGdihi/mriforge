"""Temporal samplers for 4D cine MRI (Phase 4c).

Cine volumes are loaded with frames folded into TorchIO's channel slot
(see ``src/data/datasets/cine_dataset.py``). The samplers here operate on
that channel axis to extract a contiguous window of frames per draw.

Two variants:

- :class:`TemporalUniformSampler` — training-time: each call yields one
  Subject with a randomly-chosen contiguous frame window of size
  ``frames_per_window``. Spatial extent unchanged.
- :class:`TemporalGridSampler` — inference-time: deterministic enumeration
  of overlapping frame windows covering the full sequence. Pairs with
  :class:`TemporalGridAggregator` to reassemble frame-ordered outputs.

These are not :class:`tio.PatchSampler` subclasses because TorchIO's
patch samplers slice spatial dims only — temporal axis lives in the
channel slot for our convention. The interface mimics ``__iter__``-on-
Subject so the existing queue plumbing can iterate transparently.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator

import torch
import torchio as tio

__all__ = [
    "TemporalGridAggregator",
    "TemporalGridSampler",
    "TemporalUniformSampler",
]


def _window_subject(
    subject: tio.Subject,
    start: int,
    frames_per_window: int,
    image_keys: tuple[str, ...] = ("input", "target"),
) -> tio.Subject:
    """Return a deep copy of ``subject`` with frame [start, start+window)
    selected from each image's channel axis."""
    end = start + frames_per_window
    out = copy.copy(subject)
    for key in image_keys:
        img = subject.get(key)
        if img is None or not isinstance(img, tio.ScalarImage):
            continue
        data = img.data
        if data.shape[0] < end:
            # Pad-by-reflect would corrupt cine periodicity; pad-by-repeat
            # is the cardiology convention.
            pad_count = end - data.shape[0]
            pad = data[-1:].expand(pad_count, *data.shape[1:])
            data = torch.cat([data, pad], dim=0)
        windowed = data[start:end].clone()
        out[key] = tio.ScalarImage(tensor=windowed)
    out["temporal_window_start"] = start
    out["temporal_window_end"] = end
    return out


class TemporalUniformSampler:
    """Random contiguous frame-window draws.

    Args:
        frames_per_window: Number of frames per draw (channel count of
            the emitted ScalarImage).
        image_keys: Tuple of subject image keys to window. ``"input"``
            and ``"target"`` by default.

    The generator never terminates — TorchIO Queue consumes it lazily.
    """

    def __init__(
        self,
        frames_per_window: int,
        image_keys: tuple[str, ...] = ("input", "target"),
    ):
        if frames_per_window < 1:
            raise ValueError("frames_per_window must be >= 1")
        self.frames_per_window = frames_per_window
        self.image_keys = image_keys

    def __call__(
        self,
        subject: tio.Subject,
        num_patches: int | None = None,
    ) -> Iterator[tio.Subject]:
        n_frames = self._n_frames(subject)
        max_start = max(0, n_frames - self.frames_per_window)
        count = 0
        while num_patches is None or count < num_patches:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
            yield _window_subject(subject, start, self.frames_per_window, self.image_keys)
            count += 1

    def _n_frames(self, subject: tio.Subject) -> int:
        # Prefer recorded count; fall back to the input image's C dim.
        recorded = subject.get("n_frames")
        if isinstance(recorded, int):
            return recorded
        img = subject.get("input")
        if img is None:
            raise ValueError(
                "TemporalUniformSampler requires a Subject with an 'input' "
                "ScalarImage or an 'n_frames' int attribute."
            )
        return int(img.data.shape[0])


class TemporalGridSampler:
    """Deterministic overlapping frame windows for inference.

    Enumerates start frames at stride ``frames_per_window - overlap`` and
    emits one Subject per window. The last window is clamped to the end
    of the sequence (no roll-around) — pairs with
    :class:`TemporalGridAggregator` for frame-ordered reconstruction.

    Args:
        frames_per_window: Frames per draw.
        overlap: Frames shared between adjacent windows. Must be < window.
        image_keys: Image keys to window.
    """

    def __init__(
        self,
        frames_per_window: int,
        overlap: int = 0,
        image_keys: tuple[str, ...] = ("input", "target"),
    ):
        if frames_per_window < 1:
            raise ValueError("frames_per_window must be >= 1")
        if overlap >= frames_per_window:
            raise ValueError(
                f"overlap ({overlap}) must be < frames_per_window ({frames_per_window})"
            )
        self.frames_per_window = frames_per_window
        self.overlap = overlap
        self.image_keys = image_keys

    def starts(self, n_frames: int) -> list[int]:
        """Compute the canonical list of window-start frame indices."""
        if n_frames <= self.frames_per_window:
            return [0]
        stride = self.frames_per_window - self.overlap
        starts = list(range(0, n_frames - self.frames_per_window + 1, stride))
        last_start = n_frames - self.frames_per_window
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def __call__(self, subject: tio.Subject) -> Iterator[tio.Subject]:
        n_frames = self._n_frames(subject)
        for start in self.starts(n_frames):
            windowed = _window_subject(subject, start, self.frames_per_window, self.image_keys)
            windowed["temporal_window_n_total"] = n_frames
            yield windowed

    def _n_frames(self, subject: tio.Subject) -> int:
        recorded = subject.get("n_frames")
        if isinstance(recorded, int):
            return recorded
        img = subject.get("input")
        if img is None:
            raise ValueError(
                "TemporalGridSampler requires a Subject with an 'input' "
                "ScalarImage or an 'n_frames' int attribute."
            )
        return int(img.data.shape[0])


class TemporalGridAggregator:
    """Reassemble frame-windowed predictions into a frame-ordered volume.

    Use::

        aggregator = TemporalGridAggregator(n_frames=24, spatial_shape=(W,H,D))
        for windowed_subj in sampler(subject):
            pred = model(windowed_subj["input"].data)  # (frames_per_window, W, H, D)
            aggregator.add(pred, start=windowed_subj["temporal_window_start"])
        full = aggregator.get_output()                   # (n_frames, W, H, D)

    Overlapping frames are averaged (the cardiology convention) — the
    cumulative-count tensor handles the variable-overlap arithmetic.
    """

    def __init__(
        self,
        n_frames: int,
        spatial_shape: tuple[int, int, int],
        dtype: torch.dtype = torch.float32,
    ):
        self.n_frames = n_frames
        self.spatial_shape = spatial_shape
        self._sum = torch.zeros((n_frames, *spatial_shape), dtype=dtype)
        self._count = torch.zeros((n_frames,), dtype=dtype)

    def add(self, pred: torch.Tensor, start: int) -> None:
        """Accumulate one window's prediction at the given frame offset."""
        if pred.dim() != 4:
            raise ValueError(
                f"Expected (frames_per_window, W, H, D) prediction; got shape {tuple(pred.shape)}."
            )
        end = start + pred.shape[0]
        if end > self.n_frames:
            # Clamp — late windows from TemporalGridSampler may overshoot
            # when n_frames < frames_per_window (it pads by repeat).
            usable = self.n_frames - start
            pred = pred[:usable]
            end = self.n_frames
        self._sum[start:end] += pred
        self._count[start:end] += 1

    def get_output(self) -> torch.Tensor:
        """Return averaged predictions; raises if any frame was never covered."""
        if (self._count == 0).any():
            missing = torch.nonzero(self._count == 0, as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"TemporalGridAggregator: frames {missing} were never covered "
                "by any window. Check the sampler stride / overlap config."
            )
        return self._sum / self._count[:, None, None, None]
