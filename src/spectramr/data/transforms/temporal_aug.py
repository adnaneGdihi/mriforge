"""Temporal augmentations for 4D cine MRI (Phase 4c).

Three augmentations, all TorchIO Transform subclasses so they compose
cleanly with the rest of the pipeline:

- :class:`PhaseShiftTransform` — cyclic shift along the frame axis,
  preserving the periodic cine loop semantics.
- :class:`TemporalFlipTransform` — reverse the frame order (data-aug
  pretext task; cine loops are valid backwards in time for many
  reconstruction tasks).
- :class:`FrameDropoutTransform` — zero out a random subset of frames
  to simulate missing acquisition / arrhythmia gating dropouts.

All three operate on the channel axis (where frames live by our
convention; see ``src/data/datasets/cine_dataset.py``).
"""

from __future__ import annotations

import torch
import torchio as tio

__all__ = [
    "FrameDropoutTransform",
    "PhaseShiftTransform",
    "TemporalFlipTransform",
]


class PhaseShiftTransform(tio.Transform):
    """Cyclic shift along the frame axis.

    Cardiac cine loops are periodic — shifting the starting phase is a
    valid augmentation that doesn't change the underlying signal.

    Args:
        max_shift: Maximum absolute shift in frames. Each call draws
            uniformly from ``[-max_shift, +max_shift]`` (exclusive of 0
            if ``include_zero=False``).
        image_keys: Which Subject image keys to shift in tandem (so
            input and target stay aligned).
        include_zero: Allow 0 as a draw (i.e., identity).
    """

    def __init__(
        self,
        max_shift: int = 4,
        image_keys: tuple[str, ...] = ("input", "target"),
        include_zero: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")
        self.max_shift = max_shift
        self.image_keys = image_keys
        self.include_zero = include_zero

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.max_shift == 0:
            return subject
        low = -self.max_shift
        high = self.max_shift + 1
        shift = int(torch.randint(low, high, (1,)).item())
        if shift == 0 and not self.include_zero:
            shift = 1
        for key in self.image_keys:
            img = subject.get(key)
            if img is None or not isinstance(img, tio.ScalarImage):
                continue
            data = img.data
            shifted = torch.roll(data, shifts=shift, dims=0)
            subject[key] = tio.ScalarImage(tensor=shifted, affine=img.affine)
        subject["phase_shift_applied"] = shift
        return subject


class TemporalFlipTransform(tio.Transform):
    """Reverse the order of frames along the channel axis.

    Args:
        p: Probability of applying the flip.
        image_keys: Which Subject image keys to flip in tandem.
    """

    def __init__(
        self,
        p: float = 0.5,
        image_keys: tuple[str, ...] = ("input", "target"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self.image_keys = image_keys

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if torch.rand(1).item() >= self.p:
            return subject
        for key in self.image_keys:
            img = subject.get(key)
            if img is None or not isinstance(img, tio.ScalarImage):
                continue
            data = img.data
            flipped = torch.flip(data, dims=[0])
            subject[key] = tio.ScalarImage(tensor=flipped, affine=img.affine)
        subject["temporal_flip_applied"] = True
        return subject


class FrameDropoutTransform(tio.Transform):
    """Zero out a random subset of frames to simulate missing acquisitions.

    The dropout pattern is sampled independently per Subject but shared
    across image keys (so input and target drop the same frames — this
    is the "missing-acquisition" cine setting, not the asymmetric mask).

    Args:
        dropout_rate: Probability each frame is dropped.
        image_keys: Image keys to dropout.
        only_input: When True, dropout is applied to ``"input"`` only
            and the target frames remain intact (training task: predict
            the missing frames). Defaults to True.
    """

    def __init__(
        self,
        dropout_rate: float = 0.1,
        image_keys: tuple[str, ...] = ("input", "target"),
        only_input: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        self.dropout_rate = dropout_rate
        self.image_keys = image_keys
        self.only_input = only_input

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.dropout_rate == 0.0:
            return subject
        # Compute mask using the input image's frame count.
        input_img = subject.get("input")
        if input_img is None or not isinstance(input_img, tio.ScalarImage):
            return subject
        n_frames = input_img.data.shape[0]
        keep_mask = torch.rand(n_frames) >= self.dropout_rate
        # Guarantee at least one frame survives — all-dropped is degenerate.
        if not keep_mask.any():
            keep_mask[torch.randint(0, n_frames, (1,)).item()] = True

        keys = ("input",) if self.only_input else self.image_keys
        for key in keys:
            img = subject.get(key)
            if img is None or not isinstance(img, tio.ScalarImage):
                continue
            data = img.data.clone()
            data[~keep_mask] = 0.0
            subject[key] = tio.ScalarImage(tensor=data, affine=img.affine)

        subject["frame_dropout_mask"] = keep_mask.tolist()
        return subject
