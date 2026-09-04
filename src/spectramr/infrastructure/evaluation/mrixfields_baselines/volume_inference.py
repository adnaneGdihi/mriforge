"""Faithful per-slice volume inference for MRIxFields2026 baselines (Task 5).

Port of ``Baseline/scripts/inference.py::predict_volume`` and
``Baseline/mrixfields/data/transforms.py::CenterCropOrPad``.

The round-trip is load-bearing::

    [0, 1] -> *2-1 -> G -> clip(-1, 1) * 0.5 + 0.5 -> [0, 1]

Consumed by Task 6 (pipeline runner).  The canonical-space step (orientation
round-trip) is deliberately omitted: we load volumes in canonical space and
compare pred/target there, skipping the original-orientation conversion that
the original script performs only for on-disk saving.

References:
    Original source: /tmp/mrix/Baseline_scripts_inference.py (``predict_volume``)
    Transform:       /tmp/mrix/Baseline_mrixfields_data_transforms.py (``CenterCropOrPad``)
"""

from __future__ import annotations

import numpy as np
import torch

from spectramr.infrastructure.evaluation.mrixfields_baselines.generator_loader import LoadedBaseline

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _center_crop_or_pad(a: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
    """Center-crop or zero-pad ``a`` to ``target`` shape.

    Verbatim port of ``CenterCropOrPad.__call__`` from the original repo.
    Works on arrays of any dimensionality as long as ``len(target) == a.ndim``.

    Args:
        a: Input array.
        target: Desired output shape (same number of dimensions as ``a``).

    Returns:
        New array with shape ``target`` and dtype ``a.dtype``.
    """
    out = np.zeros(target, dtype=a.dtype)
    src: list[slice] = []
    dst: list[slice] = []
    for s, t in zip(a.shape, target, strict=False):
        if s > t:
            start = (s - t) // 2
            src.append(slice(start, start + t))
            dst.append(slice(None))
        else:
            start = (t - s) // 2
            src.append(slice(None))
            dst.append(slice(start, start + s))
    out[tuple(dst)] = a[tuple(src)]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_brain_mask(
    pred: np.ndarray,
    source: np.ndarray,
    thresh: float = 1e-6,
) -> np.ndarray:
    """Zero-out background voxels using the skull-stripped source as a mask.

    The original MRIxFields2026 inputs are skull-stripped; the ResNet/StarGAN
    Tanh output leaks small non-zero values outside the brain.  Masking with
    ``source > thresh`` suppresses these artefacts.

    Args:
        pred: Predicted volume, same shape as ``source``, float32 in ``[0, 1]``.
        source: Source volume in ``[0, 1]`` used to define the brain mask.
        thresh: Voxels with ``source > thresh`` are foreground.

    Returns:
        ``pred`` with background voxels set to zero, same dtype as ``pred``.
    """
    return pred * (source > thresh).astype(pred.dtype)


@torch.no_grad()
def predict_volume(
    loaded: LoadedBaseline,
    volume: np.ndarray,
    *,
    slice_axis: int = 2,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run faithful slice-by-slice inference on a 3-D volume.

    Faithfully reproduces the original ``predict_volume`` round-trip::

        [0,1] -> slc * 2 - 1 -> [1,1,H,W] tensor -> G -> squeeze -> clip(-1,1) * 0.5 + 0.5 -> [0,1]

    For StarGAN the style code is pre-bound inside ``loaded.forward``; callers
    do not need to handle it.  For ResNet, ``crop_size`` is ``None`` and
    slices are passed at native size.

    Args:
        loaded: A ``LoadedBaseline`` produced by ``load_baseline_generator``.
            Its ``.forward`` callable accepts ``[B, 1, H, W]`` and returns the
            same shape (values in ``[-1, 1]`` before the normalisation step
            below).
        volume: Source volume in ``[0, 1]``, shape ``[H, W, D]`` (or any 3-D
            layout as long as ``slice_axis`` indexes the slice dimension).
            Must be ``float32``; will be cast if not.
        slice_axis: Axis along which to iterate (default ``2`` -> axial).
        device: PyTorch device for inference.

    Returns:
        Predicted volume, same shape and dtype as ``volume``, in ``[0, 1]``.

    Notes:
        - No orientation round-trip is performed; canonical-space comparison is
          assumed (as required by the evaluation pipeline).
        - The ``@torch.no_grad()`` decorator wraps the whole function so that
          slice-loop tensor operations are gradient-free.
    """
    device = torch.device(device)
    vol = volume.astype(np.float32)
    out = np.zeros_like(vol)
    n = vol.shape[slice_axis]
    cs = loaded.crop_size  # None for ResNet; (img_size, img_size) for StarGAN

    for i in range(n):
        # Extract the i-th slice along slice_axis.
        sl: list[slice | int] = [slice(None)] * vol.ndim
        sl[slice_axis] = i
        sl_t = tuple(sl)
        slc: np.ndarray = vol[sl_t]  # shape e.g. [H, W] for axis=2

        # Optional center-crop/pad to model input size.
        model_in = _center_crop_or_pad(slc, cs) if cs is not None else slc

        # Scale [0, 1] -> [-1, 1] and build a [1, 1, H, W] tensor.
        x = torch.from_numpy(model_in * 2.0 - 1.0).float().unsqueeze(0).unsqueeze(0).to(device)

        # Forward pass -- style handling is hidden inside loaded.forward for StarGAN.
        pred_t = loaded.forward(x)
        pred = pred_t.squeeze().cpu().numpy()

        # Un-crop: restore to original slice spatial dimensions.
        if cs is not None:
            pred = _center_crop_or_pad(pred, slc.shape)

        # Map model output [-1, 1] -> [0, 1].
        out[sl_t] = np.clip(pred, -1.0, 1.0) * 0.5 + 0.5

    return out
