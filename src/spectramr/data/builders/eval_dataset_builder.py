"""Evaluation-mode reference / sample loaders (SSOT for the eval path).

Per CLAUDE.md pitfall #11, any code turning a file (``.h5``, ``.pt``,
``.nii.gz``, ``.npy``, ``.pkl``) into a Dataset or DataLoader must live
under ``src/data/``. This module plugs the round-9-identified gap
where :mod:`spectramr.infrastructure.orchestration.campaign_evaluator`
loaded clean references, uncertainty arrays, paired eval samples,
and per-sample metrics directly via ``nib.load`` / ``h5py.File`` /
``np.load``.

Scope (Phase 1 of the data-layer unification plan):

This module exposes **role-aware** loaders the campaign evaluator
delegates to. Each role is a specific kind of artifact the evaluator
reads after training has finished:

- ``clean_reference``: ground-truth image (NIfTI / H5 / NPY / PT).
- ``paired_samples``: ``.npz`` with ``inputs`` + ``targets`` arrays.
- ``per_sample_metrics``: ``.npz`` with arbitrary metric arrays.
- ``uncertainty_bundle``: ``.npz`` with ``predictions`` /
  ``targets`` / ``uncertainties``.

Reference tensors are returned **unnormalized** — evaluation metrics
must operate on the original signal range, not the [-1, 1] training
input range that :mod:`spectramr.infrastructure.io.adapters` applies.

Manifest-style ``.pkl`` files (paths only, no MRI data) are
explicitly out of scope — those are config and remain loaded inline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Internal primitives ─────────────────────────────────────────────────────


def _load_pt(path: Path) -> torch.Tensor:
    """Load a ``.pt`` tensor (data, not a checkpoint).

    Uses ``weights_only=True`` so an attacker who controls the path
    can't trigger arbitrary code execution via pickle.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _load_npy(path: Path) -> torch.Tensor:
    """Load a ``.npy`` array as a float32 tensor."""
    return torch.from_numpy(np.load(str(path))).float()


def _load_nifti(path: Path) -> torch.Tensor:
    """Load a NIfTI volume as a float32 tensor.

    No normalization applied — evaluation references must keep the
    original intensity range. (Contrast with
    :func:`spectramr.infrastructure.io.adapters.load_nifti`, which
    rescales to [-1, 1] for training-input ingestion.)
    """
    import nibabel as nib

    img = nib.load(str(path))
    return torch.from_numpy(img.get_fdata()).float()


def _load_h5_reference(path: Path) -> torch.Tensor | None:
    """Load an H5 reference tensor.

    Tries common dataset keys in order: ``data``, ``image``,
    ``reconstruction``, ``kspace``. Returns ``None`` if no recognised
    key is present.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        for key in ("data", "image", "reconstruction", "kspace"):
            if key in f:
                return torch.from_numpy(f[key][()]).float()
    return None


# ── Role-aware public API ───────────────────────────────────────────────────


def load_clean_reference(path: str | Path) -> torch.Tensor | None:
    """Load a single ground-truth reference image / k-space tensor.

    Dispatches on file suffix:

    - ``.pt``        → :func:`_load_pt`
    - ``.npy``       → :func:`_load_npy`
    - ``.nii``, ``.nii.gz``, ``.gz`` → :func:`_load_nifti`
    - ``.h5``, ``.hdf5`` → :func:`_load_h5_reference`

    Args:
        path: Filesystem path to the reference artifact.

    Returns:
        Float32 CPU tensor in the original (unnormalized) intensity
        range, or ``None`` when the file is missing / has an
        unrecognised suffix / has no recognised H5 dataset key.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("clean_reference: %s does not exist", p)
        return None
    suffix = p.suffix.lower()
    if suffix == ".pt":
        return _load_pt(p)
    if suffix == ".npy":
        return _load_npy(p)
    if suffix in (".nii", ".gz"):
        return _load_nifti(p)
    if suffix in (".h5", ".hdf5"):
        return _load_h5_reference(p)
    logger.warning("clean_reference: unrecognized suffix %r at %s", suffix, p)
    return None


def load_paired_samples(
    path: str | Path,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load a ``.npz`` archive with ``inputs`` + ``targets`` arrays.

    The archive must contain both keys; otherwise an empty list is
    returned. Each input/target pair becomes a tuple of float32
    CPU tensors.

    Args:
        path: Filesystem path to a ``.npz`` file.

    Returns:
        List of ``(input_tensor, target_tensor)`` pairs. Empty when
        the file is missing or doesn't contain both required keys.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("paired_samples: %s does not exist", p)
        return []
    data = np.load(str(p))
    if "inputs" not in data or "targets" not in data:
        logger.warning(
            "paired_samples: %s lacks required 'inputs'/'targets' keys (found %s)",
            p,
            list(data.keys()),
        )
        return []
    inputs = torch.from_numpy(data["inputs"]).float()
    targets = torch.from_numpy(data["targets"]).float()
    return [(inputs[i], targets[i]) for i in range(len(inputs))]


def load_per_sample_metrics(path: str | Path) -> dict[str, np.ndarray] | None:
    """Load a ``.npz`` archive of per-sample metric arrays as a dict.

    Used to read cached metric tensors written by an earlier
    evaluation run (so the campaign evaluator can aggregate without
    re-running inference).

    Args:
        path: Filesystem path to the ``.npz`` archive.

    Returns:
        Dict mapping metric-name → numpy array, or ``None`` when the
        file is missing.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("per_sample_metrics: %s does not exist", p)
        return None
    data = np.load(str(p))
    return {key: data[key] for key in data.files}


def load_uncertainty_bundle(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load a ``.npz`` archive with ``predictions`` / ``targets`` /
    ``uncertainties`` arrays.

    Used by the uncertainty calibration / correlation analysis in the
    campaign evaluator.

    Args:
        path: Filesystem path to the ``.npz`` archive.

    Returns:
        Tuple of ``(predictions, targets, uncertainties)`` float32
        tensors, or ``None`` when the file is missing or any required
        key is absent.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("uncertainty_bundle: %s does not exist", p)
        return None
    data = np.load(str(p))
    required = ("predictions", "targets", "uncertainties")
    missing = [k for k in required if k not in data]
    if missing:
        logger.warning(
            "uncertainty_bundle: %s missing required keys %s (found %s)",
            p,
            missing,
            list(data.keys()),
        )
        return None
    return (
        torch.from_numpy(data["predictions"]).float(),
        torch.from_numpy(data["targets"]).float(),
        torch.from_numpy(data["uncertainties"]).float(),
    )


__all__ = [
    "load_clean_reference",
    "load_paired_samples",
    "load_per_sample_metrics",
    "load_uncertainty_bundle",
]
