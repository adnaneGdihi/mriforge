"""Inject ``acquisition_params`` onto a batch dict (Idea 2 of audit_plan_novel.md).

When ``data.expose_acquisition_params=True``, downstream strategies
(especially :class:`ScoreFieldTomographyStrategy`) expect each batch to
carry an ``acquisition_params`` tensor of shape ``[B, 5]`` containing
``(TR, TE, TI, α, B₀)``.

For datasets that read these from DICOM / JSON metadata the values
are populated upstream. For datasets that don't carry per-sample
metadata (NumPy phantoms, anonymised paired NIfTI cohorts), this
injector fills the values from the framework's static
:data:`PHYSICS_REGISTRY` so the audit ladder and the strategy can
proceed without a stub.

The injector is intentionally a *pure function*, not a TorchIO
transform — it operates on the assembled batch dict returned by the
DataLoader's collate, after per-sample transforms have already run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from spectramr.infrastructure.physics.physics_registry import (
    PHYSICS_REGISTRY,
    PhysicsParameters,
)

_DEFAULT_FALLBACK_CONTRAST: str = "T1w"


def _params_to_tensor(
    p: PhysicsParameters, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Pack a :class:`PhysicsParameters` row into a 5-vector."""
    t = torch.tensor(
        [p.TR, p.TE, p.TI, 0.0, p.B0],
        dtype=torch.float32,
        device=device,
    )
    return t


def infer_acquisition_params(
    batch: Mapping[str, Any],
    *,
    default_contrast: str = _DEFAULT_FALLBACK_CONTRAST,
) -> torch.Tensor:
    """Return acquisition parameters ``[B, 5]`` for a batch.

    Resolution order:

    1. If ``batch["acquisition_params"]`` already exists and is a
       tensor of shape ``[B, 5]``, return as-is. This honours dataset
       implementations that already wire DICOM metadata into the batch.
    2. If ``batch["source_path"]`` (or ``batch["file_path"]``) is a
       list of paths and the files have a readable DICOM / BIDS
       sidecar, use the real extractor (
       :func:`acquisition_params_extractors.extract_acquisition_params`).
    3. If ``batch["contrast"]`` or ``batch["contrast_type"]`` is a
       per-sample list of contrast names that exist in
       :data:`PHYSICS_REGISTRY`, build the matrix row-by-row.
    4. Otherwise fall back to broadcasting ``PHYSICS_REGISTRY[default_contrast]``
       across the batch dimension.

    Args:
        batch: Mapping returned by the dataloader.
        default_contrast: Registry key used in the fallback path.

    Returns:
        Float tensor ``[B, 5]`` on the same device as the batch's
        ``image`` / ``input`` tensor when present, else CPU.
    """
    existing = batch.get("acquisition_params")
    if isinstance(existing, torch.Tensor) and existing.dim() == 2 and existing.shape[-1] == 5:
        return existing

    # Find a reference tensor to determine batch size and device.
    ref = None
    for k in ("image", "input", "target", "kspace"):
        v = batch.get(k)
        if isinstance(v, torch.Tensor):
            ref = v
            break
    if ref is None:
        batch_size = 1
        device: torch.device | str = "cpu"
    else:
        batch_size = ref.shape[0]
        device = ref.device

    # Tier 2: real-dataset extraction from DICOM / BIDS-sidecar paths.
    paths = batch.get("source_path") or batch.get("file_path")
    if isinstance(paths, (list, tuple)) and len(paths) == batch_size:
        try:
            from spectramr.data.transforms.acquisition_params_extractors import (
                AcquisitionParamsExtractionError,
                extract_acquisition_params,
            )

            rows: list[torch.Tensor] = []
            extracted_all = True
            for p in paths:
                try:
                    params = extract_acquisition_params(p)
                except AcquisitionParamsExtractionError:
                    extracted_all = False
                    break
                rows.append(
                    torch.tensor(
                        list(params.to_tuple()),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            if extracted_all and rows:
                return torch.stack(rows, dim=0)
        except ImportError:
            # pydicom / json are stdlib-ish but never assume.
            pass

    contrasts = batch.get("contrast") or batch.get("contrast_type")
    if isinstance(contrasts, (list, tuple)) and len(contrasts) == batch_size:
        rows = []
        for c in contrasts:
            params = PHYSICS_REGISTRY.get(str(c)) or PHYSICS_REGISTRY[default_contrast]
            rows.append(_params_to_tensor(params, device=device))
        return torch.stack(rows, dim=0)

    fallback = PHYSICS_REGISTRY[default_contrast]
    row = _params_to_tensor(fallback, device=device)
    return row.unsqueeze(0).expand(batch_size, 5).clone()


def attach_acquisition_params(batch: dict[str, Any]) -> dict[str, Any]:
    """In-place helper: ensure ``batch["acquisition_params"]`` is populated.

    Returns the same dict for chaining. No-op if the key is already a
    well-formed ``[B, 5]`` tensor.
    """
    batch["acquisition_params"] = infer_acquisition_params(batch)
    return batch


__all__ = ["attach_acquisition_params", "infer_acquisition_params"]
