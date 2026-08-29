"""Real-dataset acquisition-parameter extractors (Idea 2 of audit_plan_novel.md).

When ``data.expose_acquisition_params: true`` is set, the strategy
expects each batch to carry ``acquisition_params`` as a ``[B, 5]``
tensor of ``(TR, TE, TI, α, B₀)``. The
:mod:`acquisition_params_injector` provides a fallback that
broadcasts a registry default; this module provides the *real*
extractors so that fallback is only invoked on data sources that
genuinely lack metadata.

Two sources are supported:

- **DICOM** — directly read the canonical DICOM tags (RepetitionTime,
  EchoTime, InversionTime, FlipAngle, MagneticFieldStrength).
- **BIDS JSON sidecar** — for NIfTI files following the BIDS
  convention, the sidecar contains the same fields under standard
  keys.

Each extractor returns ``(TR_ms, TE_ms, TI_ms, alpha_rad, B0_T)`` and
raises ``AcquisitionParamsExtractionError`` when the source does not
provide enough information to construct the vector.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AcquisitionParamsExtractionError(RuntimeError):
    """Raised when an extractor cannot recover the full 5-vector."""


@dataclass(frozen=True)
class AcquisitionParams:
    """Container for the canonical (TR, TE, TI, α, B₀) tuple.

    Stored in physical units: TR/TE/TI in ms, α in radians, B₀ in tesla.
    """

    TR: float
    TE: float
    TI: float
    alpha_rad: float
    B0: float

    def to_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.TR, self.TE, self.TI, self.alpha_rad, self.B0)


# --------------------------------------------------------------------- #
# DICOM extraction
# --------------------------------------------------------------------- #


_DICOM_TAGS: dict[str, str] = {
    "TR": "RepetitionTime",
    "TE": "EchoTime",
    "TI": "InversionTime",
    "alpha_deg": "FlipAngle",
    "B0": "MagneticFieldStrength",
}


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_acquisition_params_from_dicom(ds_or_path: Any) -> AcquisitionParams:
    """Pull the 5-vector from a DICOM dataset.

    Args:
        ds_or_path: Either a pydicom ``Dataset`` or a path to a DICOM
            file. When given a path, we load it with
            ``stop_before_pixels=True`` to keep the read cheap.

    Returns:
        :class:`AcquisitionParams` with values in canonical units.

    Raises:
        AcquisitionParamsExtractionError: When TR or TE is missing
            (these are the only strictly required fields; the rest fall
            back to safe defaults).
    """
    if isinstance(ds_or_path, (str, Path)):
        import pydicom

        ds = pydicom.dcmread(str(ds_or_path), stop_before_pixels=True)
    else:
        ds = ds_or_path

    raw: dict[str, float | None] = {}
    for canonical, dicom_field in _DICOM_TAGS.items():
        raw[canonical] = _coerce_float(getattr(ds, dicom_field, None))

    tr = raw["TR"]
    te = raw["TE"]
    if tr is None or te is None:
        raise AcquisitionParamsExtractionError(
            "DICOM record is missing RepetitionTime or EchoTime."
        )
    ti = raw["TI"] if raw["TI"] is not None else 0.0
    alpha_deg = raw["alpha_deg"] if raw["alpha_deg"] is not None else 90.0
    b0 = raw["B0"] if raw["B0"] is not None else 1.5
    return AcquisitionParams(
        TR=float(tr),
        TE=float(te),
        TI=float(ti),
        alpha_rad=math.radians(float(alpha_deg)),
        B0=float(b0),
    )


# --------------------------------------------------------------------- #
# BIDS JSON sidecar extraction
# --------------------------------------------------------------------- #


_BIDS_KEYS: dict[str, tuple[str, ...]] = {
    # canonical → list of accepted BIDS keys (canonical first)
    "TR": ("RepetitionTime", "RepetitionTimeMs"),
    "TE": ("EchoTime", "EchoTimeMs"),
    "TI": ("InversionTime", "InversionTimeMs"),
    "alpha_deg": ("FlipAngle",),
    "B0": ("MagneticFieldStrength",),
}


def _read_bids_field(meta: dict[str, Any], canonical: str) -> float | None:
    for key in _BIDS_KEYS[canonical]:
        if key in meta:
            v = _coerce_float(meta[key])
            if v is None:
                continue
            # BIDS standard expresses TR/TE/TI in *seconds*; if the
            # value looks small for a millisecond reading, convert.
            if canonical in {"TR", "TE", "TI"} and not key.endswith("Ms") and v < 50.0:
                v *= 1000.0
            return v
    return None


def extract_acquisition_params_from_bids_sidecar(path: str | Path) -> AcquisitionParams:
    """Pull the 5-vector from a BIDS JSON sidecar.

    Args:
        path: Path to the ``.json`` sidecar (NIfTI's companion).

    Returns:
        :class:`AcquisitionParams` with values in canonical units.

    Raises:
        AcquisitionParamsExtractionError: When TR or TE is missing or
            the file cannot be parsed.
    """
    p = Path(path)
    if not p.is_file():
        raise AcquisitionParamsExtractionError(f"BIDS sidecar not found: {p}")
    try:
        meta = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise AcquisitionParamsExtractionError(f"Invalid JSON in {p}: {exc}") from exc

    tr = _read_bids_field(meta, "TR")
    te = _read_bids_field(meta, "TE")
    if tr is None or te is None:
        raise AcquisitionParamsExtractionError(
            f"BIDS sidecar {p} is missing RepetitionTime or EchoTime."
        )
    ti = _read_bids_field(meta, "TI") or 0.0
    alpha_deg = _read_bids_field(meta, "alpha_deg") or 90.0
    b0 = _read_bids_field(meta, "B0") or 1.5
    return AcquisitionParams(
        TR=float(tr),
        TE=float(te),
        TI=float(ti),
        alpha_rad=math.radians(float(alpha_deg)),
        B0=float(b0),
    )


# --------------------------------------------------------------------- #
# Best-effort dispatcher
# --------------------------------------------------------------------- #


def extract_acquisition_params(path: str | Path) -> AcquisitionParams:
    """Resolve acquisition params from a path, dispatching by extension.

    - ``.dcm`` / ``.dicom`` / directories with DICOM contents → DICOM path.
    - ``.json`` (or a co-located sidecar for ``.nii`` / ``.nii.gz``) → BIDS.
    - Anything else raises :class:`AcquisitionParamsExtractionError`.

    Args:
        path: Path to the data file or DICOM dataset.

    Returns:
        :class:`AcquisitionParams`.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".gz" and p.stem.endswith(".nii"):
        nifti_root = p.with_suffix("").with_suffix("")
        sidecar = nifti_root.with_suffix(".json")
        if sidecar.is_file():
            return extract_acquisition_params_from_bids_sidecar(sidecar)
    if suffix == ".nii":
        sidecar = p.with_suffix(".json")
        if sidecar.is_file():
            return extract_acquisition_params_from_bids_sidecar(sidecar)
    if suffix == ".json":
        return extract_acquisition_params_from_bids_sidecar(p)
    if suffix in {".dcm", ".dicom"}:
        return extract_acquisition_params_from_dicom(p)
    if p.is_dir():
        for child in p.iterdir():
            if child.suffix.lower() in {".dcm", ".dicom"}:
                return extract_acquisition_params_from_dicom(child)
    raise AcquisitionParamsExtractionError(f"No DICOM / BIDS sidecar found for {p}")


__all__ = [
    "AcquisitionParams",
    "AcquisitionParamsExtractionError",
    "extract_acquisition_params",
    "extract_acquisition_params_from_bids_sidecar",
    "extract_acquisition_params_from_dicom",
]
