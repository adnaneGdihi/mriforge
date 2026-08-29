"""Tests for the real-dataset acquisition-params extractors (Idea 2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from mriforge.data.transforms.acquisition_params_extractors import (
    AcquisitionParams,
    AcquisitionParamsExtractionError,
    extract_acquisition_params,
    extract_acquisition_params_from_bids_sidecar,
)


def _write_sidecar(path: Path, **fields: object) -> None:
    path.write_text(json.dumps(fields))


def test_bids_sidecar_ms_keys(tmp_path: Path) -> None:
    sidecar = tmp_path / "sub01_T1w.json"
    _write_sidecar(
        sidecar,
        RepetitionTimeMs=8.0,
        EchoTimeMs=4.0,
        InversionTimeMs=0.0,
        FlipAngle=15.0,
        MagneticFieldStrength=3.0,
    )
    params = extract_acquisition_params_from_bids_sidecar(sidecar)
    assert isinstance(params, AcquisitionParams)
    assert params.TR == pytest.approx(8.0)
    assert params.TE == pytest.approx(4.0)
    assert params.alpha_rad == pytest.approx(math.radians(15.0))
    assert params.B0 == pytest.approx(3.0)


def test_bids_sidecar_seconds_are_converted(tmp_path: Path) -> None:
    """BIDS canonical keys are in *seconds*; values <50 should auto-convert."""
    sidecar = tmp_path / "sub02_T1w.json"
    _write_sidecar(
        sidecar,
        RepetitionTime=0.008,  # 8 ms in seconds
        EchoTime=0.004,
        FlipAngle=15.0,
        MagneticFieldStrength=3.0,
    )
    params = extract_acquisition_params_from_bids_sidecar(sidecar)
    assert params.TR == pytest.approx(8.0)
    assert params.TE == pytest.approx(4.0)


def test_bids_sidecar_missing_te_raises(tmp_path: Path) -> None:
    sidecar = tmp_path / "broken.json"
    _write_sidecar(sidecar, RepetitionTime=0.008)
    with pytest.raises(AcquisitionParamsExtractionError):
        extract_acquisition_params_from_bids_sidecar(sidecar)


def test_bids_sidecar_missing_optional_fields_uses_defaults(tmp_path: Path) -> None:
    sidecar = tmp_path / "minimal.json"
    _write_sidecar(sidecar, RepetitionTime=8.0, EchoTime=4.0)
    params = extract_acquisition_params_from_bids_sidecar(sidecar)
    # TI defaults to 0, alpha to 90°, B0 to 1.5 T.
    assert params.TI == pytest.approx(0.0)
    assert params.alpha_rad == pytest.approx(math.radians(90.0))
    assert params.B0 == pytest.approx(1.5)


def test_extract_dispatcher_handles_json(tmp_path: Path) -> None:
    sidecar = tmp_path / "sub.json"
    _write_sidecar(sidecar, RepetitionTimeMs=8.0, EchoTimeMs=4.0)
    p = extract_acquisition_params(sidecar)
    assert p.TR == pytest.approx(8.0)


def test_extract_dispatcher_handles_nifti_with_sidecar(tmp_path: Path) -> None:
    nifti = tmp_path / "scan.nii.gz"
    nifti.write_bytes(b"fake nifti header")
    sidecar = tmp_path / "scan.json"
    _write_sidecar(sidecar, RepetitionTimeMs=8.0, EchoTimeMs=4.0)
    p = extract_acquisition_params(nifti)
    assert p.TR == pytest.approx(8.0)


def test_extract_dispatcher_missing_metadata_raises(tmp_path: Path) -> None:
    bare = tmp_path / "scan.npy"
    bare.write_bytes(b"")
    with pytest.raises(AcquisitionParamsExtractionError):
        extract_acquisition_params(bare)


def test_injector_uses_real_extractor(tmp_path: Path) -> None:
    """When the batch carries a list of paths with sidecars, the injector
    pulls real values rather than the registry fallback."""
    from mriforge.data.transforms.acquisition_params_injector import (
        infer_acquisition_params,
    )

    nifti_a = tmp_path / "a.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    for f in (nifti_a, nifti_b):
        f.write_bytes(b"")
    _write_sidecar(tmp_path / "a.json", RepetitionTimeMs=10.0, EchoTimeMs=5.0)
    _write_sidecar(tmp_path / "b.json", RepetitionTimeMs=12.0, EchoTimeMs=6.0)
    batch = {
        "image": torch.zeros(2, 1, 4, 4),
        "source_path": [str(nifti_a), str(nifti_b)],
    }
    params = infer_acquisition_params(batch)
    assert params.shape == (2, 5)
    # TR is the first column.
    assert params[0, 0].item() == pytest.approx(10.0)
    assert params[1, 0].item() == pytest.approx(12.0)


def test_injector_falls_back_when_sidecars_missing(tmp_path: Path) -> None:
    """Missing sidecars should not crash — fall back to the registry path."""
    from mriforge.data.transforms.acquisition_params_injector import (
        infer_acquisition_params,
    )

    nifti = tmp_path / "orphan.nii.gz"
    nifti.write_bytes(b"")
    batch = {
        "image": torch.zeros(1, 1, 4, 4),
        "source_path": [str(nifti)],
    }
    params = infer_acquisition_params(batch)
    assert params.shape == (1, 5)
