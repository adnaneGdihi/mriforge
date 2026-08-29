"""Coverage for shared/utils/preprocessing_qc.py dataclasses (WS-8).

Closes the preprocessing_qc 0-test gap and pins the fix making
``QCResult.details`` an optional annotation (``dict[str, Any] | None``) matching
its ``None`` default. No nibabel required (dataclass-level tests only).
"""

from __future__ import annotations

from dataclasses import asdict

from mriforge.shared.utils.preprocessing_qc import (
    FileQCResult,
    PreprocessingQC,
    QCResult,
    QCThresholds,
)


def test_qcresult_details_default_is_none() -> None:
    r = QCResult(name="snr", passed=True, value=12.0)
    assert r.details is None
    assert r.threshold is None
    assert r.message == ""


def test_qcresult_asdict_roundtrip() -> None:
    r = QCResult(
        name="x", passed=False, value=1, threshold=2, message="m", details={"k": 1}
    )
    d = asdict(r)
    assert d["details"] == {"k": 1}
    assert d["passed"] is False


def test_qcthresholds_defaults() -> None:
    t = QCThresholds()
    assert t.min_brain_volume_ratio == 0.3
    assert t.max_brain_volume_ratio == 0.8
    assert t.min_snr == 10.0


def test_preprocessing_qc_uses_default_thresholds() -> None:
    qc = PreprocessingQC()
    assert isinstance(qc.thresholds, QCThresholds)


def test_file_qc_result_defaults() -> None:
    fr = FileQCResult(file_path="/p", file_name="p", checks=[], overall_passed=True)
    assert fr.processing_time == 0.0
    assert fr.checks == []
