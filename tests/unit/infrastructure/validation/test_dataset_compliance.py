"""Unit tests for infrastructure/validation/dataset_compliance.py.

Covers: ComplianceSeverity ordering, ComplianceIssue serialisation,
        VariantStatus serialisation, DatasetComplianceReport.to_dict,
        DatasetComplianceChecker._max_severity.
"""

from __future__ import annotations

import pytest

from mriforge.infrastructure.validation.dataset_compliance import (
    ComplianceIssue,
    ComplianceSeverity,
    DatasetComplianceChecker,
    DatasetComplianceReport,
    FileType,
    VariantStatus,
)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_severity_enum_values_canary():
    """Severity enum has OK < INFO < WARNING < ERROR < CRITICAL."""
    s = ComplianceSeverity
    assert s.OK.value < s.INFO.value < s.WARNING.value < s.ERROR.value < s.CRITICAL.value


# ---------------------------------------------------------------------------
# Parametrized: _max_severity
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "s1,s2,expected",
    [
        (ComplianceSeverity.OK, ComplianceSeverity.ERROR, ComplianceSeverity.ERROR),
        (ComplianceSeverity.CRITICAL, ComplianceSeverity.WARNING, ComplianceSeverity.CRITICAL),
        (ComplianceSeverity.INFO, ComplianceSeverity.INFO, ComplianceSeverity.INFO),
    ],
)
def test_max_severity(s1, s2, expected):
    checker = DatasetComplianceChecker()
    result = checker._max_severity(s1, s2)
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "ft,value",
    [
        (FileType.H5, "h5"),
        (FileType.NIFTI, "nifti"),
        (FileType.DICOM, "dicom"),
        (FileType.UNKNOWN, "unknown"),
    ],
)
def test_file_type_values(ft, value):
    assert ft.value == value


# ---------------------------------------------------------------------------
# Edge: serialisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compliance_issue_to_dict():
    issue = ComplianceIssue(
        severity=ComplianceSeverity.WARNING,
        category="file_count_low",
        message="Only 5 files found",
        path="/some/path",
        details={"found": 5, "expected": 100},
    )
    d = issue.to_dict()
    assert d["severity"] == "WARNING"
    assert d["category"] == "file_count_low"
    assert d["path"] == "/some/path"
    assert d["details"]["found"] == 5


@pytest.mark.unit
def test_variant_status_to_dict():
    vs = VariantStatus(
        variant_type="normalized",
        path="/data/normalized",
        exists=True,
        file_count=50,
        total_size_mb=1024.5678,
    )
    d = vs.to_dict()
    assert d["exists"] is True
    assert d["file_count"] == 50
    assert d["total_size_mb"] == pytest.approx(1024.57, abs=0.01)


@pytest.mark.unit
def test_dataset_compliance_report_to_dict():
    from datetime import datetime

    report = DatasetComplianceReport(
        dataset_key="fastmri_knee",
        dataset_name="FastMRI Knee",
        timestamp=datetime.now().isoformat(),
        overall_severity=ComplianceSeverity.OK,
        is_compliant=True,
    )
    d = report.to_dict()
    assert d["dataset_key"] == "fastmri_knee"
    assert d["is_compliant"] is True
    assert d["overall_severity"] == "OK"


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_severity_max_with_symmetric_call():
    """_max_severity is symmetric: max(A,B) == max(B,A)."""
    checker = DatasetComplianceChecker()
    a, b = ComplianceSeverity.ERROR, ComplianceSeverity.WARNING
    assert checker._max_severity(a, b) == checker._max_severity(b, a)


# ---------------------------------------------------------------------------
# _check_contrasts: partial match must not crash; empty match must not pass
# ---------------------------------------------------------------------------


def _blank_report() -> DatasetComplianceReport:
    from datetime import datetime

    return DatasetComplianceReport(
        dataset_key="ds",
        dataset_name="DS",
        timestamp=datetime.now().isoformat(),
        overall_severity=ComplianceSeverity.OK,
        is_compliant=True,
    )


@pytest.mark.unit
def test_check_contrasts_partial_match_does_not_crash(tmp_path):
    """A partial contrast match previously called bare ``max()`` on plain-Enum
    members, raising TypeError. It must warn and bump severity, not crash."""
    from types import SimpleNamespace

    (tmp_path / "T1_scan").mkdir()  # matches T1, not T2
    checker = DatasetComplianceChecker(dataset_root=tmp_path)
    report = _blank_report()
    descriptor = SimpleNamespace(data_root=str(tmp_path), contrasts=["T1", "T2"])

    checker._check_contrasts(descriptor, report)  # must not raise TypeError

    assert any(i.category == "contrasts_incomplete" for i in report.issues)
    assert report.overall_severity == ComplianceSeverity.WARNING


@pytest.mark.unit
def test_check_contrasts_zero_match_still_warns(tmp_path):
    """A dataset with NONE of the expected contrasts present must warn — the old
    ``found and …`` guard let "found nothing" pass silently."""
    from types import SimpleNamespace

    (tmp_path / "unrelated").mkdir()  # matches neither T1 nor T2
    checker = DatasetComplianceChecker(dataset_root=tmp_path)
    report = _blank_report()
    descriptor = SimpleNamespace(data_root=str(tmp_path), contrasts=["T1", "T2"])

    checker._check_contrasts(descriptor, report)

    assert any(i.category == "contrasts_incomplete" for i in report.issues), (
        "zero-contrast dataset silently graded compliant (empty-input false-pass)"
    )
    assert report.overall_severity == ComplianceSeverity.WARNING
