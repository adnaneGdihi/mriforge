"""Dataset existence and compliance validation system.

This module provides comprehensive validation of datasets against known dataset
definitions and actual filesystem structure. It checks:

- Directory existence for all data roots and input directories
- File count and type validation (h5, nifti, dicom, png, etc.)
- Data completeness (e.g., expected contrast-specific subdirectories)
- Variant availability (original, normalized, registered, kspace, etc.)
- Data integrity (readable, non-corrupted accessible files)
- Size estimation and quota usage

The compliance system supports:
- Dry-run validation (filesystem checks only)
- Detailed reporting with severity levels
- JSON/CSV export for pipeline integration
- Dataset health monitoring and alerting
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import h5py  # type: ignore
except ImportError:
    h5py = None

try:
    import nibabel as nib  # type: ignore
except ImportError:
    nib = None


class ComplianceSeverity(Enum):
    """Severity levels for compliance issues."""

    OK = 0  # No issues
    INFO = 1  # Informational, not a problem
    WARNING = 2  # Minor issue, may affect some tasks
    ERROR = 3  # Significant issue, dataset partially unusable
    CRITICAL = 4  # Dataset unusable for intended tasks


class FileType(Enum):
    """Recognized data file types."""

    H5 = "h5"
    NIFTI = "nifti"
    DICOM = "dicom"
    PNG = "png"
    JPEG = "jpeg"
    JSON = "json"
    YAML = "yaml"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class ComplianceIssue:
    """A single compliance issue or check result."""

    severity: ComplianceSeverity
    category: str  # e.g., "directory_missing", "file_count_low", "format_invalid"
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return {
            "severity": self.severity.name,
            "category": self.category,
            "message": self.message,
            "path": self.path,
            "details": self.details,
        }


@dataclass
class VariantStatus:
    """Status of a single preprocessing variant."""

    variant_type: str
    path: str
    exists: bool
    file_count: int = 0
    total_size_mb: float = 0.0
    severity: ComplianceSeverity = ComplianceSeverity.OK
    issues: list[ComplianceIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return {
            "variant_type": self.variant_type,
            "path": self.path,
            "exists": self.exists,
            "file_count": self.file_count,
            "total_size_mb": round(self.total_size_mb, 2),
            "severity": self.severity.name,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class DatasetComplianceReport:
    """Comprehensive compliance report for a single dataset."""

    dataset_key: str
    dataset_name: str
    timestamp: str
    overall_severity: ComplianceSeverity
    is_compliant: bool

    # Structural checks
    root_exists: bool = False
    root_path: str = ""
    root_size_mb: float = 0.0

    # Content validation
    primary_file_count: int = 0
    file_type_distribution: dict[str, int] = field(default_factory=dict)
    expected_contrasts: list[str] = field(default_factory=list)
    found_contrasts: list[str] = field(default_factory=list)

    # Variants status
    variants: list[VariantStatus] = field(default_factory=list)

    # Task readiness
    task_readiness: dict[str, bool] = field(default_factory=dict)

    # Issues and details
    issues: list[ComplianceIssue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return {
            "dataset_key": self.dataset_key,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "overall_severity": self.overall_severity.name,
            "is_compliant": self.is_compliant,
            "root_exists": self.root_exists,
            "root_path": self.root_path,
            "root_size_mb": round(self.root_size_mb, 2),
            "primary_file_count": self.primary_file_count,
            "file_type_distribution": self.file_type_distribution,
            "expected_contrasts": self.expected_contrasts,
            "found_contrasts": self.found_contrasts,
            "variants": [v.to_dict() for v in self.variants],
            "task_readiness": self.task_readiness,
            "issues": [issue.to_dict() for issue in self.issues],
            "details": self.details,
        }


class DatasetComplianceChecker:
    """Validates dataset existence and compliance."""

    @staticmethod
    def _max_severity(sev1: ComplianceSeverity, sev2: ComplianceSeverity) -> ComplianceSeverity:
        """Return the more severe of two severity levels."""
        return sev1 if sev1.value >= sev2.value else sev2

    def __init__(
        self,
        logger: logging.Logger | None = None,
        dataset_root: str | Path | None = None,
    ) -> None:
        """Initialize the compliance checker.

        Args:
            logger: Optional logger instance. If None, creates a default logger.
            dataset_root: Base directory for relative dataset paths. If None,
                         uses current working directory.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.dataset_root = Path(dataset_root or Path.cwd())
        self.logger.debug(f"Initialized compliance checker with root: {self.dataset_root}")

    def check_dataset(self, descriptor: Any) -> DatasetComplianceReport:
        """Check compliance of a single dataset.

        Args:
            descriptor: Dataset descriptor instance.

        Returns:
            DatasetComplianceReport with detailed compliance information.
        """
        report = DatasetComplianceReport(
            dataset_key=descriptor.key,
            dataset_name=descriptor.display_name,
            timestamp=datetime.now().isoformat(),
            overall_severity=ComplianceSeverity.OK,
            is_compliant=True,
        )

        # Check root directory
        root_path = self._resolve_path(descriptor.data_root)
        report.root_path = str(root_path)

        if not root_path.exists():
            self.logger.warning(f"Root directory missing for {descriptor.key}: {root_path}")
            report.root_exists = False
            issue = ComplianceIssue(
                severity=ComplianceSeverity.CRITICAL,
                category="root_missing",
                message=f"Root directory does not exist: {root_path}",
                path=str(root_path),
            )
            report.issues.append(issue)
            report.overall_severity = self._max_severity(
                report.overall_severity, ComplianceSeverity.CRITICAL
            )
            report.is_compliant = False
            return report

        report.root_exists = True
        report.root_size_mb = self._calculate_directory_size(root_path)

        # Check primary input directories
        self._check_input_directories(descriptor, report)

        # Check expected contrasts
        if descriptor.contrasts:
            report.expected_contrasts = list(descriptor.contrasts)
            self._check_contrasts(descriptor, report)

        # Check preprocessing variants
        self._check_variants(descriptor, report)

        # Check task readiness
        self._check_task_readiness(descriptor, report)

        # Determine final compliance status
        self._finalize_report(report)

        return report

    def check_multiple_datasets(
        self, descriptors: Iterable[Any]
    ) -> dict[str, DatasetComplianceReport]:
        """Check multiple datasets and return aggregated reports.

        Args:
            descriptors: Iterable of Dataset descriptor instances.

        Returns:
            Dictionary mapping dataset keys to compliance reports.
        """
        reports = {}
        for descriptor in descriptors:
            reports[descriptor.key] = self.check_dataset(descriptor)
        return reports

    def _check_input_directories(self, descriptor: Any, report: DatasetComplianceReport) -> None:
        """Check input_lr_dir, input_hr_dir, and related directories."""
        dirs_to_check = [
            ("input_lr_dir", descriptor.input_lr_dir),
            ("input_hr_dir", descriptor.input_hr_dir),
            ("lr_volume_dir", descriptor.lr_volume_dir),
            ("hr_volume_dir", descriptor.hr_volume_dir),
        ]

        for attr_name, dir_path in dirs_to_check:
            if not dir_path:
                continue

            resolved = self._resolve_path(dir_path)
            if not resolved.exists():
                severity = (
                    ComplianceSeverity.WARNING
                    if "volume" in attr_name
                    else ComplianceSeverity.ERROR
                )
                issue = ComplianceIssue(
                    severity=severity,
                    category=f"{attr_name}_missing",
                    message=f"{attr_name} directory does not exist: {resolved}",
                    path=str(resolved),
                )
                report.issues.append(issue)
                if severity == ComplianceSeverity.ERROR:
                    report.overall_severity = self._max_severity(
                        report.overall_severity, ComplianceSeverity.ERROR
                    )
                    report.is_compliant = False
            else:
                file_count = len(list(resolved.rglob("*")))
                if file_count == 0:
                    issue = ComplianceIssue(
                        severity=ComplianceSeverity.WARNING,
                        category=f"{attr_name}_empty",
                        message=f"{attr_name} directory is empty: {resolved}",
                        path=str(resolved),
                    )
                    report.issues.append(issue)
                    report.overall_severity = self._max_severity(
                        report.overall_severity, ComplianceSeverity.WARNING
                    )

    def _check_contrasts(self, descriptor: Any, report: DatasetComplianceReport) -> None:
        """Check for contrast-specific subdirectories or files."""
        root_path = self._resolve_path(descriptor.data_root)
        found = set()

        for item in root_path.iterdir():
            name_lower = item.name.lower()
            for contrast in descriptor.contrasts:
                if contrast.lower() in name_lower or name_lower in contrast.lower():
                    found.add(contrast)

        report.found_contrasts = list(found)
        # Warn on ANY missing contrast, including the "found nothing" case. The
        # previous ``found and …`` guard let a dataset with zero expected
        # contrasts present pass silently while a partial match warned — i.e.
        # "found nothing" graded healthier than "found some" (the empty-input
        # false-pass shape).
        if len(found) < len(descriptor.contrasts):
            missing = set(descriptor.contrasts) - found
            issue = ComplianceIssue(
                severity=ComplianceSeverity.WARNING,
                category="contrasts_incomplete",
                message=f"Missing contrasts: {', '.join(sorted(missing))}",
                details={"found": list(found), "missing": list(missing)},
            )
            report.issues.append(issue)
            # ComplianceSeverity is a plain Enum (not IntEnum), so bare ``max()``
            # on its members raises TypeError; use the value-comparing helper the
            # other severity-combine sites use.
            report.overall_severity = self._max_severity(
                report.overall_severity, ComplianceSeverity.WARNING
            )

    def _check_variants(self, descriptor: Any, report: DatasetComplianceReport) -> None:
        """Check availability of preprocessing variants."""
        if not descriptor.preprocessing_variants:
            return

        for variant_type, variant_path in descriptor.preprocessing_variants.items():
            resolved = self._resolve_path(variant_path)
            variant_status = VariantStatus(
                variant_type=variant_type,
                path=str(resolved),
                exists=resolved.exists(),
            )

            if not resolved.exists():
                variant_status.severity = ComplianceSeverity.INFO
                variant_status.issues.append(
                    ComplianceIssue(
                        severity=ComplianceSeverity.INFO,
                        category="variant_missing",
                        message=f"Variant '{variant_type}' not yet generated: {resolved}",
                        path=str(resolved),
                    )
                )
            else:
                variant_status.file_count = len([f for f in resolved.rglob("*") if f.is_file()])
                variant_status.total_size_mb = self._calculate_directory_size(resolved)
                if variant_status.file_count == 0:
                    variant_status.severity = ComplianceSeverity.WARNING
                    variant_status.issues.append(
                        ComplianceIssue(
                            severity=ComplianceSeverity.WARNING,
                            category="variant_empty",
                            message=f"Variant '{variant_type}' directory is empty: {resolved}",
                            path=str(resolved),
                        )
                    )

            report.variants.append(variant_status)

    def _check_task_readiness(self, descriptor: Any, report: DatasetComplianceReport) -> None:
        """Check if dataset is ready for each declared task."""
        if not descriptor.task_requirements:
            return

        root_path = self._resolve_path(descriptor.data_root)

        for task_name in descriptor.task_requirements.keys():
            ready = root_path.exists()

            # Special logic for preprocessed datasets
            if "preprocessed" in descriptor.key:
                # Preprocessed datasets need their variant directories
                if descriptor.preprocessing_variants:
                    ready = any(
                        self._resolve_path(path).exists()
                        for path in descriptor.preprocessing_variants.values()
                    )

            report.task_readiness[task_name] = ready

    def _finalize_report(self, report: DatasetComplianceReport) -> None:
        """Finalize the report with aggregate severity level."""
        # Determine final severity from all issues
        if report.issues:
            severities = [issue.severity for issue in report.issues]
            max_severity = max(severities, key=lambda s: s.value)

            if max_severity == ComplianceSeverity.CRITICAL:
                report.overall_severity = ComplianceSeverity.CRITICAL
                report.is_compliant = False
            elif max_severity == ComplianceSeverity.ERROR:
                report.overall_severity = ComplianceSeverity.ERROR
                # May still be partially compliant
                report.is_compliant = (
                    len([i for i in report.issues if i.severity == ComplianceSeverity.ERROR]) == 0
                )
            elif max_severity == ComplianceSeverity.WARNING:
                report.overall_severity = ComplianceSeverity.WARNING
                report.is_compliant = True

        # Add summary details
        report.details["total_issues"] = len(report.issues)
        report.details["critical_issues"] = sum(
            1 for i in report.issues if i.severity == ComplianceSeverity.CRITICAL
        )
        report.details["error_issues"] = sum(
            1 for i in report.issues if i.severity == ComplianceSeverity.ERROR
        )
        report.details["warning_issues"] = sum(
            1 for i in report.issues if i.severity == ComplianceSeverity.WARNING
        )
        report.details["ready_tasks"] = sum(1 for ready in report.task_readiness.values() if ready)
        report.details["available_variants"] = sum(1 for v in report.variants if v.exists)

    def _resolve_path(self, path_str: str | None) -> Path:
        """Resolve a path string to an absolute Path."""
        if not path_str:
            return Path()

        path = Path(path_str)
        if path.is_absolute():
            return path

        # If the path starts with "databases/", "experiments/" etc that match
        # a top-level directory, resolve from repo root, not dataset_root
        if path_str.startswith(("databases/", "experiments/", "cache/")):
            return self.dataset_root.parent / path

        return self.dataset_root / path

    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate total size of directory in MB."""
        if not directory.exists():
            return 0.0

        try:
            total_bytes = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            return total_bytes / (1024 * 1024)
        except (OSError, PermissionError) as e:
            self.logger.debug(f"Could not calculate size of {directory}: {e}")
            return 0.0

    def generate_summary_report(
        self, reports: dict[str, DatasetComplianceReport]
    ) -> dict[str, Any]:
        """Generate an aggregated summary from multiple compliance reports.

        Args:
            reports: Dictionary of dataset key to compliance report.

        Returns:
            Dictionary with summary statistics and recommendations.
        """
        total_datasets = len(reports)
        compliant_count = sum(1 for r in reports.values() if r.is_compliant)

        severity_counts = {
            "OK": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }

        for report in reports.values():
            severity_counts[report.overall_severity.name] += 1

        total_size_mb = sum(r.root_size_mb for r in reports.values())
        total_files = sum(r.primary_file_count for r in reports.values())

        # Identify problematic datasets
        critical_datasets = [
            k for k, r in reports.items() if r.overall_severity == ComplianceSeverity.CRITICAL
        ]
        error_datasets = [
            k for k, r in reports.items() if r.overall_severity == ComplianceSeverity.ERROR
        ]

        # Identify missing variants
        missing_variants = {}
        for key, report in reports.items():
            missing = [v.variant_type for v in report.variants if not v.exists]
            if missing:
                missing_variants[key] = missing

        return {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": total_datasets,
            "compliant_datasets": compliant_count,
            "compliance_percentage": round(100 * compliant_count / total_datasets, 1),
            "severity_distribution": severity_counts,
            "total_size_mb": round(total_size_mb, 2),
            "total_files": total_files,
            "critical_datasets": critical_datasets,
            "error_datasets": error_datasets,
            "missing_variants_by_dataset": missing_variants,
            "recommendations": self._generate_recommendations(
                reports, critical_datasets, error_datasets
            ),
        }

    def _generate_recommendations(
        self,
        reports: dict[str, DatasetComplianceReport],
        critical: list[str],
        errors: list[str],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if critical:
            recommendations.append(
                f"Address critical issues in {len(critical)} dataset(s): {', '.join(critical)}. "
                "These datasets cannot be used until fixed."
            )

        if errors:
            recommendations.append(
                f"Fix errors in {len(errors)} dataset(s): {', '.join(errors)}. "
                "These datasets may be partially usable."
            )

        # Check for missing variants
        missing_preprocessed = sum(
            1 for r in reports.values() if "preprocessed" in r.dataset_key and not r.variants
        )
        if missing_preprocessed > 0:
            recommendations.append(
                f"{missing_preprocessed} preprocessed dataset(s) have missing variants. "
                "Run preprocessing pipeline to generate them."
            )

        return recommendations

    def export_to_json(
        self,
        reports: dict[str, DatasetComplianceReport],
        output_path: str | Path,
    ) -> None:
        """Export compliance reports to JSON file.

        Args:
            reports: Dictionary of compliance reports.
            output_path: Path to write JSON output.
        """
        summary = self.generate_summary_report(reports)
        output = {
            "summary": summary,
            "datasets": {key: report.to_dict() for key, report in reports.items()},
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Compliance report exported to {output_path}")

    def export_to_csv(
        self,
        reports: dict[str, DatasetComplianceReport],
        output_path: str | Path,
    ) -> None:
        """Export simplified compliance summary to CSV file.

        Args:
            reports: Dictionary of compliance reports.
            output_path: Path to write CSV output.
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset_key",
                    "dataset_name",
                    "severity",
                    "compliant",
                    "root_exists",
                    "root_size_mb",
                    "file_count",
                    "issues_count",
                    "ready_tasks",
                ],
            )
            writer.writeheader()

            for key, report in reports.items():
                writer.writerow(
                    {
                        "dataset_key": report.dataset_key,
                        "dataset_name": report.dataset_name,
                        "severity": report.overall_severity.value,
                        "compliant": report.is_compliant,
                        "root_exists": report.root_exists,
                        "root_size_mb": round(report.root_size_mb, 2),
                        "file_count": report.primary_file_count,
                        "issues_count": len(report.issues),
                        "ready_tasks": sum(1 for ready in report.task_readiness.values() if ready),
                    }
                )

        self.logger.info(f"Compliance summary exported to {output_path}")
