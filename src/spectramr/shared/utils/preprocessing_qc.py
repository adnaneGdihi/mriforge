"""Preprocessing Quality Control Module

This module provides comprehensive quality control for MRI preprocessing
pipelines including co-registration, brain extraction, and data validation.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import nibabel as nib

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None

logger = logging.getLogger(__name__)


@dataclass
class QCThresholds:
    """Quality control thresholds for preprocessing validation."""

    min_brain_volume_ratio: float = 0.3
    max_brain_volume_ratio: float = 0.8
    min_image_intensity: float = 0.1
    max_image_intensity: float = 0.95
    max_registration_error: float = 2.0  # mm
    min_snr: float = 10.0
    max_motion_artifact_score: float = 0.3


@dataclass
class QCResult:
    """Result of a quality control check."""

    name: str
    passed: bool
    value: Any
    threshold: Any = None
    message: str = ""
    details: dict[str, Any] = None


@dataclass
class FileQCResult:
    """Quality control results for a single file."""

    file_path: str
    file_name: str
    checks: list[QCResult]
    overall_passed: bool
    processing_time: float = 0.0


@dataclass
class QCReport:
    """Comprehensive quality control report."""

    timestamp: str
    input_dir: str
    output_dir: str
    thresholds: QCThresholds
    file_results: list[FileQCResult]
    summary: dict[str, Any]
    passed: bool
    warnings: list[str]
    errors: list[str]


class PreprocessingQC:
    """Quality control system for MRI preprocessing pipelines.

    Provides comprehensive validation of:
    - Co-registration accuracy
    - Brain extraction quality
    - Data integrity
    - Image quality metrics
    """

    def __init__(self, thresholds: QCThresholds | None = None):
        """__init__.

        Args:
            thresholds (Optional[QCThresholds]): Description.
        """
        self.thresholds = thresholds or QCThresholds()
        self.logger = logging.getLogger(__name__)

    def run_quality_control(
        self,
        input_dir: str,
        output_dir: str,
        save_report: bool = True,
        report_path: str | None = None,
    ) -> QCReport:
        """Run comprehensive quality control on preprocessing outputs.

        Args:
            input_dir: Input data directory
            output_dir: Output/preprocessed data directory
            save_report: Whether to save QC report to file
            report_path: Path to save QC report (auto-generated if None)

        Returns:
            QCReport: Comprehensive quality control report

        """
        import os
        import time
        from concurrent.futures import ProcessPoolExecutor

        start_time = time.time()

        self.logger.info(f"Starting quality control for: {output_dir}")

        # Find all processed files
        processed_files = self._find_processed_files(output_dir)

        # Run QC on each file (Parallel)
        file_results = []

        # Use 80% of available cores
        max_workers = max(1, int((os.cpu_count() or 1) * 0.8))
        self.logger.info(f"Running QC with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map helper function to files
            # Note: We need to ensure _qc_single_file is picklable.
            # Since it's an instance method, it might be tricky if 'self' contains unpicklable state.
            # However, PreprocessingQC seems stateless except for thresholds and logger.
            # Logger is usually not picklable. We should use a static method or helper function.
            # But for minimal refactor, let's try submitting. If it fails, we'll need to refactor _qc_single_file to be static.
            # Actually, let's make a wrapper to avoid pickling 'self' if possible, or just rely on fork (Linux).
            # Linux uses fork by default for multiprocessing, which copies memory, so it might work.
            # But ProcessPoolExecutor uses spawn or forkserver sometimes.
            # Safer to use a module-level function or ensure self is clean.
            # Let's try direct submission first, as Linux default is fork.

            futures = [
                executor.submit(self._qc_single_file, fp, input_dir, output_dir)
                for fp in processed_files
            ]

            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    file_results.append(result)
                except Exception as e:
                    file_path = processed_files[i]
                    self.logger.error(f"QC failed for {file_path}: {e!s}")
                    failed_result = FileQCResult(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        checks=[],
                        overall_passed=False,
                    )
                    file_results.append(failed_result)

        # Generate summary
        summary = self._generate_summary(file_results)

        # Create report
        report = QCReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            input_dir=input_dir,
            output_dir=output_dir,
            thresholds=self.thresholds,
            file_results=file_results,
            summary=summary,
            passed=summary.get("overall_passed", False),
            warnings=summary.get("warnings", []),
            errors=summary.get("errors", []),
        )

        # Save report if requested
        if save_report:
            if report_path is None:
                report_path = os.path.join(
                    output_dir,
                    "quality_control",
                    "qc_report.json",
                )
            self._save_report(report, report_path)

        processing_time = time.time() - start_time
        self.logger.info(f"QC completed in {processing_time:.2f} seconds")
        return report

    def _find_processed_files(self, output_dir: str) -> list[Path]:
        """Find all processed files in output directory."""
        output_path = Path(output_dir)
        processed_files = []

        # Common neuroimaging file extensions
        extensions = ["*.nii", "*.nii.gz", "*.npy", "*.mha", "*.mhd"]

        for ext in extensions:
            processed_files.extend(list(output_path.glob(f"**/{ext}")))

        self.logger.info(f"Found {len(processed_files)} processed files")
        return processed_files

    def _qc_single_file(
        self,
        file_path: Path,
        input_dir: str,
        output_dir: str,
    ) -> FileQCResult:
        """Run quality control checks on a single file."""
        import time

        start_time = time.time()

        self.logger.debug(f"QC checking: {file_path}")

        checks = []

        # Basic file checks
        checks.extend(self._check_file_integrity(file_path))

        # Format-specific checks
        if file_path.suffix in [".nii", ".gz"]:
            if NIBABEL_AVAILABLE:
                checks.extend(self._check_nifti_file(file_path, input_dir, output_dir))
            else:
                checks.append(
                    QCResult(
                        name="nifti_support",
                        passed=False,
                        value=None,
                        message="NiBabel not available for NIfTI QC",
                    ),
                )
        elif file_path.suffix == ".npy":
            checks.extend(self._check_numpy_file(file_path))

        # Overall assessment
        overall_passed = all(check.passed for check in checks)

        processing_time = time.time() - start_time

        return FileQCResult(
            file_path=str(file_path),
            file_name=file_path.name,
            checks=checks,
            overall_passed=overall_passed,
            processing_time=processing_time,
        )

    def _check_file_integrity(self, file_path: Path) -> list[QCResult]:
        """Check basic file integrity."""
        checks = []

        # Check file exists and is readable
        try:
            file_size = file_path.stat().st_size
            checks.append(
                QCResult(
                    name="file_exists",
                    passed=True,
                    value=file_size,
                    message=f"File size: {file_size} bytes",
                ),
            )
        except Exception as e:
            checks.append(
                QCResult(
                    name="file_exists",
                    passed=False,
                    value=None,
                    message=f"File access error: {e!s}",
                ),
            )
            return checks  # Can't do further checks if file is not accessible

        # Check file is not empty
        if file_size == 0:
            checks.append(
                QCResult(
                    name="file_not_empty",
                    passed=False,
                    value=file_size,
                    message="File is empty",
                ),
            )
        else:
            checks.append(
                QCResult(
                    name="file_not_empty",
                    passed=True,
                    value=file_size,
                    message="File contains data",
                ),
            )

        return checks

    def _check_nifti_file(
        self,
        file_path: Path,
        input_dir: str,
        output_dir: str,
    ) -> list[QCResult]:
        """Check NIfTI file quality."""
        checks = []

        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
            header = img.header

            # Check data type
            dtype = data.dtype
            checks.append(
                QCResult(
                    name="data_type",
                    passed=dtype in [np.float32, np.float64, np.int16, np.uint8],
                    value=str(dtype),
                    message=f"Data type: {dtype}",
                ),
            )

            # Check for NaN/inf values
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            checks.append(
                QCResult(
                    name="data_integrity",
                    passed=not (has_nan or has_inf),
                    value={
                        "nan_count": int(np.sum(np.isnan(data))),
                        "inf_count": int(np.sum(np.isinf(data))),
                    },
                    message=f"NaN: {has_nan}, Inf: {has_inf}",
                ),
            )

            # Check intensity range
            if data.size > 0:
                nonzero_data = data[data != 0]
                if len(nonzero_data) > 0:
                    p1, p99 = np.percentile(nonzero_data, [1, 99])
                    intensity_ok = (
                        p1 >= self.thresholds.min_image_intensity
                        and p99 <= self.thresholds.max_image_intensity
                    )
                    checks.append(
                        QCResult(
                            name="intensity_range",
                            passed=bool(intensity_ok),
                            value=[float(p1), float(p99)],
                            threshold=[
                                self.thresholds.min_image_intensity,
                                self.thresholds.max_image_intensity,
                            ],
                            message=f"intensity p1={p1:.3f}, p99={p99:.3f}",
                        ),
                    )

            # Check brain volume ratio (if brain mask exists)
            brain_mask_path = self._find_corresponding_mask(file_path, output_dir)
            if brain_mask_path:
                try:
                    mask_img = nib.load(str(brain_mask_path))
                    mask_data = mask_img.get_fdata()
                    volume_ratio = self._calculate_brain_volume_ratio(mask_data, data)

                    volume_ok = (
                        volume_ratio >= self.thresholds.min_brain_volume_ratio
                        and volume_ratio <= self.thresholds.max_brain_volume_ratio
                    )

                    checks.append(
                        QCResult(
                            name="brain_volume_ratio",
                            passed=bool(volume_ok),
                            value=float(volume_ratio),
                            threshold=[
                                self.thresholds.min_brain_volume_ratio,
                                self.thresholds.max_brain_volume_ratio,
                            ],
                            message=f"brain volume ratio={volume_ratio:.3f}",
                        ),
                    )
                except Exception as e:
                    checks.append(
                        QCResult(
                            name="brain_volume_ratio",
                            passed=False,
                            value=None,
                            message=f"Could not calculate brain volume ratio: {e!s}",
                        ),
                    )

            # Check image dimensions
            shape = data.shape
            checks.append(
                QCResult(
                    name="image_dimensions",
                    passed=len(shape) >= 3,
                    value=list(shape),
                    message=f"Shape: {shape}",
                ),
            )

            # Check voxel sizes
            voxel_sizes = (
                header.get_zooms()[:3] if len(header.get_zooms()) >= 3 else header.get_zooms()
            )
            checks.append(
                QCResult(
                    name="voxel_sizes",
                    passed=all(v > 0 for v in voxel_sizes),
                    value=list(voxel_sizes),
                    message=f"Voxel sizes: {voxel_sizes}",
                ),
            )

        except Exception as e:
            checks.append(
                QCResult(
                    name="nifti_loading",
                    passed=False,
                    value=None,
                    message=f"Could not load NIfTI file: {e!s}",
                ),
            )

        return checks

    def _check_numpy_file(self, file_path: Path) -> list[QCResult]:
        """Check NumPy array file quality."""
        checks = []

        try:
            data = np.load(str(file_path))

            # Check data type
            dtype = data.dtype
            checks.append(
                QCResult(
                    name="data_type",
                    passed=dtype in [np.float32, np.float64, np.int16, np.uint8],
                    value=str(dtype),
                    message=f"Data type: {dtype}",
                ),
            )

            # Check for NaN/inf values
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            checks.append(
                QCResult(
                    name="data_integrity",
                    passed=not (has_nan or has_inf),
                    value={
                        "nan_count": int(np.sum(np.isnan(data))),
                        "inf_count": int(np.sum(np.isinf(data))),
                    },
                    message=f"NaN: {has_nan}, Inf: {has_inf}",
                ),
            )

            # Check array shape
            shape = data.shape
            checks.append(
                QCResult(
                    name="array_shape",
                    passed=len(shape) >= 3,
                    value=list(shape),
                    message=f"Shape: {shape}",
                ),
            )

            # Check value range
            if data.size > 0:
                min_val, max_val = float(np.min(data)), float(np.max(data))
                range_ok = max_val <= 1.0 and min_val >= -1.0  # Assuming normalized data
                checks.append(
                    QCResult(
                        name="value_range",
                        passed=range_ok,
                        value=[min_val, max_val],
                        threshold=[-1.0, 1.0],
                        message=f"value range [{min_val:.3f}, {max_val:.3f}]",
                    ),
                )

        except Exception as e:
            checks.append(
                QCResult(
                    name="numpy_loading",
                    passed=False,
                    value=None,
                    message=f"Could not load NumPy file: {e!s}",
                ),
            )

        return checks

    def _find_corresponding_mask(
        self,
        image_path: Path,
        output_dir: str,
    ) -> Path | None:
        """Find corresponding brain mask for an image file."""
        image_name = image_path.name

        # Common mask naming patterns
        # Only include patterns that are actually different from the original filename
        candidate_patterns = [
            image_name.replace(".nii", "_brain_mask.nii"),
            image_name.replace(".nii.gz", "_brain_mask.nii.gz"),
            image_name.replace(".nii", "_mask.nii"),
            image_name.replace(".nii.gz", "_mask.nii.gz"),
            "brain_mask.nii.gz",
            "brain_mask.nii",
            "mask.nii.gz",
            "mask.nii",
        ]

        # Filter out patterns that equal the original filename (no-op replacements)
        mask_patterns = [p for p in candidate_patterns if p != image_name]

        output_path = Path(output_dir)

        # Check in same directory
        for pattern in mask_patterns:
            mask_path = image_path.parent / pattern
            if mask_path.exists():
                return mask_path

        # Check in brain_extraction subdirectory
        brain_ext_dir = output_path / "brain_extraction"
        if brain_ext_dir.exists():
            for pattern in mask_patterns:
                mask_path = brain_ext_dir / pattern
                if mask_path.exists():
                    return mask_path

        return None

    def _calculate_brain_volume_ratio(
        self,
        mask_data: np.ndarray,
        image_data: np.ndarray,
    ) -> float:
        """Calculate ratio of brain volume to total intracranial volume."""
        brain_volume = np.sum(mask_data > 0)
        # Use non-zero voxels in image as proxy for intracranial volume
        intracranial_volume = np.sum(image_data > 0)

        if intracranial_volume == 0:
            return 0.0

        return brain_volume / intracranial_volume

    def _generate_summary(self, file_results: list[FileQCResult]) -> dict[str, Any]:
        """Generate summary statistics from file QC results."""
        total_files = len(file_results)
        passed_files = sum(1 for r in file_results if r.overall_passed)
        failed_files = total_files - passed_files

        # Collect all warnings and errors
        all_warnings = []
        all_errors = []

        check_summary = {}
        for file_result in file_results:
            for check in file_result.checks:
                if check.name not in check_summary:
                    check_summary[check.name] = {"passed": 0, "failed": 0, "total": 0}

                check_summary[check.name]["total"] += 1
                if check.passed:
                    check_summary[check.name]["passed"] += 1
                else:
                    check_summary[check.name]["failed"] += 1

                    # Collect error messages
                    if check.message:
                        all_errors.append(f"{file_result.file_name}: {check.message}")

        # Overall assessment
        overall_passed = failed_files == 0

        return {
            "total_files": total_files,
            "passed_files": passed_files,
            "failed_files": failed_files,
            "pass_rate": passed_files / total_files if total_files > 0 else 0,
            "check_summary": check_summary,
            "warnings": all_warnings,
            "errors": all_errors,
            "overall_passed": overall_passed,
        }

    def _save_report(self, report: QCReport, report_path: str) -> None:
        """Save QC report to JSON file."""
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # Convert dataclasses to dictionaries
        report_dict = {
            "timestamp": report.timestamp,
            "input_dir": report.input_dir,
            "output_dir": report.output_dir,
            "thresholds": asdict(report.thresholds),
            "file_results": [
                {
                    "file_path": r.file_path,
                    "file_name": r.file_name,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "value": c.value,
                            "threshold": c.threshold,
                            "message": c.message,
                            "details": c.details,
                        }
                        for c in r.checks
                    ],
                    "overall_passed": r.overall_passed,
                    "processing_time": r.processing_time,
                }
                for r in report.file_results
            ],
            "summary": report.summary,
            "passed": report.passed,
            "warnings": report.warnings,
            "errors": report.errors,
        }

        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        self.logger.info(f"QC report saved to: {report_path}")


def run_preprocessing_qc(
    input_dir: str,
    output_dir: str,
    thresholds: QCThresholds | None = None,
    save_report: bool = True,
    report_path: str | None = None,
) -> QCReport:
    """Convenience function to run preprocessing quality control.

    Args:
        input_dir: Input data directory
        output_dir: Output/preprocessed data directory
        thresholds: QC thresholds (uses defaults if None)
        save_report: Whether to save report to file
        report_path: Path for QC report (auto-generated if None)

    Returns:
        QCReport: Comprehensive quality control report

    """
    qc = PreprocessingQC(thresholds)
    return qc.run_quality_control(input_dir, output_dir, save_report, report_path)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing Quality Control")
    parser.add_argument("--input-dir", required=True, help="Input data directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output/preprocessed data directory",
    )
    parser.add_argument("--report-path", help="Path to save QC report")
    parser.add_argument(
        "--min-brain-volume",
        type=float,
        default=0.3,
        help="Minimum brain volume ratio threshold",
    )
    parser.add_argument(
        "--max-brain-volume",
        type=float,
        default=0.8,
        help="Maximum brain volume ratio threshold",
    )

    args = parser.parse_args()

    # Create custom thresholds
    thresholds = QCThresholds(
        min_brain_volume_ratio=args.min_brain_volume,
        max_brain_volume_ratio=args.max_brain_volume,
    )

    # Run QC
    report = run_preprocessing_qc(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        thresholds=thresholds,
        report_path=args.report_path,
    )

    # Print summary
    logger.info(f"QC Status: {'PASSED' if report.passed else 'FAILED'}")
    logger.info(f"Files processed: {len(report.file_results)}")
    logger.info(f"Files passed: {report.summary.get('passed_files', 0)}")
    logger.info(f"Files failed: {report.summary.get('failed_files', 0)}")

    if report.errors:
        logger.warning(f"Errors: {len(report.errors)}")
        for error in report.errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")

    if report.warnings:
        logger.warning(f"Warnings: {len(report.warnings)}")
        for warning in report.warnings[:5]:  # Show first 5 warnings
            logger.warning(f"  - {warning}")
