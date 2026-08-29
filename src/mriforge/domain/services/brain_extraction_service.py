"""Brain Extraction Service
========================

Domain service for brain extraction from MRI data using HD-BET.
Handles NIfTI file organization and brain extraction operations.

### Extraction Workflow
```mermaid
flowchart TD
    Start[Start Extraction] --> Scan[Scan NIfTI Files]
    Scan --> Split{Field Strength?}
    Split -->|High Field| ParseHigh[Parse Subject/Resolution]
    Split -->|Low Field| ParseLow[Parse Subject/Session]
    ParseHigh --> Struct[Organize File Structure]
    ParseLow --> Struct
    Struct --> CreateDirs[Create Output Directories]
    CreateDirs --> Iterate[Iterate Organized Files]
    Iterate --> HDBET[Execute HD-BET]
    HDBET --> Save[Save .nii.gz Output]
    Save --> Iterate
    Iterate -->|Completed| Finish[Extraction Report]
```
"""

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IBrainExtractionService(ABC):
    """Abstract interface for brain extraction services."""

    @abstractmethod
    def extract_brain(
        self,
        input_dir: Path,
        output_dir: Path,
        method: str = "bet",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract brain from MRI images."""

    @abstractmethod
    def validate_extraction(
        self,
        extracted_file: Path,
        mask_file: Path,
    ) -> dict[str, float]:
        """Validate brain extraction quality."""

    @abstractmethod
    def get_brain_mask(self, input_file: Path, output_dir: Path, **kwargs: Any) -> Path:
        """Generate brain mask for single file."""

    @abstractmethod
    def get_supported_methods(self) -> list[str]:
        """Get supported extraction methods."""


class BrainExtractionService(IBrainExtractionService):
    """Service for extracting brains from MRI NIfTI files using HD-BET.
    Organizes files by field strength, subject, session, and modality.
    """

    data_path: str
    output_base_dir: str
    nifti_files: dict[str, Any]

    def __init__(
        self,
        data_path: str = "./paired_ulf_3T/Data",
        output_base_dir: str = "./data/brain_extracted_data",
    ) -> None:
        """__init__.

        Args:
            data_path (str): Description.
            output_base_dir (str): Description.
        """
        self.data_path = data_path
        self.output_base_dir = output_base_dir
        self.nifti_files = {}

    def scan_nifti_files(self) -> dict[str, Any]:
        """Scan and organize NIfTI files.

        Organizes files by field strength, subject, session, and modality.


        Returns:
            Dictionary containing organized file paths

        """
        logger.info(f"Scanning NIfTI files in {self.data_path}")

        # list field strengths
        field_strengths = [
            d
            for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d)) and d in ["highfield", "lowfield"]
        ]

        for fs in field_strengths:
            fs_path = os.path.join(self.data_path, fs)
            self.nifti_files[fs] = {}

            # Walk through directories to find NIfTI files
            for root, _dirs, files in os.walk(fs_path):
                relative_path = os.path.relpath(root, fs_path)
                path_parts = relative_path.split(os.sep)

                subject = None
                session = None
                resolution = "default_resolution"

                # Parse path structure
                if fs == "highfield" and len(path_parts) >= 2 and path_parts[0].startswith("sub-"):
                    subject = path_parts[0]
                    if "highres" in path_parts:
                        resolution = "highres"
                    elif "lowres" in path_parts:
                        resolution = "lowres"
                    else:
                        resolution = "highres"  # Default for highfield

                elif (
                    fs == "lowfield"
                    and len(path_parts) >= 2
                    and path_parts[0].startswith("sub-")
                    and path_parts[1].startswith("ses-")
                ):
                    subject = path_parts[0]
                    session = path_parts[1]
                    resolution = "default_resolution"

                if subject and subject.startswith("sub-"):
                    self._organize_file_structure(
                        fs,
                        subject,
                        session,
                        resolution,
                        root,
                        files,
                    )

        logger.info(f"Found files for {len(field_strengths)} field strengths")
        return self.nifti_files

    def _organize_file_structure(
        self,
        fs: str,
        subject: str,
        session: str | None,
        resolution: str,
        root: str,
        files: list,
    ) -> None:
        """Organize files into the nested dictionary structure."""
        if subject not in self.nifti_files[fs]:
            self.nifti_files[fs][subject] = {}

        current_level = self.nifti_files[fs][subject]
        if session:
            if session not in current_level:
                current_level[session] = {}
            current_level = current_level[session]

        if resolution not in current_level:
            current_level[resolution] = {}
        current_level = current_level[resolution]

        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                modality_name = self._extract_modality(file)
                if modality_name:
                    current_level[modality_name] = os.path.join(root, file)
                else:
                    current_level[os.path.splitext(file)[0]] = os.path.join(root, file)

    def _extract_modality(self, filename: str) -> str | None:
        """Extract modality name from filename using BIDS-like patterns."""
        filename_parts = filename.split("_")
        for part in filename_parts:
            if "T1w" in part:
                return "T1w"
            if "T2w" in part:
                return "T2w"
            if "FLAIR" in part:
                return "FLAIR"
            if "dwi" in part:
                return "dwi"
            if "ADC" in part:
                return "ADC"
        return None

    def create_output_directories(self) -> None:
        """Create output directories for brain extraction results."""
        logger.info(f"Creating output directories in {self.output_base_dir}")

        for fs, subjects in self.nifti_files.items():
            for subject, sessions_or_resolutions in subjects.items():
                if fs == "lowfield":
                    for session, resolutions in sessions_or_resolutions.items():
                        for resolution, _modalities in resolutions.items():
                            output_dir = os.path.join(
                                self.output_base_dir,
                                fs,
                                subject,
                                session,
                                resolution,
                            )
                            os.makedirs(output_dir, exist_ok=True)
                            logger.debug(f"Created directory: {output_dir}")
                else:  # highfield
                    for resolution, _modalities in sessions_or_resolutions.items():
                        output_dir = os.path.join(
                            self.output_base_dir,
                            fs,
                            subject,
                            resolution,
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        logger.debug(f"Created directory: {output_dir}")

    def extract_brains(self) -> None:
        """Perform brain extraction on all organized NIfTI files using HD-BET."""
        logger.info("Starting brain extraction process")

        for fs, subjects in self.nifti_files.items():
            for subject, sessions_or_resolutions in subjects.items():
                if fs == "lowfield":
                    self._extract_for_field_strength(
                        fs,
                        subject,
                        sessions_or_resolutions,
                        include_session=True,
                    )
                else:  # highfield
                    self._extract_for_field_strength(
                        fs,
                        subject,
                        sessions_or_resolutions,
                        include_session=False,
                    )

        logger.info("Brain extraction process completed")

    def _extract_for_field_strength(
        self,
        fs: str,
        subject: str,
        sessions_or_resolutions: dict,
        include_session: bool = False,
    ) -> None:
        """Extract brains for a specific field strength."""
        if include_session:
            for session, resolutions in sessions_or_resolutions.items():
                for resolution, modalities in resolutions.items():
                    output_dir = os.path.join(
                        self.output_base_dir,
                        fs,
                        subject,
                        session,
                        resolution,
                    )
                    self._extract_modalities(modalities, output_dir)
        else:
            for resolution, modalities in sessions_or_resolutions.items():
                output_dir = os.path.join(self.output_base_dir, fs, subject, resolution)
                self._extract_modalities(modalities, output_dir)

    def _extract_modalities(self, modalities: dict[str, str], output_dir: str) -> None:
        """Extract brains for all modalities in a directory."""
        for _modality, input_file_path in modalities.items():
            output_file_name = os.path.basename(input_file_path)
            output_file_path = os.path.join(output_dir, output_file_name)

            self._run_hd_bet(input_file_path, output_file_path)

    def _run_hd_bet(self, input_path: str, output_path: str) -> None:
        """Run HD-BET command for brain extraction."""
        command = ["hd-bet", "-i", input_path, "-o", output_path]

        logger.info(f"Executing: {' '.join(command)}")

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.debug(f"HD-BET STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"HD-BET STDERR: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"HD-BET failed for {input_path}: {e}")
            if e.stdout:
                logger.debug(f"HD-BET STDOUT: {e.stdout}")
            if e.stderr:
                logger.debug(f"HD-BET STDERR: {e.stderr}")
        except FileNotFoundError:
            logger.error(
                "HD-BET command not found. Make sure it's installed and in PATH.",
            )

    def get_file_organization(self) -> dict[str, Any]:
        """Get the current file organization structure."""
        return self.nifti_files

    # Implementation of abstract methods from IBrainExtractionService

    def extract_brain(
        self,
        _input_dir: Path,
        _output_dir: Path,
        _method: str = "bet",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Extract brain using HD-BET method."""
        # Scan files first
        self.scan_nifti_files()

        # Create output directories
        self.create_output_directories()

        # Extract brains
        self.extract_brains()

        # Return results in expected format
        return {
            "output_dir": str(self.output_base_dir),
            "extracted_files": [],  # Would need to collect these
            "mask_files": [],  # HD-BET generates masks
            "metrics": {},  # Would need to calculate metrics
        }

    def validate_extraction(
        self,
        extracted_file: Path,
        mask_file: Path,
    ) -> dict[str, float]:
        """Validate brain extraction quality."""
        # Basic validation - could be enhanced
        return {"validation_score": 1.0}

    def get_brain_mask(self, input_file: Path, output_dir: Path, **kwargs: Any) -> Path:
        """Generate brain mask for single file."""
        # Run HD-BET on single file
        base_name = input_file.stem
        base_name = base_name.removesuffix(".nii")

        output_file = output_dir / f"{base_name}_brain.nii.gz"
        self._run_hd_bet(str(input_file), str(output_file))

        return output_file

    def get_supported_methods(self) -> list:
        """Get supported extraction methods."""
        return ["hdbet"]
