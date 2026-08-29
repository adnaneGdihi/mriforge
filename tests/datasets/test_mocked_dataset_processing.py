"""Tests for dataset processing with mocked cluster datasets."""

from pathlib import Path

import pytest


class TestMockedDatasetProcessing:
    """Test dataset processing with mocked cluster datasets."""

    def test_brats_sr_loader_with_mock_data(self, mock_brats_sr_png):
        """Test BratsSrLoader path validation with mocked PNG data."""

        # Just verify the path exists and has expected structure
        path = Path(mock_brats_sr_png)
        assert path.exists()
        assert path.is_dir()

        # Check for PNG files
        png_files = list(path.rglob("*.png"))
        assert len(png_files) > 0, "No PNG files found in brats_sr mock data"

    def test_fastmri_hdf5_loader_with_mock_data(self, mock_fastmri_hdf5):
        """Test FastMRILoader path validation with mocked HDF5 data."""

        path = Path(mock_fastmri_hdf5)
        assert path.exists()
        assert path.is_dir()

        # Check for HDF5 files
        hdf5_files = list(path.rglob("*.h5"))
        assert len(hdf5_files) > 0, "No HDF5 files found in fastmri mock data"

    def test_fastmri_dicom_loader_with_mock_data(self, mock_fastmri_dicom):
        """Test FastMRILoader path validation with mocked DICOM data."""

        path = Path(mock_fastmri_dicom)
        assert path.exists()
        assert path.is_dir()

        # Check for DICOM files
        dicom_files = list(path.rglob("*.dcm"))
        assert len(dicom_files) > 0, "No DICOM files found in fastmri mock data"

    def test_hcp_loader_with_mock_data(self, mock_hcp_nifti):
        """Test HCPLoader path validation with mocked NIfTI data."""

        path = Path(mock_hcp_nifti)
        assert path.exists()
        assert path.is_dir()

        # Check for NIfTI files
        nifti_files = list(path.rglob("*.nii.gz"))
        assert len(nifti_files) > 0, "No NIfTI files found in hcp mock data"

    def test_m4raw_loader_with_mock_data(self, mock_m4raw_h5):
        """Test M4RawLoader path validation with mocked HDF5 data."""

        path = Path(mock_m4raw_h5)
        assert path.exists()
        assert path.is_dir()

        # Check for HDF5 files
        hdf5_files = list(path.rglob("*.h5"))
        assert len(hdf5_files) > 0, "No HDF5 files found in m4raw mock data"

    def test_ulf_paired_loader_with_mock_data(self, mock_ulf_paired_nifti):
        """Test UlfPairedLoader path validation with mocked NIfTI data."""

        path = Path(mock_ulf_paired_nifti)
        assert path.exists()
        assert path.is_dir()

        # Check for NIfTI files
        nifti_files = list(path.rglob("*.nii.gz"))
        assert len(nifti_files) > 0, "No NIfTI files found in ulf_paired mock data"

    def test_dataset_path_validation(self, mock_cluster_datasets):
        """Test that dataset paths are valid and accessible."""
        datasets = {
            "brats_sr": mock_cluster_datasets / "brats_sr",
            "fastmri": mock_cluster_datasets / "fastmri",
            "hcp": mock_cluster_datasets / "hcp",
            "m4raw": mock_cluster_datasets / "m4raw",
            "ulf_paired": mock_cluster_datasets / "ulf_paired",
        }

        for name, path in datasets.items():
            dataset_path = path
            assert dataset_path.exists(), f"Dataset {name} path does not exist: {path}"
            assert (
                dataset_path.is_dir()
            ), f"Dataset {name} path is not a directory: {path}"

            # Check that directory is not empty
            contents = list(dataset_path.rglob("*"))
            assert len(contents) > 0, f"Dataset {name} directory is empty: {path}"

    def test_file_extensions_match_expected(self, mock_cluster_datasets):
        """Test that files have expected extensions for each dataset type."""
        # Define expected extensions for each dataset (including compound extensions)
        expected_extensions = {
            "brats_sr": [".png"],
            "fastmri": [".h5", ".dcm"],
            "hcp": [".nii.gz", ".json"],
            "m4raw": [".h5"],
            "ulf_paired": [".nii.gz", ".json", ".bval", ".bvec"],
        }

        base_path = mock_cluster_datasets

        for dataset_name, extensions in expected_extensions.items():
            dataset_path = base_path / dataset_name
            assert dataset_path.exists(), f"Dataset {dataset_name} not found"

            # Find all files in the dataset
            all_files = list(dataset_path.rglob("*"))
            data_files = [f for f in all_files if f.is_file()]

            assert len(data_files) > 0, f"No data files found in {dataset_name}"

            # Check that all files have expected extensions
            for file_path in data_files:
                # For compound extensions, check the full suffix
                full_suffix = (
                    "".join(file_path.suffixes)
                    if file_path.suffixes
                    else file_path.suffix
                )
                assert (
                    full_suffix in extensions
                ), f"Unexpected file extension in {dataset_name}: {file_path} (got {full_suffix})"

    def test_bids_structure_compliance(self, mock_ulf_paired_nifti):
        """Test that ulf_paired follows BIDS naming conventions."""
        pytest.skip("Mock fixtures not implemented")
        ulf_path = Path(mock_ulf_paired_nifti)

        # Check 3T_data structure
        t3_path = ulf_path / "ulf_paired_64mt_3t" / "Data" / "3T_data"
        assert t3_path.exists()

        # Find all NIfTI files and check BIDS naming
        nii_files = list(t3_path.rglob("*.nii.gz"))
        for nii_file in nii_files:
            filename = nii_file.name
            # Should follow BIDS pattern: sub-XX[_ses-XX][_acq-XX][_run-XX]_mod.nii.gz
            assert filename.startswith("sub-"), f"Non-BIDS filename: {filename}"
            assert (
                "_T1w" in filename
                or "_T2w" in filename
                or "_FLAIR" in filename
                or "_adc" in filename
                or "_dwi" in filename
            ), f"Unexpected modality in filename: {filename}"

        # Check 64mT_data structure
        mt64_path = ulf_path / "ulf_paired_64mt_3t" / "Data" / "64mT_data"
        assert mt64_path.exists()

        nii_files_64mt = list(mt64_path.rglob("*.nii.gz"))
        for nii_file in nii_files_64mt:
            filename = nii_file.name
            assert filename.startswith("sub-"), f"Non-BIDS filename: {filename}"
            assert "ses-" in filename, f"Missing session in filename: {filename}"
        """Test BratsSrLoader path validation with mocked PNG data."""
        # Just verify the path exists and has expected structure
        path = Path(mock_brats_sr_png)
        assert path.exists()
        assert path.is_dir()

        # Check for PNG files
        png_files = list(path.rglob("*.png"))
        assert len(png_files) > 0, "No PNG files found in brats_sr mock data"

    def test_fastmri_hdf5_loader_with_mock_data(self, mock_fastmri_hdf5):
        """Test FastMRILoader path validation with mocked HDF5 data."""
        path = Path(mock_fastmri_hdf5)
        assert path.exists()
        assert path.is_dir()

        # Check for HDF5 files
        hdf5_files = list(path.rglob("*.h5"))
        assert len(hdf5_files) > 0, "No HDF5 files found in fastmri mock data"

    def test_fastmri_dicom_loader_with_mock_data(self, mock_fastmri_dicom):
        """Test FastMRILoader path validation with mocked DICOM data."""
        path = Path(mock_fastmri_dicom)
        assert path.exists()
        assert path.is_dir()

        # Check for DICOM files
        dicom_files = list(path.rglob("*.dcm"))
        assert len(dicom_files) > 0, "No DICOM files found in fastmri mock data"

    def test_hcp_loader_with_mock_data(self, mock_hcp_nifti):
        """Test HCPLoader path validation with mocked NIfTI data."""
        path = Path(mock_hcp_nifti)
        assert path.exists()
        assert path.is_dir()

        # Check for NIfTI files
        nifti_files = list(path.rglob("*.nii.gz"))
        assert len(nifti_files) > 0, "No NIfTI files found in hcp mock data"

    def test_m4raw_loader_with_mock_data(self, mock_m4raw_h5):
        """Test M4RawLoader path validation with mocked HDF5 data."""
        path = Path(mock_m4raw_h5)
        assert path.exists()
        assert path.is_dir()

        # Check for HDF5 files
        hdf5_files = list(path.rglob("*.h5"))
        assert len(hdf5_files) > 0, "No HDF5 files found in m4raw mock data"

    def test_ulf_paired_loader_with_mock_data(self, mock_ulf_paired_nifti):
        """Test UlfPairedLoader path validation with mocked NIfTI data."""
        path = Path(mock_ulf_paired_nifti)
        assert path.exists()
        assert path.is_dir()

        # Check for NIfTI files
        nifti_files = list(path.rglob("*.nii.gz"))
        assert len(nifti_files) > 0, "No NIfTI files found in ulf_paired mock data"

    def test_dataset_path_validation(self, mock_cluster_datasets):
        """Test that dataset paths are valid and accessible."""
        datasets = {
            "brats_sr": mock_cluster_datasets / "brats_sr",
            "fastmri": mock_cluster_datasets / "fastmri",
            "hcp": mock_cluster_datasets / "hcp",
            "m4raw": mock_cluster_datasets / "m4raw",
            "ulf_paired": mock_cluster_datasets / "ulf_paired",
        }

        for name, path in datasets.items():
            dataset_path = path
            assert dataset_path.exists(), f"Dataset {name} path does not exist: {path}"
            assert (
                dataset_path.is_dir()
            ), f"Dataset {name} path is not a directory: {path}"

            # Check that directory is not empty
            contents = list(dataset_path.rglob("*"))
            assert len(contents) > 0, f"Dataset {name} directory is empty: {path}"

    def test_file_extensions_match_expected(self, mock_cluster_datasets):
        """Test that files have expected extensions for each dataset type."""
        # Define expected extensions for each dataset (including compound extensions)
        expected_extensions = {
            "brats_sr": [".png"],
            "fastmri": [".h5", ".dcm"],
            "hcp": [".nii.gz", ".json"],
            "m4raw": [".h5"],
            "ulf_paired": [".nii.gz", ".json", ".bval", ".bvec"],
        }

        base_path = mock_cluster_datasets

        for dataset_name, extensions in expected_extensions.items():
            dataset_path = base_path / dataset_name
            assert dataset_path.exists(), f"Dataset {dataset_name} not found"

            # Find all files in the dataset
            all_files = list(dataset_path.rglob("*"))
            data_files = [f for f in all_files if f.is_file()]

            assert len(data_files) > 0, f"No data files found in {dataset_name}"

            # Check that all files have expected extensions
            for file_path in data_files:
                # For compound extensions, check the full suffix
                full_suffix = (
                    "".join(file_path.suffixes)
                    if file_path.suffixes
                    else file_path.suffix
                )
                assert (
                    full_suffix in extensions
                ), f"Unexpected file extension in {dataset_name}: {file_path} (got {full_suffix})"

    def test_bids_structure_compliance(self, mock_ulf_paired_nifti):
        """Test that ulf_paired follows BIDS naming conventions."""
        ulf_path = Path(mock_ulf_paired_nifti)

        # Check 3T_data structure
        t3_path = ulf_path / "ulf_paired_64mt_3t" / "Data" / "3T_data"
        assert t3_path.exists()

        # Find all NIfTI files and check BIDS naming
        nii_files = list(t3_path.rglob("*.nii.gz"))
        for nii_file in nii_files:
            filename = nii_file.name
            # Should follow BIDS pattern: sub-XX[_ses-XX][_acq-XX][_run-XX]_mod.nii.gz
            assert filename.startswith("sub-"), f"Non-BIDS filename: {filename}"
            assert (
                "_T1w" in filename
                or "_T2w" in filename
                or "_FLAIR" in filename
                or "_adc" in filename
                or "_dwi" in filename
            ), f"Unexpected modality in filename: {filename}"

        # Check 64mT_data structure
        mt64_path = ulf_path / "ulf_paired_64mt_3t" / "Data" / "64mT_data"
        assert mt64_path.exists()

        nii_files_64mt = list(mt64_path.rglob("*.nii.gz"))
        for nii_file in nii_files_64mt:
            filename = nii_file.name
            assert filename.startswith("sub-"), f"Non-BIDS filename: {filename}"
            assert "ses-" in filename, f"Missing session in filename: {filename}"
