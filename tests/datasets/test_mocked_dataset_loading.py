"""Tests for mocked dataset loading using pyfakefs fixtures."""

from pathlib import Path


class TestMockedDatasetLoading:
    """Test dataset loading with mocked cluster datasets."""

    def test_brats_sr_png_files_exist(self, mock_brats_sr_png):
        """Test that brats_sr PNG files are created correctly."""

        brats_path = Path(mock_brats_sr_png)

        assert brats_path.exists()
        assert brats_path.is_dir()

        # Check patients exist
        patients = ["BRATS_001", "BRATS_002", "BRATS_003"]
        for patient in patients:
            patient_dir = brats_path / patient
            assert patient_dir.exists()
            assert patient_dir.is_dir()

            # Check PNG files exist
            png_files = list(patient_dir.glob("*.png"))
            assert len(png_files) == 10  # 10 slices per patient

            for png_file in png_files:
                assert png_file.suffix == ".png"
                assert png_file.stat().st_size > 0  # Has content

    def test_fastmri_hdf5_files_exist(self, mock_fastmri_hdf5):
        """Test that fastmri HDF5 files are created correctly."""

        fastmri_path = Path(mock_fastmri_hdf5)

        assert fastmri_path.exists()

        # Check HDF5 files exist
        expected_files = [
            "brain_multicoil_train/file001.h5",
            "brain_multicoil_val/file002.h5",
            "knee_multicoil_train/file003.h5",
            "knee_multicoil_val/file004.h5",
        ]

        for file_path in expected_files:
            full_path = fastmri_path / file_path
            assert full_path.exists()
            assert full_path.suffix == ".h5"
            assert full_path.stat().st_size > 0

    def test_fastmri_dicom_structure(self, mock_fastmri_dicom):
        """Test that fastmri DICOM structure is created correctly."""

        dicom_path = Path(mock_fastmri_dicom)

        assert dicom_path.exists()

        patients = ["patient001", "patient002"]
        for patient in patients:
            patient_dir = dicom_path / patient
            assert patient_dir.exists()

            for scan in ["scan01", "scan02"]:
                scan_dir = patient_dir / scan
                assert scan_dir.exists()

                for seq in ["T1", "T2"]:
                    seq_dir = scan_dir / seq
                    assert seq_dir.exists()

                    # Check DICOM files
                    dcm_files = list(seq_dir.glob("*.dcm"))
                    assert len(dcm_files) == 5  # 5 slices per sequence

                    for dcm_file in dcm_files:
                        assert dcm_file.suffix == ".dcm"
                        assert dcm_file.stat().st_size > 0

    def test_hcp_nifti_structure(self, mock_hcp_nifti):
        """Test that hcp NIfTI structure is created correctly."""

        hcp_path = Path(mock_hcp_nifti)

        assert hcp_path.exists()

        # Check nested structure
        resampled_path = hcp_path / "slice_sequences_image" / "resampled_image"
        assert resampled_path.exists()

        metadata_dir = resampled_path / "metadata"
        assert metadata_dir.exists()

        resampled_data_dir = resampled_path / "resampled"
        assert resampled_data_dir.exists()

        # Check subjects
        subjects = ["100206", "100307", "100408"]
        for subject in subjects:
            subject_dir = resampled_data_dir / subject
            assert subject_dir.exists()

            modalities = ["T1w", "T2w", "dwi"]
            for mod in modalities:
                nii_file = subject_dir / f"{subject}_{mod}.nii.gz"
                json_file = subject_dir / f"{subject}_{mod}.json"

                assert nii_file.exists()
                assert nii_file.stat().st_size > 0
                assert json_file.exists()
                assert json_file.stat().st_size > 0

    def test_m4raw_h5_structure(self, mock_m4raw_h5):
        """Test that m4raw HDF5 structure is created correctly."""

        m4raw_path = Path(mock_m4raw_h5)

        assert m4raw_path.exists()

        subdirs = ["gre_data", "motion", "multicoil_test", "multicoil_train"]
        for subdir in subdirs:
            subdir_path = m4raw_path / subdir
            assert subdir_path.exists()

            # Check HDF5 files
            h5_files = list(subdir_path.glob("*.h5"))
            assert len(h5_files) == 5  # 5 files per subdir

            for h5_file in h5_files:
                assert h5_file.suffix == ".h5"
                assert h5_file.stat().st_size > 0

    def test_ulf_paired_3t_structure(self, mock_ulf_paired_nifti):
        """Test that ulf_paired 3T_data BIDS structure is created correctly."""

        ulf_path = Path(mock_ulf_paired_nifti)
        t3_path = ulf_path / "ulf_paired_64mt_3t" / "Data" / "3T_data"

        assert t3_path.exists()

        # Check dataset description
        desc_file = t3_path / "dataset_description.json"
        assert desc_file.exists()
        assert desc_file.stat().st_size > 0

        # Check subjects
        subjects_3t = ["0011", "0015", "0023", "0025", "0027"]
        for subj in subjects_3t:
            subj_dir = t3_path / f"sub-{subj}"
            assert subj_dir.exists()

            # Check anat directory
            anat_dir = subj_dir / "anat"
            assert anat_dir.exists()

            modalities = ["FLAIR", "T1w", "T2w"]
            for mod in modalities:
                # High-res files
                hr_nii = anat_dir / f"sub-{subj}_acq-highres_{mod}.nii.gz"
                hr_json = anat_dir / f"sub-{subj}_acq-highres_{mod}.json"
                assert hr_nii.exists()
                assert hr_json.exists()

                # Low-res files
                lr_nii = anat_dir / f"sub-{subj}_acq-lowres_{mod}.nii.gz"
                lr_json = anat_dir / f"sub-{subj}_acq-lowres_{mod}.json"
                assert lr_nii.exists()
                assert lr_json.exists()

            # Check dwi directory
            dwi_dir = subj_dir / "dwi"
            assert dwi_dir.exists()

            # Check DWI files
            adc_nii = dwi_dir / f"sub-{subj}_acq-highres_adc.nii.gz"
            adc_json = dwi_dir / f"sub-{subj}_acq-highres_adc.json"
            dwi_bval = dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.bval"
            dwi_bvec = dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.bvec"
            dwi_nii = dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.nii.gz"
            dwi_json = dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.json"

            assert adc_nii.exists()
            assert adc_json.exists()
            assert dwi_bval.exists()
            assert dwi_bvec.exists()
            assert dwi_nii.exists()
            assert dwi_json.exists()

    def test_ulf_paired_64mt_structure(self, mock_ulf_paired_nifti):
        """Test that ulf_paired 64mT_data BIDS structure is created correctly."""

        ulf_path = Path(mock_ulf_paired_nifti)
        mt64_path = ulf_path / "ulf_paired_64mt_3t" / "Data" / "64mT_data"

        assert mt64_path.exists()

        # Check dataset description
        desc_file = mt64_path / "dataset_description.json"
        assert desc_file.exists()

        # Check subjects
        subjects_64mt = ["0001", "0002", "0003", "0004", "0005"]
        for subj in subjects_64mt:
            subj_dir = mt64_path / f"sub-{subj}"
            assert subj_dir.exists()

            # Determine sessions for this subject
            sessions = ["01", "02"] if subj in ["0001", "0012", "0015"] else ["01"]

            for ses in sessions:
                ses_dir = subj_dir / f"ses-{ses}"
                assert ses_dir.exists()

                # Check anat directory
                anat_dir = ses_dir / "anat"
                assert anat_dir.exists()

                modalities = ["FLAIR", "T1w", "T2w"]
                for mod in modalities:
                    nii_file = anat_dir / f"sub-{subj}_ses-{ses}_run-1_{mod}.nii.gz"
                    json_file = anat_dir / f"sub-{subj}_ses-{ses}_run-1_{mod}.json"
                    assert nii_file.exists()
                    assert json_file.exists()

                    # Additional localizer for T1w
                    if mod == "T1w":
                        loc_nii = (
                            anat_dir
                            / f"sub-{subj}_ses-{ses}_run-1_{mod}_acq-localizer.nii.gz"
                        )
                        loc_json = (
                            anat_dir
                            / f"sub-{subj}_ses-{ses}_run-1_{mod}_acq-localizer.json"
                        )
                        assert loc_nii.exists()
                        assert loc_json.exists()

                # Check dwi directory
                dwi_dir = ses_dir / "dwi"
                assert dwi_dir.exists()

                # Check DWI files
                # All subjects have at least run-1
                for run in ["1"]:
                    adc_nii = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_ADC.nii.gz"
                    adc_json = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_ADC.json"
                    dwi_bval = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.bval"
                    dwi_bvec = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.bvec"
                    dwi_nii = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.nii.gz"
                    dwi_json = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.json"

                    assert adc_nii.exists()
                    assert adc_json.exists()
                    assert dwi_bval.exists()
                    assert dwi_bvec.exists()
                    assert dwi_nii.exists()
                    assert dwi_json.exists()

                # Some subjects have run-2 (those with multiple sessions)
                if len(sessions) > 1 or ses == "02":
                    run = "2"
                    adc_nii = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_ADC.nii.gz"
                    adc_json = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_ADC.json"
                    dwi_bval = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.bval"
                    dwi_bvec = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.bvec"
                    dwi_nii = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.nii.gz"
                    dwi_json = dwi_dir / f"sub-{subj}_ses-{ses}_run-{run}_dwi.json"

                    assert adc_nii.exists()
                    assert adc_json.exists()
                    assert dwi_bval.exists()
                    assert dwi_bvec.exists()
                    assert dwi_nii.exists()
                    assert dwi_json.exists()

    def test_all_datasets_in_cluster_structure(self, mock_cluster_datasets):
        """Test that all datasets exist in the cluster structure."""
        cluster_path = mock_cluster_datasets
        assert cluster_path.exists()

        expected_datasets = ["brats_sr", "fastmri", "hcp", "m4raw", "ulf_paired"]

        for dataset in expected_datasets:
            dataset_path = cluster_path / dataset
            assert dataset_path.exists(), f"Dataset {dataset} not found"
            assert dataset_path.is_dir(), f"Dataset {dataset} is not a directory"
