import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from spectramr.data.datasets.preprocessed_dataset import PreprocessedMRIDataset


class TestTargetFiltering(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)

        # Create artifacts
        (self.root / "images_input").mkdir()
        (self.root / "images_target").mkdir()

        # Create NIfTI helper
        affine = np.eye(4)
        data = np.zeros((10, 10, 10), dtype=np.float32)
        img = nib.Nifti1Image(data, affine)

        # Input: Just one subject
        nib.save(img, self.root / "images_input" / "sub-01.nii.gz")

        # Target: Two contrasts for the same subject
        # Note: In a real directory, if they map to same ID, they might overwrite each other
        # in the internal dict if filtered. valid behavior relies on filters removing duplicates.
        nib.save(img, self.root / "images_target" / "sub-01_T1w.nii.gz")
        nib.save(img, self.root / "images_target" / "sub-01_T2w.nii.gz")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_filter_t1(self):
        """Test filtering target matching T1w."""
        ds = PreprocessedMRIDataset(
            output_dir=self.root,
            task_type="reconstruction",
            input_artifact="images_input",
            target_artifact="images_target",
            target_contrasts=["T1w"],  # Should select sub-01_T1w.nii.gz
            load_coil_sensitivity=False,
            load_statistics=False,
        )

        self.assertEqual(len(ds), 1)
        # Check internal sample record directly since Subject.path is unreliable/None
        sample_record = ds._samples[0]
        target_path = str(sample_record.target_path)

        self.assertIn("T1w", target_path)
        self.assertNotIn("T2w", target_path)

    def test_filter_t2(self):
        """Test filtering target matching T2w."""
        ds = PreprocessedMRIDataset(
            output_dir=self.root,
            task_type="reconstruction",
            input_artifact="images_input",
            target_artifact="images_target",
            target_contrasts=["T2w"],  # Should select sub-01_T2w.nii.gz
            load_coil_sensitivity=False,
            load_statistics=False,
        )

        self.assertEqual(len(ds), 1)
        sample_record = ds._samples[0]
        target_path = str(sample_record.target_path)
        self.assertIn("T2w", target_path)

    def test_no_filter_collision(self):
        """Test behavior when no filter is applied (uncertain which one survives dict overwrite)."""
        # If we don't filter, both T1w and T2w map to 'sub-01'.
        # The last one iterated wins. (Sorted by filename -> T2w likely last).
        ds = PreprocessedMRIDataset(
            output_dir=self.root,
            task_type="reconstruction",
            input_artifact="images_input",
            target_artifact="images_target",
            load_coil_sensitivity=False,
            load_statistics=False,
        )
        self.assertEqual(len(ds), 1)
        # We don't strictly assert which one wins, but one should exist.


if __name__ == "__main__":
    unittest.main()
