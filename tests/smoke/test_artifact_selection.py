import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from spectramr.data.datasets.preprocessed_dataset import PreprocessedMRIDataset
from types import SimpleNamespace


class TestArtifactSelection(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)

        # Create dummy artifacts
        (self.root / "custom_input").mkdir()
        (self.root / "custom_target").mkdir()

        # Create dummy NIfTI files
        affine = np.eye(4)
        data = np.zeros((10, 10, 10), dtype=np.float32)
        img = nib.Nifti1Image(data, affine)

        nib.save(img, self.root / "custom_input" / "sub-01.nii.gz")
        nib.save(img, self.root / "custom_target" / "sub-01.nii.gz")

        # Create 'default' artifacts to prove we are NOT using them
        (self.root / "compressed_kspace").mkdir()
        (self.root / "gt_images").mkdir()
        (self.root / "compressed_kspace" / "default.pt").touch()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_artifact_override(self):
        """Test that input_artifact and target_artifact args override defaults."""

        # Init dataset with overrides
        ds = PreprocessedMRIDataset(
            output_dir=self.root,
            task_type="reconstruction",  # Defaults to compressed_kspace -> gt_images
            input_artifact="custom_input",
            target_artifact="custom_target",
            load_coil_sensitivity=False,
            load_statistics=False,
        )

        # Check internal state
        self.assertEqual(ds.input_artifact_override, "custom_input")
        self.assertEqual(ds.target_artifact_override, "custom_target")

        # Check validation logic (should pass since files exist)
        # Check if samples were found
        self.assertTrue(len(ds) > 0)

        # Check that it actually loaded the correct files
        sample = ds[0]
        # PreprocessedMRIDataset returns sample dict or object?
        # It references file paths

        # Check that it actually loaded the correct files
        sample = ds[0]
        # Verify keys
        self.assertIn("input", sample)
        self.assertIn("target", sample)

        # Verify path
        input_image = sample["input"]
        # Just check we loaded the data
        self.assertEqual(input_image.data.numel(), 1000)

        # We can also check that the other file corresponds
        # But PreprocessedMRIDataset doesn't expose input_files list directly
        # It's inside _samples which is list of dicts.
        pass

    def test_factory_passing(self):
        """Test that factory passes config values to dataset.

        Use the real ``DataConfigSchema`` Pydantic model rather than a
        SimpleNamespace so every field gets a sensible default. The
        previous SimpleNamespace approach forced this test to enumerate
        every attribute the factory might read — which drifts every time
        the factory adds a new field.
        """
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.infrastructure.builders.context import BuilderContext
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            DataPipelineDirector,
        )

        mock_config = DataConfigSchema(
            dataset_type="preprocessed",
            data_root=str(self.root),
            input_artifact="custom_input",
            target_artifact="custom_target",
            patch_size=(64, 64),
            queue_length=10,
            max_queue_length=10,
            samples_per_volume=1,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )

        # We need to mock PreprocessedMRIDataset to check args passed
        with patch(
            "spectramr.data.datasets.preprocessed_dataset.PreprocessedMRIDataset"
        ) as MockDataset:
            # Set length to avoid DataLoader error
            MockDataset.return_value.__len__.return_value = 10

            # Was ConsolidatedDatasetFactory (deleted, 6a-iii). The
            # DeprecationWarning suppression went with it: the director is the
            # supported path and emits none, so a filterwarnings promotion has
            # nothing left to catch here.
            # The director reads `config.data.*`: BuilderContext expects a
            # TrainingSettings-shaped object, not a bare DataConfigSchema, so the
            # old call produced `data.data` and raised. Wrap it at the seam.
            settings_like = SimpleNamespace(data=mock_config)
            director = DataPipelineDirector(BuilderContext(config=settings_like))
            _ = director.build_dataloaders()

            # Verify call args
            _, kwargs = MockDataset.call_args
            self.assertEqual(kwargs["input_artifact"], "custom_input")
            self.assertEqual(kwargs["target_artifact"], "custom_target")


if __name__ == "__main__":
    unittest.main()
