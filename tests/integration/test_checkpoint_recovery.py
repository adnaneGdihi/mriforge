import pathlib
import shutil
import tempfile
import time
import unittest

# NOTE: Integration tests should NOT use remediation mocks.
# They test real service behavior with real dependencies.


class TestCheckpointRecoveryDeterminism(unittest.TestCase):
    """Test deterministic checkpoint recovery."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = pathlib.Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_finds_highest_step_not_latest_mtime(self):
        """Verify recovery uses step number, not filesystem timestamp."""
        from mriforge.config.schemas.checkpoint import CheckpointConfigSchema
        from mriforge.infrastructure.services.checkpoint_service import CheckpointService

        config = CheckpointConfigSchema(
            checkpoint_dir=str(self.tmp_path),
            save_interval=10,
        )
        service = CheckpointService(config)

        # Create checkpoints with OLDER timestamp but HIGHER step
        ckpt_100 = self.tmp_path / "checkpoint_step_100.safetensors"
        ckpt_50 = self.tmp_path / "checkpoint_step_50.safetensors"

        # Write step 100 first
        ckpt_100.write_text("step 100")
        time.sleep(0.1)  # Ensure different mtime
        # Write step 50 second (newer mtime, but LOWER step)
        ckpt_50.write_text("step 50")

        # Find latest should return step 100, not step 50
        latest = service.find_latest_checkpoint(str(self.tmp_path))

        self.assertIsNotNone(latest)
        # Convert to string explicitly to handle potential Mock objects in some environments
        latest_str = str(latest)
        self.assertIn("step_100", latest_str, f"Expected step_100, got {latest_str}")

    def test_handles_epoch_checkpoints(self):
        """Verify epoch checkpoints are handled correctly."""
        from mriforge.config.schemas.checkpoint import CheckpointConfigSchema
        from mriforge.infrastructure.services.checkpoint_service import CheckpointService

        config = CheckpointConfigSchema(
            checkpoint_dir=str(self.tmp_path),
            save_interval=10,
        )
        service = CheckpointService(config)

        # Create epoch checkpoints
        ckpt_10 = self.tmp_path / "checkpoint_epoch_10.safetensors"
        ckpt_5 = self.tmp_path / "checkpoint_epoch_5.safetensors"

        ckpt_10.write_text("epoch 10")
        time.sleep(0.1)
        ckpt_5.write_text("epoch 5")  # Newer mtime

        latest = service.find_latest_checkpoint(str(self.tmp_path))

        self.assertIsNotNone(latest)
        self.assertIn("epoch_10", str(latest))

    def test_mixed_epoch_step_checkpoints(self):
        """Verify step checkpoints are correctly compared with epoch checkpoints."""
        from mriforge.config.schemas.checkpoint import CheckpointConfigSchema
        from mriforge.infrastructure.services.checkpoint_service import CheckpointService

        config = CheckpointConfigSchema(
            checkpoint_dir=str(self.tmp_path),
            save_interval=10,
        )
        service = CheckpointService(config)

        # Create mixed checkpoints
        # Epoch 2 = conceptually 2 * 100000 = 200000
        # Step 500 = 500
        ckpt_epoch = self.tmp_path / "checkpoint_epoch_2.safetensors"
        ckpt_step = self.tmp_path / "checkpoint_step_500.safetensors"

        ckpt_step.write_text("step 500")
        ckpt_epoch.write_text("epoch 2")

        latest = service.find_latest_checkpoint(str(self.tmp_path))

        # Epoch 2 should win (200000 > 500)
        self.assertIsNotNone(latest)
        self.assertIn("epoch_2", str(latest))


if __name__ == "__main__":
    unittest.main()
