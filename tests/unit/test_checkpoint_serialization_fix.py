import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from mriforge.config.schemas.checkpoint import CheckpointConfigSchema
from mriforge.infrastructure.services.checkpoint_service import CheckpointService


class TestCheckpointSerialization(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = CheckpointConfigSchema(
            checkpoint_dir=self.test_dir, save_interval=10, keep_last_n=2, keep_best_n=1
        )
        self.service = CheckpointService(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tensor_serialization_in_metadata(self):
        """Test that passing tensors for loss/metric values works."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Create tensor values
        loss_tensor = torch.tensor(0.5, requires_grad=True)
        metric_tensor = torch.tensor(0.95)

        # Save checkpoint
        path = self.service.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            loss=loss_tensor,
            step=100,
            metric_name="accuracy",
            metric_value=metric_tensor,
        )

        # Load metadata back
        filename = Path(path).name
        meta = self.service._metadata[filename]

        # Verify types
        self.assertIsInstance(meta.loss, float)
        self.assertIsInstance(meta.metric_value, float)
        self.assertAlmostEqual(meta.loss, 0.5)
        self.assertAlmostEqual(meta.metric_value, 0.95)

        print("\n✅ Checkpoint serialization test passed!")


if __name__ == "__main__":
    unittest.main()
