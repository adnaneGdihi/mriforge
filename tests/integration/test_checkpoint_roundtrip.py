"""Integration tests for Checkpoint Save/Load Roundtrip."""

import os
import tempfile

import pytest
import torch

from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
from spectramr.infrastructure.services.checkpoint_service import CheckpointService


@pytest.mark.integration
def test_checkpoint_roundtrip():
    """Test saving and loading a checkpoint preserves state."""

    # 1. Setup Service
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create valid config
        config = CheckpointConfigSchema(
            checkpoint_dir=tmpdir,
            save_interval=10,
            keep_last_n=3,
            format="pth",  # Use pth for simplicity in test
        )

        service = CheckpointService(config=config)

        # 2. Create State
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        epoch = 5
        loss = 0.5
        step = 100

        # 3. Save
        # service.save_checkpoint takes model, optimizer directly
        path = service.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            step=step,
            file_path=os.path.join(tmpdir, "test_checkpoint.pth"),
        )
        assert os.path.exists(path)

        # 4. Load
        # Create new model/optimizer to load into
        loaded_model = torch.nn.Linear(10, 10)
        loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.1)

        # Verify they are different initially
        assert not torch.equal(
            next(model.parameters()), next(loaded_model.parameters())
        )

        loaded_state = service.load_checkpoint(
            model=loaded_model, optimizer=loaded_optimizer, file_path=path
        )

        # 5. Verify
        assert loaded_state["epoch"] == epoch
        assert loaded_state["step"] == step
        assert loaded_state["loss"] == loss

        # Verify weight equality
        for k, v in model.state_dict().items():
            assert torch.equal(loaded_model.state_dict()[k], v)
