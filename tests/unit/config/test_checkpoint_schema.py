import pytest
from pydantic import ValidationError

from spectramr.config.schemas.checkpoint import CheckpointConfigSchema


class TestCheckpointConfigSchema:
    def test_defaults(self):
        schema = CheckpointConfigSchema()
        assert schema.enabled is True
        assert schema.checkpoint_dir == "./checkpoints"
        assert schema.save_interval == 1000
        assert schema.keep_last_n == 5

    def test_custom_values(self):
        schema = CheckpointConfigSchema(
            enabled=False,
            checkpoint_dir="/tmp/ckpt",
            save_interval=500,
            keep_last_n=10,
            pretrained_path="model.pt",
        )
        assert schema.enabled is False
        assert schema.checkpoint_dir == "/tmp/ckpt"
        assert schema.save_interval == 500
        assert schema.keep_last_n == 10
        assert schema.pretrained_path == "model.pt"

    def test_validation(self):
        with pytest.raises(ValidationError):
            CheckpointConfigSchema(save_interval=0)
        with pytest.raises(ValidationError):
            CheckpointConfigSchema(keep_last_n=0)

    def test_extra_fields(self):
        # extra="ignore"
        schema = CheckpointConfigSchema(extra_field="value")
        assert not hasattr(schema, "extra_field")
