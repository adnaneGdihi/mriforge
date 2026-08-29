import pytest
from pydantic import ValidationError

from mriforge.config.schemas.early_stopping import EarlyStoppingConfigSchema
from mriforge.config.schemas.enums import MetricMode


class TestEarlyStoppingConfigSchema:
    def test_defaults(self):
        schema = EarlyStoppingConfigSchema()
        assert schema.enabled is False
        assert schema.patience == 10
        assert schema.metric == "val_loss"
        assert schema.mode == MetricMode.MIN

    def test_custom_values(self):
        schema = EarlyStoppingConfigSchema(
            enabled=True,
            patience=20,
            metric="val_psnr",
            mode=MetricMode.MAX,
            min_delta=0.01,
        )
        assert schema.enabled is True
        assert schema.patience == 20
        assert schema.metric == "val_psnr"
        assert schema.mode == MetricMode.MAX
        assert schema.min_delta == 0.01

    def test_validation(self):
        with pytest.raises(ValidationError):
            EarlyStoppingConfigSchema(patience=0)
        with pytest.raises(ValidationError):
            EarlyStoppingConfigSchema(min_delta=-0.1)

    def test_extra_fields(self):
        # extra="forbid"
        with pytest.raises(ValidationError):
            EarlyStoppingConfigSchema(extra_field="value")
