import pytest
from pydantic import ValidationError

from spectramr.config.schemas.acceleration import AccelerationConfigSchema
from spectramr.config.schemas.enums import AccelerationSchedule


class TestAccelerationConfigSchema:
    def test_defaults(self):
        schema = AccelerationConfigSchema()
        assert schema.base_acceleration == 4.0
        assert schema.max_acceleration == 8.0
        assert schema.acceleration_schedule == AccelerationSchedule.LINEAR
        assert schema.acceleration_type == "variable_density"

    def test_custom_values(self):
        schema = AccelerationConfigSchema(
            base_acceleration=2.0,
            max_acceleration=4.0,
            acceleration_schedule=AccelerationSchedule.EXPONENTIAL,
            acceleration_type="radial",
            mask_types=["radial", "spiral"],
            mask_combination_mode="random",
        )
        assert schema.base_acceleration == 2.0
        assert schema.max_acceleration == 4.0
        assert schema.acceleration_schedule == AccelerationSchedule.EXPONENTIAL
        assert schema.acceleration_type == "radial"
        assert schema.mask_types == ["radial", "spiral"]
        assert schema.mask_combination_mode == "random"

    def test_validation(self):
        with pytest.raises(ValidationError):
            AccelerationConfigSchema(base_acceleration=0)
        with pytest.raises(ValidationError):
            AccelerationConfigSchema(center_fraction=1.5)
        with pytest.raises(ValidationError):
            AccelerationConfigSchema(schedule_steps=0)

    def test_extra_fields(self):
        # extra="ignore" - extra fields are silently ignored for flexibility
        schema = AccelerationConfigSchema(extra_field="value")
        assert not hasattr(schema, "extra_field")  # Ignored, not added
