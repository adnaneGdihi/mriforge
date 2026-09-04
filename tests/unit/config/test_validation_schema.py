import pytest
from pydantic import ValidationError

from spectramr.config.schemas.validation import ValidationConfigSchema


class TestValidationConfigSchema:
    def test_defaults(self):
        schema = ValidationConfigSchema()
        assert schema.enabled is True
        assert schema.split == 0.2
        assert schema.schedule.interval_steps == 1000

    def test_custom_values(self):
        schema = ValidationConfigSchema(
            enabled=False,
            split=0.1,
            eval_interval=500,
            validation_batch_size=8,
            validation_metric="psnr",
        )
        assert schema.enabled is False
        assert schema.split == 0.1
        assert schema.schedule.interval_steps == 500
        assert schema.loader.batch_size == 8
        assert schema.validation_metric == "psnr"

    def test_validation(self):
        with pytest.raises(ValidationError):
            ValidationConfigSchema(split=1.5)
        with pytest.raises(ValidationError):
            ValidationConfigSchema(eval_interval=0)

    def test_sync_intervals(self):
        # Test canonical field names (aliases removed)
        schema = ValidationConfigSchema(eval_interval=500)
        assert schema.schedule.interval_steps == 500

        schema = ValidationConfigSchema(frequency_epochs=5)
        assert schema.schedule.interval_epochs == 5

        schema = ValidationConfigSchema(split=0.3)
        assert schema.split == 0.3

    def test_extra_fields(self):
        # extra="forbid"
        with pytest.raises(ValidationError):
            ValidationConfigSchema(extra_field="value")


class TestInputDependenceGate:
    """L4 measurement-independence (DC-blob) gate knob."""

    def test_default_disabled(self):
        """The gate is off by default (None)."""
        schema = ValidationConfigSchema()
        assert schema.gates.input_dependence_tol is None

    def test_enable_with_positive_float(self):
        """A positive float enables the gate."""
        schema = ValidationConfigSchema(gates={"input_dependence_tol": 0.01})
        assert schema.gates.input_dependence_tol == 0.01

    def test_zero_is_rejected(self):
        """gt=0: a non-positive tolerance is a ValidationError.

        Declared canonically on purpose. `validation.input_dependence_tol` was
        promoted to `raise` once its corpus count hit zero, and the retirement
        error is ALSO a `ValidationError` -- so the legacy spelling would keep
        this test green while never reaching the `gt=0` constraint it exists to
        pin.
        """
        with pytest.raises(ValidationError):
            ValidationConfigSchema(gates={"input_dependence_tol": 0.0})

    def test_negative_is_rejected(self):
        """A negative tolerance is a ValidationError."""
        with pytest.raises(ValidationError):
            ValidationConfigSchema(gates={"input_dependence_tol": -0.5})

    def test_field_is_part_of_forbid_schema(self):
        """The knob lives on the extra='forbid' validation schema.

        This is the regression that proves a YAML ``validation:`` block can
        carry ``input_dependence_tol`` without the audit rejecting it.
        """
        schema = ValidationConfigSchema(
            enabled=True, gates={"input_dependence_tol": 0.02}
        )
        assert schema.gates.input_dependence_tol == 0.02
        # A sibling undeclared key is still forbidden.
        with pytest.raises(ValidationError):
            ValidationConfigSchema(input_dependence_tolerance=0.02)
