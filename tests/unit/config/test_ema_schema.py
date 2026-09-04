import pytest
from pydantic import ValidationError

from spectramr.config.schemas.ema import EMAConfigSchema


class TestEMAConfigSchema:
    def test_defaults(self):
        schema = EMAConfigSchema()
        assert schema.enabled is False
        assert schema.decay == 0.999
        assert schema.update_frequency == 1

    def test_custom_values(self):
        schema = EMAConfigSchema(
            enabled=True,
            decay=0.99,
            update_frequency=10,
            enable_adaptive_ema=True,
            # warmup_steps IS the adaptive ramp length, so an adaptive
            # declaration without it is now refused (#1294) rather than
            # collapsing to a fixed final_decay.
            warmup_steps=1000,
            initial_decay=0.9,
            final_decay=0.999,
        )
        assert schema.enabled is True
        assert schema.decay == 0.99
        assert schema.update_frequency == 10
        assert schema.enable_adaptive_ema is True
        assert schema.warmup_steps == 1000
        assert schema.initial_decay == 0.9
        assert schema.final_decay == 0.999

    def test_adaptive_without_ramp_length_is_refused(self):
        """A zero-length ramp is a declaration that does nothing."""
        with pytest.raises(ValidationError, match="warmup_steps"):
            EMAConfigSchema(enable_adaptive_ema=True)

    def test_adaptive_with_unwired_stability_knobs_is_refused(self):
        """stability_threshold / adaptation_rate configure a feedback schedule
        that has no signal source; refusing beats silently ignoring them."""
        with pytest.raises(ValidationError, match="stability-feedback"):
            EMAConfigSchema(
                enable_adaptive_ema=True,
                warmup_steps=100,
                stability_threshold=0.1,
            )

    def test_stability_knobs_alone_are_still_accepted(self):
        """They remain harmless (and inert) while adaptive EMA is off — the
        raise is scoped to the declaration that cannot be honoured."""
        schema = EMAConfigSchema(stability_threshold=0.1, adaptation_rate=0.02)
        assert schema.enable_adaptive_ema is False

    def test_validation_decay(self):
        with pytest.raises(ValidationError):
            EMAConfigSchema(decay=1.1)
        with pytest.raises(ValidationError):
            EMAConfigSchema(decay=0.4)  # < 0.5

    def test_validation_adaptive_decay(self):
        with pytest.raises(ValidationError):
            EMAConfigSchema(initial_decay=0.9, final_decay=0.8)

    def test_extra_fields(self):
        # extra="forbid"
        with pytest.raises(ValidationError):
            EMAConfigSchema(extra_field="value")
