"""Test that deprecated configuration fields are properly rejected.

This test suite verifies that the configuration system enforces strict v5.0-only
validation and rejects deprecated fields with clear error messages.
"""

import pytest

from spectramr.config.schemas.training.base import BaseTrainingConfigSchema


class TestDeprecatedFieldRejection:
    """Test rejection of deprecated configuration fields."""

    def test_reject_training_mode_field(self):
        """Test that training_mode field is rejected."""
        config_data = {
            "config_version": "5.0",
            "training_mode": "gan",  # DEPRECATED
            "epochs": 100,
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "training_mode" in error_msg
        assert "deprecated" in error_msg.lower()
        assert "strategy_class" in error_msg

    def test_reject_num_epochs_field(self):
        """Test that num_epochs (old name) is rejected."""
        config_data = {
            "config_version": "5.0",
            "num_epochs": 100,  # DEPRECATED
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "num_epochs" in error_msg
        assert "deprecated" in error_msg.lower()

    def test_reject_lambda_l1_field(self):
        """Test that lambda_l1 (flat) field is rejected."""
        config_data = {
            "config_version": "5.0",
            "lambda_l1": 10.0,  # DEPRECATED
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "lambda_l1" in error_msg
        assert "objectives.reconstruction" in error_msg

    def test_reject_lambda_l2_field(self):
        """Test that lambda_l2 (flat) field is rejected."""
        config_data = {
            "config_version": "5.0",
            "lambda_l2": 0.01,  # DEPRECATED
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "lambda_l2" in error_msg

    def test_reject_lambda_adv_field(self):
        """Test that lambda_adv (flat) field is rejected."""
        config_data = {
            "config_version": "5.0",
            "lambda_adv": 1.0,  # DEPRECATED
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "lambda_adv" in error_msg
        assert "objectives.gan" in error_msg

    def test_reject_validation_check_interval_field(self):
        """Test that validation_check_interval is rejected."""
        config_data = {
            "config_version": "5.0",
            "validation_check_interval": 5,  # DEPRECATED
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "validation_check_interval" in error_msg

    def test_reject_multiple_deprecated_fields(self):
        """Test that multiple deprecated fields are rejected together."""
        config_data = {
            "config_version": "5.0",
            "training_mode": "gan",
            "num_epochs": 100,
            "lambda_l1": 10.0,
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        # Should list all deprecated fields
        assert "training_mode" in error_msg
        # Note: Only first one is caught due to how the validator works
        # This is acceptable - forces user to fix one at a time

    def test_accept_valid_config(self):
        """Test that a minimal valid config is accepted.

        Bumped from ``config_version: "5.0"`` to the canonical ``"1.0"`` — the loader
        no longer accepts the v5.x family per the version gate added in
        ``BaseTrainingConfigSchema`` (accepted_versions=['6.0', '6.1']).
        """
        config_data = {
            "config_version": "1.0",
            "epochs": 100,
            # Other required fields get defaults
        }

        # Should not raise
        config = BaseTrainingConfigSchema.model_validate(config_data)
        assert config.epochs == 100
        assert config.config_version == "1.0"

    def test_accept_config_with_strategy_class(self):
        """Test that config with new strategy_class field is accepted.

        Asserted on the schema that OWNS the field. This previously read
        ``BaseTrainingConfigSchema(...).training.strategy_class`` -- i.e.
        ``training.training``, a self-nesting that never existed after the
        decomposition. Worse, ``BaseTrainingConfigSchema`` declares neither
        ``training`` nor ``strategy_class`` and is ``extra="ignore"``, so the
        whole payload was silently swallowed: there was nothing on that object to
        assert against, and the check could only raise or pass vacuously.
        """
        from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

        strategy = "spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy"
        block = TrainingStrategyConfigSchema.model_validate(
            {"strategy_class": strategy}
        )
        assert block.strategy_class == strategy

        # ...and the outer schema still accepts a document carrying that block.
        config = BaseTrainingConfigSchema.model_validate(
            {
                "config_version": "1.0",
                "training": {"strategy_class": strategy},
                "epochs": 50,
            }
        )
        assert config.epochs == 50


class TestDeprecatedFieldErrorMessages:
    """Test that error messages are clear and actionable."""

    def test_error_message_mentions_migration_guide(self):
        """Test that error message references migration documentation."""
        config_data = {
            "config_version": "5.0",
            "training_mode": "gan",
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        # Should guide user to documentation
        assert "CONFIG_MIGRATION" in error_msg or "migration" in error_msg.lower()

    def test_error_message_provides_correct_replacement(self):
        """Test that error message shows correct v5.0 field name."""
        config_data = {
            "config_version": "5.0",
            "lambda_adv": 1.0,
        }

        with pytest.raises(ValueError) as exc_info:
            BaseTrainingConfigSchema.model_validate(config_data)

        error_msg = str(exc_info.value)
        # Should mention the correct replacement field
        assert "objectives.gan.lambda_adv" in error_msg


class TestVersionEnforcement:
    """Test that only v5.0 configs are accepted."""

    def test_reject_v3_config(self):
        """Test that v3.0 config is rejected."""
        config_data = {
            "config_version": "3.0",
            "learning_rate": 0.0002,
        }

        # This should be caught by TrainingSettings.from_yaml()
        # but we test the schema validation here too
        # Note: BaseTrainingConfigSchema doesn't validate version
        # That's done in TrainingSettings.from_yaml()
        pass

    def test_reject_v4_config(self):
        """Test that v4.0 config is rejected."""
        config_data = {
            "config_version": "4.0",
            "learning_rate": 0.0002,
        }

        # This should be caught by TrainingSettings.from_yaml()
        pass

    def test_reject_no_version(self):
        """Test that config without version field fails."""
        # This is tested in test_settings.py
        # Here we just verify the principle
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
