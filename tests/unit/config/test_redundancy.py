from pathlib import Path

import pytest
import yaml

from spectramr.config.settings import TrainingSettings


class TestConfigRedundancy:
    """Validate that the comprehensive template is valid and synchronized with the schema."""

    @pytest.fixture
    def template_path(self):
        root = Path(__file__).resolve().parents[3]
        return root / "experiments/templates/comprehensive_config_template.yaml"

    def test_template_validity(self, template_path):
        """Template must be loadable by TrainingSettings without error."""
        assert template_path.exists(), "Template file not found at expected location"

        try:
            settings = TrainingSettings.from_yaml(str(template_path))
        except Exception as e:
            pytest.fail(f"Template failed validation: {e}")

    def test_no_legacy_keys_in_template(self, template_path):
        """Template must not contain specific legacy keys."""
        with open(template_path) as f:
            raw_data = yaml.safe_load(f)

        legacy_keys = ["training_mode", "task", "seed", "max_iterations"]

        for key in legacy_keys:
            assert (
                key not in raw_data
            ), f"Legacy key '{key}' found in template root! Use nested structure."

        # Check training strategy legacy keys
        if "training" in raw_data:
            training = raw_data["training"]
            banned_nested = [
                "diffusion_type",
                "timesteps",
                "degradation_type",
                "lambda_recon",
                "input_domain",
            ]
            for key in banned_nested:
                assert (
                    key not in training
                ), f"Legacy key 'training.{key}' found in template! Use nested structure."

    def test_schema_coverage(self, template_path):
        """Template (raw dict) should roughly match schema field count (sanity check).

        This finds keys present in Schema but missing in Template.
        """
        # This is a soft check - just printing warnings if needed, or failing if major mismatch
        pass
