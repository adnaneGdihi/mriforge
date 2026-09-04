"""Unit tests for configuration parameter resolution.

This module validates that configuration parameters are correctly resolved
across different config sections and that deprecated paths trigger warnings.
"""

import pytest
from pydantic import ValidationError

from spectramr.config.settings import TrainingSettings


class TestTrainingDiffusionPathResolution:
    """Test that training.diffusion parameters are correctly read."""

    def test_training_diffusion_timesteps_priority(self):
        """Verify training.diffusion.timesteps reads from the SSOT path.

        Direct construction skips the YAML-aware ``config_version`` strip,
        so pop it manually. The legacy ``objectives`` block is no longer
        accepted (removed in v6) — see ``test_objectives_key_is_rejected``.
        """
        config_dict = {
            "config_version": "1.0",
            "model": {
                "model_type": "kspace_cold_diffusion_generator",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {"dataset_type": "kspace", "batch_size": 4},
            "optimization": {"optimizer_type": "adam", "learning_rate": 1e-4},
            "logging": {"log_interval": 100},
            "training": {"diffusion": {"timesteps": 500, "noise_schedule": "cosine"}},
        }
        config_dict.pop("config_version")
        settings = TrainingSettings(**config_dict)
        assert settings.training.diffusion.timesteps == 500
        assert settings.training.diffusion.noise_schedule == "cosine"

    def test_objectives_key_is_rejected(self):
        """The legacy ``objectives`` top-level block was removed in v6;
        callers must use ``losses.*`` and ``training.diffusion.*`` instead.
        """
        config_dict = {
            "model": {
                "model_type": "kspace_cold_diffusion_generator",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {"dataset_type": "kspace", "batch_size": 4},
            "optimization": {"optimizer_type": "adam", "learning_rate": 1e-4},
            "logging": {"log_interval": 100},
            "objectives": {"diffusion": {"timesteps": 1000}},
        }
        with pytest.raises(ValidationError, match=r"objectives"):
            TrainingSettings(**config_dict)


class TestConfigSchemaDefaults:
    """Test that config schema provides correct defaults."""

    def test_reconstruction_losses_defaults(self):
        """Verify ``ReconstructionLossesConfig`` ships the documented defaults.

        ``enable_l1`` defaults to ``False`` (opt-in) and the YAML must
        flip it on — this prevents an experiment from silently picking
        up L1 just by omitting the flag. The lambda values are still
        the documented baseline (``lambda_l1=10.0``) so that turning
        the flag on doesn't require also specifying a weight.
        """
        from spectramr.config.schemas.loss import ReconstructionLossesConfig

        config = ReconstructionLossesConfig()
        assert (
            config.lambda_weighted_kspace_l1 == 0.0
        ), "Default should be 0.0 (disabled)"
        assert config.lambda_l1 == 10.0, "Default L1 weight should be 10.0"
        assert config.enable_l1 is False, "L1 should be opt-in (flag-driven)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
