from unittest.mock import patch

import pytest

from spectramr.config.schemas.training import create_training_config


class TestCreateTrainingConfig:
    """Test suite for create_training_config factory function.

    WS-1 round-2 (2026-07-02): ``training_mode`` is a v5.0-removed field that
    the schema's ``reject_deprecated_fields`` validator raises on, so the
    factory now STRIPS it from a copy before ``model_validate`` — the mock is
    called with the training_mode-free dict, not the raw input. And an unknown
    mode RAISES instead of silently degrading to reconstruction (pitfall #9).
    """

    def test_create_gan_config(self):
        """GAN dispatch; training_mode stripped before validation."""
        config_dict = {"training_mode": "gan", "epochs": 100}
        with patch("spectramr.config.schemas.training.TrainingConfigGAN") as MockGAN:
            create_training_config(config_dict)
            MockGAN.model_validate.assert_called_once_with({"epochs": 100})

    def test_create_diffusion_config(self):
        config_dict = {"training_mode": "diffusion"}
        with patch(
            "spectramr.config.schemas.training.TrainingConfigDiffusion"
        ) as MockDiffusion:
            create_training_config(config_dict)
            MockDiffusion.model_validate.assert_called_once_with({})

    def test_create_vae_config(self):
        config_dict = {"training_mode": "vae"}
        with patch("spectramr.config.schemas.training.TrainingConfigVAE") as MockVAE:
            create_training_config(config_dict)
            MockVAE.model_validate.assert_called_once_with({})

    def test_create_reconstruction_config(self):
        config_dict = {"training_mode": "reconstruction"}
        with patch(
            "spectramr.config.schemas.training.TrainingConfigReconstruction"
        ) as MockRecon:
            create_training_config(config_dict)
            MockRecon.model_validate.assert_called_once_with({})

    def test_create_ssl_config(self):
        config_dict = {"training_mode": "ssl"}
        with patch("spectramr.config.schemas.training.TrainingConfigSSL") as MockSSL:
            create_training_config(config_dict)
            MockSSL.model_validate.assert_called_once_with({})

    def test_create_default_config(self):
        """No training_mode -> reconstruction default; nothing to strip."""
        config_dict = {}
        with patch(
            "spectramr.config.schemas.training.TrainingConfigReconstruction"
        ) as MockRecon:
            create_training_config(config_dict)
            MockRecon.model_validate.assert_called_once_with({})

    def test_create_unknown_mode_raises(self):
        """An unknown/typo'd mode must NOT silently degrade (pitfall #9)."""
        with pytest.raises(ValueError, match="Unknown training_mode"):
            create_training_config({"training_mode": "unknown_mode"})

    def test_recognised_mode_without_schema_warns_and_uses_default(self, caplog):
        """A VALID mode with no dedicated schema loads the default, observably."""
        import logging

        # 'pinn' is a valid training mode with no dedicated paradigm schema in
        # the dispatch table. (Was 'sensitivity_estimation' until 2026-07-19,
        # when that orphan was removed -- it named no strategy, and 'pinn' ->
        # ConcretePINNSensitivityStrategy is the real name for the same thing.)
        config_dict = {"training_mode": "pinn"}
        with patch(
            "spectramr.config.schemas.training.TrainingConfigReconstruction"
        ) as MockRecon:
            with caplog.at_level(logging.WARNING):
                create_training_config(config_dict)
            MockRecon.model_validate.assert_called_once_with({})
        assert any("no dedicated schema" in r.message for r in caplog.records)

    def test_create_kspace_cold_diffusion_config(self):
        """kspace_cold_diffusion maps to the Diffusion schema."""
        config_dict = {"training_mode": "kspace_cold_diffusion"}
        with patch(
            "spectramr.config.schemas.training.TrainingConfigDiffusion"
        ) as MockDiffusion:
            create_training_config(config_dict)
            MockDiffusion.model_validate.assert_called_once_with({})


class TestDeterministicKnobDefault:
    """``training.deterministic`` defaults to True (2026-06 determinism SSOT).

    The canonical train path has always FORCED full determinism (a hardcoded
    cudnn override in ``main.py``), while the schema default said ``False`` —
    so the knob was a pitfall-#15 silent no-op. Wiring the knob preserves the
    forced behaviour by flipping the default to True; setting ``False`` in
    YAML is now an honoured opt-out for speed.
    """

    def test_deterministic_defaults_true(self):
        from spectramr.config.schemas.training.base import BaseTrainingConfigSchema

        assert BaseTrainingConfigSchema.model_fields["deterministic"].default is True
