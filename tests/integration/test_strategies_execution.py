"""Integration tests for strategy execution."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from mriforge.bootstrap import build_container
from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.training.builders.director import TrainingEnvironmentDirector
from mriforge.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


class TestStrategyExecution:
    """Integration tests for strategy execution."""

    @pytest.fixture
    def temp_experiment_dir(self):
        """Create temporary directory for experiment artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def synthetic_dataloader(self):
        """Create synthetic DataLoader."""
        batch_size = 2
        num_samples = 4
        # (B, C, H, W)
        kspace_data = torch.randn(num_samples, 2, 64, 64)
        targets = torch.randn(num_samples, 2, 64, 64)
        masks = torch.ones(num_samples, 1, 64, 64)

        dataset = TensorDataset(kspace_data, targets, masks)
        # Use drop_last=True to ensure batch size consistency
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def create_base_config(self, output_dir):
        """Create base configuration (v6.0 compliant)."""
        return {
            "config_version": "1.0",
            "metadata": {
                "name": "test_integration",
                "description": "Integration test",
                "tags": ["test"],
            },
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
                "base_channels": 8,  # Minimal channels
                "num_res_units": 1,
                "spatial_dims": 2,
            },
            "data": {
                "dataset_type": "synthetic",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": [64, 64],
                "acceleration": 4,
                "center_fraction": 0.08,
                "normalize_kspace": True,
                "patch_size": [64, 64, 1],
            },
            "optimization": {
                "learning_rate": 1e-4,
                "optimizer_type": "adam",
                "scheduler_type": "constant",
                # GAN specific
                "gradient_clip_value": 1.0,
                "gradient_clip_method": "norm",
                "enable_gradient_clipping": True,
                "gradient_accumulation_steps": 1,
            },
            # `seed` is a fact about the RUN. Both `training.seed` and the top-level
            # `seed` were retired to `run.seed` on 2026-07-31 with posture=raise, so a
            # fixture declaring either fails to construct.
            "run": {"seed": 42},
            "training": {
                # strategy_class will be set in tests
                "epochs": 2,
                "output_dir": str(output_dir),
            },
            "losses": {
                "reconstruction": {
                    "lambda_l1": 1.0,
                },
            },
            "physics": {
                "data_consistency": {
                    "enabled": True,
                    "train_noise_level": 0.01,
                },
            },
            "checkpoint": {
                "save_top_k": 1,
                "monitor": "val_loss",
                "mode": "min",
            },
            "validation": {
                "compute_image_metrics": False,
            },
            "logging": {
                "log_interval": 1,
                # `logging.debug_log_steps` -> `logging.snapshots.log_steps`,
                # posture="raise", so the flat spelling no longer loads.
                # Non-vacuous: the canonical default is [0, 1, 2, 10], so a
                # fixture asking for [0] still proves the value was read.
                "snapshots": {"log_steps": [0]},
                "log_validation_images": False,
            },
            # Required infrastructure fields. `services` is NOT one of them
            # any more: it is not a `TrainingSettings` field and has no
            # `RENAMES` record, so under extra="forbid" it rejects the whole
            # config. It survives only in the legacy `active/`/`validated/`
            # corpus (211 arms), which is unloadable for other reasons anyway.
            "loss_logging": {},
            "metrics": {},
            "early_stopping": {},
            "ema": {},
        }

    def test_reconstruction_execution(self, temp_experiment_dir, synthetic_dataloader):
        """Test ReconstructionTrainingStrategy execution."""
        config_dict = self.create_base_config(temp_experiment_dir)
        config_dict["training"][
            "strategy_class"
        ] = "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
        # Needed for ModelBuilder to validate correct model set
        config_dict["training"]["training_mode"] = "reconstruction"

        config_path = temp_experiment_dir / "recon_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = TrainingSettings.from_yaml(str(config_path))
        build_container(config)

        director = TrainingEnvironmentDirector(config)
        env = director.build_environment()

        strategy = ReconstructionTrainingStrategy(env=env)
        # strategy.setup()  # Not needed as env is fully built

        # Run minimal training loop
        epoch = 0
        for i, batch in enumerate(synthetic_dataloader):
            input_batch, target_batch, mask = batch

            step_configs = strategy.train_step(
                batch=batch,
                epoch=epoch,
                input_batch=input_batch,
                target_batch=target_batch,
                iteration=i,
                batch_data={"mask": mask},
            )

            losses = {}
            for config in step_configs:
                loss = config["closure"]()
                losses["g_total_loss"] = loss
                losses.update(strategy.get_last_metrics())

            assert "loss" in losses or "g_total_loss" in losses
            val = losses.get("g_total_loss", losses.get("loss"))
            if isinstance(val, torch.Tensor):
                assert val.item() > 0
            else:
                assert val > 0

    def test_gan_execution(self, temp_experiment_dir, synthetic_dataloader):
        """Test GANTrainingStrategy execution."""
        config_dict = self.create_base_config(temp_experiment_dir)
        config_dict["training"][
            "strategy_class"
        ] = "mriforge.infrastructure.training.strategies.gan.GANTrainingStrategy"
        config_dict["training"]["disc_start_step"] = 0
        # config_dict["training"]["training_mode"] = "gan" # Extra field issue in builder

        # Configure GAN losses
        config_dict["losses"]["gan"] = {
            "lambda_adv": 0.1,
            "gan_loss_type": "wgan-gp",
        }

        # Add discriminator config to training.gan (for ModelBuilder)
        # ModelBuilder looks at config.training.gan.discriminator
        # Since we use dict, ModelBuilder getattr will fail but default to patchgan, which is fine
        config_dict["training"]["gan"] = {
            "discriminator": {"type": "patchgan", "kwargs": {"ndf": 16, "n_layers": 2}}
        }

        config_path = temp_experiment_dir / "gan_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = TrainingSettings.from_yaml(str(config_path))

        # Hack: Set task/training_mode on frozen config to ensure ModelBuilder detects GAN
        # This is needed because 'task' is not in TrainingSettings schema but ModelBuilder checks it,
        # and 'training_mode' as extra field might not be accessible via getattr in Pydantic v2.
        object.__setattr__(config, "task", "gan")
        if config.training:
            object.__setattr__(config.training, "training_mode", "gan")

        build_container(config)

        director = TrainingEnvironmentDirector(config)
        env = director.build_environment()

        # Assert discriminator was created
        assert env.discriminator is not None

        strategy = GANTrainingStrategy(env=env)
        # strategy.setup()

        # Run minimal training loop
        epoch = 0
        for i, batch in enumerate(synthetic_dataloader):
            input_batch, target_batch, mask = batch

            step_configs = strategy.train_step(
                batch=batch,
                epoch=epoch,
                input_batch=input_batch,
                target_batch=target_batch,
                iteration=i,
            )

            losses = {}
            for config in step_configs:
                loss = config["closure"]()
                if config.get("name") == "discriminator":
                    losses["d_total_loss"] = loss
                elif config.get("name") == "generator":
                    losses["g_total_loss"] = loss
                losses.update(strategy.get_last_metrics())

            # GAN returns metrics dict with floats
            assert "g_total_loss" in losses
            if "d_total_loss" in losses:
                val = losses["d_total_loss"]
                if isinstance(val, torch.Tensor):
                    assert val.item() > 0
                else:
                    assert val > 0

    def test_diffusion_execution(self, temp_experiment_dir, synthetic_dataloader):
        """Test DiffusionTrainingStrategy execution."""
        config_dict = self.create_base_config(temp_experiment_dir)
        config_dict["training"][
            "strategy_class"
        ] = "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy"
        config_dict["training"]["training_mode"] = "diffusion"
        config_dict["training"]["diffusion"] = {
            "timesteps": 10,  # Small number for test
            "noise_schedule": "linear",
            "type": "GaussianDiffusion",
        }
        config_dict["model"][
            "model_type"
        ] = "standard_unet"  # Can be used for diffusion too if verifying simple execution

        config_path = temp_experiment_dir / "diffusion_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = TrainingSettings.from_yaml(str(config_path))
        build_container(config)

        director = TrainingEnvironmentDirector(config)
        env = director.build_environment()

        strategy = DiffusionTrainingStrategy(env=env)
        # strategy.setup()

        # Run minimal training loop
        epoch = 0
        for i, batch in enumerate(synthetic_dataloader):
            input_batch, target_batch, mask = batch

            step_configs = strategy.train_step(
                batch=batch,
                epoch=epoch,
                input_batch=input_batch,
                target_batch=target_batch,
                iteration=i,
                batch_data={"mask": mask},
            )

            losses = {}
            for config in step_configs:
                loss = config["closure"]()
                losses["g_total_loss"] = loss
                losses.update(strategy.get_last_metrics())

            assert "loss" in losses or "g_total_loss" in losses
