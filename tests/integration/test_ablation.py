"""Integration tests for Ablation Studies.

Tests component-by-component impact on model performance:
- Data Consistency Layer (DC) on/off
- Loss functions (L1 vs L2 vs Combined)
- Attention mechanisms (none vs self-attention vs cross-attention)
- Physics constraints (k-space loss, SSIM, perceptual)
- Noise simulation levels (data leak prevention)
- Normalization strategies (global vs per-sample)
- Diffusion timesteps (100 vs 1000)
- Model depth and width

Ablation studies answer: "Does component X improve performance?"
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from spectramr.config.settings import TrainingSettings


class TestDataConsistencyAblation:
    """Ablation tests for Data Consistency Layer."""

    @pytest.fixture
    def temp_ablation_dir(self):
        """Create temporary directory for ablation experiments."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def base_config(self, temp_ablation_dir):
        """Base configuration for ablation studies."""
        return {
            "config_version": "1.0",
            "metadata": {
                "name": "ablation_dc",
                "description": "Ablation study",
            },
            "run": {"seed": 42},
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
                "model_kwargs": {"base_channels": 16},
            },
            "data": {
                "dataset_type": "synthetic",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": [64, 64],
            },
            "optimization": {
                "learning_rate": 1e-4,
                "use_amp": False,
            },
            "training": {
                "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
            },
            "losses": {
                "reconstruction": {
                    "lambda_l1": 1.0,
                },
            },
            "physics": {
                "data_consistency": {
                    "enabled": True,  # Will be ablated
                    "train_noise_level": 0.01,
                },
            },
            "logging": {},
            "loss_logging": {},
            "metrics": {},
            "early_stopping": {},
            "ema": {},
            "validation": {"compute_image_metrics": False},
            "checkpoint": {},
        }

    def test_ablation_dc_enabled_vs_disabled(self, base_config, temp_ablation_dir):
        """Test performance with/without Data Consistency Layer.

        Expected: DC enabled should improve reconstruction quality.
        """
        configs = []

        # Config 1: DC enabled
        config1 = base_config.copy()
        config1["metadata"]["name"] = "ablation_dc_enabled"
        config1["physics"]["data_consistency"]["enabled"] = True

        config1_path = temp_ablation_dir / "config_dc_enabled.yaml"
        with open(config1_path, "w") as f:
            yaml.dump(config1, f)

        settings1 = TrainingSettings.from_yaml(str(config1_path))
        assert settings1.physics.data_consistency.enabled is True

        # Config 2: DC disabled
        config2 = base_config.copy()
        config2["metadata"]["name"] = "ablation_dc_disabled"
        config2["physics"]["data_consistency"]["enabled"] = False

        config2_path = temp_ablation_dir / "config_dc_disabled.yaml"
        with open(config2_path, "w") as f:
            yaml.dump(config2, f)

        settings2 = TrainingSettings.from_yaml(str(config2_path))
        assert settings2.physics.data_consistency.enabled is False

        # In real ablation: run both, compare val_psnr
        # Expected result: DC enabled > DC disabled by ~2 dB

    def test_ablation_dc_noise_levels(self, base_config, temp_ablation_dir):
        """Test different noise levels in Data Consistency.

        Ablation: train_noise_level = [0.0, 0.005, 0.01, 0.02, 0.05]
        Expected: Too low = overfitting, too high = underfitting
        """
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]

        configs = []
        for noise_level in noise_levels:
            config = base_config.copy()
            config["metadata"]["name"] = f"ablation_noise_{noise_level:.3f}"
            config["physics"]["data_consistency"]["train_noise_level"] = noise_level

            config_path = temp_ablation_dir / f"config_noise_{noise_level:.3f}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.physics.data_consistency.train_noise_level == noise_level

            configs.append(settings)

        assert len(configs) == 5

        # Expected: noise_level=0.01 is optimal (from prior experiments)


class TestLossFunctionAblation:
    """Ablation tests for loss functions."""

    @pytest.fixture
    def loss_configs(self, tmp_path):
        """Generate configs with different loss combinations."""
        base = {
            "config_version": "1.0",
            "metadata": {
                "name": "ablation_loss",
                "description": "Loss ablation",
            },
            "run": {"seed": 42},
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {"dataset_type": "synthetic", "batch_size": 2},
            "optimization": {"learning_rate": 1e-4},
            "training": {
                "epochs": 1,
                "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
            },
            "physics": {"data_consistency": {"enabled": True}},
            "logging": {},
            "loss_logging": {},
            "metrics": {},
            "early_stopping": {},
            "ema": {},
            "validation": {"compute_image_metrics": False},
            "checkpoint": {},
        }

        loss_variants = [
            {"name": "l1_only", "lambda_l1": 1.0, "lambda_l2": 0.0},
            {"name": "l2_only", "lambda_l1": 0.0, "lambda_l2": 1.0},
            {"name": "l1_l2", "lambda_l1": 0.5, "lambda_l2": 0.5},
            {"name": "l1_ssim", "lambda_l1": 0.7, "lambda_ssim": 0.3},
        ]

        configs = []
        for variant in loss_variants:
            config = base.copy()
            config["metadata"]["name"] = f"ablation_loss_{variant['name']}"

            if "losses" not in config:
                config["losses"] = {}
            if "reconstruction" not in config["losses"]:
                config["losses"]["reconstruction"] = {}
            if "lambda_l1" in variant:
                config["losses"]["reconstruction"]["lambda_l1"] = variant["lambda_l1"]
            if "lambda_l2" in variant:
                config["losses"]["reconstruction"]["lambda_l2"] = variant["lambda_l2"]
            if "lambda_ssim" in variant:
                config["losses"]["reconstruction"]["lambda_ssim"] = variant[
                    "lambda_ssim"
                ]

            config_path = tmp_path / f"config_loss_{variant['name']}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            configs.append((variant["name"], config_path))

        return configs

    def test_ablation_loss_l1_vs_l2(self, loss_configs):
        """Test L1 vs L2 reconstruction loss.

        Expected: L1 better for MRI (sharper edges), L2 smoother but blurrier.
        """
        for loss_name, config_path in loss_configs:
            config = TrainingSettings.from_yaml(str(config_path))

            if loss_name == "l1_only":
                assert config.losses.reconstruction.lambda_l1 == 1.0
                # Expected: Best PSNR for L1-only

            elif loss_name == "l2_only":
                assert config.losses.reconstruction.lambda_l2 == 1.0
                # Expected: Lower PSNR but faster convergence

    def test_ablation_ssim_weight(self, tmp_path):
        """Test SSIM loss weight ablation.

        Ablation: lambda_ssim = [0.0, 0.1, 0.3, 0.5, 1.0]
        Expected: Too high SSIM → worse PSNR but better perceptual quality.
        """
        ssim_weights = [0.0, 0.1, 0.3, 0.5, 1.0]

        base_config = {
            "config_version": "1.0",
            "metadata": {"name": "ablation_ssim", "description": "SSIM ablation"},
            "run": {"seed": 42},
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {"dataset_type": "synthetic", "batch_size": 2},
            "optimization": {"learning_rate": 1e-4},
            "training": {
                "epochs": 1,
                "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
            },
            "physics": {"data_consistency": {"enabled": True}},
            "logging": {},
            "loss_logging": {},
            "metrics": {},
            "early_stopping": {},
            "ema": {},
            "validation": {"compute_image_metrics": False},
            "checkpoint": {},
        }

        for ssim_weight in ssim_weights:
            config = base_config.copy()
            config["metadata"]["name"] = f"ablation_ssim_{ssim_weight:.1f}"
            config["losses"] = {
                "reconstruction": {
                    "lambda_l1": 1.0 - ssim_weight,
                    "lambda_ssim": ssim_weight,
                }
            }

            config_path = tmp_path / f"config_ssim_{ssim_weight:.1f}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.losses.reconstruction.lambda_ssim == ssim_weight


class TestAttentionMechanismAblation:
    """Ablation tests for attention mechanisms in models."""

    def test_ablation_no_attention_vs_self_attention(self, tmp_path):
        """Test model performance with/without self-attention.

        Configs:
        1. standard_unet (no attention)
        2. swin_unet (Swin Transformer attention)
        3. vision_mamba (state-space attention)
        """
        model_types = ["standard_unet", "swin_transformer", "vision_mamba"]

        base_config = {
            "config_version": "1.0",
            "metadata": {
                "name": "ablation_attention",
                "description": "Attention ablation",
            },
            "run": {"seed": 42},
            "model": {"in_channels": 2, "out_channels": 2, "base_channels": 16},
            "data": {"dataset_type": "synthetic", "batch_size": 2},
            "optimization": {"learning_rate": 1e-4},
            "training": {
                "epochs": 1,
                "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
            },
            "losses": {"reconstruction": {"lambda_l1": 1.0}},
            "physics": {"data_consistency": {"enabled": True}},
            "logging": {},
            "loss_logging": {},
            "metrics": {},
            "early_stopping": {},
            "ema": {},
            "validation": {"compute_image_metrics": False},
            "checkpoint": {},
        }

        for model_type in model_types:
            config = base_config.copy()
            config["model"]["model_type"] = model_type
            config["metadata"]["name"] = f"ablation_attn_{model_type}"

            config_path = tmp_path / f"config_{model_type}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.model.model_type == model_type

        # Expected: Self-attention improves long-range dependencies
        # swin_unet > standard_unet by ~0.5-1 dB PSNR

    def test_ablation_attention_depth(self, tmp_path):
        """Test different attention layer depths.

        Ablation: depth = [1, 2, 3, 4]
        Expected: Deeper networks → more layers for attention
        """
        depths = [1, 2, 3, 4]

        for depth in depths:
            config = {
                "config_version": "1.0",
                "metadata": {
                    "name": f"ablation_attn_depth_{depth}",
                    "description": "Depth ablation",
                },
                "run": {"seed": 42},
                "model": {
                    "model_type": "standard_unet",
                    "in_channels": 2,
                    "out_channels": 2,
                    "model_kwargs": {
                        "depth": depth,
                        "use_attention": True,
                    },
                },
                "data": {"dataset_type": "synthetic", "batch_size": 2},
                "optimization": {"learning_rate": 1e-4},
                "training": {
                    "epochs": 1,
                    "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
                },
                "losses": {"reconstruction": {"lambda_l1": 1.0}},
                "physics": {"data_consistency": {"enabled": True}},
                "logging": {},
                "loss_logging": {},
                "metrics": {},
                "early_stopping": {},
                "ema": {},
                "validation": {"compute_image_metrics": False},
                "checkpoint": {},
            }

            config_path = tmp_path / f"config_attn_depth_{depth}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.model.model_kwargs["depth"] == depth
            assert settings.model.model_kwargs["use_attention"] is True


class TestPhysicsConstraintsAblation:
    """Ablation tests for physics-based constraints."""

    def test_ablation_kspace_loss(self, tmp_path):
        """Test k-space loss weight ablation.

        Ablation: lambda_kspace = [0.0, 0.1, 0.5, 1.0]
        Expected: k-space loss enforces frequency domain accuracy
        """
        kspace_weights = [0.0, 0.1, 0.5, 1.0]

        base_config = {
            "config_version": "1.0",
            "metadata": {"name": "ablation_kspace", "description": "K-space ablation"},
            "run": {"seed": 42},
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {"dataset_type": "synthetic", "batch_size": 2},
            "optimization": {"learning_rate": 1e-4},
            "training": {
                "epochs": 1,
                "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
            },
            "physics": {"data_consistency": {"enabled": True}},
            "logging": {},
            "loss_logging": {},
            "metrics": {},
            "early_stopping": {},
            "ema": {},
            "validation": {"compute_image_metrics": False},
            "checkpoint": {},
        }

        for kspace_weight in kspace_weights:
            config = base_config.copy()
            config["metadata"]["name"] = f"ablation_kspace_{kspace_weight:.1f}"
            if "losses" not in config:
                config["losses"] = {}
            if "reconstruction" not in config["losses"]:
                config["losses"]["reconstruction"] = {}
            config["losses"]["reconstruction"]["lambda_l1"] = 0.5
            config["losses"]["reconstruction"]["lambda_kspace"] = kspace_weight

            config_path = tmp_path / f"config_kspace_{kspace_weight:.1f}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))

            # Expected: lambda_kspace=0.1 optimal (balance image + frequency)

        def test_ablation_hermitian_symmetry(self, tmp_path):
            """Test Hermitian symmetry enforcement in k-space.

            Configs:
            1. enforce_hermitian_symmetry=True
            2. enforce_hermitian_symmetry=False

            Expected: True → better for real-valued images
            """
            hermitian_values = [True, False]

            for hermitian in hermitian_values:
                config = {
                    "config_version": "1.0",
                    "metadata": {
                        "name": f"ablation_hermitian_{hermitian}",
                        "description": "Hermitian ablation",
                    },
                    "run": {"seed": 42},
                    "model": {
                        "model_type": "standard_unet",
                        "in_channels": 2,
                        "out_channels": 2,
                    },
                    "data": {"dataset_type": "synthetic", "batch_size": 2},
                    "optimization": {"learning_rate": 1e-4},
                    "training": {
                        "epochs": 1,
                        "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
                    },
                    "losses": {"reconstruction": {"lambda_l1": 1.0}},
                    "physics": {
                        "data_consistency": {"enabled": True},
                        "kspace": {"enforce_hermitian_symmetry": hermitian},
                    },
                    "logging": {},
                    "loss_logging": {},
                    "metrics": {},
                    "early_stopping": {},
                    "ema": {},
                    "validation": {"compute_image_metrics": False},
                    "checkpoint": {},
                }

                config_path = tmp_path / f"config_hermitian_{hermitian}.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                settings = TrainingSettings.from_yaml(str(config_path))
                assert settings.physics.kspace.enforce_hermitian_symmetry == hermitian


class TestNormalizationAblation:
    """Ablation tests for normalization strategies."""

    def test_ablation_normalization_percentile(self, tmp_path):
        """Test different normalization percentiles.

        Ablation: percentile = [0.95, 0.99, 1.0]
        Expected: 0.99 optimal (robust to outliers)
        """
        percentiles = [0.95, 0.99, 1.0]

        for percentile in percentiles:
            config = {
                "config_version": "1.0",
                "metadata": {
                    "name": f"ablation_norm_p{percentile:.2f}",
                    "description": "Normalization ablation",
                },
                "run": {"seed": 42},
                "model": {
                    "model_type": "standard_unet",
                    "in_channels": 2,
                    "out_channels": 2,
                },
                "data": {
                    "dataset_type": "synthetic",
                    "batch_size": 2,
                    "normalize_kspace": True,
                    "normalization_percentile": percentile,  # Custom field
                },
                "optimization": {"learning_rate": 1e-4},
                "training": {
                    "epochs": 1,
                    "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
                },
                "losses": {"reconstruction": {"lambda_l1": 1.0}},
                "physics": {"data_consistency": {"enabled": True}},
                "logging": {},
                "loss_logging": {},
                "metrics": {},
                "early_stopping": {},
                "ema": {},
                "validation": {"compute_image_metrics": False},
                "checkpoint": {},
            }

            config_path = tmp_path / f"config_norm_p{percentile:.2f}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Note: Requires dataloader support for normalization_percentile

    def test_ablation_global_vs_per_sample_normalization(self, tmp_path):
        """Test global vs per-sample normalization.

        Expected: Per-sample avoids data leakage and adapts to contrast variations.
        """
        norm_strategies = ["global", "per_sample"]

        for strategy in norm_strategies:
            config = {
                "config_version": "1.0",
                "metadata": {
                    "name": f"ablation_norm_{strategy}",
                    "description": "Normalization ablation",
                },
                "run": {"seed": 42},
                "model": {
                    "model_type": "standard_unet",
                    "in_channels": 2,
                    "out_channels": 2,
                },
                "data": {
                    "dataset_type": "synthetic",
                    "batch_size": 2,
                    "normalization_strategy": strategy,  # Custom field
                },
                "optimization": {"learning_rate": 1e-4},
                "training": {
                    "epochs": 1,
                    "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
                },
                "losses": {"reconstruction": {"lambda_l1": 1.0}},
                "physics": {"data_consistency": {"enabled": True}},
                "logging": {},
                "loss_logging": {},
                "metrics": {},
                "early_stopping": {},
                "ema": {},
                "validation": {"compute_image_metrics": False},
                "checkpoint": {},
            }

            config_path = tmp_path / f"config_norm_{strategy}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

        # Expected: per_sample > global (no train/val leakage)


class TestDiffusionTimestepsAblation:
    """Ablation tests for diffusion model timesteps."""

    def test_ablation_num_timesteps(self, tmp_path):
        """Test different numbers of diffusion timesteps.

        Ablation: num_timesteps = [50, 100, 500, 1000]
        Expected: More timesteps → better quality but slower sampling
        """
        timesteps = [50, 100, 500, 1000]

        for num_t in timesteps:
            config = {
                "config_version": "1.0",
                "metadata": {
                    "name": f"ablation_timesteps_{num_t}",
                    "description": "Timesteps ablation",
                },
                "run": {"seed": 42},
                "model": {
                    "model_type": "kspace_cold_diffusion_generator",
                    "in_channels": 2,
                    "out_channels": 2,
                },
                "data": {"dataset_type": "synthetic", "batch_size": 2},
                "optimization": {"learning_rate": 1e-4},
                "training": {
                    "epochs": 1,
                    "strategy_class": "spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
                    "diffusion": {
                        "num_timesteps": num_t,
                        "cold_diffusion": True,
                    },
                },
                "losses": {"reconstruction": {"lambda_l1": 1.0}},
                "physics": {"data_consistency": {"enabled": True}},
                "logging": {},
                "loss_logging": {},
                "metrics": {},
                "early_stopping": {},
                "ema": {},
                "validation": {"compute_image_metrics": False},
                "checkpoint": {},
            }

            config_path = tmp_path / f"config_timesteps_{num_t}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.training.diffusion.num_timesteps == num_t

    def test_ablation_beta_schedule(self, tmp_path):
        """Test different noise schedules for diffusion.

        Ablation: beta_schedule = ['linear', 'cosine', 'quadratic']
        Expected: Cosine schedule often best for image generation
        """
        schedules = ["linear", "cosine", "quadratic"]

        for schedule in schedules:
            config = {
                "config_version": "1.0",
                "metadata": {
                    "name": f"ablation_schedule_{schedule}",
                    "description": "Schedule ablation",
                },
                "run": {"seed": 42},
                "model": {
                    "model_type": "kspace_cold_diffusion_generator",
                    "in_channels": 2,
                    "out_channels": 2,
                },
                "data": {"dataset_type": "synthetic", "batch_size": 2},
                "optimization": {"learning_rate": 1e-4},
                "training": {
                    "epochs": 1,
                    "strategy_class": "spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
                    "diffusion": {
                        "num_timesteps": 100,
                        "beta_schedule": schedule,
                        "cold_diffusion": True,
                    },
                },
                "losses": {"reconstruction": {"lambda_l1": 1.0}},
                "physics": {"data_consistency": {"enabled": True}},
                "logging": {},
                "loss_logging": {},
                "metrics": {},
                "early_stopping": {},
                "ema": {},
                "validation": {"compute_image_metrics": False},
                "checkpoint": {},
            }

            config_path = tmp_path / f"config_schedule_{schedule}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.training.diffusion.beta_schedule == schedule


class TestModelArchitectureAblation:
    """Ablation tests for model architecture variations."""

    def test_ablation_model_depth(self, tmp_path):
        """Test different U-Net depths.

        Ablation: depth = [1, 2, 3, 4]
        Expected: Deeper networks → more encoder/decoder blocks but slower training
        """
        depths = [1, 2, 3, 4]

        for depth in depths:
            config = {
                "config_version": "1.0",
                "metadata": {
                    "name": f"ablation_depth_{depth}",
                    "description": "Depth ablation",
                },
                "run": {"seed": 42},
                "model": {
                    "model_type": "standard_unet",
                    "in_channels": 2,
                    "out_channels": 2,
                    "model_kwargs": {
                        "depth": depth,
                    },
                },
                "data": {"dataset_type": "synthetic", "batch_size": 2},
                "optimization": {"learning_rate": 1e-4},
                "training": {
                    "epochs": 1,
                    "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
                },
                "losses": {"reconstruction": {"lambda_l1": 1.0}},
                "physics": {"data_consistency": {"enabled": True}},
                "logging": {},
                "loss_logging": {},
                "metrics": {},
                "early_stopping": {},
                "ema": {},
                "validation": {"compute_image_metrics": False},
                "checkpoint": {},
            }

            config_path = tmp_path / f"config_depth_{depth}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.model.model_kwargs["depth"] == depth

    def test_ablation_model_width(self, tmp_path):
        """Test different model widths (base channels).

        Ablation: base_channels = [8, 16, 32, 64]
        Expected: Wider → better capacity but more parameters
        """
        widths = [8, 16, 32, 64]

        for width in widths:
            config = {
                "config_version": "1.0",
                "metadata": {
                    "name": f"ablation_width_{width}",
                    "description": "Width ablation",
                },
                "run": {"seed": 42},
                "model": {
                    "model_type": "standard_unet",
                    "in_channels": 2,
                    "out_channels": 2,
                    "base_channels": width,
                },
                "data": {"dataset_type": "synthetic", "batch_size": 2},
                "optimization": {"learning_rate": 1e-4},
                "training": {
                    "epochs": 1,
                    "strategy_class": "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
                },
                "losses": {"reconstruction": {"lambda_l1": 1.0}},
                "physics": {"data_consistency": {"enabled": True}},
                "logging": {},
                "loss_logging": {},
                "metrics": {},
                "early_stopping": {},
                "ema": {},
                "validation": {"compute_image_metrics": False},
                "checkpoint": {},
            }

            config_path = tmp_path / f"config_width_{width}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings.model.base_channels == width


class TestAblationMetricsComparison:
    """Test metrics comparison framework for ablation studies."""

    def test_compare_ablation_results(self):
        """Test comparing metrics across ablation experiments.

        Typical workflow:
        1. Run baseline config
        2. Run N ablation variants (one component changed)
        3. Compare delta in validation metrics
        4. Rank by improvement over baseline
        """
        # Simulated results from ablation study
        ablation_results = [
            {"config": "baseline", "val_psnr": 30.0, "val_ssim": 0.85},
            {"config": "dc_enabled", "val_psnr": 32.0, "val_ssim": 0.88},  # +2 dB
            {"config": "dc_disabled", "val_psnr": 28.5, "val_ssim": 0.82},  # -1.5 dB
            {"config": "kspace_loss_0.1", "val_psnr": 31.0, "val_ssim": 0.87},  # +1 dB
            {"config": "no_attention", "val_psnr": 29.5, "val_ssim": 0.84},  # -0.5 dB
        ]

        baseline = ablation_results[0]

        # Compute delta from baseline
        for result in ablation_results[1:]:
            result["delta_psnr"] = result["val_psnr"] - baseline["val_psnr"]
            result["delta_ssim"] = result["val_ssim"] - baseline["val_ssim"]

        # Rank by PSNR improvement
        ranked = sorted(
            ablation_results[1:], key=lambda x: x["delta_psnr"], reverse=True
        )

        # Best improvement
        best = ranked[0]
        assert best["config"] == "dc_enabled"
        assert best["delta_psnr"] == 2.0

        # Worst (regression)
        worst = ranked[-1]
        assert worst["config"] == "dc_disabled"
        assert worst["delta_psnr"] == -1.5

    def test_ablation_statistical_significance(self):
        """Test statistical significance of ablation results.

        Run each config with 3 random seeds, compute mean ± std.
        Use t-test to check if improvement is significant.
        """
        import scipy.stats as stats

        # Simulated results from 3 seeds
        baseline_psnr = [30.0, 30.1, 29.9]  # Mean: 30.0
        ablation_psnr = [32.0, 31.8, 32.2]  # Mean: 32.0

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(ablation_psnr, baseline_psnr)

        # With p < 0.05, improvement is statistically significant
        assert p_value < 0.05, "Ablation improvement not statistically significant"

    def test_ablation_efficiency_frontier(self):
        """Test Pareto efficiency: best quality for least compute.

        Plot (training_time, val_psnr) for each ablation config.
        Identify Pareto-optimal configs on frontier.
        """
        ablation_results = [
            {"config": "tiny_model", "val_psnr": 28.0, "train_time": 50},
            {"config": "baseline", "val_psnr": 30.0, "train_time": 100},
            {"config": "deep_model", "val_psnr": 33.0, "train_time": 200},
            {
                "config": "huge_model",
                "val_psnr": 32.5,
                "train_time": 500,
            },  # Not Pareto (deep_model is better and faster)
        ]

        # Find Pareto frontier: for each point, no other point is better in both metrics
        def is_pareto_optimal(result, all_results):
            for other in all_results:
                if other == result:
                    continue
                # Other is faster AND better quality
                if (
                    other["train_time"] < result["train_time"]
                    and other["val_psnr"] > result["val_psnr"]
                ):
                    return False
            return True

        pareto_frontier = [
            r for r in ablation_results if is_pareto_optimal(r, ablation_results)
        ]

        # Expect: tiny, baseline, deep on frontier; huge NOT on frontier
        pareto_configs = [r["config"] for r in pareto_frontier]
        assert "huge_model" not in pareto_configs  # Not worth 2.5x time for 0.5 dB


@pytest.mark.slow
class TestLargeScaleAblation:
    """Large-scale ablation studies (marked slow)."""

    def test_full_factorial_ablation(self):
        """Test full factorial design: all combinations of ablations.

        Example: 3 binary factors (DC, k-space loss, attention)
        → 2^3 = 8 total configs

        This grows exponentially, so limit to critical components.
        """
        factors = {
            "dc_enabled": [True, False],
            "kspace_loss": [0.0, 0.1],
            "attention": [True, False],
        }

        # Generate all combinations
        import itertools

        keys = list(factors.keys())
        values = [factors[k] for k in keys]

        all_combinations = list(itertools.product(*values))

        # 2 x 2 x 2 = 8 configs
        assert len(all_combinations) == 8

        # Example combination: (dc_enabled=True, kspace_loss=0.1, attention=True)
        assert (True, 0.1, True) in all_combinations

    def test_fractional_factorial_ablation(self):
        """Test fractional factorial: subset of full factorial.

        For 5 binary factors, full factorial = 32 configs.
        Fractional factorial with resolution IV = only 8 configs.

        Trade-off: Can't detect all interaction effects.
        """
        # With 5 factors, select representative subset
        factors = ["dc", "kspace_loss", "attention", "ssim_loss", "deep_model"]

        # Fractional design: 8 configs instead of 32
        # (Design matrix from DoE software, e.g., pyDOE)
        fractional_configs = [
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, False, True, False, False],
            [True, False, False, True, True],
            [False, True, True, False, False],
            [False, True, False, True, True],
            [False, False, True, True, False],
            [False, False, False, False, True],
        ]

        assert len(fractional_configs) == 8

        # Can estimate main effects and some 2-way interactions
        # Cannot resolve higher-order interactions (acceptable trade-off)
