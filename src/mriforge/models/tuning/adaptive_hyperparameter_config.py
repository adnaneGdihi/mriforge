"""Adaptive Hyperparameter Configuration System
===========================================

Comprehensive hyperparameter configuration system that adapts based on:
- Model type (KAN, diffusion, transformer, standard GAN, etc.)
- Training strategy (GAN, reconstruction, diffusion, VAE, etc.)
- Data characteristics (2D/3D, modality, size, etc.)
- Task type (super-resolution, reconstruction, generation, etc.)

This system provides intelligent defaults and optimization ranges for HPO.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelCategory(Enum):
    """Model architecture categories."""

    KAN_BASED = "kan_based"
    DIFFUSION_BASED = "diffusion_based"
    TRANSFORMER_BASED = "transformer_based"
    STANDARD_GAN = "standard_gan"
    VAE_BASED = "vae_based"
    RECONSTRUCTION_BASED = "reconstruction_based"


class TrainingStrategy(Enum):
    """Training strategy types."""

    GAN = "gan"
    RECONSTRUCTION = "reconstruction"
    DIFFUSION = "diffusion"
    VAE = "vae"
    MAE = "mae"
    VQ_VAE = "vq_vae"
    DOMAIN_ADAPTATION = "domain_adaptation"
    KSPACE_COLD_DIFFUSION = "kspace_cold_diffusion"


class DataModality(Enum):
    """Data modality types for MRI data."""

    MRI_2D = "mri_2d"
    MRI_3D = "mri_3d"
    MRI_KSPACE = "mri_kspace"
    MRI_MULTICONTRAST = "mri_multicontrast"


class TaskType(Enum):
    """Task types."""

    SUPER_RESOLUTION = "super_resolution"
    RECONSTRUCTION = "reconstruction"
    GENERATION = "generation"
    DENOISING = "denoising"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


@dataclass
class HyperparameterRange:
    """Represents a hyperparameter range for optimization."""

    min_val: float | int
    max_val: float | int
    default: float | int | str
    log_scale: bool = False
    categorical: list[Any] | None = None

    def to_optuna_suggest(self, name: str, trial: Any) -> Any:
        """Convert to optuna suggestion."""
        if self.categorical:
            return trial.suggest_categorical(name, self.categorical)
        if isinstance(self.min_val, int) and isinstance(self.max_val, int):
            return trial.suggest_int(
                name,
                self.min_val,
                self.max_val,
                log=self.log_scale,
            )
        return trial.suggest_float(
            name,
            self.min_val,
            self.max_val,
            log=self.log_scale,
        )


@dataclass
class ModelHyperparameters:
    """Hyperparameters specific to a model type."""

    learning_rate: HyperparameterRange
    weight_decay: HyperparameterRange
    gradient_clip: HyperparameterRange
    batch_size: HyperparameterRange
    optimizer_type: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            min_val=0,
            max_val=0,
            default="adamw",
            categorical=["adamw", "adam", "sgd"],
        ),
    )
    lr_scheduler: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            min_val=0,
            max_val=0,
            default="cosine",
            categorical=["cosine", "linear", "step", "plateau"],
        ),
    )
    # Model-specific parameters
    model_specific: dict[str, HyperparameterRange] = field(default_factory=dict)


@dataclass
class TrainingHyperparameters:
    """Hyperparameters specific to training strategy."""

    epochs: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(10, 500, 100),
    )
    warmup_steps: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(0, 10000, 1000),
    )
    loss_weights: dict[str, HyperparameterRange] = field(default_factory=dict)
    # Strategy-specific parameters
    strategy_specific: dict[str, HyperparameterRange] = field(default_factory=dict)


@dataclass
class DataHyperparameters:
    """Hyperparameters that depend on data characteristics."""

    augmentation_probability: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(0.0, 1.0, 0.5),
    )
    num_workers: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(1, 16, 4),
    )
    prefetch_factor: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(1, 4, 2),
    )
    # Data-specific parameters
    data_specific: dict[str, HyperparameterRange] = field(default_factory=dict)


class AdaptiveHyperparameterConfig:
    """Adaptive hyperparameter configuration system."""

    def __init__(self) -> None:
        """__init__."""
        self.logger = logging.getLogger(__name__)
        self._initialize_hyperparameter_databases()

    def _initialize_hyperparameter_databases(self) -> None:
        """Initialize hyperparameter databases for different model/training combinations."""
        self.model_hyperparams = self._create_model_hyperparameters()
        self.training_hyperparams = self._create_training_hyperparameters()
        self.data_hyperparams = self._create_data_hyperparameters()

    def _create_model_hyperparameters(self) -> dict[str, ModelHyperparameters]:
        """Create model-specific hyperparameter configurations."""
        return {
            # KAN-based models - very conservative parameters
            "kan": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-7, 1e-5, 5e-6, log_scale=True),
                weight_decay=HyperparameterRange(1e-6, 1e-4, 1e-5, log_scale=True),
                gradient_clip=HyperparameterRange(0.1, 1.0, 0.5),
                batch_size=HyperparameterRange(1, 8, 4),
                model_specific={
                    "num_bases": HyperparameterRange(4, 16, 8),
                    "basis_trainable": HyperparameterRange(
                        0,
                        1,
                        1,
                        categorical=[True, False],
                    ),
                    "wavelet_family": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["db4", "haar", "coif1"],
                    ),
                },
            ),
            "vit_kan": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-7, 1e-5, 2e-6, log_scale=True),
                weight_decay=HyperparameterRange(1e-6, 1e-4, 5e-6, log_scale=True),
                gradient_clip=HyperparameterRange(0.1, 1.0, 0.3),
                batch_size=HyperparameterRange(1, 4, 2),
                model_specific={
                    "patch_size": HyperparameterRange(8, 32, 16),
                    "num_heads": HyperparameterRange(4, 16, 8),
                    "mlp_ratio": HyperparameterRange(2.0, 8.0, 4.0),
                },
            ),
            "swin_transformer_kan": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-7, 1e-5, 1e-6, log_scale=True),
                weight_decay=HyperparameterRange(1e-6, 1e-4, 1e-5, log_scale=True),
                gradient_clip=HyperparameterRange(0.1, 1.0, 0.5),
                batch_size=HyperparameterRange(1, 4, 2),
                model_specific={
                    "window_size": HyperparameterRange(4, 16, 8),
                    "num_heads": HyperparameterRange(4, 12, 6),
                    "mlp_ratio": HyperparameterRange(2.0, 6.0, 4.0),
                },
            ),
            # Diffusion models - moderate parameters
            "diffusion": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-6, 1e-4, 2e-5, log_scale=True),
                weight_decay=HyperparameterRange(1e-5, 1e-3, 1e-4, log_scale=True),
                gradient_clip=HyperparameterRange(0.5, 2.0, 1.0),
                batch_size=HyperparameterRange(2, 16, 8),
                model_specific={
                    "timesteps": HyperparameterRange(500, 2000, 1000),
                    "noise_schedule": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["cosine", "linear", "quadratic"],
                    ),
                    "cond_drop_prob": HyperparameterRange(0.0, 0.5, 0.1),
                    "guidance_scale": HyperparameterRange(1.0, 15.0, 7.5),
                },
            ),
            "cold_diffusion": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-6, 1e-4, 5e-5, log_scale=True),
                weight_decay=HyperparameterRange(1e-5, 1e-3, 5e-4, log_scale=True),
                gradient_clip=HyperparameterRange(0.5, 2.0, 1.5),
                batch_size=HyperparameterRange(1, 8, 4),
                model_specific={
                    "kspace_mask_type": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["random", "equispaced", "gaussian"],
                    ),
                    "acceleration_factor": HyperparameterRange(2, 16, 4),
                    "noise_level": HyperparameterRange(0.01, 0.5, 0.1),
                },
            ),
            # Transformer-based models
            "vit": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-6, 1e-4, 1e-5, log_scale=True),
                weight_decay=HyperparameterRange(1e-5, 1e-3, 1e-4, log_scale=True),
                gradient_clip=HyperparameterRange(0.5, 2.0, 1.0),
                batch_size=HyperparameterRange(2, 16, 8),
                model_specific={
                    "patch_size": HyperparameterRange(8, 32, 16),
                    "num_heads": HyperparameterRange(8, 16, 12),
                    "mlp_ratio": HyperparameterRange(2.0, 8.0, 4.0),
                    "drop_path_rate": HyperparameterRange(0.0, 0.3, 0.1),
                },
            ),
            "swin": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-6, 1e-4, 1e-5, log_scale=True),
                weight_decay=HyperparameterRange(1e-5, 1e-3, 1e-4, log_scale=True),
                gradient_clip=HyperparameterRange(0.5, 2.0, 1.0),
                batch_size=HyperparameterRange(2, 16, 8),
                model_specific={
                    "window_size": HyperparameterRange(4, 16, 8),
                    "num_heads": HyperparameterRange(6, 12, 8),
                    "mlp_ratio": HyperparameterRange(2.0, 6.0, 4.0),
                    "drop_path_rate": HyperparameterRange(0.0, 0.3, 0.1),
                },
            ),
            # Standard GAN models - more aggressive parameters
            "unet": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-5, 1e-3, 2e-4, log_scale=True),
                weight_decay=HyperparameterRange(1e-6, 1e-4, 1e-5, log_scale=True),
                gradient_clip=HyperparameterRange(1.0, 5.0, 1.0),
                batch_size=HyperparameterRange(4, 32, 16),
                model_specific={
                    "num_layers": HyperparameterRange(3, 8, 5),
                    "features_start": HyperparameterRange(32, 128, 64),
                    "use_attention": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=[True, False],
                    ),
                },
            ),
            "enhanced_unet": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-5, 1e-3, 1e-4, log_scale=True),
                weight_decay=HyperparameterRange(1e-6, 1e-4, 5e-6, log_scale=True),
                gradient_clip=HyperparameterRange(1.0, 5.0, 2.0),
                batch_size=HyperparameterRange(4, 32, 12),
                model_specific={
                    "num_layers": HyperparameterRange(4, 9, 6),
                    "features_start": HyperparameterRange(48, 192, 96),
                    "residual_connections": HyperparameterRange(
                        0,
                        1,
                        1,
                        categorical=[True, False],
                    ),
                },
            ),
            # VAE models
            "vae": ModelHyperparameters(
                learning_rate=HyperparameterRange(1e-5, 1e-3, 1e-4, log_scale=True),
                weight_decay=HyperparameterRange(1e-6, 1e-4, 1e-5, log_scale=True),
                gradient_clip=HyperparameterRange(0.5, 2.0, 1.0),
                batch_size=HyperparameterRange(4, 32, 16),
                model_specific={
                    "latent_dim": HyperparameterRange(64, 1024, 256),
                    "beta_kl": HyperparameterRange(0.1, 10.0, 1.0, log_scale=True),
                    "encoder_type": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["standard", "attention", "kan"],
                    ),
                },
            ),
        }

    def _create_training_hyperparameters(self) -> dict[str, TrainingHyperparameters]:
        """Create training strategy-specific hyperparameter configurations."""
        return {
            "gan": TrainingHyperparameters(
                epochs=HyperparameterRange(50, 300, 100),
                warmup_steps=HyperparameterRange(0, 5000, 1000),
                loss_weights={
                    "lambda_gan": HyperparameterRange(0.1, 10.0, 1.0),
                    "lambda_l1": HyperparameterRange(1.0, 100.0, 10.0, log_scale=True),
                    "lambda_perceptual": HyperparameterRange(0.1, 50.0, 10.0),
                },
                strategy_specific={
                    "gan_loss_type": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["wgan-gp", "lsgan", "vanilla"],
                    ),
                    "discriminator_updates": HyperparameterRange(1, 5, 1),
                    "generator_updates": HyperparameterRange(1, 3, 1),
                },
            ),
            "reconstruction": TrainingHyperparameters(
                epochs=HyperparameterRange(30, 200, 80),
                warmup_steps=HyperparameterRange(0, 2000, 500),
                loss_weights={
                    "lambda_l1": HyperparameterRange(0.1, 10.0, 1.0),
                    "lambda_perceptual": HyperparameterRange(0.1, 20.0, 5.0),
                    "lambda_mse": HyperparameterRange(0.1, 5.0, 1.0),
                },
                strategy_specific={
                    "use_ssim_loss": HyperparameterRange(
                        0,
                        1,
                        1,
                        categorical=[True, False],
                    ),
                    "ssim_weight": HyperparameterRange(0.1, 2.0, 0.5),
                },
            ),
            "diffusion": TrainingHyperparameters(
                epochs=HyperparameterRange(20, 100, 50),
                warmup_steps=HyperparameterRange(0, 10000, 2000),
                loss_weights={
                    "lambda_simple": HyperparameterRange(0.1, 10.0, 1.0),
                    "lambda_vlb": HyperparameterRange(0.01, 1.0, 0.001),
                },
                strategy_specific={
                    "timesteps": HyperparameterRange(500, 2000, 1000),
                    "noise_schedule": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["cosine", "linear", "sigmoid"],
                    ),
                    "use_ema": HyperparameterRange(0, 1, 1, categorical=[True, False]),
                    "ema_decay": HyperparameterRange(0.9, 0.999, 0.999),
                },
            ),
            "vae": TrainingHyperparameters(
                epochs=HyperparameterRange(50, 200, 100),
                warmup_steps=HyperparameterRange(0, 5000, 1000),
                loss_weights={
                    "beta_kl": HyperparameterRange(0.1, 10.0, 1.0, log_scale=True),
                    "reconstruction_weight": HyperparameterRange(0.1, 10.0, 1.0),
                },
                strategy_specific={
                    "anneal_kl": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=[True, False],
                    ),
                    "kl_start": HyperparameterRange(0.0, 1.0, 0.0),
                    "kl_end": HyperparameterRange(0.1, 5.0, 1.0),
                    "kl_anneal_steps": HyperparameterRange(1000, 50000, 10000),
                },
            ),
            "kspace_cold_diffusion": TrainingHyperparameters(
                epochs=HyperparameterRange(30, 150, 80),
                warmup_steps=HyperparameterRange(0, 8000, 1500),
                loss_weights={
                    "lambda_l1": HyperparameterRange(0.5, 20.0, 5.0),
                    "lambda_perceptual": HyperparameterRange(0.1, 10.0, 2.0),
                    "lambda_kspace": HyperparameterRange(0.1, 5.0, 1.0),
                },
                strategy_specific={
                    "acceleration_factor": HyperparameterRange(2, 16, 4),
                    "mask_type": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["random", "equispaced", "gaussian"],
                    ),
                    "noise_std": HyperparameterRange(0.01, 0.3, 0.1),
                },
            ),
        }

    def _create_data_hyperparameters(self) -> dict[str, DataHyperparameters]:
        """Create data-specific hyperparameter configurations for MRI modalities."""
        return {
            "mri_2d": DataHyperparameters(
                augmentation_probability=HyperparameterRange(0.3, 0.8, 0.5),
                num_workers=HyperparameterRange(2, 8, 4),
                prefetch_factor=HyperparameterRange(1, 3, 2),
                data_specific={
                    "intensity_norm": HyperparameterRange(
                        0,
                        1,
                        1,
                        categorical=[True, False],
                    ),
                    "bias_field_correction": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=[True, False],
                    ),
                    "motion_correction": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=[True, False],
                    ),
                },
            ),
            "mri_3d": DataHyperparameters(
                augmentation_probability=HyperparameterRange(0.2, 0.6, 0.3),
                num_workers=HyperparameterRange(1, 4, 2),
                prefetch_factor=HyperparameterRange(1, 2, 1),
                data_specific={
                    "slice_thickness": HyperparameterRange(1.0, 5.0, 2.0),
                    "interpolation": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["linear", "cubic", "nearest"],
                    ),
                    "registration": HyperparameterRange(
                        0,
                        1,
                        1,
                        categorical=[True, False],
                    ),
                },
            ),
            "mri_kspace": DataHyperparameters(
                augmentation_probability=HyperparameterRange(0.1, 0.4, 0.2),
                num_workers=HyperparameterRange(1, 6, 2),
                prefetch_factor=HyperparameterRange(1, 2, 1),
                data_specific={
                    "acceleration_factor": HyperparameterRange(2, 16, 4),
                    "mask_type": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["random", "equispaced", "gaussian"],
                    ),
                    "noise_std": HyperparameterRange(0.01, 0.3, 0.1),
                    "phase_encoding": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["linear", "centric", "segmented"],
                    ),
                },
            ),
            "mri_multicontrast": DataHyperparameters(
                augmentation_probability=HyperparameterRange(0.4, 0.9, 0.6),
                num_workers=HyperparameterRange(3, 10, 5),
                prefetch_factor=HyperparameterRange(2, 4, 2),
                data_specific={
                    "contrast_weighting": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=["t1", "t2", "flair", "pd"],
                    ),
                    "sequence_alignment": HyperparameterRange(
                        0,
                        1,
                        1,
                        categorical=[True, False],
                    ),
                    "intensity_matching": HyperparameterRange(
                        0,
                        1,
                        1,
                        categorical=[True, False],
                    ),
                    "motion_compensation": HyperparameterRange(
                        0,
                        1,
                        0,
                        categorical=[True, False],
                    ),
                },
            ),
        }

    def get_hyperparameters_for_config(
        self,
        model_type: str,
        training_strategy: str,
        data_modality: str = "mri_2d",
        task_type: str = "super_resolution",
    ) -> dict[str, Any]:
        """Get comprehensive hyperparameter configuration for given setup.

        Args:
            model_type: Type of model (e.g., 'kan', 'diffusion', 'unet')
            training_strategy: Training strategy (e.g., 'gan', 'reconstruction')
            data_modality: Data modality (e.g., 'mri_2d', 'mri_3d', 'mri_kspace', 'mri_multicontrast')
            task_type: Task type (e.g., 'super_resolution', 'reconstruction')

        Returns:
            Dictionary with all hyperparameter configurations

        """
        # Get base hyperparameters
        model_params = self.model_hyperparams.get(
            model_type,
            self.model_hyperparams["unet"],
        )
        training_params = self.training_hyperparams.get(
            training_strategy,
            self.training_hyperparams["reconstruction"],
        )
        data_params = self.data_hyperparams.get(
            data_modality,
            self.data_hyperparams["mri_2d"],
        )

        # Combine into comprehensive config
        config = {
            "model": {
                "learning_rate": model_params.learning_rate.default,
                "weight_decay": model_params.weight_decay.default,
                "gradient_clip": model_params.gradient_clip.default,
                "batch_size": model_params.batch_size.default,
                "optimizer_type": model_params.optimizer_type.default,
                "lr_scheduler": model_params.lr_scheduler.default,
                **{k: v.default for k, v in model_params.model_specific.items()},
            },
            "training": {
                "epochs": training_params.epochs.default,
                "warmup_steps": training_params.warmup_steps.default,
                **{k: v.default for k, v in training_params.loss_weights.items()},
                **{k: v.default for k, v in training_params.strategy_specific.items()},
            },
            "data": {
                "augmentation_probability": data_params.augmentation_probability.default,
                "num_workers": data_params.num_workers.default,
                "prefetch_factor": data_params.prefetch_factor.default,
                **{k: v.default for k, v in data_params.data_specific.items()},
            },
            "optimization_ranges": {
                "model": {
                    k: (v.min_val, v.max_val)
                    for k, v in model_params.__dict__.items()
                    if isinstance(v, HyperparameterRange)
                },
                "training": {
                    k: (v.min_val, v.max_val)
                    for k, v in training_params.__dict__.items()
                    if isinstance(v, HyperparameterRange)
                },
                "data": {
                    k: (v.min_val, v.max_val)
                    for k, v in data_params.__dict__.items()
                    if isinstance(v, HyperparameterRange)
                },
            },
        }

        # Apply task-specific adjustments
        config = self._apply_task_adjustments(config, task_type)

        return config

    def _apply_task_adjustments(
        self,
        config: dict[str, Any],
        task_type: str,
    ) -> dict[str, Any]:
        """Apply task-specific hyperparameter adjustments."""
        if task_type == "super_resolution":
            # SR typically needs higher learning rates and more epochs
            config["model"]["learning_rate"] *= 2.0
            config["training"]["epochs"] = int(config["training"]["epochs"] * 1.5)
        elif task_type == "denoising":
            # Denoising benefits from lower learning rates
            config["model"]["learning_rate"] *= 0.5
            config["model"]["weight_decay"] *= 2.0
        elif task_type == "segmentation":
            # Segmentation often uses smaller batches and different loss
            # weights
            config["model"]["batch_size"] = max(1, config["model"]["batch_size"] // 2)
            if "lambda_l1" in config["training"]:
                config["training"]["lambda_l1"] *= 0.5

        return config

    def get_optuna_search_space(
        self,
        model_type: str,
        training_strategy: str,
        data_modality: str = "mri_2d",
        task_type: str = "super_resolution",
    ) -> dict[str, Any]:
        """Get Optuna search space for hyperparameter optimization.

        Args:
            model_type: Type of model
            training_strategy: Training strategy
            data_modality: Data modality
            task_type: Task type

        Returns:
            Dictionary mapping parameter names to Optuna suggestion functions

        """
        model_params = self.model_hyperparams.get(
            model_type,
            self.model_hyperparams["unet"],
        )
        training_params = self.training_hyperparams.get(
            training_strategy,
            self.training_hyperparams["reconstruction"],
        )
        data_params = self.data_hyperparams.get(
            data_modality,
            self.data_hyperparams["mri_2d"],
        )

        search_space = {}

        # Add model parameters
        for param_name, param_range in model_params.__dict__.items():
            if isinstance(param_range, HyperparameterRange):
                search_space[f"model_{param_name}"] = param_range

        # Add training parameters
        for param_name, param_range in training_params.__dict__.items():
            if isinstance(param_range, HyperparameterRange):
                search_space[f"training_{param_name}"] = param_range

        # Add data parameters
        for param_name, param_range in data_params.__dict__.items():
            if isinstance(param_range, HyperparameterRange):
                search_space[f"data_{param_name}"] = param_range

        return search_space

    def validate_hyperparameter_config(
        self,
        config: dict[str, Any],
        model_type: str,
        training_strategy: str,
    ) -> list[str]:
        """Validate hyperparameter configuration for compatibility.

        Args:
            config: Hyperparameter configuration
            model_type: Model type
            training_strategy: Training strategy

        Returns:
            list of validation error messages

        """
        errors = []

        # Check model-specific constraints
        if model_type in ["kan", "vit_kan", "swin_transformer_kan"]:
            lr = config.get("model", {}).get("learning_rate", 1e-4)
            if lr > 1e-4:
                errors.append(f"KAN models should use learning rate <= 1e-4, got {lr}")

        # Check training strategy constraints
        if training_strategy == "diffusion":
            timesteps = config.get("training", {}).get("timesteps", 1000)
            if timesteps < 500:
                errors.append(
                    f"Diffusion models should use timesteps >= 500, got {timesteps}",
                )

        # Check data constraints
        batch_size = config.get("model", {}).get("batch_size", 16)
        if batch_size > 32:
            errors.append(f"Batch size {batch_size} may be too large for GPU memory")

        return errors


# Global instance for easy access
_adaptive_config = AdaptiveHyperparameterConfig()


def get_adaptive_hyperparameters(
    model_type: str,
    training_strategy: str,
    data_modality: str = "mri_2d",
    task_type: str = "super_resolution",
) -> dict[str, Any]:
    """Get adaptive hyperparameters for the specified configuration.

    Args:
        model_type: Type of model (e.g., 'kan', 'diffusion', 'unet')
        training_strategy: Training strategy (e.g., 'gan', 'reconstruction')
        data_modality: MRI data modality ('mri_2d', 'mri_3d', 'mri_kspace', 'mri_multicontrast')
        task_type: Task type (e.g., 'super_resolution', 'reconstruction')

    Returns:
        Dictionary with adaptive hyperparameter configuration

    """
    return _adaptive_config.get_hyperparameters_for_config(
        model_type=model_type,
        training_strategy=training_strategy,
        data_modality=data_modality,
        task_type=task_type,
    )
