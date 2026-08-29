"""Uncertainty Quantification for MRI Super-Resolution
==================================================

Implements Monte Carlo Dropout and Deep Ensembles for predictive uncertainty
estimation in GAN-based MRI super-resolution models.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from mriforge.models.interfaces.models import IDiscriminator, IGenerator
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


class UncertaintyTrainingHook:
    """Training hook for uncertainty quantification.

    Integrates uncertainty estimation into the training loop
    and provides uncertainty-aware metrics.
    """

    def __init__(self, n_mc_samples: int = 10, uncertainty_weight: float = 0.1):
        """__init__.

        Args:
            n_mc_samples (int): Description.
            uncertainty_weight (float): Description.
        """
        self.n_mc_samples = n_mc_samples
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_history: list[float] = []

    def on_batch_end(
        self,
        model: nn.Module,
        batch_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Called at the end of each training batch.

        Args:
            model: The model being trained
            batch_data: Batch input data

        Returns:
            Additional metrics dictionary

        """
        metrics = {}

        # Estimate uncertainty for current batch
        if hasattr(model, "estimate_uncertainty"):
            with torch.no_grad():
                input_data = batch_data.get("lr", batch_data.get("input"))
                if input_data is not None:
                    mean_pred, uncertainty = model.estimate_uncertainty(
                        input_data,
                        n_samples=self.n_mc_samples,
                    )

                    # Calculate uncertainty metrics
                    mean_uncertainty = torch.mean(uncertainty).item()
                    max_uncertainty = torch.max(uncertainty).item()

                    metrics.update(
                        {
                            "mean_uncertainty": mean_uncertainty,
                            "max_uncertainty": max_uncertainty,
                            "uncertainty_std": torch.std(uncertainty).item(),
                        },
                    )

                    # Store uncertainty history
                    self.uncertainty_history.append(mean_uncertainty)

        return metrics

    def get_uncertainty_trend(self) -> list[float]:
        """Get uncertainty trend over training."""
        return self.uncertainty_history.copy()


class EnhancedUncertaintyWrapper(nn.Module):
    """Enhanced uncertainty wrapper with comprehensive uncertainty estimation.

    Combines multiple uncertainty quantification methods and provides
    integration with training pipelines.
    """

    def __init__(
        self,
        base_model: IGenerator | IDiscriminator,
        mc_dropout_p: float = 0.1,
        use_deep_ensemble: bool = False,
        n_ensemble: int = 5,
        use_epistemic_uncertainty: bool = True,
        use_aleatoric_uncertainty: bool = False,
    ):
        """__init__.

        Args:
            base_model (Union[IGenerator, IDiscriminator]): Description.
            mc_dropout_p (float): Description.
            use_deep_ensemble (bool): Description.
            n_ensemble (int): Description.
            use_epistemic_uncertainty (bool): Description.
            use_aleatoric_uncertainty (bool): Description.
        """
        super().__init__()
        self.base_model = base_model
        self.mc_dropout_p = mc_dropout_p
        self.use_deep_ensemble = use_deep_ensemble
        self.n_ensemble = n_ensemble
        self.use_epistemic_uncertainty = use_epistemic_uncertainty
        self.use_aleatoric_uncertainty = use_aleatoric_uncertainty

        # Declare attribute types
        self.mc_dropout: MCDropout | nn.Identity
        self.ensemble_models: nn.ModuleList | None
        self.aleatoric_head: nn.Module | None

        # MC Dropout layers
        if mc_dropout_p > 0:
            self.mc_dropout = MCDropout(mc_dropout_p)
        else:
            self.mc_dropout = nn.Identity()

        # Deep ensemble (if enabled)
        if use_deep_ensemble:
            self.ensemble_models = self._create_ensemble()
        else:
            self.ensemble_models = None

        # Aleatoric uncertainty head (if enabled)
        if use_aleatoric_uncertainty:
            self.aleatoric_head = self._create_aleatoric_head()
        else:
            self.aleatoric_head = None

        # Training mode flags
        self.uncertainty_enabled = True

    def _create_ensemble(self) -> nn.ModuleList:
        """Create ensemble of models with different initializations."""
        ensemble = nn.ModuleList()

        for _ in range(self.n_ensemble):
            # Create a copy of the base model with different initialization
            model_copy = self._copy_model_with_random_init(self.base_model)
            if isinstance(model_copy, nn.Module):
                ensemble.append(model_copy)
            else:
                # If not a Module, wrap it
                ensemble.append(nn.Sequential(model_copy))  # type: ignore

        return ensemble

    def _copy_model_with_random_init(
        self,
        model: IGenerator | IDiscriminator | nn.Module,
    ) -> IGenerator | IDiscriminator | nn.Module:
        """Create a copy of the model with random weight initialization."""
        import copy

        model_copy = copy.deepcopy(model)

        # Re-initialize weights with different random seed
        def init_weights(module: nn.Module) -> None:
            """init_weights.

            Args:
                module (nn.Module): Description.
            """
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Only apply to nn.Module instances
        if isinstance(model_copy, nn.Module):
            model_copy.apply(init_weights)
        return model_copy

    def _create_aleatoric_head(self) -> nn.Module:
        """Create head for aleatoric uncertainty estimation."""
        # Get output channels from base model
        if hasattr(self.base_model, "out_channels"):
            out_channels = getattr(self.base_model, "out_channels", 1)
        else:
            out_channels = 1  # Default

        return nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with uncertainty estimation.

        forward method for EnhancedUncertaintyWrapper.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if not self.uncertainty_enabled:
            return self.base_model.forward(x)

        # Apply MC dropout
        x = self.mc_dropout(x)

        # Base model prediction
        pred = self.base_model.forward(x)

        # Add aleatoric uncertainty if enabled
        if self.use_aleatoric_uncertainty and self.aleatoric_head is not None:
            log_var = self.aleatoric_head(pred)
            # During training, add noise based on uncertainty
            if self.training:
                noise = torch.randn_like(pred) * torch.exp(0.5 * log_var)
                pred = pred + noise

        return pred

    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Comprehensive uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean_prediction, total_uncertainty, uncertainty_components)

        """
        uncertainty_components = {}

        # Epistemic uncertainty (model uncertainty)
        if self.use_epistemic_uncertainty:
            epistemic_uncertainty = self._estimate_epistemic_uncertainty(x, n_samples)
            uncertainty_components["epistemic"] = epistemic_uncertainty
        else:
            epistemic_uncertainty = torch.zeros_like(x)

        # Aleatoric uncertainty (data uncertainty)
        if self.use_aleatoric_uncertainty and self.aleatoric_head is not None:
            aleatoric_uncertainty = self._estimate_aleatoric_uncertainty(x)
            uncertainty_components["aleatoric"] = aleatoric_uncertainty
        else:
            aleatoric_uncertainty = torch.zeros_like(x)

        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # Mean prediction (use ensemble or single model)
        if self.use_deep_ensemble and self.ensemble_models is not None:
            mean_pred = self._ensemble_prediction(x)
        else:
            mean_pred = self.base_model.forward(x)

        return mean_pred, total_uncertainty, uncertainty_components

    def _estimate_epistemic_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Estimate epistemic uncertainty using MC dropout or ensemble."""
        if self.use_deep_ensemble and self.ensemble_models is not None:
            # Use ensemble variance
            predictions: list[torch.Tensor] = []
            for model in self.ensemble_models:
                pred = model.forward(x)
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=0)
            return torch.var(predictions, dim=0, unbiased=False)
        # Use MC dropout variance
        predictions: list[torch.Tensor] = []
        for _ in range(n_samples):
            pred = self.forward(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        return torch.var(predictions, dim=0, unbiased=False)

    def _estimate_aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate aleatoric uncertainty from the uncertainty head."""
        if self.aleatoric_head is None:
            return torch.zeros_like(x)

        with torch.no_grad():
            pred = self.base_model.forward(x)
            log_var = self.aleatoric_head(pred)
            return torch.exp(log_var)  # Convert log variance to variance

    def _ensemble_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble prediction."""
        if self.ensemble_models is None:
            return self.base_model.forward(x)

        predictions = []
        for model in self.ensemble_models:
            pred = model.forward(x)
            predictions.append(pred)

        return torch.mean(torch.stack(predictions, dim=0), dim=0)

    def enable_uncertainty(self) -> None:
        """Enable uncertainty estimation."""
        self.uncertainty_enabled = True
        if hasattr(self, "mc_dropout"):
            self.mc_dropout.training = True

    def disable_uncertainty(self) -> None:
        """Disable uncertainty estimation."""
        self.uncertainty_enabled = False
        if hasattr(self, "mc_dropout"):
            self.mc_dropout.training = False

    def get_uncertainty_metrics(self) -> dict[str, Any]:
        """Get uncertainty-related metrics and configuration."""
        return {
            "mc_dropout_p": self.mc_dropout_p,
            "use_deep_ensemble": self.use_deep_ensemble,
            "n_ensemble": self.n_ensemble,
            "use_epistemic_uncertainty": self.use_epistemic_uncertainty,
            "use_aleatoric_uncertainty": self.use_aleatoric_uncertainty,
            "uncertainty_enabled": self.uncertainty_enabled,
        }


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer for uncertainty estimation.

    Applies dropout during inference to generate multiple predictions
    for uncertainty quantification.
    """

    def __init__(self, p: float = 0.1):
        """__init__.

        Args:
            p (float): Description.
        """
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.training = True  # Always in training mode for MC Dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MCDropout.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.dropout(x)


class UncertaintyQuantificationMixin:
    """Mixin class providing uncertainty quantification methods.

    Can be mixed into any generator or discriminator to add
    uncertainty estimation capabilities.
    """

    def enable_mc_dropout(self) -> None:
        """Enable Monte Carlo dropout for uncertainty estimation."""
        self.mc_dropout_enabled = True
        self.training = True  # Keep dropout active during inference

    def disable_mc_dropout(self) -> None:
        """Disable Monte Carlo dropout."""
        self.mc_dropout_enabled = False
        self.training = False

    def mc_forward(self, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Monte Carlo forward pass for uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples

        Returns:
            Mean prediction across samples

        """
        if not hasattr(self, "mc_dropout_enabled") or not self.mc_dropout_enabled:
            return self.forward(x)  # type: ignore

        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x)  # type: ignore
            predictions.append(pred)

        return torch.stack(predictions, dim=0)

    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate predictive uncertainty using Monte Carlo dropout.

        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean_prediction, uncertainty_map)

        """
        predictions = self.mc_forward(x, n_samples)

        # Calculate mean prediction
        mean_pred = torch.mean(predictions, dim=0)

        # Calculate uncertainty (variance across samples)
        uncertainty = torch.var(predictions, dim=0, unbiased=False)

        return mean_pred, uncertainty


@register_model(name="mc_dropout", training_mode="reconstruction")
class MCDropoutGenerator(nn.Module, UncertaintyQuantificationMixin, IGenerator):
    """Generator with Monte Carlo Dropout for uncertainty quantification.

    Wraps any generator to add uncertainty estimation capabilities.
    """

    def __init__(self, base_generator: IGenerator, dropout_p: float = 0.1):
        """__init__.

        Args:
            base_generator (IGenerator): Description.
            dropout_p (float): Description.
        """
        super().__init__()
        self.base_generator = base_generator
        self.mc_dropout = MCDropout(dropout_p)
        self.dropout_p = dropout_p

    @property
    def name(self) -> str:
        """Returns the model name."""
        return f"MCDropout({self.base_generator.name})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional MC dropout.

        forward method for MCDropoutGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.mc_dropout(x)
        return self.base_generator.forward(x)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples with uncertainty."""
        # Apply dropout to latent space if provided
        if z is not None:
            z = self.mc_dropout(z)
        return self.base_generator.generate(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape."""
        return self.base_generator.get_output_shape(input_shape)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return self.base_generator.get_parameter_count()


@register_model(name="deep_ensemble", training_mode="reconstruction")
class DeepEnsembleGenerator(nn.Module):
    """Deep Ensemble for uncertainty quantification.

    Maintains multiple model instances trained with different
    initializations for ensemble uncertainty estimation.
    """

    def __init__(
        self,
        generator_factory: Callable[..., nn.Module],
        n_ensemble: int = 5,
        generator_params: dict[str, Any] | None = None,
    ):
        """__init__.

        Args:
            generator_factory (Callable[..., nn.Module]): Description.
            n_ensemble (int): Description.
            generator_params (Optional[dict[str, Any]]): Description.
        """
        super().__init__()
        self.n_ensemble = n_ensemble
        self.generator_params = generator_params or {}

        # Create ensemble members
        self.ensemble = nn.ModuleList(
            [generator_factory(**self.generator_params) for _ in range(n_ensemble)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.

        forward method for DeepEnsembleGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        predictions = []
        for generator in self.ensemble:
            pred = generator.forward(x)
            predictions.append(pred)

        # Return mean prediction
        return torch.mean(torch.stack(predictions, dim=0), dim=0)

    def generate(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples from ensemble."""
        predictions = []
        for generator in self.ensemble:
            pred = generator.generate(batch_size=batch_size, **kwargs)  # type: ignore
            predictions.append(pred)

        return torch.mean(torch.stack(predictions, dim=0), dim=0)

    def estimate_uncertainty(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using ensemble variance.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean_prediction, uncertainty_map)

        """
        predictions = []
        for generator in self.ensemble:
            pred = generator.forward(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        # Calculate mean prediction
        mean_pred = torch.mean(predictions, dim=0)

        # Calculate uncertainty (variance across ensemble)
        uncertainty = torch.var(predictions, dim=0, unbiased=False)

        return mean_pred, uncertainty

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape from first ensemble member."""
        return self.ensemble[0].get_output_shape(input_shape)

    def get_parameter_count(self) -> int:
        """Get total parameter count across ensemble."""
        return sum(gen.get_parameter_count() for gen in self.ensemble)


class UncertaintyAwareDiscriminator(nn.Module, UncertaintyQuantificationMixin):
    """Discriminator with uncertainty quantification capabilities.

    Extends base discriminator with MC dropout for uncertainty estimation.
    """

    def __init__(self, base_discriminator: IDiscriminator, dropout_p: float = 0.1):
        """__init__.

        Args:
            base_discriminator (IDiscriminator): Description.
            dropout_p (float): Description.
        """
        super().__init__()
        self.base_discriminator = base_discriminator
        self.mc_dropout = MCDropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional MC dropout.

        forward method for UncertaintyAwareDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.mc_dropout(x)
        return self.base_discriminator.forward(x)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return self.base_discriminator.get_parameter_count()


class UncertaintyCalibration:
    """Uncertainty calibration utilities.

    Provides methods for calibrating uncertainty estimates
    and computing calibration metrics.
    """

    @staticmethod
    def expected_calibration_error(
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error (ECE).

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            n_bins: Number of bins for calibration

        Returns:
            ECE score

        """
        # Calculate prediction errors
        errors = torch.abs(predictions - targets)

        # Bin predictions by uncertainty
        bin_boundaries = torch.linspace(0, uncertainties.max(), n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (uncertainties >= bin_boundaries[i]) & (
                uncertainties < bin_boundaries[i + 1]
            )

            if bin_mask.sum() > 0:
                bin_errors = errors[bin_mask]
                bin_uncertainties = uncertainties[bin_mask]

                # Average error and uncertainty in bin
                avg_error = bin_errors.mean()
                avg_uncertainty = bin_uncertainties.mean()

                # Add to ECE
                bin_weight = bin_mask.sum() / len(predictions)
                ece += bin_weight * torch.abs(avg_error - avg_uncertainty)

        return float(ece)

    @staticmethod
    def uncertainty_correlation(
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
    ) -> float:
        """Calculate correlation between uncertainty and prediction error.

        Args:
            uncertainties: Uncertainty estimates
            errors: Prediction errors

        Returns:
            Pearson correlation coefficient

        """
        # Flatten tensors
        uncertainties_flat = uncertainties.flatten()
        errors_flat = errors.flatten()

        # Calculate correlation
        uncertainties_mean = uncertainties_flat.mean()
        errors_mean = errors_flat.mean()

        uncertainties_std = uncertainties_flat.std()
        errors_std = errors_flat.std()

        if uncertainties_std == 0 or errors_std == 0:
            return 0.0

        covariance = torch.mean(
            (uncertainties_flat - uncertainties_mean) * (errors_flat - errors_mean),
        )

        correlation = covariance / (uncertainties_std * errors_std)
        return correlation.item()


class UncertaintyQuantificationFactory:
    """Factory for creating uncertainty quantification wrappers."""

    @staticmethod
    def create_mc_dropout_generator(
        base_generator: IGenerator,
        dropout_p: float = 0.1,
    ) -> MCDropoutGenerator:
        """Create MC dropout generator wrapper."""
        return MCDropoutGenerator(base_generator, dropout_p)

    @staticmethod
    def create_deep_ensemble_generator(
        generator_factory: Callable[..., nn.Module],
        n_ensemble: int = 5,
        generator_params: dict[str, Any] | None = None,
    ) -> DeepEnsembleGenerator:
        """Create deep ensemble generator."""
        return DeepEnsembleGenerator(generator_factory, n_ensemble, generator_params)

    @staticmethod
    def create_uncertainty_discriminator(
        base_discriminator: IDiscriminator,
        dropout_p: float = 0.1,
    ) -> UncertaintyAwareDiscriminator:
        """Create uncertainty-aware discriminator."""
        return UncertaintyAwareDiscriminator(base_discriminator, dropout_p)


# Example usage and testing
if __name__ == "__main__":
    from mriforge.models.factories.model_factory import get_model_factory

    # Get model factory
    factory = get_model_factory()

    # Create base generator
    base_gen = factory.create_generator("standard_unet", in_channels=1, out_channels=1)

    # Create uncertainty quantification wrappers
    mc_gen = UncertaintyQuantificationFactory.create_mc_dropout_generator(
        base_gen,
        dropout_p=0.1,
    )

    ensemble_gen = UncertaintyQuantificationFactory.create_deep_ensemble_generator(
        lambda **kwargs: factory.create_generator(  # type: ignore
            "standard_unet",
            **kwargs,
        ),
        n_ensemble=3,
        generator_params={"in_channels": 1, "out_channels": 1},
    )

    # Test uncertainty estimation
    x = torch.randn(2, 1, 256, 256)

    # MC Dropout uncertainty
    mc_gen.enable_mc_dropout()
    mean_pred, uncertainty = mc_gen.estimate_uncertainty(x, n_samples=5)
    logger.debug(
        f"MC Dropout - Mean shape: {mean_pred.shape}, Uncertainty shape: {uncertainty.shape}"
    )

    # Deep Ensemble uncertainty
    mean_pred, uncertainty = ensemble_gen.estimate_uncertainty(x)
    logger.debug(
        f"Deep Ensemble - Mean shape: {mean_pred.shape}, Uncertainty shape: {uncertainty.shape}"
    )

    logger.debug("Uncertainty quantification implementation complete!")
