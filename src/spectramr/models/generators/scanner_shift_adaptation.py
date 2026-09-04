"""Test-Time Adaptation for MRI Scanner Shift Correction
=====================================================

Implements test-time adaptation techniques to handle domain shifts
between different MRI scanners and acquisition protocols.
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class ScannerShiftAdapter(nn.Module):
    """Test-time adaptation module for scanner shift correction.

    Learns to adapt model parameters to new scanner domains
    using a few test samples.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adaptation_steps: int = 10,
        learning_rate: float = 1e-4,
        adaptation_layers: list[str] | None = None,
    ):
        """__init__.

        Args:
            base_model (nn.Module): Description.
            adaptation_steps (int): Description.
            learning_rate (float): Description.
            adaptation_layers (Optional[list[str]]): Description.
        """
        super().__init__()
        self.base_model = base_model
        self.adaptation_steps = adaptation_steps
        self.learning_rate = learning_rate

        # Identify layers to adapt (default: BatchNorm and final layers)
        if adaptation_layers is None:
            self.adaptation_layers = self._identify_adaptation_layers()
        else:
            self.adaptation_layers = adaptation_layers

        # Create adaptation parameters
        self.adaptation_params = nn.ParameterDict()
        for layer_name in self.adaptation_layers:
            # ParameterDict doesn't allow "." in keys
            flat_name = layer_name.replace(".", "_")
            layer = self._get_layer_by_name(layer_name)
            if isinstance(layer, nn.BatchNorm2d):
                # Adapt batch norm statistics
                self.adaptation_params[f"{flat_name}_running_mean"] = nn.Parameter(
                    layer.running_mean.clone(),
                )
                self.adaptation_params[f"{flat_name}_running_var"] = nn.Parameter(
                    layer.running_var.clone(),
                )
            elif isinstance(layer, nn.Conv2d):
                # Adapt convolution weights and biases
                self.adaptation_params[f"{flat_name}_weight"] = nn.Parameter(
                    layer.weight.clone(),
                )
                if layer.bias is not None:
                    self.adaptation_params[f"{flat_name}_bias"] = nn.Parameter(
                        layer.bias.clone(),
                    )

    def _identify_adaptation_layers(self) -> list[str]:
        """Identify layers suitable for adaptation."""
        adaptation_layers = []

        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.Conv2d)):
                adaptation_layers.append(name)

        return adaptation_layers

    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get layer by name from base model."""
        names = name.split(".")
        module = self.base_model
        for n in names:
            module = getattr(module, n)
        return module

    def _apply_adaptation(self):
        """Link adaptation parameters to base model."""
        self._original_state = {}
        for layer_name in self.adaptation_layers:
            layer = self._get_layer_by_name(layer_name)
            flat_name = layer_name.replace(".", "_")
            self._original_state[layer_name] = {}

            if isinstance(layer, nn.BatchNorm2d):
                # Update batch norm statistics
                running_mean_key = f"{flat_name}_running_mean"
                running_var_key = f"{flat_name}_running_var"

                if running_mean_key in self.adaptation_params:
                    self._original_state[layer_name]["running_mean"] = layer.running_mean
                    # Buffers are just attributes, usually
                    layer.running_mean = self.adaptation_params[running_mean_key]
                if running_var_key in self.adaptation_params:
                    self._original_state[layer_name]["running_var"] = layer.running_var
                    layer.running_var = self.adaptation_params[running_var_key]

            elif isinstance(layer, nn.Conv2d):
                # Update convolution parameters
                weight_key = f"{flat_name}_weight"
                bias_key = f"{flat_name}_bias"

                if weight_key in self.adaptation_params:
                    self._original_state[layer_name]["weight"] = layer.weight
                    del layer.weight  # Delete parameter registration
                    layer.weight = self.adaptation_params[weight_key]

                if bias_key in self.adaptation_params and layer.bias is not None:
                    self._original_state[layer_name]["bias"] = layer.bias
                    del layer.bias
                    layer.bias = self.adaptation_params[bias_key]

    def _restore_original(self):
        """Restore original model parameters (unlink)."""
        if not hasattr(self, "_original_state"):
            return

        for layer_name, state in self._original_state.items():
            layer = self._get_layer_by_name(layer_name)

            if isinstance(layer, nn.BatchNorm2d):
                if "running_mean" in state:
                    if hasattr(layer, "running_mean"):
                        del layer.running_mean
                    layer.register_buffer("running_mean", state["running_mean"])
                if "running_var" in state:
                    if hasattr(layer, "running_var"):
                        del layer.running_var
                    layer.register_buffer("running_var", state["running_var"])

            elif isinstance(layer, nn.Conv2d):
                if "weight" in state:
                    if hasattr(layer, "weight"):
                        del layer.weight  # remove our adaptation param link
                    layer.weight = state["weight"]

                if "bias" in state:
                    if hasattr(layer, "bias") and layer.bias is not None:
                        del layer.bias
                    layer.bias = state["bias"]

        self._original_state = {}

    def adapt(
        self,
        test_samples: torch.Tensor,
        mc_samples: int = 1,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Perform test-time adaptation on test samples.

        Args:
            test_samples: Batch of test images
            mc_samples: Number of Monte Carlo samples for uncertainty
            return_uncertainty: Whether to return uncertainty estimate

        Returns:
            Adapted predictions or (predictions, uncertainty)

        """
        # Store original parameters
        original_params = {}
        for name, param in self.adaptation_params.named_parameters():
            original_params[name] = param.clone()

        from spectramr.infrastructure.training.builders import OptimizationBuilder

        # Create optimizer for adaptation parameters
        optimizer = OptimizationBuilder.create_single_optimizer(
            self.adaptation_params.parameters(),
            learning_rate=self.learning_rate,
        )

        try:
            # Link parameters (replace base model weights with adaptation params)
            self._apply_adaptation()

            # Adaptation loop
            for _step in range(self.adaptation_steps):
                # Forward pass
                predictions = self.base_model(test_samples)

                # Compute adaptation loss (e.g., entropy minimization)
                loss = self._compute_adaptation_loss(predictions)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # MC Dropout inference uses the CURRENT (adapted) parameters

            if mc_samples > 1:
                # MC Dropout inference
                # Enable dropout only, keep BN in eval mode
                self.base_model.eval()
                for m in self.base_model.modules():
                    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                        m.train()

                batch_preds = []
                for _ in range(mc_samples):
                    with torch.no_grad():
                        batch_preds.append(self.base_model(test_samples))

                # Stack and compute stats
                stacked_preds = torch.stack(batch_preds)
                final_predictions = torch.mean(stacked_preds, dim=0)
                uncertainty = torch.var(stacked_preds, dim=0)

                self.base_model.eval()  # Restore full eval mode
            else:
                self.base_model.eval()
                with torch.no_grad():
                    final_predictions = self.base_model(test_samples)
                uncertainty = torch.zeros_like(final_predictions)

        finally:
            # Restore original parameters (unlink)
            self._restore_original()

            # Reset adaptation params to initial state
            with torch.no_grad():
                for name, param in original_params.items():
                    self.adaptation_params.get_parameter(name).copy_(param)

        if return_uncertainty:
            return final_predictions, uncertainty

        return final_predictions

    def _compute_adaptation_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute adaptation loss for test-time adaptation.

        Uses entropy minimization as the adaptation objective.
        """
        # Convert to probabilities (assuming output is in [-1, 1])
        probs = torch.sigmoid(predictions)

        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # Minimize entropy (encourage confident predictions)
        return torch.mean(entropy)


class DomainNormalizer(nn.Module):
    """Domain normalization module for scanner shift correction.

    Learns to normalize features from different scanner domains
    to a common representation.
    """

    def __init__(self, num_features: int, num_domains: int = 10):
        """__init__.

        Args:
            num_features (int): Description.
            num_domains (int): Description.
        """
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains

        # Domain-specific normalization parameters
        self.domain_norms = nn.ModuleList(
            [nn.BatchNorm2d(num_features) for _ in range(num_domains)],
        )

        # Domain classifier for automatic domain detection
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply domain-specific normalization.

        forward method for DomainNormalizer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Predict domain
        domain_logits = self.domain_classifier(x)
        domain_probs = F.softmax(domain_logits, dim=1)

        # Apply domain-specific normalization
        normalized_features = []
        for i in range(self.num_domains):
            domain_norm = self.domain_norms[i](x)
            domain_weight = domain_probs[:, i].view(-1, 1, 1, 1)
            normalized_features.append(domain_norm * domain_weight)

        return torch.sum(torch.stack(normalized_features, dim=0), dim=0)


@register_model(name="scanner_adaptive_generator", training_mode="domain_adaptation")
class ScannerAdaptiveGenerator(nn.Module, IGenerator):
    """Generator with scanner adaptation capabilities.

    Combines base generator with test-time adaptation modules.
    """

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "scanner_adaptive_generator"

    def __init__(
        self,
        base_generator: IGenerator,
        adaptation_config: dict[str, Any] | None = None,
    ):
        """__init__.

        Args:
            base_generator (IGenerator): Description.
            adaptation_config (Optional[dict[str, Any]]): Description.
        """
        super().__init__()
        self.base_generator = base_generator

        # Default adaptation configuration
        if adaptation_config is None:
            adaptation_config = {
                "adaptation_steps": 10,
                "learning_rate": 1e-4,
                "use_domain_norm": True,
                "num_domains": 5,
            }

        self.adaptation_config = adaptation_config

        # Create adaptation module
        self.adapter = ScannerShiftAdapter(
            base_generator,
            adaptation_steps=adaptation_config["adaptation_steps"],
            learning_rate=adaptation_config["learning_rate"],
        )

        # Domain normalizer (optional)
        if adaptation_config.get("use_domain_norm", False):
            # Estimate number of features from base generator
            with torch.no_grad():
                # TODO: Remove dummy_input dependency by inferring shape dynamically
                dummy_input = torch.randn(1, 1, 256, 256)
                features = base_generator.forward(dummy_input)
                num_features = features.shape[1]

            self.domain_normalizer = DomainNormalizer(
                num_features=num_features,
                num_domains=adaptation_config["num_domains"],
            )
        else:
            self.domain_normalizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional domain normalization.

        forward method for ScannerAdaptiveGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Apply domain normalization if available
        if self.domain_normalizer is not None:
            x = self.domain_normalizer(x)

        return self.base_generator.forward(x)

    def generate(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples (no adaptation for generation)."""
        return self.base_generator.generate(batch_size=batch_size, **kwargs)

    def adapt_and_predict(
        self,
        test_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt to test samples and make predictions.

        Args:
            test_samples: Test images from new scanner domain

        Returns:
            Tuple of (adapted_predictions, uncertainty_estimate)

        """
        # Perform test-time adaptation with MC Dropout
        # Use 10 samples for robust uncertainty estimation
        adapted_predictions, uncertainty = self.adapter.adapt(
            test_samples, mc_samples=10, return_uncertainty=True
        )

        return adapted_predictions, uncertainty

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape."""
        return self.base_generator.get_output_shape(input_shape)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        count = self.base_generator.get_parameter_count()
        count += sum(p.numel() for p in self.adapter.parameters())
        if self.domain_normalizer is not None:
            count += sum(p.numel() for p in self.domain_normalizer.parameters())
        return count


class ScannerShiftCorrectionFactory:
    """Factory for creating scanner shift correction modules."""

    @staticmethod
    def create_adaptive_generator(
        base_generator: IGenerator,
        adaptation_config: dict[str, Any] | None = None,
    ) -> ScannerAdaptiveGenerator:
        """Create scanner-adaptive generator."""
        return ScannerAdaptiveGenerator(base_generator, adaptation_config)

    @staticmethod
    def create_domain_normalizer(
        num_features: int,
        num_domains: int = 10,
    ) -> DomainNormalizer:
        """Create domain normalizer."""
        return DomainNormalizer(num_features, num_domains)

    @staticmethod
    def create_shift_adapter(
        base_model: nn.Module,
        adaptation_steps: int = 10,
        learning_rate: float = 1e-4,
    ) -> ScannerShiftAdapter:
        """Create scanner shift adapter."""
        return ScannerShiftAdapter(base_model, adaptation_steps, learning_rate)


class ScannerDomainDataset:
    """Dataset for training domain adaptation modules.

    Contains images from multiple scanner domains for training
    domain normalization and adaptation modules.
    """

    def __init__(self, domain_datasets: list[torch.utils.data.Dataset]):
        """__init__.

        Args:
            domain_datasets (list[torch.utils.data.Dataset]): Description.
        """
        self.domain_datasets = domain_datasets
        self.domain_labels = []

        # Assign domain labels
        for i, dataset in enumerate(domain_datasets):
            self.domain_labels.extend([i] * len(dataset))

    def __len__(self):
        """__len__.

        Returns:
            Any: Description.
        """
        return sum(len(dataset) for dataset in self.domain_datasets)

    def __getitem__(self, idx):
        # Find which domain this index belongs to
        """__getitem__.

        Args:
            idx (Any): Description.
        Returns:
            Any: Description.
        """
        cumulative_len = 0
        for domain_idx, dataset in enumerate(self.domain_datasets):
            if idx < cumulative_len + len(dataset):
                sample_idx = idx - cumulative_len
                sample = dataset[sample_idx]

                # Add domain label
                if isinstance(sample, tuple):
                    image, *rest = sample
                    return (image, domain_idx, *rest)
                return (sample, domain_idx)

            cumulative_len += len(dataset)

        raise IndexError("Index out of range")


# Example usage and testing
if __name__ == "__main__":
    from ..factories.model_factory import get_model_factory

    # Get model factory
    factory = get_model_factory()

    # Create base generator
    base_gen = factory.create_generator("standard_unet", in_channels=1, out_channels=1)

    # Create scanner-adaptive generator
    adaptive_config = {
        "adaptation_steps": 5,
        "learning_rate": 1e-4,
        "use_domain_norm": True,
        "num_domains": 3,
    }

    adaptive_gen = ScannerShiftCorrectionFactory.create_adaptive_generator(
        base_gen,
        adaptive_config,
    )

    # Test adaptation
    test_samples = torch.randn(4, 1, 256, 256)

    logger.debug("Testing scanner shift adaptation...")
    adapted_pred, uncertainty = adaptive_gen.adapt_and_predict(test_samples)

    logger.debug(f"Original shape: {test_samples.shape}")
    logger.debug(f"Adapted predictions shape: {adapted_pred.shape}")
    logger.debug(f"Uncertainty shape: {uncertainty.shape}")
    logger.debug(f"Parameter count: {adaptive_gen.get_parameter_count()}")

    logger.debug("Scanner shift correction implementation complete!")
