import logging
from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

__all__ = ["SymbolicRegressionWrapper"]

# NOTE: Defer factory import to avoid circular imports during registry initialization.


@register_model(name="symbolic_regression_wrapper", training_mode="reconstruction")
class SymbolicRegressionWrapper(nn.Module, IGenerator):
    """
    Experiment 119: Physics-Discovery (PySR)

    Wraps a base generator and provides functionality to analyze residuals
    using Symbolic Regression (simulated or via PySR if available) to discover
    missing physics terms.
    """

    def __init__(
        self,
        base_model_type: str = "standard_unet",
        base_model_kwargs: dict[str, Any] = None,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        """__init__.

        Args:
            base_model_type (str): Description.
            base_model_kwargs (dict[str, Any]): Description.
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if base_model_kwargs is None:
            base_model_kwargs = {}

        base_model_kwargs["in_channels"] = in_channels
        base_model_kwargs["out_channels"] = out_channels

        from mriforge.models.factories.model_factory import get_model_factory

        factory = get_model_factory()
        self.generator = factory.create_generator(base_model_type, **base_model_kwargs)

        self.logger = logging.getLogger(__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SymbolicRegressionWrapper.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.generator(x)

    def analyze_residuals(self, x: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Analyze residuals between prediction and ground truth to discover physics terms.
        """
        with torch.no_grad():
            y_pred = self.generator(x)
            residual = y_true - y_pred

            # In a real implementation, we would pass 'residual' and 'x' (or features of x)
            # to PySR to find f(x) ~ residual.

            # Log statistics
            mse = torch.mean(residual**2).item()
            self.logger.info(f"Analyzing residuals: MSE={mse:.6f}")

            # PySR integration for equation discovery
            try:
                from pysr import PySRRegressor

                # Flatten tensors for PySR (expects 2D numpy arrays)
                x_flat = x.reshape(x.size(0), -1).cpu().numpy()
                residual_flat = residual.reshape(residual.size(0), -1).cpu().numpy()

                # Use first few principal components if too high-dimensional
                max_features = 10
                if x_flat.shape[1] > max_features:
                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=max_features)
                    x_flat = pca.fit_transform(x_flat)

                # Run symbolic regression
                model = PySRRegressor(
                    niterations=40,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sin", "cos", "exp", "log", "sqrt"],
                    maxsize=20,
                    verbosity=0,
                    temp_equation_file=True,
                )

                # Fit on first output channel
                model.fit(
                    x_flat,
                    residual_flat[:, 0] if residual_flat.ndim > 1 else residual_flat,
                )

                self.logger.info(f"Discovered equations: {model.equations_}")
                self._discovered_equations = model.equations_

            except ImportError:
                self.logger.warning("PySR not installed. Install with: pip install pysr")
            except Exception as e:
                self.logger.warning(f"PySR failed: {e}")

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "symbolic_regression_wrapper"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return self.generator.get_parameter_count()

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return self.generator.get_output_shape(input_shape)
