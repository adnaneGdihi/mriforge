from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["ReflexionReconstruction"]

# NOTE: Avoid importing ModelFactory at module import time to prevent circular imports.
# Import inside __init__ when needed.


@register_model(name="reflexion_reconstruction", training_mode="reconstruction")
class ReflexionReconstruction(nn.Module, IGenerator):
    """
    Experiment 116: "System 2" Reflexion Recon

    A reconstruction model that produces an initial image, generates a "critique"
    (detects hallucinations via consistency check), and iteratively refines the result.

    Workflow:
    1. Draft: x_0 = G(y)
    2. Critique: Calculate error map H = |F(x_0) - y|
    3. Refinement: x_new = x_old - alpha * grad(DataConsistency)
    """

    def __init__(
        self,
        base_model_type: str = "standard_unet",
        base_model_kwargs: dict[str, Any] = None,
        critique_threshold: float = 0.05,
        max_refinement_steps: int = 5,
        step_size: float = 0.1,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        """__init__.

        Args:
            base_model_type (str): Description.
            base_model_kwargs (dict[str, Any]): Description.
            critique_threshold (float): Description.
            max_refinement_steps (int): Description.
            step_size (float): Description.
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.critique_threshold = critique_threshold
        self.max_refinement_steps = max_refinement_steps
        self.step_size = step_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if base_model_kwargs is None:
            base_model_kwargs = {}

        # Ensure base model has correct channels
        base_model_kwargs["in_channels"] = in_channels
        base_model_kwargs["out_channels"] = out_channels

        # Instantiate base generator (deferred import to avoid circular dependency)
        from spectramr.models.factories.model_factory import get_model_factory

        factory = get_model_factory()
        self.generator = factory.create_generator(base_model_type, **base_model_kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input zero-filled image or initial guess [B, C, H, W]
            mask: Sampling mask [B, 1, H, W] (optional, required for data consistency)

        forward method for ReflexionReconstruction.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Initial Draft
        x_curr = self.generator(x)

        if mask is None:
            # If no mask provided, we can't do data consistency check easily
            # unless we assume x is the zero-filled image and we can infer mask
            # For now, return draft if no mask
            return x_curr

        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        # 2. Iterative Refinement (System 2)
        # We need the original k-space measurement 'y' to check consistency.
        # Assuming 'x' is the zero-filled image, y = FFT(x) (masked)

        y_measured = fft2c(x) * mask

        for _ in range(self.max_refinement_steps):
            # Critique: Data Consistency Error
            # F(x_curr) using canonical centered FFT
            k_curr = fft2c(x_curr)

            # Error in k-space (only at sampled locations)
            # diff = (k_curr - y_measured) * mask
            # But we want to do this in image space or just take gradient

            # We can use a simple gradient descent step on data consistency
            # x_new = x_curr - alpha * grad( || (F(x_curr) - y) * mask ||^2 )

            # Analytical gradient of DC term is: F_inv( (F(x_curr) - y) * mask )

            k_diff = (k_curr * mask) - y_measured
            grad_dc = ifft2c(k_diff).real  # Assuming real image

            # Check magnitude of error (Critique)
            error_magnitude = torch.norm(grad_dc)
            if error_magnitude < self.critique_threshold:
                break

            # Refinement Step
            x_curr = x_curr - self.step_size * grad_dc

        return x_curr

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "reflexion_reconstruction"

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
