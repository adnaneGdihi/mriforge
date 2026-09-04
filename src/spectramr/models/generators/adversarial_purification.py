import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["AdversarialPurificationDiffusion"]


@register_model(name="adversarial_purification", training_mode="diffusion")
class AdversarialPurificationDiffusion(nn.Module, IGenerator):
    """
    Adversarial Purification using Diffusion Models (Experiment 106).

    This model wraps a pre-trained diffusion model.
    It implements the purification process:
    1. Add noise to the input image (diffuse) up to timestep t*.
    2. Denoise from t* back to 0.

    This removes adversarial perturbations which are usually high-frequency and fragile.
    """

    def __init__(
        self,
        base_model: nn.Module,
        purification_timestep: int = 100,  # t*
    ):
        """__init__.

        Args:
            base_model (nn.Module): Description.
            purification_timestep (int): Description.
        """
        super().__init__()
        self.diffusion = base_model
        self.purification_timestep = purification_timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Adversarially perturbed image [B, C, H, W]
        Returns:
            Purified image

        forward method for AdversarialPurificationDiffusion.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Diffuse (Add noise)
        t = torch.tensor([self.purification_timestep], device=x.device).repeat(x.shape[0])
        noise = torch.randn_like(x)

        # q_sample gets x_t from x_0
        x_t = self.diffusion.q_sample(x, t, noise)

        # 2. Denoise (Reverse process)
        # We need to run p_sample loop from t down to 0
        current_x = x_t

        for i in reversed(range(self.purification_timestep + 1)):
            t_step = torch.tensor([i], device=x.device).repeat(x.shape[0])
            current_x = self.diffusion.p_sample(current_x, t_step)

        return current_x

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return self.diffusion.get_parameter_count()

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "adversarial_purification"
