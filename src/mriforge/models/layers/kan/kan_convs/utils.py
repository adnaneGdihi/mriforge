import torch
import torch.nn as nn


class NoiseInjection(nn.Module):
    """
    Noise injection layer that adds Gaussian noise to the input with a given probability.
    """

    def __init__(self, p: float = 0.0, alpha: float = 0.05):
        """__init__.

        Args:
            p (float): Description.
            alpha (float): Description.
        """
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for NoiseInjection.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.training and self.p > 0:
            mask = (torch.rand_like(x) < self.p).float()
            noise = torch.randn_like(x) * self.alpha
            return x + mask * noise
        return x
