import torch
from torch import nn


class RadialBasisFunction(nn.Module):
    """RadialBasisFunction class."""

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        """__init__.

        Args:
            grid_min (float): Description.
            grid_max (float): Description.
            num_grids (int): Description.
            denominator (float): Description.
        """
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for RadialBasisFunction.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))
