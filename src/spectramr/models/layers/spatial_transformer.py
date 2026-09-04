import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    N-Dimensional Differentiable Spatial Transformer Network (STN).
    Warps an input image 'x' using a dense displacement field 'flow'.
    """

    def __init__(self, size, mode="bilinear"):
        """__init__.

        Args:
            size (Any): Description.
            mode (Any): Description.
        """
        super().__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # Register as buffer to save in state_dict but not update via gradient
        self.register_buffer("grid", grid)
        self.mode = mode

    def forward(self, src, flow):
        # flow shape: [B, Dims, H, W, (D)]
        """forward.

        Args:
            src (Any): Description.
            flow (Any): Description.
        Returns:
            Any: Description.

        forward method for SpatialTransformer.

        Executes PyTorch tensor operations.

        Args:
            src (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            flow (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize to [-1, 1] for grid_sample standard
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Move channels to last dim for grid_sample: [B, H, W, D, C]
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
