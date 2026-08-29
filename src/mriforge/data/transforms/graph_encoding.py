import torch
import torch.nn as nn


class GraphBatchEncoder(nn.Module):
    """
    Converts a spatial MRI batch (B, C, H, W) into a Graph batch (B, N, C_out).

    Data Flow:
    Input:  Batch from TorchIO Loader [B, C, H, W] (e.g., 2 channels for Re/Im)
    Output: Graph Tensor [B, N, 5] where N = H*W
            Features: [Re, Im, Kx, Ky, DCF]

    Design:
    - Runs on GPU (inherits nn.Module)
    - Generates coordinates on-the-fly to handle dynamic shapes
    """

    def __init__(self, normalize_coords=True):
        """__init__.

        Args:
            normalize_coords (Any): Description.
        """
        super().__init__()
        self.normalize_coords = normalize_coords

    def forward(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch_tensor: (B, C, H, W) - K-space data
        Returns:
            nodes: (B, H*W, 5)
        """
        if batch_tensor.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {batch_tensor.shape}")

        B, C, H, W = batch_tensor.shape
        N = H * W
        device = batch_tensor.device

        # 1. Flatten Signal [B, C, H, W] -> [B, N, C]
        # Permute to (B, H, W, C) first to keep spatial locality in sequence
        signal = batch_tensor.permute(0, 2, 3, 1).reshape(B, N, C)

        # Ensure we have 2 channels (Re, Im)
        if C == 1:  # If complex magnitude or single channel
            signal = torch.cat([signal, torch.zeros_like(signal)], dim=-1)
        elif C > 2:  # Reduce or slice if extra channels exist
            signal = signal[..., :2]

        # 2. Generate Coordinates (Kx, Ky)
        # We generate on-the-fly to match the specific H,W of the batch (handling dynamic shapes)
        if self.normalize_coords:
            kx = torch.linspace(-1, 1, W, device=device)
            ky = torch.linspace(-1, 1, H, device=device)
        else:
            kx = torch.arange(W, device=device).float()
            ky = torch.arange(H, device=device).float()

        # Grid: (H, W, 2) -> Flatten -> (N, 2)
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(N, 2)

        # Expand coords to Batch: (1, N, 2) -> (B, N, 2)
        coords = coords.unsqueeze(0).expand(B, -1, -1)

        # 3. Generate DCF (Density Compensation Function)
        # For Cartesian grid data, the sampling density is uniform (1.0).
        # In non-Cartesian trajectories (radial/spiral), this would be calculated
        # based on the trajectory. Here we assume Cartesian input.
        dcf = torch.ones(B, N, 1, device=device)

        # 4. Concatenate: [B, N, 2] + [B, N, 2] + [B, N, 1] -> [B, N, 5]
        # Signal(2) + Coords(2) + DCF(1) = 5 Features
        nodes = torch.cat([signal, coords, dcf], dim=-1)

        return nodes
