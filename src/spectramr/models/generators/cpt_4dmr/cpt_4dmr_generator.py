"""CPT-4DMR Generator
====================

Combined generator that uses Spatial Anatomy Network (SAN) and
Temporal Motion Network (TMN) to reconstruct 4D cardiac MRI.

Core equation: I(x, t) = SAN(x + TMN(x, t, resp_signal))

This allows continuous-time reconstruction without motion binning,
sharing k-space data across all time points.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

from .spatial_anatomy_net import SpatialAnatomyNet
from .temporal_motion_net import TemporalMotionNet


@register_model(name="cpt_4dmr", training_mode="reconstruction")
class CPT4DMRGenerator(nn.Module, IGenerator):
    """CPT-4DMR: Continuous Spatio-Temporal 4D MRI Generator.

    Reconstructs 4D (3D + time) MRI volumes by:
    1. Learning a canonical anatomical volume (SAN)
    2. Learning motion as continuous deformation fields (TMN)
    3. Warping canonical anatomy to each time point

    Args:
        spatial_dim: Spatial dimension (2 for 2D+t, 3 for 3D+t)
        san_hidden_dim: SAN hidden dimension
        san_num_layers: SAN number of layers
        tmn_hidden_dim: TMN hidden dimension
        tmn_num_layers: TMN number of layers
        out_channels: Output channels (typically 1)
        use_respiratory: Condition on respiratory signal
        max_displacement: Maximum motion displacement
        jacobian_weight: Weight for Jacobian regularization loss
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        san_hidden_dim: int = 256,
        san_num_layers: int = 4,
        tmn_hidden_dim: int = 128,
        tmn_num_layers: int = 4,
        out_channels: int = 1,
        use_respiratory: bool = True,
        max_displacement: float = 0.3,
        jacobian_weight: float = 0.1,
        **kwargs,
    ):
        """__init__.

        Args:
            spatial_dim (int): Description.
            san_hidden_dim (int): Description.
            san_num_layers (int): Description.
            tmn_hidden_dim (int): Description.
            tmn_num_layers (int): Description.
            out_channels (int): Description.
            use_respiratory (bool): Description.
            max_displacement (float): Description.
            jacobian_weight (float): Description.
        """
        super().__init__()

        self.spatial_dim = spatial_dim
        self.out_channels = out_channels
        self.jacobian_weight = jacobian_weight

        # Spatial Anatomy Network (canonical volume)
        self.san = SpatialAnatomyNet(
            spatial_dim=spatial_dim,
            hidden_dim=san_hidden_dim,
            num_layers=san_num_layers,
            out_channels=out_channels,
            use_positional_encoding=True,
        )

        # Temporal Motion Network (deformation fields)
        self.tmn = TemporalMotionNet(
            spatial_dim=spatial_dim,
            hidden_dim=tmn_hidden_dim,
            num_layers=tmn_num_layers,
            use_respiratory=use_respiratory,
            max_displacement=max_displacement,
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "cpt_4dmr"

    def forward(
        self,
        coords: torch.Tensor,
        time: torch.Tensor | None = None,
        resp_signal: torch.Tensor | None = None,
        return_dvf: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct signal at given coordinates and time.

        Args:
            coords: Spatial coordinates [B, N, D] or image [B, C, H, W]
            time: Normalized time [B] in [0, 1], None = canonical (t=0)
            resp_signal: Respiratory surrogate [B], optional
            return_dvf: Whether to return DVF along with intensity

        Returns:
            Intensity [B, N, 1] or [B, C, H, W]
            DVF [B, N, D] (if return_dvf=True)

        forward method for CPT4DMRGenerator.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            resp_signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_dvf (bool): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle image input (convert to coordinates)
        is_image_input = coords.dim() == 4
        if is_image_input:
            B, C, H, W = coords.shape
            coords = self._make_coord_grid(H, W, coords.device)
            coords = coords.expand(B, -1, -1)

        B, N, D = coords.shape

        # Get motion deformation
        if time is not None:
            dvf = self.tmn(coords, time, resp_signal)
            warped_coords = coords + dvf
        else:
            # No time = canonical anatomy (identity transform)
            dvf = torch.zeros_like(coords)
            warped_coords = coords

        # Query canonical anatomy at warped coordinates
        intensity = self.san(warped_coords)  # [B, N, C]

        # Reshape if image input
        if is_image_input:
            intensity = intensity.permute(0, 2, 1).contiguous().view(B, self.out_channels, H, W)
            if return_dvf:
                dvf = dvf.permute(0, 2, 1).contiguous().view(B, D, H, W)

        if return_dvf:
            return intensity, dvf
        return intensity

    def _make_coord_grid(
        self,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create normalized coordinate grid.

        Args:
            H, W: Grid dimensions
            device: Target device

        Returns:
            Coordinates [1, H*W, 2]
        """
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)

        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]

        return grid.view(1, H * W, 2)

    def generate(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        resp_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate output (IGenerator interface).

        Args:
            x: Input (unused for INR, just provides shape/device)
            time: Time point [B]
            resp_signal: Respiratory signal [B]

        Returns:
            Generated image [B, C, H, W]
        """
        return self.forward(x, time, resp_signal)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        B, C, H, W = input_shape
        return (B, self.out_channels, H, W)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def compute_regularization_loss(
        self,
        coords: torch.Tensor,
        time: torch.Tensor,
        resp_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute regularization losses for training.

        Includes:
        - Jacobian determinant loss (diffeomorphism)
        - Smoothness regularization

        Args:
            coords: Sample coordinates [B, N, D]
            time: Time points [B]
            resp_signal: Respiratory signal [B]

        Returns:
            Total regularization loss
        """
        # Jacobian loss (enforce diffeomorphism)
        jac_loss = self.tmn.compute_jacobian_loss(coords, time, resp_signal)

        return self.jacobian_weight * jac_loss

    def generate_4d_sequence(
        self,
        resolution: tuple[int, int],
        num_frames: int,
        resp_signal: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Generate a full 4D sequence.

        Args:
            resolution: (H, W) spatial resolution
            num_frames: Number of time frames
            resp_signal: Respiratory signal [num_frames], optional
            device: Target device

        Returns:
            4D sequence [1, num_frames, C, H, W]
        """
        device = device or next(self.parameters()).device
        H, W = resolution

        # Create coordinate grid
        coords = self._make_coord_grid(H, W, device)  # [1, H*W, 2]

        # Generate frames
        frames = []
        for t_idx in range(num_frames):
            t = torch.tensor([t_idx / (num_frames - 1)], device=device)

            if resp_signal is not None:
                resp = resp_signal[t_idx : t_idx + 1]
            else:
                resp = None

            intensity = self.forward(coords, t, resp)  # [1, H*W, 1]
            frame = intensity.permute(0, 2, 1).contiguous().view(1, self.out_channels, H, W)
            frames.append(frame)

        return torch.stack(frames, dim=1)  # [1, T, C, H, W]
