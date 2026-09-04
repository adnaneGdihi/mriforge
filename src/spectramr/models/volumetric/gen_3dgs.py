"""Gen-3DGS: Generative 3D Gaussian Splatting VAE.

Innovation VII (SOTA 2.0): Explicit Volumetric Generative Model.

Mathematical Foundation:
    - Primitives: 3D Gaussians (mu, Sigma, alpha, c)
    - Structured Grid: Gaussians are anchored to a voxel grid.
    - Parameterization per voxel:
        - offsets: delta_mu in [-0.5, 0.5] (tanh)
        - scale: s in R+ (softplus)
        - rotation: q in SO(3) (normalized quaternion)
        - opacity: alpha in [0, 1] (sigmoid)
        - intensity: c in R (linear/sigmoid)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


class GaussianParameters:
    """Helper to interpret tensor output as Gaussian Parameters."""

    def __init__(self, tensor: torch.Tensor):
        # tensor: [B, 12, D, H, W]
        # Channels map:
        # 0-2: mu offsets
        # 3-5: scale (log)
        # 6-9: rotation (quaternion)
        # 10: opacity (logit)
        # 11: intensity (value)

        """__init__.

        Args:
            tensor (torch.Tensor): Description.
        """
        self.raw = tensor

    @property
    def xyz(self) -> torch.Tensor:
        """Get absolute coordinates of Gaussians."""
        B, C, D, H, W = self.raw.shape
        device = self.raw.device

        # Grid coordinates
        # We need to generate a grid of centers
        # z, y, x order matching D, H, W
        dz = torch.linspace(0, 1, D, device=device)
        dy = torch.linspace(0, 1, H, device=device)
        dx = torch.linspace(0, 1, W, device=device)

        grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing="ij")

        grid = torch.stack([grid_x, grid_y, grid_z], dim=0)  # [3, D, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, 3, D, H, W]

        # Offsets: tanh mapped to [-1/grid_size, 1/grid_size] roughly?
        # Or just relative to cell size.
        # Let's assume global coordinates normalized to [0, 1].
        # Cell size is roughly 1/D, 1/H, 1/W.

        offsets = torch.tanh(self.raw[:, 0:3])  # [-1, 1]

        # Scale offsets by cell size to stay local?
        scale_fac = torch.tensor([1.0 / W, 1.0 / H, 1.0 / D], device=device).view(1, 3, 1, 1, 1)

        return grid + offsets * scale_fac * 0.5  # Constrain to cell

    @property
    def scale(self) -> torch.Tensor:
        """scale.

        Returns:
            torch.Tensor: Description.
        """
        return F.softplus(self.raw[:, 3:6])  # Positive dimensionality

    @property
    def rotation(self) -> torch.Tensor:
        """rotation.

        Returns:
            torch.Tensor: Description.
        """
        q = self.raw[:, 6:10]
        return F.normalize(q, dim=1)  # Normalized Quaternion

    @property
    def opacity(self) -> torch.Tensor:
        """opacity.

        Returns:
            torch.Tensor: Description.
        """
        return torch.sigmoid(self.raw[:, 10:11])

    @property
    def intensity(self) -> torch.Tensor:
        """intensity.

        Returns:
            torch.Tensor: Description.
        """
        return self.raw[:, 11:12]  # Raw intensity


class Encoder3D(nn.Module):
    """3D CNN Encoder."""

    def __init__(self, in_channels=1, latent_dim=128):
        """__init__.

        Args:
            in_channels (Any): Description.
            latent_dim (Any): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, stride=2),  # /2
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2),  # /4
            nn.BatchNorm3d(64),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2),  # /8
            nn.BatchNorm3d(128),
            nn.SiLU(),
            nn.Conv3d(128, latent_dim, kernel_size=3, padding=1),
            nn.Tanh(),  # Constrain latent
        )
        self.use_checkpoint = True

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Encoder3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if getattr(self, "use_checkpoint", False) and self.training:
            return torch.utils.checkpoint.checkpoint(self.net, x, use_reentrant=False)
        return self.net(x)


class Decoder3D(nn.Module):
    """3D CNN Decoder to Gaussian/Voxel Grid."""

    def __init__(self, latent_dim=128, out_channels=12):
        """__init__.

        Args:
            latent_dim (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__()
        # Input is /8 resolution
        self.net = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, padding=1, stride=2),  # *2
            nn.BatchNorm3d(64),
            nn.SiLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, padding=1, stride=2),  # *4
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.ConvTranspose3d(
                32, 16, kernel_size=4, padding=1, stride=2
            ),  # *8 -> Original resolution grid
            nn.BatchNorm3d(16),
            nn.SiLU(),
            nn.Conv3d(16, out_channels, kernel_size=1),  # Pixel-wise projection
        )
        self.use_checkpoint = True

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Decoder3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if getattr(self, "use_checkpoint", False) and self.training:
            return torch.utils.checkpoint.checkpoint(self.net, x, use_reentrant=False)
        return self.net(x)


@register_model(name="gen_3dgs", training_mode="synthesis")
class Gen3DGS(nn.Module):
    """Generative 3D Gaussian Splatting Model."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
        """
        super().__init__()
        self.encoder = Encoder3D(in_channels, latent_dim)
        # Output channels = 12 (3 pos + 3 scale + 4 rot + 1 op + 1 int)
        self.decoder = Decoder3D(latent_dim, out_channels=12)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, GaussianParameters]:
        """
        Args:
           x: Input MRI volume [B, C, D, H, W]

        Returns:
           mu: Latent distribution [B, L, d, h, w] (if VAE, here simple AE for now)
           params: GaussianParameters

        forward method for Gen3DGS.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, GaussianParameters]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode
        z = self.encoder(x)

        # Decode to Grid Parameters
        raw_params = self.decoder(z)

        # In a real VAE, we'd sample z.
        # Here we verify the backbone connectivity.

        params = GaussianParameters(raw_params)

        return z, params

    def get_parameter_shapes(self, input_shape):
        """Helper to debug output shapes."""
        # Simple dry run
        dummy = torch.zeros(1, *input_shape)  # No device
        z, p = self(dummy)
        return z.shape, p.raw.shape


def create_gen_3dgs(in_channels: int = 1, latent_dim: int = 128) -> Gen3DGS:
    """create_gen_3dgs.

    Args:
        in_channels (int): Description.
        latent_dim (int): Description.
    Returns:
        Gen3DGS: Description.
    """
    return Gen3DGS(in_channels, latent_dim)
