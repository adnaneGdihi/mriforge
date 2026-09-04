"""Geo-FNO: Geometry-Aware Fourier Neural Operator.

Extends FNO with geometry-adaptive coordinate transform for
non-Cartesian domains common in MRI applications.

Key Features:
- Learnable coordinate deformation for irregular geometries
- Physics-informed spectral layers
- Multi-scale Fourier processing

Reference:
- Li et al. "Geometry-aware Fourier Neural Operator"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer.

    Performs convolution in Fourier space for global receptive field.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        modes1: Number of Fourier modes in first dimension
        modes2: Number of Fourier modes in second dimension
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            modes1 (int): Description.
            modes2 (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Learnable Fourier coefficients
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        # input: [B, C_in, H, W], weights: [C_in, C_out, H, W]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution.

        forward method for SpectralConv2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )

        # Top modes
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )

        # Bottom modes
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(H, W))


class CoordinateDeformer(nn.Module):
    """Learnable coordinate deformation for irregular geometries.

    Maps physical coordinates to computational domain.

    Args:
        hidden_dim: Hidden network dimension
        num_layers: Number of MLP layers
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        """__init__.

        Args:
            hidden_dim (int): Description.
            num_layers (int): Description.
        """
        super().__init__()

        layers = []
        in_dim = 2  # (x, y) coordinates

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 2
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.Tanh())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Deform coordinates.

        Args:
            coords: [B, 2, H, W] or [N, 2] coordinate tensor

        Returns:
            Deformed coordinates (same shape)

        forward method for CoordinateDeformer.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if coords.dim() == 4:
            B, C, H, W = coords.shape
            coords_flat = coords.permute(0, 2, 3, 1).reshape(-1, 2)
            deformed = self.net(coords_flat)
            return deformed.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        else:
            return self.net(coords)


class FNOBlock(nn.Module):
    """FNO building block with spectral convolution and bypass."""

    def __init__(
        self,
        channels: int,
        modes1: int = 12,
        modes2: int = 12,
    ):
        """__init__.

        Args:
            channels (int): Description.
            modes1 (int): Description.
            modes2 (int): Description.
        """
        super().__init__()

        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for FNOBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


@register_model(name="geo_fno", training_mode="reconstruction")
class GeoFNOGenerator(nn.Module, IGenerator):
    """Geometry-Aware Fourier Neural Operator.

    Extends FNO with learnable coordinate deformation for
    irregular domains (e.g., brain surface, tumors).

    Args:
        in_channels: Input channels
        out_channels: Output channels
        hidden_channels: Hidden layer channels
        modes: Number of Fourier modes
        num_layers: Number of FNO layers
        use_deformation: Whether to use geometry deformation
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        modes: int = 12,
        num_layers: int = 4,
        use_deformation: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_channels (int): Description.
            modes (int): Description.
            num_layers (int): Description.
            use_deformation (bool): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deformation = use_deformation

        # Coordinate deformer
        if use_deformation:
            self.deformer = CoordinateDeformer()

        # Lifting layer
        # Input: image + coordinates (if deformation) = in_channels + 2
        lift_channels = in_channels + 2 if use_deformation else in_channels
        self.lift = nn.Conv2d(lift_channels, hidden_channels, 1)

        # FNO layers
        self.fno_blocks = nn.ModuleList(
            [FNOBlock(hidden_channels, modes, modes) for _ in range(num_layers)]
        )

        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def get_grid(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate coordinate grid.

        Args:
            shape: (B, H, W)
            device: Target device

        Returns:
            [B, 2, H, W] coordinate tensor in [-1, 1]
        """
        B, H, W = shape

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        grid = torch.stack([xx, yy], dim=0)  # [2, H, W]
        return grid.unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] input image

        Returns:
            [B, C_out, H, W] output image

        forward method for GeoFNOGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Get coordinate grid
        grid = self.get_grid((B, H, W), x.device)

        # Apply deformation
        if self.use_deformation:
            deformed_grid = self.deformer(grid)
            x_aug = torch.cat([x, deformed_grid], dim=1)
        else:
            x_aug = x

        # Lift to hidden dimension
        h = self.lift(x_aug)

        # FNO layers
        for block in self.fno_blocks:
            h = block(h) + h  # Residual

        # Project to output
        return self.project(h)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
        return "GeoFNOGenerator"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        B, C, H, W = input_shape
        return (B, self.out_channels, H, W)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "CoordinateDeformer",
    "FNOBlock",
    "GeoFNOGenerator",
    "SpectralConv2d",
]
