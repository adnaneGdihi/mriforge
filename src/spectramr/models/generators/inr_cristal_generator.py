"""INR-CRISTAL: Joint Coil Sensitivity and Image Estimation via INR.

Implicit Neural Representation for Coil-Combined image Reconstruction
with Integrated Sensitivity learning Through Alternating Layers.

Key Features:
- Jointly estimates coil sensitivities and image
- Uses INR for continuous representation
- No need for pre-computed sensitivity maps

Reference:
- INR-CRISTAL: Joint sensitivity and image estimation via INR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class SensitivityINR(nn.Module):
    """INR network for coil sensitivity estimation.

    Represents coil sensitivities as continuous functions
    S_c(x, y) for each coil c.

    Args:
        num_coils: Number of receiver coils
        hidden_dim: Hidden dimension
        num_layers: Number of MLP layers
    """

    def __init__(
        self,
        num_coils: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_frequencies: int = 64,
    ):
        """__init__.

        Args:
            num_coils (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            num_frequencies (int): Description.
        """
        super().__init__()

        self.num_coils = num_coils

        # Positional encoding
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0, num_frequencies // 2 - 1, num_frequencies // 2),
        )

        in_dim = 2 + 2 * num_frequencies  # x,y + sin/cos encodings

        # Shared MLP
        layers = []
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else num_coils * 2
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply Fourier positional encoding."""
        # coords: [N, 2]
        proj = coords.unsqueeze(-1) * self.freq_bands  # [N, 2, F]
        sin_enc = torch.sin(2 * torch.pi * proj).flatten(-2)  # [N, 2*F]
        cos_enc = torch.cos(2 * torch.pi * proj).flatten(-2)  # [N, 2*F]
        return torch.cat([coords, sin_enc, cos_enc], dim=-1)

    def forward(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate sensitivity maps at coordinates.

        Args:
            coords: [N, 2] or [B, N, 2] spatial coordinates

        Returns:
            [N, num_coils, 2] or [B, N, num_coils, 2] complex sensitivities

        forward method for SensitivityINR.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if coords.dim() == 3:
            B, N, _ = coords.shape
            coords_flat = coords.view(-1, 2)
            encoded = self.positional_encoding(coords_flat)
            out = self.mlp(encoded)
            out = out.view(B, N, self.num_coils, 2)
        else:
            encoded = self.positional_encoding(coords)
            out = self.mlp(encoded)
            out = out.view(-1, self.num_coils, 2)

        return out


class ImageINR(nn.Module):
    """INR network for image estimation.

    Represents the underlying MR image as a continuous function I(x, y).

    Args:
        out_channels: Number of output channels (real/imaginary)
        hidden_dim: Hidden dimension
        num_layers: Number of SIREN layers
    """

    def __init__(
        self,
        out_channels: int = 2,  # Real + Imaginary
        hidden_dim: int = 256,
        num_layers: int = 5,
        omega_0: float = 30.0,
    ):
        """__init__.

        Args:
            out_channels (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            omega_0 (float): Description.
        """
        super().__init__()

        self.omega_0 = omega_0

        # SIREN-style network
        self.first_layer = nn.Linear(2, hidden_dim)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        )

        self.output_layer = nn.Linear(hidden_dim, out_channels)

        # Initialize weights for SIREN
        self._init_siren_weights()

    def _init_siren_weights(self):
        """SIREN weight initialization."""
        with torch.no_grad():
            self.first_layer.weight.uniform_(-1 / 2, 1 / 2)
            for layer in self.hidden_layers:
                bound = (6 / layer.weight.shape[1]) ** 0.5 / self.omega_0
                layer.weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Estimate image at coordinates.

        Args:
            coords: [N, 2] or [B, N, 2] spatial coordinates

        Returns:
            [N, 2] or [B, N, 2] complex image values

        forward method for ImageINR.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch_mode = coords.dim() == 3
        if batch_mode:
            B, N, _ = coords.shape
            coords = coords.view(-1, 2)

        # SIREN forward
        h = torch.sin(self.omega_0 * self.first_layer(coords))

        for layer in self.hidden_layers:
            h = torch.sin(self.omega_0 * layer(h))

        out = self.output_layer(h)

        if batch_mode:
            out = out.view(B, N, -1)

        return out


@register_model(name="inr_cristal", training_mode="reconstruction")
class INRCRISTALGenerator(nn.Module, IGenerator):
    """INR-CRISTAL: Joint Coil Sensitivity and Image Estimation.

    Estimates both coil sensitivity maps and the underlying image
    directly from k-space measurements using INRs.

    Args:
        in_channels: Input channels (k-space real/imag)
        out_channels: Output channels
        num_coils: Number of receiver coils
        hidden_dim: Hidden dimension for INRs
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        num_coils: int = 8,
        hidden_dim: int = 256,
        num_image_layers: int = 5,
        num_sens_layers: int = 4,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_coils (int): Description.
            hidden_dim (int): Description.
            num_image_layers (int): Description.
            num_sens_layers (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_coils = num_coils

        # Image INR
        self.image_inr = ImageINR(
            out_channels=2,  # Complex
            hidden_dim=hidden_dim,
            num_layers=num_image_layers,
        )

        # Sensitivity INR
        self.sens_inr = SensitivityINR(
            num_coils=num_coils,
            hidden_dim=hidden_dim // 2,
            num_layers=num_sens_layers,
        )

        # Refinement network (operates on image domain)
        self.refine = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def create_grid(
        self,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create coordinate grid.

        Returns:
            [H*W, 2] coordinates in [-1, 1]
        """
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        return coords

    def forward_from_coords(
        self,
        coords: torch.Tensor,
        return_sens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass at specific coordinates.

        Args:
            coords: [N, 2] coordinates
            return_sens: Whether to return sensitivity maps

        Returns:
            image: [N, 2] complex image values
            sens: [N, C, 2] sensitivity maps (if return_sens)
        """
        image = self.image_inr(coords)

        if return_sens:
            sens = self.sens_inr(coords)
            return image, sens

        return image, None

    def forward(
        self,
        x: torch.Tensor,
        return_sens: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] input (zeroed-filled or low-res)
            return_sens: Whether to return sensitivity maps

        Returns:
            [B, C_out, H, W] reconstructed image

        forward method for INRCRISTALGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_sens (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        device = x.device

        # Create coordinate grid
        coords = self.create_grid(H, W, device)  # [H*W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]

        # Estimate image at all coordinates
        image = self.image_inr(coords)  # [B, H*W, 2]
        image = image.view(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # Refine
        out = self.refine(image)

        return out

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
        return "INRCRISTALGenerator"

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

    def fit_kspace(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        num_iters: int = 1000,
        lr: float = 1e-3,
    ) -> torch.Tensor:
        """Fit INR directly to k-space measurements.

        Args:
            kspace: [B, C, H, W, 2] multi-coil k-space
            mask: [B, 1, H, W] or [H, W] sampling mask
            num_iters: Optimization iterations
            lr: Learning rate

        Returns:
            Reconstructed image
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        B, C, H, W, _ = kspace.shape
        device = kspace.device

        coords = self.create_grid(H, W, device)
        coords = coords.unsqueeze(0).expand(B, -1, -1)

        for _ in range(num_iters):
            optimizer.zero_grad()

            # Estimate image and sensitivities
            image, sens = self.forward_from_coords(coords.view(-1, 2), return_sens=True)

            # Reshape
            image = image.view(B, H, W, 2)
            sens = sens.view(B, H, W, self.num_coils, 2)

            # Apply coil sensitivities: S * I
            # Complex multiplication
            coil_images_real = sens[..., 0] * image[..., 0:1] - sens[..., 1] * image[..., 1:2]
            coil_images_imag = sens[..., 0] * image[..., 1:2] + sens[..., 1] * image[..., 0:1]
            coil_images = torch.stack([coil_images_real, coil_images_imag], dim=-1)
            coil_images = coil_images.permute(0, 3, 1, 2, 4)  # [B, C, H, W, 2]

            # FFT using physics module for proper centering
            coil_images_complex = torch.complex(coil_images[..., 0], coil_images[..., 1])
            pred_kspace = fft2c(coil_images_complex)
            pred_kspace = torch.stack([pred_kspace.real, pred_kspace.imag], dim=-1)

            # Data consistency loss
            gt_kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])
            pred_kspace_complex = torch.complex(pred_kspace[..., 0], pred_kspace[..., 1])

            loss = F.mse_loss(
                pred_kspace_complex[mask.expand_as(pred_kspace_complex) > 0].abs(),
                gt_kspace_complex[mask.expand_as(gt_kspace_complex) > 0].abs(),
            )

            loss.backward()
            optimizer.step()

        # Final reconstruction
        with torch.no_grad():
            return self.forward(torch.zeros(B, 2, H, W, device=device))


__all__ = [
    "INRCRISTALGenerator",
    "ImageINR",
    "SensitivityINR",
]
