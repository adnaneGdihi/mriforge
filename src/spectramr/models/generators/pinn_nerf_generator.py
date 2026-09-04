import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.blocks.pinn_nerf_block import PINNNeRFBlock
from spectramr.models.registry import register_model


class DilatedResidualBlock(nn.Module):
    """Residual block with dilated convolution for receptive field expansion.

    Unlike MaxPool, this preserves spatial resolution and phase information.
    """

    def __init__(self, channels: int, dilation: int = 1):
        """__init__.

        Args:
            channels (int): Description.
            dilation (int): Description.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DilatedResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class DilatedResidualEncoder(nn.Module):
    """Phase-preserving encoder using dilated convolutions.

    [ARCHITECTURE FIX] Replaces MaxPool encoder to preserve:
    - Full spatial resolution (no downsampling)
    - Phase information (critical for MRI de-aliasing)
    - High-frequency texture details

    Uses increasing dilations [1, 2, 4] for receptive field expansion.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: list[int] = [64, 128, 256],
        dilations: list[int] = [1, 2, 4],
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            features (list[int]): Description.
            dilations (list[int]): Description.
        """
        super().__init__()

        layers = []
        current_in = in_channels

        # Initial projection
        layers.append(nn.Conv2d(current_in, features[0], kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_in = features[0]

        # Dilated residual blocks (NO pooling)
        for i, feat in enumerate(features):
            dilation = dilations[i % len(dilations)]

            # Channel projection if needed
            if current_in != feat:
                layers.append(nn.Conv2d(current_in, feat, kernel_size=1))
                current_in = feat

            # Dilated residual block
            layers.append(DilatedResidualBlock(feat, dilation=dilation))

        self.layers = nn.Sequential(*layers)
        self.out_channels = features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DilatedResidualEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.layers(x)


@register_model(
    name="pinn_nerf",
    training_mode="reconstruction",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class PINNNeRFGenerator(nn.Module):
    """Generator using PINN-NeRF for continuous MRI reconstruction.

    [ARCHITECTURE FIX] Uses DilatedResidualEncoder instead of MaxPool
    to preserve phase information required for k-space consistency.

    Structure:
    1. Encoder (Dilated ResNet) extracts phase-preserving feature map.
    2. Grid Sampler samples features at query coordinates.
    3. PINNNeRFBlock (SIREN) predicts values at query coordinates.
    4. (Optional) Reshape back to image for standard training loop.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_freqs: int = 10,
        output_domain: str = "image",  # [PHYSICS] New output domain control
        **kwargs,  # Accept but ignore extra parameters from model factory (e.g., acceleration_config)
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (list[int]): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            num_freqs (int): Description.
            output_domain (str): Description.
        """
        super().__init__()

        self.output_domain = output_domain

        # [ARCHITECTURE FIX] Dilated encoder - NO MaxPool
        # Preserves spatial resolution and phase for MRI de-aliasing
        self.encoder = DilatedResidualEncoder(
            in_channels=in_channels,
            features=features,
            dilations=[1, 2, 4],
        )

        self.feature_dim = self.encoder.out_channels

        self.nerf_block = PINNNeRFBlock(
            in_features=self.feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_freqs=num_freqs,
            out_channels=out_channels,
        )

        # Cache for coordinate grids
        self._grid_cache = {}

    def forward(
        self,
        x: torch.Tensor,
        query_coords: torch.Tensor = None,
        out_shape: tuple = None,
        sensitivity_maps: torch.Tensor = None,  # [PHYSICS] Side-load smaps
    ) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W)
            query_coords: Optional (B, N_points, 2) - Arbitrary locations to sample
            out_shape: Optional (H_out, W_out) - Target resolution override
            sensitivity_maps: Optional (B, Coils, H, W) for physics-based k-space recon
        Returns:
            Reconstructed image (B, out_channels, H, W) or (B, N, out_channels)
            OR K-Space (B, Coils, H, W) if output_domain='kspace'
        """
        B, C, H, W = x.shape

        # Handle complex k-space input: convert to magnitude
        if torch.is_complex(x):
            x = x.abs()
            B, C, H, W = x.shape

        # 1. Extract features (full resolution preserved)
        feat_map = self.encoder(x)  # (B, F, H, W) - same spatial dims!

        # 2. Create coordinates grid
        is_flat_query = False

        if query_coords is not None:
            coords = query_coords
            if coords.dim() == 3:
                is_flat_query = True
                coords_sampling = coords.unsqueeze(1)
            else:
                coords_sampling = coords
        elif out_shape is not None:
            h_out, w_out = out_shape
            coords = self._make_grid(B, h_out, w_out, x.device)
            coords_sampling = coords
        else:
            coords = self._make_grid(B, H, W, x.device)
            coords_sampling = coords

        # 3. Sample features at coordinates
        sampled_features = F.grid_sample(
            feat_map,
            coords_sampling,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Prepare for MLP
        if is_flat_query:
            sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)
            coords_flat = coords
        else:
            b, f, h_g, w_g = sampled_features.shape
            sampled_features = sampled_features.permute(0, 2, 3, 1).reshape(b, -1, f)
            coords_flat = coords_sampling.reshape(b, -1, 2)

        # 4. Predict with SIREN
        out = self.nerf_block(sampled_features, coords_flat)

        # 5. Reshape to image if grid structure exists
        if not is_flat_query:
            b_coords, h_out, w_out, _ = coords_sampling.shape
            out = out.reshape(B, h_out, w_out, -1).permute(0, 3, 1, 2)

        # [PHYSICS INTEGRATION] K-Space Reconstruction
        if self.output_domain == "kspace" and not is_flat_query:
            # out: (B, C, H, W) usually C=2 (Complex) or C=1 (Mag)
            # If C=1, we need phase or assume 0?
            # If C=2, we have Re/Im.

            # Convert to complex
            if out.shape[1] == 2:
                img_complex = torch.complex(out[:, 0], out[:, 1]).unsqueeze(1)  # (B, 1, H, W)
            else:
                # If C=1, we need phase or assume 0?
                img_complex = torch.complex(out, torch.zeros_like(out))  # (B, C, H, W)

            # Multiply by sensitivity maps
            if sensitivity_maps is not None:
                # smaps: (B, Coils, H, W)
                # img_complex: (B, 1, H, W) or (B, C, H, W)
                coil_imgs = img_complex * sensitivity_maps  # broadcast
            else:
                coil_imgs = img_complex  # Single coil case?

            # FFT
            return fft2c(coil_imgs)

        return out

    def _make_grid(self, B, H, W, device):
        """_make_grid.

        Args:
            B (Any): Description.
            H (Any): Description.
            W (Any): Description.
            device (Any): Description.
        Returns:
            Any: Description.
        """
        if isinstance(device, torch.device):
            device_key = (
                f"{device.type}:{device.index}" if device.index is not None else device.type
            )
        else:
            device_key = str(device)

        cache_key = (H, W, device_key)

        if cache_key not in self._grid_cache:
            y = torch.linspace(-1, 1, H, device=device)
            x = torch.linspace(-1, 1, W, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1)
            self._grid_cache[cache_key] = grid

        return self._grid_cache[cache_key].unsqueeze(0).expand(B, -1, -1, -1)
