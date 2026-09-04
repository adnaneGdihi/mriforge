"""Enhanced Deep Diffusion Models
Implements three variants of diffusion models with enhanced skip connections:
1. Gaussian Noise Diffusion
2. Chi-Square Noise Diffusion
3. Gaussian Noise with KAN Embeddings
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.blocks.embeddings import (
    SinusoidalPositionEmbedding as SinusoidalPositionEmbeddings,
)

# SinusoidalPositionEmbeddings is provided by blocks.embeddings and imported above.


class KANTimeEmbedding(nn.Module):
    """KAN-based time embedding for more expressive temporal representations."""

    def __init__(self, dim: int, hidden_dim: int = None):
        """__init__.

        Args:
            dim (int): Description.
            hidden_dim (int): Description.
        """
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim * 2

        # Simple KAN-inspired layers with learnable activation functions
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            KANActivation(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            KANActivation(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, time):
        # Normalize time to [-1, 1] range
        """forward.

        Args:
            time (Any): Description.
        Returns:
            Any: Description.

        forward method for KANTimeEmbedding.

        Executes PyTorch tensor operations.

        Args:
            time (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        time = time.float().unsqueeze(-1) / 1000.0
        return self.time_mlp(time)


class KANActivation(nn.Module):
    """KAN-inspired learnable activation function."""

    def __init__(self, dim: int, num_splines: int = 8):
        """__init__.

        Args:
            dim (int): Description.
            num_splines (int): Description.
        """
        super().__init__()
        self.dim = dim
        self.num_splines = num_splines

        # Learnable spline coefficients
        self.spline_weights = nn.Parameter(torch.randn(dim, num_splines))
        self.bias = nn.Parameter(torch.zeros(dim))

        # Learnable spline basis for adaptivity
        self.spline_grid = nn.Parameter(torch.linspace(-3, 3, num_splines))

    def forward(self, x):
        # Compute RBF-like basis functions
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KANActivation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_expanded = x.unsqueeze(-1)  # [B, dim, 1]

        # Adaptive width for RBF for better generalization
        grid_range = self.spline_grid.max() - self.spline_grid.min()
        sigma = grid_range / (self.num_splines - 1) + 1e-6  # Add epsilon for stability

        # Gaussian RBF basis with adaptive width
        basis = torch.exp(-0.5 * ((x_expanded - self.spline_grid) / sigma) ** 2)

        # Weighted combination
        output = torch.einsum("bdi,di->bd", basis, self.spline_weights) + self.bias
        return output


class TimeAwareResidualBlock(nn.Module):
    """Enhanced residual block with time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_dim (int): Description.
            dropout (float): Description.
            use_attention (bool): Description.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

        # Calculate num_groups that divides out_channels evenly
        num_groups1 = min(32, out_channels)
        while out_channels % num_groups1 != 0 and num_groups1 > 1:
            num_groups1 -= 1
        num_groups2 = min(32, out_channels)
        while out_channels % num_groups2 != 0 and num_groups2 > 1:
            num_groups2 -= 1

        self.norm1 = nn.GroupNorm(num_groups1, out_channels)
        self.norm2 = nn.GroupNorm(num_groups2, out_channels)

        self.dropout = nn.Dropout(dropout)

        # Channel attention
        if use_attention:
            self.attention = ChannelAttention(out_channels)
        else:
            self.attention = nn.Identity()

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, time_emb):
        """forward.

        Args:
            x (Any): Description.
            time_emb (Any): Description.
        Returns:
            Any: Description.

        forward method for TimeAwareResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            time_emb (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = self.skip_conv(x)

        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)

        # Add time conditioning
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        out = out + time_out

        out = F.silu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.attention(out)

        # Residual connection
        out = out + residual
        return F.silu(out)


class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature enhancement."""

    def __init__(self, channels: int, reduction: int = 16):
        """__init__.

        Args:
            channels (int): Description.
            reduction (int): Description.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ChannelAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class DenseSkipConnection(nn.Module):
    """Dense skip connection block for better gradient flow."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        time_dim: int,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            growth_rate (int): Description.
            num_layers (int): Description.
            time_dim (int): Description.
            dropout (float): Description.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        current_channels = in_channels

        for _i in range(num_layers):
            # Calculate num_groups that divides current_channels evenly
            num_groups = min(32, current_channels)
            while current_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1

            layer = nn.Sequential(
                nn.GroupNorm(num_groups, current_channels),
                nn.SiLU(),
                nn.Conv2d(current_channels, growth_rate, 3, padding=1),
                nn.Dropout(dropout),
            )
            self.layers.append(layer)
            current_channels += growth_rate

        # Calculate num_groups for transition layer
        num_groups_trans = min(32, current_channels)
        while current_channels % num_groups_trans != 0 and num_groups_trans > 1:
            num_groups_trans -= 1

        # Transition layer to reduce channels
        self.transition = nn.Sequential(
            nn.GroupNorm(num_groups_trans, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, in_channels, 1),
            nn.Dropout(dropout),
        )

        # Time conditioning
        self.time_mlp = nn.Linear(time_dim, in_channels)

    def forward(self, x, time_emb):
        """forward.

        Args:
            x (Any): Description.
            time_emb (Any): Description.
        Returns:
            Any: Description.

        forward method for DenseSkipConnection.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            time_emb (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = [x]

        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)

        out = torch.cat(features, 1)
        out = self.transition(out)

        # Add time conditioning
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        out = out + time_out

        return out + x  # Residual connection


class EnhancedDeepDiffusionUNet(nn.Module):
    """Enhanced Deep U-Net for Diffusion Models

    Features:
    - Dense skip connections for better gradient flow
    - Time conditioning throughout the network
    - Channel attention mechanisms
    - Support for different noise types
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        channel_mult: tuple = (1, 2, 4, 8),
        time_embedding_type: str = "sinusoidal",  # "sinusoidal" or "kan"
        noise_type: str = "gaussian",  # "gaussian", "chi_square", or "gaussian_kan"
        use_dense_skip: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            model_channels (int): Description.
            num_res_blocks (int): Description.
            dropout (float): Description.
            channel_mult (tuple): Description.
            time_embedding_type (str): Description.
            noise_type (str): Description.
            use_dense_skip (bool): Description.
        """
        super().__init__()

        valid_time_embedding_types = ("sinusoidal", "kan")
        if time_embedding_type not in valid_time_embedding_types:
            raise ValueError(
                f"Unknown time_embedding_type '{time_embedding_type}'. "
                f"Valid options: {valid_time_embedding_types}"
            )
        valid_noise_types = ("gaussian", "chi_square", "gaussian_kan")
        if noise_type not in valid_noise_types:
            raise ValueError(
                f"Unknown noise_type '{noise_type}'. Valid options: {valid_noise_types}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.noise_type = noise_type
        self.use_dense_skip = use_dense_skip

        # Time embeddings
        time_embed_dim = model_channels * 4
        if time_embedding_type == "kan":
            self.time_embed = KANTimeEmbedding(time_embed_dim)
        else:
            self.time_embed = nn.Sequential(
                SinusoidalPositionEmbeddings(model_channels),
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = model_channels
        channels_list = [ch]

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = ch if j == 0 else out_ch
                block_layers = [
                    TimeAwareResidualBlock(
                        in_ch,
                        out_ch,
                        time_embed_dim,
                        dropout,
                        use_attention=True,
                    ),
                ]
                if use_dense_skip:
                    block_layers.append(
                        DenseSkipConnection(
                            out_ch,
                            out_ch // 4,
                            2,
                            time_embed_dim,
                            dropout,
                        ),
                    )
                level_blocks.append(nn.Sequential(*block_layers))
                if j == 0:
                    ch = out_ch
            self.encoder_blocks.append(level_blocks)
            channels_list.append(ch)

            if i < len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        # Middle block
        self.middle_block = nn.Sequential(
            TimeAwareResidualBlock(ch, ch, time_embed_dim, dropout, use_attention=True),
            (
                DenseSkipConnection(ch, ch // 4, 3, time_embed_dim, dropout)
                if use_dense_skip
                else nn.Identity()
            ),
            TimeAwareResidualBlock(ch, ch, time_embed_dim, dropout, use_attention=True),
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # Start with the channel count from the last encoder level
        ch = model_channels * channel_mult[-1]

        # Reverse the channel_mult for decoder (deepest to shallowest)
        reversed_mults = list(reversed(channel_mult))

        for i, mult in enumerate(reversed_mults):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()

            # First block in each level concatenates skip connection
            skip_ch = channels_list[-(i + 1)] if i < len(channels_list) - 1 else 0
            in_ch = ch + skip_ch

            # Create blocks for this level
            for j in range(num_res_blocks + 1):
                block_layers = [
                    TimeAwareResidualBlock(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        time_embed_dim,
                        dropout,
                        use_attention=True,
                    ),
                ]
                if use_dense_skip:
                    block_layers.append(
                        DenseSkipConnection(
                            out_ch,
                            out_ch // 4,
                            2,
                            time_embed_dim,
                            dropout,
                        ),
                    )
                level_blocks.append(nn.Sequential(*block_layers))

            self.decoder_blocks.append(level_blocks)
            ch = out_ch

            # Add upsample for all but last level
            if i < len(reversed_mults) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
                )

        # Calculate num_groups that divides model_channels evenly
        num_groups = min(32, model_channels)
        while model_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
            nn.Tanh(),  # Added Tanh activation for output
        )

        # Noise type specific initialization
        self._initialize_noise_specific_components()

    def _initialize_noise_specific_components(self):
        """Initialize components specific to noise type."""
        if self.noise_type == "chi_square":
            # Chi-square noise requires different scaling
            self.noise_scale = nn.Parameter(torch.ones(1) * 0.5)
        elif self.noise_type == "gaussian_kan":
            # Additional KAN layers for Gaussian noise with KAN embeddings
            self.kan_noise_processor = nn.Sequential(
                KANActivation(self.model_channels),
                nn.Conv2d(self.model_channels, self.model_channels, 1),
                KANActivation(self.model_channels),
            )

    def _apply_block_sequence(self, x, time_emb, block_sequence):
        """Apply a sequence of blocks that may need time_emb parameter."""
        # Known blocks that accept time_emb
        time_aware_blocks = (TimeAwareResidualBlock, DenseSkipConnection)

        # Handle Sequential modules by applying each submodule individually
        if isinstance(block_sequence, nn.Sequential):
            for module in block_sequence:
                if isinstance(module, time_aware_blocks):
                    x = module(x, time_emb)
                else:
                    x = module(x)
            return x

        # Handle list of modules
        for module in block_sequence:
            if isinstance(module, time_aware_blocks):
                x = module(x, time_emb)
            else:
                x = module(x)
        return x

    def forward(self, x, timesteps, cond=None):
        """Forward pass of the diffusion model.

        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Time step tensor [B]
            cond: Optional conditioning tensor (ignored for compatibility)

        Returns:
            Predicted noise or denoised image

        forward method for EnhancedDeepDiffusionUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Apply noise-specific processing
        if self.noise_type == "gaussian_kan" and hasattr(self, "kan_noise_processor"):
            x = self.kan_noise_processor(x)
        elif self.noise_type == "chi_square":
            x = x * self.noise_scale

        # Input projection
        h = self.input_conv(x)

        # Encoder
        skip_connections = []
        for level_blocks, down_sample in zip(self.encoder_blocks, self.down_samples, strict=False):
            for block in level_blocks:
                h = self._apply_block_sequence(h, time_emb, block)
            skip_connections.append(h)
            h = down_sample(h)

        # Last encoder level - also produces a skip connection
        for block in self.encoder_blocks[-1]:
            h = self._apply_block_sequence(h, time_emb, block)
        skip_connections.append(h)

        # Middle block
        h = self._apply_block_sequence(h, time_emb, self.middle_block)

        # Decoder
        for i, (level_blocks, up_sample) in enumerate(
            zip(self.decoder_blocks, self.up_samples + [None], strict=False)
        ):
            # Concatenate skip connection if available
            if skip_connections:
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)

            # Apply all blocks for this level
            for block in level_blocks:
                h = self._apply_block_sequence(h, time_emb, block)

            # Upsample if not the last level
            if up_sample is not None:
                h = up_sample(h)

        # Output projection
        return self.output_conv(h)


class GaussianDiffusionUNet(EnhancedDeepDiffusionUNet):
    """Gaussian noise diffusion model."""

    def __init__(self, **kwargs):
        """__init__."""
        kwargs["noise_type"] = "gaussian"
        kwargs["time_embedding_type"] = "sinusoidal"
        super().__init__(**kwargs)


class ChiSquareDiffusionUNet(EnhancedDeepDiffusionUNet):
    """Chi-square noise diffusion model."""

    def __init__(self, **kwargs):
        """__init__."""
        kwargs["noise_type"] = "chi_square"
        kwargs["time_embedding_type"] = "sinusoidal"
        super().__init__(**kwargs)


class GaussianKANDiffusionUNet(EnhancedDeepDiffusionUNet):
    """Gaussian noise diffusion model with KAN embeddings."""

    def __init__(self, **kwargs):
        """__init__."""
        kwargs["noise_type"] = "gaussian_kan"
        kwargs["time_embedding_type"] = "kan"
        super().__init__(**kwargs)


def create_diffusion_model(variant: str = "gaussian", **kwargs):
    """Factory function to create diffusion model variants.

    Args:
        variant: "gaussian", "chi_square", or "gaussian_kan"
        **kwargs: Additional model parameters

    Returns:
        Diffusion model instance

    """
    if variant == "gaussian":
        return GaussianDiffusionUNet(**kwargs)
    if variant == "chi_square":
        return ChiSquareDiffusionUNet(**kwargs)
    if variant == "gaussian_kan":
        return GaussianKANDiffusionUNet(**kwargs)
    raise ValueError(
        f"Unknown variant: {variant}. Choose from 'gaussian', 'chi_square', 'gaussian_kan'",
    )


# Noise scheduling functions for different noise types
class NoiseScheduler:
    """Noise scheduler for different noise types."""

    @staticmethod
    def gaussian_noise_schedule(
        timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """Linear beta schedule for Gaussian noise."""
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return betas, alphas, alphas_cumprod

    @staticmethod
    def chi_square_noise_schedule(timesteps: int, df: float = 2.0, scale: float = 0.02):
        """Chi-square noise schedule with degrees of freedom."""
        # Chi-square distribution parameters
        t = torch.linspace(0, 1, timesteps)
        # Scale chi-square noise based on timestep
        noise_scales = scale * (1 + df * t)
        return noise_scales

    @staticmethod
    def sample_gaussian_noise(shape, device):
        """Sample Gaussian noise."""
        return torch.randn(shape, device=device)

    @staticmethod
    def sample_chi_square_noise(shape, device, df: float = 2.0):
        """Sample Chi-square noise."""
        # Generate chi-square noise using Gamma distribution
        noise = torch.distributions.Gamma(df / 2, 0.5).sample(shape).to(device)
        # Center around zero
        noise = noise - df
        return noise


# Alias function for model factory compatibility
def create_enhanced_deep_diffusion(variant: str = "gaussian", **kwargs):
    """Alias for create_diffusion_model for model factory compatibility."""
    return create_diffusion_model(variant=variant, **kwargs)
