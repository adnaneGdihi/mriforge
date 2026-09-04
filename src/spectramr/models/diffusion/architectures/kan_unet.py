"""KAN-based U-Net architecture."""

from torch import nn

from spectramr.models.blocks.convolutional import DoubleConv
from spectramr.models.blocks.unet import Down, OutConv, Up
from spectramr.models.diffusion import SinusoidalPositionEmbeddings
from spectramr.models.registry import register_model

from .kan_blocks import DownWithKAN, UpWithKAN


@register_model(
    name="kan_unet",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class RefinedKANUNet(nn.Module):
    """
    A U-Net using KAN-based convolutional blocks only in the bottleneck.

    Optimized for efficiency by using standard Convs for high-resolution feature maps
    and KANs for low-resolution, high-semantic density bottleneck features.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        base_dim=32,
        time_dim=128,
        small_kan=False,
        cond_channels=0,
        bilinear=False,
        kan_type="FastKAN",
        bottleneck_only=None,  # Consumed here, ignored since this model is bottleneck-only by definition
        features=None,  # Consumed here: legacy multi-scale feature spec; this model derives widths from base_dim only
        base_filters=None,  # Consumed here: legacy alias for base_dim — preserved for YAML compatibility
        **kan_kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            base_dim (Any): Description.
            time_dim (Any): Description.
            small_kan (Any): Description.
            cond_channels (Any): Description.
            bilinear (Any): Description.
            kan_type (Any): Description.
        """
        super().__init__()
        # Honor base_filters as alias for base_dim when caller supplied it explicitly.
        if base_filters is not None:
            base_dim = int(base_filters)
        # `features` (e.g. [32, 64, 128, 256, 512]) is purely informational for this
        # bottleneck-only architecture — capture & drop so it never reaches KAN/BatchNorm.
        _ = features
        if small_kan:
            base_dim //= 2
            time_dim = int(time_dim / 2)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # A small, stable network to process the conditional image for classifier-free guidance
        if cond_channels > 0:
            self.cond_net = nn.Sequential(
                nn.Conv2d(cond_channels, base_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(base_dim // 2, base_dim, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(base_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.cond_net = None

        # Encoder (Standard Conv)
        # We use standard DoubleConv/Down blocks for spatial downsampling
        self.inc = DoubleConv(in_channels, base_dim)  # Standard
        self.down1 = Down(base_dim, base_dim * 2)  # Standard
        self.down2 = Down(base_dim * 2, base_dim * 4)  # Standard

        # Bottleneck (KAN)
        # Use KAN at the deepest layer where spatial resolution is lowest (e.g. 16x16)
        self.down3 = DownWithKAN(
            base_dim * 4,
            base_dim * 8,
            time_emb_dim=time_dim,
            kan_type=kan_type,
            **kan_kwargs,
        )

        # Decoder (Mixed)
        # First upsampling consumes bottleneck features, so we can use KAN or Standard.
        # To be safe and efficient, we transition back to standard quickly.
        # But let's keep KAN for the first upsampling to decode semantic features.
        self.up1 = UpWithKAN(
            base_dim * 8,
            base_dim * 4,
            time_emb_dim=time_dim,
            bilinear=bilinear,
            kan_type=kan_type,
            **kan_kwargs,
        )

        # Remaining Decoder (Standard Conv)
        self.up2 = Up(
            base_dim * 4,
            base_dim * 2,
            bilinear=bilinear,
        )
        self.up3 = Up(
            base_dim * 2,
            base_dim,
            bilinear=bilinear,
        )
        self.outc = OutConv(base_dim, out_channels)

    def forward(self, x, t=None, cond=None, **kwargs):
        """Forward pass through KAN U-Net.

        Args:
            x: Input tensor [B, C, H, W].
            t: Diffusion timesteps [B]. Defaults to zeros if not provided.
            cond: Optional conditioning image.
        Returns:
            Reconstructed tensor [B, C_out, H, W].

        forward method for RefinedKANUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        import torch

        x = x.to(next(self.parameters()).device)
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        t = t.to(next(self.parameters()).device)
        if cond is not None:
            cond = cond.to(next(self.parameters()).device)

        t_emb = self.time_mlp(t)

        # Add condition embedding for guidance (if applicable)
        if self.cond_net is not None and cond is not None:
            cond_emb = self.cond_net(cond)
            t_emb = t_emb + cond_emb

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Bottleneck
        x4 = self.down3(x3, t_emb)  # KAN block takes time embedding

        # Decoder
        x = self.up1(x4, x3, t_emb)  # KAN block takes time embedding
        x = self.up2(x, x2)  # Standard block doesn't take time embedding (unless modified)
        x = self.up3(x, x1)  # Standard block doesn't take time embedding

        return self.outc(x)


# Alias for backward compatibility
RefinedDiffusionUNet = RefinedKANUNet
