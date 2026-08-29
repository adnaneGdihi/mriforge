"""Minimal UNet used for score-based diffusion configs.
This keeps the public API stable for tests expecting `ScoreBasedUNet`.
"""

from torch import nn

from mriforge.models.blocks.block_registry import create_block
from mriforge.models.diffusion.base_diffusion import (
    DoubleConvWithTime,
    DownWithTime,
    UpWithTime,
)
from mriforge.models.reconstruction.unet import SinusoidalPositionEmbeddings


class ScoreBasedUNet(nn.Module):
    """A light U-Net with time conditioning suitable for score-based diffusion.

    Signature mirrors other UNets: forward(x, t, cond=None) -> tensor
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_dim=128,
        base_dim=32,
        cond_channels=0,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            time_dim (Any): Description.
            base_dim (Any): Description.
            cond_channels (Any): Description.
        """
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.cond_net = None
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

        self.inc = DoubleConvWithTime(in_channels, base_dim, time_emb_dim=time_dim)
        self.down1 = DownWithTime(base_dim, base_dim * 2, time_emb_dim=time_dim)
        self.down2 = DownWithTime(base_dim * 2, base_dim * 4, time_emb_dim=time_dim)
        self.up1 = UpWithTime(base_dim * 4, base_dim * 2, time_emb_dim=time_dim)
        self.up2 = UpWithTime(base_dim * 2, base_dim, time_emb_dim=time_dim)
        self.outc = create_block("unet_out_conv", in_channels=base_dim, out_channels=out_channels)

    def forward(self, x, t, cond=None):
        # Ensure input tensors are on the correct device
        """forward.

        Args:
            x (Any): Description.
            t (Any): Description.
            cond (Any): Description.
        Returns:
            Any: Description.

        forward method for ScoreBasedUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x.to(next(self.parameters()).device)
        t = t.to(next(self.parameters()).device)
        if cond is not None:
            cond = cond.to(next(self.parameters()).device)

        t_emb = self.time_mlp(t)
        if self.cond_net is not None and cond is not None:
            t_emb = t_emb + self.cond_net(cond)

        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x = self.up1(x3, x2, t_emb)
        x = self.up2(x, x1, t_emb)
        return self.outc(x)
