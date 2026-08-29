import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

__all__ = ["SplitClientEncoder", "SplitServerDecoder"]


@register_model(name="split_client_encoder", training_mode="reconstruction")
class SplitClientEncoder(nn.Module, IGenerator):
    """
    Client-side Encoder for Federated Split Learning (Experiment 105).

    This part resides on the hospital/client side. It processes the raw input
    and sends intermediate activations (cut_layer) to the server.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        # The cut is here. We send the output of down2 to the server.

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Description.

        forward method for SplitClientEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x3, [x1, x2]  # Return skips to keep locally if needed, or send if safe

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "split_client_encoder"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        # Output is downsampled by factor of 4 (two Down layers)
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (
            input_shape[0],
            self.base_channels * 4,
            input_shape[2] // 4,
            input_shape[3] // 4,
        )


@register_model(name="split_server_decoder", training_mode="reconstruction")
class SplitServerDecoder(nn.Module, IGenerator):
    """
    Server-side Decoder for Federated Split Learning.

    Receives activations from the client, processes the bottleneck and decoder,
    and computes the loss.
    """

    def __init__(self, out_channels=1, base_channels=64):
        """__init__.

        Args:
            out_channels (Any): Description.
            base_channels (Any): Description.
        """
        super().__init__()
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // 2)
        self.up1 = Up(base_channels * 16, base_channels * 8 // 2)
        self.up2 = Up(base_channels * 8, base_channels * 4 // 2)
        self.up3 = Up(base_channels * 4, base_channels * 2 // 2)
        self.up4 = Up(base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x_client, skips_client=None):
        # x_client is output of down2 (channels * 4)

        """forward.

        Args:
            x_client (Any): Description.
            skips_client (Any): Description.
        Returns:
            Any: Description.

        forward method for SplitServerDecoder.

        Executes PyTorch tensor operations.

        Args:
            x_client (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            skips_client (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x4 = self.down3(x_client)
        x5 = self.down4(x4)

        # For true split learning, skips might need to be handled carefully.
        # If skips are not sent (privacy), we might use U-Net without skips or
        # use a different architecture.
        # If we assume we can receive skips (obfuscated?):
        if skips_client is None:
            # Privacy-Preserving Mode ("No-Peek")
            # If skips are not provided (to protect patient privacy), we initialize
            # local zero-tensors matching the expected shapes.
            # This enables the server to train a decoder without inspecting intermediate client features.

            b, c, h, w = x_client.shape
            # x_client is output of down2 (base * 4)
            # base = c // 4
            base = c // 4

            # Expected shapes based on SplitClientEncoder architecture:
            # x_client (x3): [B, base*4, H, W]
            # x2 (skip[1]): [B, base*2, H*2, W*2]
            # x1 (skip[0]): [B, base, H*4, W*4]

            skips_client = [
                # Skip 0 (x1)
                torch.zeros(b, base, h * 4, w * 4, device=x_client.device),
                # Skip 1 (x2)
                torch.zeros(b, base * 2, h * 2, w * 2, device=x_client.device),
            ]

        x = self.up1(x5, x4)
        x = self.up2(x, x_client)
        x = self.up3(x, skips_client[1])
        x = self.up4(x, skips_client[0])
        logits = self.outc(x)
        return logits

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def generate(self, x_client, skips_client=None) -> torch.Tensor:
        """generate.

        Args:
            x_client (Any): Description.
            skips_client (Any): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x_client, skips_client)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "split_server_decoder"


# --- Helper Blocks (Standard U-Net parts) ---


class DoubleConv(nn.Module):
    """DoubleConv class."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            mid_channels (Any): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for DoubleConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.double_conv(x)


class Down(nn.Module):
    """Down class."""

    def __init__(self, in_channels, out_channels):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Down.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up class."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            bilinear (Any): Description.
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """forward.

        Args:
            x1 (Any): Description.
            x2 (Any): Description.
        Returns:
            Any: Description.

        forward method for Up.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
