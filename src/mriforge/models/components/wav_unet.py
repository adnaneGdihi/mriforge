import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mriforge.models.layers.kan.kan_convs.wav_kan import WavKANConv2DLayer


class DoubleConv(nn.Module):
    """DoubleConv class."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_class=WavKANConv2DLayer,
        dropout=0.0,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            conv_class (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.conv = nn.Sequential(
            conv_class(
                input_dim=in_channels,
                output_dim=out_channels,
                kernel_size=3,
                groups=1,
                padding=1,
                stride=1,
                dilation=1,
                dropout=0,
                wavelet_type="mexican_hat",
                norm_layer=nn.BatchNorm2d,
                wav_version="fast",
            ),
            nn.ReLU(inplace=True),
            conv_class(
                input_dim=out_channels,
                output_dim=out_channels,
                kernel_size=3,
                padding=1,
                wavelet_type="mexican_hat",
            ),
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
        return self.conv(x)


class DownSample(nn.Module):
    """DownSample class."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_class=WavKANConv2DLayer,
        dropout=0.0,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            conv_class (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, conv_class=conv_class, dropout=0.0),
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for DownSample.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    """UpSample class."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_class=WavKANConv2DLayer,
        dropout=0.0,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            conv_class (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            conv_class=conv_class,
            dropout=0.0,
        )

    def forward(self, x1, x2):
        """forward.

        Args:
            x1 (Any): Description.
            x2 (Any): Description.
        Returns:
            Any: Description.

        forward method for UpSample.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.up(x1)

        # Handling the case where sizes don't match perfectly
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Wav_Unet(nn.Module):
    """Wav_Unet class."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        bilinear=False,
        opt=None,
        norm_layer=nn.InstanceNorm2d,
        conv_class=WavKANConv2DLayer,
        dropout_rate=0.2,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            bilinear (Any): Description.
            opt (Any): Description.
            norm_layer (Any): Description.
            conv_class (Any): Description.
            dropout_rate (Any): Description.
        """
        super().__init__()
        self.opt = opt

        # Note: bias usage determination removed as it's not currently used
        # if isinstance(norm_layer, functools.partial):
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        # Dropout layer
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Encoder with increased capacity
        self.inc = DoubleConv(
            in_channels,
            32,
            conv_class=conv_class,
            dropout=dropout_rate,
        )
        self.down1 = DownSample(32, 64, conv_class=conv_class, dropout=dropout_rate)
        self.down2 = DownSample(
            64,
            128,
            conv_class=conv_class,
            dropout=dropout_rate,
        )  # extra downsample for increased depth

        # Decoder with corresponding layers
        self.up1 = UpSample(128, 64, conv_class=conv_class, dropout=dropout_rate)
        self.up2 = UpSample(64, 32, conv_class=conv_class, dropout=dropout_rate)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.output_activation = nn.Tanh()

    def weight_norm(x):
        """weight_norm.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        return x

    def forward(self, x, layers=None):
        """forward.

        Args:
            x (Any): Description.
            layers (Any): Description.
        Returns:
            Any: Description.

        forward method for Wav_Unet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            layers (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if layers is None:
            layers = []
        x1 = self.inc(x)  # Encoder level 1
        x2 = self.down1(x1)  # Encoder level 2
        x3 = self.down2(x2)  # Bottleneck
        x = self.up1(x3, x2)  # Decoder level 1
        x = self.up2(x, x1)  # Decoder level 2
        x = self.outc(x)  # Final output mapping

        # Apply tanh activation to ensure output is in [-1, 1] range
        return self.output_activation(x)

    def forward_checkpoint(self, x):
        """forward_checkpoint.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        x1 = checkpoint(lambda x: self.dropout(self.inc(x)), x)
        x2 = checkpoint(lambda x: self.dropout(self.down1(x)), x1)
        x3 = checkpoint(lambda x: self.dropout(self.down2(x)), x2)
        x = checkpoint(lambda a, b: self.dropout(self.up1(a, b)), x3, x2)
        x = checkpoint(lambda a, b: self.dropout(self.up2(a, b)), x, x1)
        x = self.outc(x)

        # Apply tanh activation to ensure output is in [-1, 1] range
        return self.output_activation(x)
