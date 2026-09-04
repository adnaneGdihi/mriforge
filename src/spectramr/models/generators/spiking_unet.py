"""Spiking U-Net for MRI Reconstruction

Implements a U-Net architecture using Leaky Integrate-and-Fire (LIF) neurons
for energy-efficient neuromorphic computing.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["LIFNeuron", "SpikingUNet"]


class SurrogateHeaviside(torch.autograd.Function):
    """
    Heaviside function with ArcTan surrogate gradient.
    Forward: Heaviside(x)
    Backward: (1 / (1 + (pi * x)^2))
    """

    @staticmethod
    def forward(ctx, x):
        """forward.

        Args:
            ctx (Any): Description.
            x (Any): Description.
        Returns:
            Any: Description.
        """
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """backward.

        Args:
            ctx (Any): Description.
            grad_output (Any): Description.
        Returns:
            Any: Description.
        """
        (x,) = ctx.saved_tensors
        # alpha controls the width of the surrogate gradient
        alpha = 2.0
        grad_x = grad_output / (1 + (alpha * x).pow(2))
        return grad_x


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron layer."""

    def __init__(
        self,
        threshold: float = 1.0,
        leak: float = 0.9,
        reset_mechanism: str = "subtract",
    ):
        """__init__.

        Args:
            threshold (float): Description.
            leak (float): Description.
            reset_mechanism (str): Description.
        """
        super().__init__()
        self.threshold = threshold
        self.leak = leak
        self.reset_mechanism = reset_mechanism
        self.membrane_potential = None
        self.spike_fn = SurrogateHeaviside.apply

    def reset_state(self, batch_size: int, *shape: int, device: torch.device):
        """Reset membrane potential."""
        self.membrane_potential = torch.zeros(batch_size, *shape, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LIF neuron dynamics.

        forward method for LIFNeuron.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.membrane_potential is None or self.membrane_potential.shape != x.shape:
            self.reset_state(x.shape[0], *x.shape[1:], device=x.device)

        # Integrate: V(t) = leak * V(t-1) + I(t)
        self.membrane_potential = self.leak * self.membrane_potential + x

        # Fire: spike when V >= threshold with surrogate gradient
        # V - threshold is inputs to Heaviside.
        # If V >= threshold, input >= 0 -> spike=1
        spikes = self.spike_fn(self.membrane_potential - self.threshold)

        # Reset
        if self.reset_mechanism == "subtract":
            self.membrane_potential = self.membrane_potential - spikes * self.threshold
        else:  # zero reset
            self.membrane_potential = self.membrane_potential * (1 - spikes)

        return spikes


class SpikingConvBlock(nn.Module):
    """Convolutional block with spiking neurons."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SpikingConvBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv(x)
        x = self.bn(x)
        return self.lif(x)


@register_model(name="spiking_unet", training_mode="reconstruction")
class SpikingUNet(nn.Module, IGenerator):
    """U-Net with spiking neurons for neuromorphic MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        time_steps: int = 10,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (int): Description.
            time_steps (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_steps = time_steps

        # Input encoding (rate coding)
        self.input_encoder = nn.Conv2d(in_channels, features, 3, 1, 1)

        # Encoder
        self.enc1 = SpikingConvBlock(features, features)
        self.enc2 = SpikingConvBlock(features, features * 2)
        self.enc3 = SpikingConvBlock(features * 2, features * 4)

        # Bottleneck
        self.bottleneck = SpikingConvBlock(features * 4, features * 4)

        # Decoder
        self.dec3 = SpikingConvBlock(features * 8, features * 2)
        self.dec2 = SpikingConvBlock(features * 4, features)
        self.dec1 = SpikingConvBlock(features * 2, features)

        # Output decoding (spike rate to analog)
        self.output_decoder = nn.Conv2d(features, out_channels, 1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    @staticmethod
    def _center_crop_to(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """_center_crop_to.

        Args:
            x (torch.Tensor): Description.
            target_h (int): Description.
            target_w (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        _, _, h, w = x.shape
        if h == target_h and w == target_w:
            return x
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return x[:, :, top : top + target_h, left : left + target_w]

    @classmethod
    def _concat_align(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenate along channels after center-cropping to the common spatial size."""
        ta_h, ta_w = a.shape[-2], a.shape[-1]
        b = cls._center_crop_to(b, ta_h, ta_w)
        a = cls._center_crop_to(a, ta_h, ta_w)
        return torch.cat([a, b], dim=1)

    def _reset_neurons(self):
        """Reset all LIF neurons."""
        for module in self.modules():
            if isinstance(module, LIFNeuron):
                module.membrane_potential = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal dynamics.

        forward method for SpikingUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Encode input to initial spike train
        x_encoded = self.input_encoder(x)

        # Accumulator for output spikes
        output_accumulator = torch.zeros(B, self.out_channels, H, W, device=x.device)

        # Reset all neurons
        self._reset_neurons()

        # Run for multiple time steps
        for t in range(self.time_steps):
            # Encoder path with skip connections
            e1 = self.enc1(x_encoded)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))

            # Bottleneck
            b = self.bottleneck(self.pool(e3))

            # Decoder path with skip connections
            up_b = self.upsample(b)
            d3 = self.dec3(self._concat_align(up_b, e3))
            up_d3 = self.upsample(d3)
            d2 = self.dec2(self._concat_align(up_d3, e2))
            up_d2 = self.upsample(d2)
            d1 = self.dec1(self._concat_align(up_d2, e1))

            # Decode spikes to analog output
            output_t = self.output_decoder(d1)
            output_accumulator += output_t

        # Average over time steps (rate coding)
        output = output_accumulator / self.time_steps
        return torch.tanh(output)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "spiking_unet"

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
