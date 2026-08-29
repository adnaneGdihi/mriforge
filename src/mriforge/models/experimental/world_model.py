import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model


class FourierNeuralOperator2D(nn.Module):
    """
    Paradigm 1 & 4: Resolution-Agnostic Generator.
    Learns the continuous function of the MRI signal in frequency domain.
    """

    def __init__(self, in_channels, out_channels, modes1=16, modes2=16, width=64):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            modes1 (Any): Description.
            modes2 (Any): Description.
            width (Any): Description.
        """
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: [Batch, C, H, W]
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for FourierNeuralOperator2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, Width, H, W]

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)


class SpectralConv2d(nn.Module):
    """SpectralConv2d class."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            modes1 (Any): Description.
            modes2 (Any): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """compl_mul2d.

        Args:
            input (Any): Description.
            weights (Any): Description.
        Returns:
            Any: Description.
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SpectralConv2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batchsize = x.shape[0]
        # Compute Fourier coeff
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


@register_model(name="generative_world_model", training_mode="experimental")
class GenerativeWorldModel(nn.Module):
    """
    Paradigm 1: Generative World Model using Swin-Transformer Encoder and FNO Decoder.
    Simulates the anatomical reality from sparse observations.
    """

    def __init__(self, img_size=256):
        """__init__.

        Args:
            img_size (Any): Description.
        """
        super().__init__()
        # Mock Swin Transformer Encoder (Replace with actual timm implementation in prod)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 64
        )

        # FNO Generator (Resolution Agnostic)
        self.generator = FourierNeuralOperator2D(256, 1, width=64)

    def forward(self, x):
        # x: [Batch, 1, H, W] ULF Image
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for GenerativeWorldModel.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        latent = self.encoder(x)
        # Latent: [Batch, 256, H/4, W/4]

        # FNO takes the latent physics representation and "hallucinates" the continuous signal
        # We might need to upsample latent to query arbitrary points
        out = self.generator(latent)

        # Ideally FNO outputs at any resolution, here we assume it matches training grid
        # Upsample to original size if FNO operates in latent space
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear")
        return out
