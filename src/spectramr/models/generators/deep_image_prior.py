import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import StandardUNetGenerator
from spectramr.models.registry import register_model

__all__ = ["DeepImagePriorGenerator"]


@register_model(name="deep_image_prior", training_mode="reconstruction")
class DeepImagePriorGenerator(nn.Module, IGenerator):
    """
    Deep Image Prior (DIP) Generator (Experiment 107).

    In DIP, the network weights *are* the representation.
    We optimize the weights to map a fixed random input z to the observed image y.

    This class is essentially a wrapper that holds the fixed input z
    and provides the standard U-Net architecture.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        input_noise_shape: tuple = (1, 32, 256, 256),  # Fixed noise input shape
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            input_noise_shape (tuple): Description.
        """
        super().__init__()
        # DIP usually uses a U-Net with skip connections
        # The input channels to the U-Net match the noise shape channels
        self.generator = StandardUNetGenerator(
            in_channels=input_noise_shape[1], out_channels=out_channels
        )

        # Fixed input noise z
        self.register_buffer("net_input", torch.randn(input_noise_shape) * 0.1)

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Ignored (usually). In DIP, we optimize weights, so input is fixed.
               However, to be compatible with pipelines that pass 'input',
               we can optionally ignore it or use it if we are doing something hybrid.
               For pure DIP, we ignore x and use self.net_input.

        forward method for DeepImagePriorGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # We expand net_input to match batch size of x if x is provided
        if x is not None:
            batch_size = x.shape[0]
            net_input = self.net_input.expand(batch_size, -1, -1, -1)
        else:
            net_input = self.net_input

        return self.generator(net_input)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return self.generator.get_parameter_count()

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (
            input_shape[0],
            self.generator.out_channels,
            input_shape[2],
            input_shape[3],
        )

    def generate(self, x: torch.Tensor = None) -> torch.Tensor:
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
        return "deep_image_prior"
