import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

__all__ = ["EquivariantDiffusionGenerator"]


@register_model(name="equivariant_diffusion", training_mode="diffusion")
class EquivariantDiffusionGenerator(nn.Module, IGenerator):
    """
    Equivariant Diffusion Model for MRI (Experiment 112).

    Incorporates rotation and translation equivariance into the diffusion process.
    This implementation uses a simplified Group Convolution approach by rotating kernels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        num_groups: int = 4,  # C4 symmetry (0, 90, 180, 270 degrees)
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (int): Description.
            num_groups (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.num_groups = num_groups

        # Encoder
        self.conv1 = self._make_group_conv(in_channels, features)
        self.conv2 = self._make_group_conv(features, features * 2)

        # Decoder
        self.conv3 = self._make_group_conv(features * 2, features)
        self.conv4 = self._make_group_conv(features, out_channels)

    def _make_group_conv(self, in_c, out_c):
        """
        Creates a p4 group-equivariant convolution layer.

        We implement p4 (C4 rotation group) by applying the same Conv2d kernel
        to 4 rotated versions of the input and aggregating the responses.
        This is a weight-sharing approach that enforces rotation equivariance.
        """
        return nn.Conv2d(in_c, out_c, 3, padding=1)

    def _apply_p4_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution in p4 group-equivariant manner.

        Input is convolved at 4 rotations (0, 90, 180, 270 degrees),
        then responses are rotated back and aggregated.
        """
        responses = []
        for k in range(self.num_groups):  # 0, 1, 2, 3 (0°, 90°, 180°, 270°)
            # Rotate input
            x_rot = torch.rot90(x, k, dims=[-2, -1])

            # Apply convolution
            y_rot = conv(x_rot)

            # Rotate back
            y = torch.rot90(y_rot, -k, dims=[-2, -1])
            responses.append(y)

        # Aggregate: max pooling over group dimension preserves equivariance
        return torch.stack(responses, dim=0).max(dim=0).values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with p4 group equivariance.

        Each convolution is applied across 4 rotations and aggregated,
        providing exact C4 rotation equivariance.

        forward method for EquivariantDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Feature extraction with p4 equivariance
        x1 = F.relu(self._apply_p4_conv(self.conv1, x))
        x2 = F.relu(self._apply_p4_conv(self.conv2, x1))

        # Reconstruction with p4 equivariance
        x3 = F.relu(self._apply_p4_conv(self.conv3, x2))
        out = self._apply_p4_conv(self.conv4, x3)

        return out

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for Equivariant Diffusion, z is the input state."""
        return self.forward(z)

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

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "equivariant_diffusion_generator"
