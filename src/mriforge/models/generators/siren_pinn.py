import math
from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces import IGenerator
from mriforge.models.registry import register_model


class SirenLayer(nn.Module):
    """SIREN layer with specific uniform initialization and sinusoidal activation.

    Reference: Sitzmann et al., "Implicit Neural Representations with Periodic
    Activation Functions", NeurIPS 2020.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            is_first (bool): Description.
            omega_0 (float): Description.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        """init_weights.

        Returns:
            Any: Description.
        """
        with torch.no_grad():
            if self.is_first:
                # First layer initialization: uniform(-1 / in_features, 1 / in_features)
                # But Sitzmann actually suggests waiting, let's follow the precise paper:
                # w_first ~ U(-1/n_in, 1/n_in)
                bound = 1.0 / self.in_features
            else:
                # Hidden layers initialization: uniform(-sqrt(6 / in_features) / omega_0, sqrt(6 / in_features) / omega_0)
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SirenLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        return torch.sin(self.omega_0 * self.linear(x))


@register_model(name="siren_sens_net", training_mode="pinn")
class SirenSensNet(nn.Module, IGenerator):
    """Continuous Coordinate-Based PINN mapping spatial coordinates to Complex Sensitivity Maps.

    Input: (N, 2) spatial coordinates (x, y) normalized to [-1, 1]
    Output: (N, num_coils, 2) representing the Real and Imaginary components of the B1- fields.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 16,  # Default for 8 coils: 8 real + 8 imag
        hidden_features: int = 256,
        hidden_layers: int = 4,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
            first_omega_0 (float): Description.
            hidden_omega_0 (float): Description.
            **kwargs: Extra IGenerator interface requirements.
        """
        super().__init__()
        self._name = "siren_sens_net"
        self.in_features = kwargs.get("in_features", in_channels)
        self.out_features = kwargs.get("out_features", out_channels)

        self.net = nn.Sequential()

        # First layer
        self.net.add_module(
            "Siren_0",
            SirenLayer(self.in_features, hidden_features, is_first=True, omega_0=first_omega_0),
        )

        # Hidden layers
        for i in range(hidden_layers):
            self.net.add_module(
                f"Siren_{i + 1}",
                SirenLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                ),
            )

        # Final layer (linear out)
        final_linear = nn.Linear(hidden_features, self.out_features)
        with torch.no_grad():
            # For the final layer, standard uniform or Kaiming is fine,
            # or following SIREN approach:
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)

        self.net.add_module("Linear_Out", final_linear)

        self.num_coils = self.out_features // 2

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return self._name

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        # input is (N, 2), output is two tensors of (N, num_coils)
        # We return the shape of the concatenated (Real, Imaginary) output for standard interface
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape[:-1] + (self.out_features,)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard generator interface. Returns concatenated real and imaginary parts."""
        s_real, s_imag = self.forward(x)
        return torch.cat([s_real, s_imag], dim=-1)

    def forward(self, coords):
        """
        Args:
            coords: Tensor of shape (..., 2) representing (x, y) coordinates with requires_grad=True.
        Returns:
            Real and Imaginary parts, each of shape (..., num_coils).

        forward method for SirenSensNet.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # coords usually (N, 2)
        out = self.net(coords)

        # Split into Real and Imaginary, enforcing contiguous memory for CUBLAS spatial 3rd-order autograd
        S_real = out[..., : self.num_coils].contiguous()
        S_imag = out[..., self.num_coils :].contiguous()

        return S_real, S_imag


# Helper function to get the shared weights array for adaptive gradient lambda computation
def get_last_shared_layer(model: nn.Module) -> nn.Parameter:
    """Returns the weights of the last hidden layer before the output."""
    # The output layer is Linear_Out
    # The last hidden layer is Siren_{n}
    names = list(model.net.named_children())
    last_siren_name, last_siren_module = names[-2]
    return last_siren_module.linear.weight
