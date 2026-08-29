"""Hypernetwork and HyperSIREN for Implicit Neural Representations.

This module implements:
1. HyperSIREN: A SIREN model where weights/biases are inputs (functional).
2. Hypernetwork: Predicts parameters for HyperSIREN from latent codes.
3. ShiftModulation: Implements CoNeS-style shift modulation (s_i).
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from mriforge.models.registry import register_model


class Sine(nn.Module):
    """Sine activation function with scaling factor."""

    def __init__(self, omega_0: float = 30.0):
        """__init__.

        Args:
            omega_0 (float): Description.
        """
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        Args:
            x (Tensor): Description.
        Returns:
            Tensor: Description.

        forward method for Sine.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.omega_0 * x)


class HyperLinear(nn.Module):
    """Linear layer where weights and biases are provided as input."""

    def __init__(self, in_features: int, out_features: int):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(
        self,
        input: Float[Tensor, "B N In"],
        weight: Float[Tensor, "B Out In"],
        bias: Float[Tensor, "B Out"] | None = None,
    ) -> Float[Tensor, "B N Out"]:
        """Apply linear transform using batch-specific weights.

        Args:
            input: Input features [Batch, N_points, In_features]
            weight: Weights [Batch, Out_features, In_features]
            bias: Biases [Batch, Out_features] (Optional)

        forward method for HyperLinear.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            weight (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            bias (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B N Out']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # x: [B, N, I]
        # w^T: [B, I, O]
        # x @ w^T + b

        out = torch.bmm(input, weight.permute(0, 2, 1))

        if bias is not None:
            out = out + bias.unsqueeze(1)

        return out


class HyperSIREN(nn.Module):
    """SIREN model with external weights (Hypernetwork-compatible)."""

    def __init__(
        self,
        in_features: int = 3,  # (x, y, z)
        hidden_features: int = 64,
        hidden_layers: int = 3,
        out_features: int = 1,  # (intensity)
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        outermost_linear: bool = True,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
            out_features (int): Description.
            first_omega_0 (float): Description.
            hidden_omega_0 (float): Description.
            outermost_linear (bool): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.outermost_linear = outermost_linear

        self.net = nn.ModuleList()
        # No internal weights, just structure definition
        self.layers_meta = []

        # Input layer
        self.layers_meta.append((in_features, hidden_features))
        self.net.append(HyperLinear(in_features, hidden_features))
        self.net.append(Sine(first_omega_0))

        # Hidden layers
        for _ in range(hidden_layers):
            self.layers_meta.append((hidden_features, hidden_features))
            self.net.append(HyperLinear(hidden_features, hidden_features))
            self.net.append(Sine(hidden_omega_0))

        # Output layer
        self.layers_meta.append((hidden_features, out_features))
        self.net.append(HyperLinear(hidden_features, out_features))
        if not outermost_linear:
            self.net.append(Sine(hidden_omega_0))

    def get_param_shapes(self) -> list[tuple[tuple[int, int], tuple[int]]]:
        """Return shapes of weights and biases for each layer."""
        shapes = []
        for in_f, out_f in self.layers_meta:
            shapes.append(((out_f, in_f), (out_f,)))
        return shapes

    def forward(
        self,
        coords: Float[Tensor, "B N 3"],
        params: dict[str, Tensor],
    ) -> Float[Tensor, "B N 1"]:
        """Forward pass with provided parameters map.

        Args:
            coords: Spatial coordinates.
            params: Dictionary containing 'w0', 'b0', 'w1', 'b1', etc.

        forward method for HyperSIREN.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            params (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B N 1']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = coords

        layer_idx = 0
        for module in self.net:
            if isinstance(module, HyperLinear):
                w = params[f"w{layer_idx}"]
                b = params[f"b{layer_idx}"]
                x = module(x, w, b)
                layer_idx += 1
            elif isinstance(module, Sine):
                x = module(x)

        return x


class ShiftModulation(nn.Module):
    """Predicts shift parameters for CoNeS modulation."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int):
        """__init__.

        Args:
            latent_dim (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
        """
        super().__init__()
        # Predicts s_i for each layer
        # Output dim = num_layers * hidden_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_layers * hidden_dim),
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, z: Tensor) -> Float[Tensor, "B Layers Hidden"]:
        """forward.

        Args:
            z (Tensor): Description.
        Returns:
            Float[Tensor, 'B Layers Hidden']: Description.

        forward method for ShiftModulation.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B Layers Hidden']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        shifts = self.net(z)
        return shifts.reshape(-1, self.num_layers, self.hidden_dim)


@register_model(name="inr_hypernetwork", training_mode="reconstruction")
class Hypernetwork(nn.Module):
    """Generates weights for HyperSIREN from latent codes."""

    def __init__(
        self,
        z_geo_dim: int,
        z_phy_dim: int,
        hyper_siren: HyperSIREN,
    ):
        """__init__.

        Args:
            z_geo_dim (int): Description.
            z_phy_dim (int): Description.
            hyper_siren (HyperSIREN): Description.
        """
        super().__init__()
        self.z_dim = z_geo_dim + z_phy_dim
        self.param_shapes = hyper_siren.get_param_shapes()

        # Calculate total number of parameters needed
        self.total_params = 0
        self.slices = {}

        start = 0
        for idx, ((out_f, in_f), (bias_f,)) in enumerate(self.param_shapes):
            w_size = out_f * in_f
            b_size = bias_f

            self.slices[f"w{idx}"] = (start, start + w_size, (out_f, in_f))
            start += w_size

            self.slices[f"b{idx}"] = (start, start + b_size, (bias_f,))
            start += b_size

        self.total_params = start

        # 3-layer MLP as hypernetwork
        self.generator = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.total_params),
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # SIREN weights require specific init range
        # Here we init the last layer small to start with "standard" init
        """_init_weights.

        Returns:
            Any: Description.
        """
        with torch.no_grad():
            self.generator[-1].weight.normal_(0.0, 0.001)
            self.generator[-1].bias.zero_()

    def forward(
        self, z_geo: Float[Tensor, "B Dg"], z_phy: Float[Tensor, "B Dp"]
    ) -> dict[str, Tensor]:
        """Generate parameters for SIREN.

        Args:
            z_geo: Anatomy latent code.
            z_phy: Physics latent code.

        Returns:
            Dictionary of weights and biases.

        forward method for Hypernetwork.

        Executes PyTorch tensor operations.

        Args:
            z_geo (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            z_phy (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        z = torch.cat([z_geo, z_phy], dim=1)
        flat_params = self.generator(z)

        params = {}
        batch_size = z.shape[0]

        for name, (start, end, shape) in self.slices.items():
            param = flat_params[:, start:end]
            params[name] = param.reshape(batch_size, *shape)

        return params
