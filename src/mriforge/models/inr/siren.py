"""SIREN: Sinusoidal Representation Networks with FiLM Conditioning.

Implements periodic activation functions for high-frequency detail preservation
in Implicit Neural Representations (INR).

Based on:
- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SineLayer(nn.Module):
    """Sine activation layer with learnable frequency.

    Implements: sin(omega * (Wx + b))

    The omega parameter controls the frequency of the sine activation,
    allowing the network to learn high-frequency functions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            omega_0: Frequency multiplier (default: 30.0)
            is_first: Whether this is the first layer (affects initialization)
            bias: Whether to include bias
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in_features, 1/in_features]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in_features)/omega, sqrt(6/in_features)/omega]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sine activation.

        forward method for SineLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.omega_0 * self.linear(x))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Applies affine transformation conditioned on style code:
    FiLM(F | z) = gamma(z) * F + beta(z)

    This allows global conditioning (e.g., MRI contrast) to modulate
    local features without changing the network architecture.
    """

    def __init__(
        self,
        feature_dim: int,
        style_dim: int,
    ):
        """
        Args:
            feature_dim: Dimension of features to modulate
            style_dim: Dimension of conditioning vector
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.style_dim = style_dim

        # Learnable mappings from style to scale/shift
        self.gamma_proj = nn.Linear(style_dim, feature_dim)
        self.beta_proj = nn.Linear(style_dim, feature_dim)

        # Identity-at-init: gamma = gamma_proj(style) must equal 1 (passthrough) and
        # beta = 0 for ANY style, so FiLM starts as the identity and does not perturb
        # the SIREN's variance-preserving init. That requires W=0, b=1 on gamma_proj
        # (and W=0, b=0 on beta_proj). The previous line wrote ones into a TEMPORARY
        # (`weight.data.mean(...)`), leaving gamma_proj.weight at random Kaiming init —
        # so FiLM was NOT identity (gamma ~ random, scaling features by ~0.08) and the
        # conditioning was always-on from step 0.
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, features: Tensor, style: Tensor) -> Tensor:
        """
        Args:
            features: [B, ..., feature_dim] features to modulate
            style: [B, style_dim] conditioning vector

        Returns:
            Modulated features [B, ..., feature_dim]

        forward method for FiLMLayer.

        Executes PyTorch tensor operations.

        Args:
            features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        gamma = self.gamma_proj(style)  # [B, feature_dim]
        beta = self.beta_proj(style)  # [B, feature_dim]

        # Reshape for broadcasting
        # features: [B, N, feature_dim] or [B, feature_dim]
        while gamma.dim() < features.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * features + beta


class SIRENWithFiLM(nn.Module):
    """SIREN network with FiLM conditioning for style modulation.

    Combines:
    1. Sine activations for high-frequency detail (spatial/anatomical)
    2. FiLM layers for global conditioning (physics/contrast)

    This is the INR decoder for the GLVS framework.
    """

    def __init__(
        self,
        in_features: int = 3,  # Coordinates (x, y, z)
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 1,  # Intensity
        style_dim: int = 64,  # Latent physics/modality code
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        use_film: bool = True,
        film_every_n: int = 1,  # Apply FiLM every N layers
    ):
        """
        Args:
            in_features: Input coordinate dimension
            hidden_features: Hidden layer width
            hidden_layers: Number of hidden layers
            out_features: Output dimension
            style_dim: Dimension of style conditioning vector
            omega_0: Frequency for first layer
            omega_hidden: Frequency for hidden layers
            use_film: Whether to use FiLM conditioning
            film_every_n: Apply FiLM every N hidden layers
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.style_dim = style_dim
        self.use_film = use_film
        self.film_every_n = film_every_n

        # Build network
        layers = []
        film_layers = nn.ModuleList()

        # First layer
        layers.append(SineLayer(in_features, hidden_features, omega_0=omega_0, is_first=True))

        # Hidden layers with optional FiLM
        for i in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    omega_0=omega_hidden,
                    is_first=False,
                )
            )

            # Add FiLM layer every N layers
            if use_film and (i + 1) % film_every_n == 0:
                film_layers.append(FiLMLayer(hidden_features, style_dim))
            elif use_film:
                film_layers.append(None)  # Placeholder

        self.layers = nn.ModuleList(layers)
        self.film_layers = film_layers

        # Final linear layer (no sine activation)
        self.final_layer = nn.Linear(hidden_features, out_features)
        self._init_final_layer()

    def _init_final_layer(self) -> None:
        """Initialize final layer weights."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.hidden_features) / 30.0
            self.final_layer.weight.uniform_(-bound, bound)
            if self.final_layer.bias is not None:
                self.final_layer.bias.uniform_(-bound, bound)

    def forward(
        self,
        coords: Tensor,
        style: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            coords: [B, N, in_features] spatial coordinates
            style: [B, style_dim] optional style conditioning

        Returns:
            values: [B, N, out_features] predicted values

        forward method for SIRENWithFiLM.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.layers[0](coords)  # First sine layer

        film_idx = 0
        for i, layer in enumerate(self.layers[1:]):
            x = layer(x)

            # Apply FiLM if available
            if self.use_film and film_idx < len(self.film_layers):
                film = self.film_layers[film_idx]
                if film is not None and style is not None:
                    x = film(x, style)
                film_idx += 1

        return self.final_layer(x)

    def forward_with_intermediate(
        self,
        coords: Tensor,
        style: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward with intermediate activations for visualization."""
        intermediates = []

        x = self.layers[0](coords)
        intermediates.append(x)

        film_idx = 0
        for i, layer in enumerate(self.layers[1:]):
            x = layer(x)

            if self.use_film and film_idx < len(self.film_layers):
                film = self.film_layers[film_idx]
                if film is not None and style is not None:
                    x = film(x, style)
                film_idx += 1

            intermediates.append(x)

        output = self.final_layer(x)
        return output, intermediates


from mriforge.models.blocks.embeddings import (
    CoordinatePositionalEncoding as PositionalEncoding,
)


class GLVS_INR_Decoder(nn.Module):
    """Complete INR decoder for Generic Latent Volume Synthesis.

    Combines:
    1. Positional encoding (optional) or Hash Grid
    2. SIREN with FiLM conditioning
    3. Local anatomy tensor sampling

    This is the "Phi" function in the GLVS framework:
    I(x) = Phi(gamma(x); sigma(s, x), z_style)
    """

    def __init__(
        self,
        coord_dim: int = 3,
        anatomy_dim: int = 32,  # K tissue classes
        style_dim: int = 64,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 1,
        use_positional_encoding: bool = True,
        num_frequencies: int = 10,
        omega_0: float = 30.0,
    ):
        """__init__.

        Args:
            coord_dim (int): Description.
            anatomy_dim (int): Description.
            style_dim (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
            out_features (int): Description.
            use_positional_encoding (bool): Description.
            num_frequencies (int): Description.
            omega_0 (float): Description.
        """
        super().__init__()

        # Positional encoding
        self.use_pe = use_positional_encoding
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(
                in_features=coord_dim,
                num_frequencies=num_frequencies,
            )
            coord_input_dim = self.pos_enc.out_features
        else:
            self.pos_enc = None
            coord_input_dim = coord_dim

        # Total input: encoded coords + local anatomy features
        in_features = coord_input_dim + anatomy_dim

        # SIREN with FiLM
        self.siren = SIRENWithFiLM(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            style_dim=style_dim,
            omega_0=omega_0,
            use_film=True,
        )

    def forward(
        self,
        coords: Tensor,  # [B, N, 3] spatial coordinates
        anatomy: Tensor,  # [B, N, K] local anatomy features (sampled from s)
        style: Tensor,  # [B, style_dim] global physics conditioning
    ) -> Tensor:
        """
        Args:
            coords: [B, N, 3] query coordinates
            anatomy: [B, N, K] anatomy features at those coordinates
            style: [B, style_dim] style/modality vector

        Returns:
            intensity: [B, N, 1] predicted intensity values

        forward method for GLVS_INR_Decoder.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            anatomy (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Positional encoding
        if self.use_pe:
            coords_enc = self.pos_enc(coords)
        else:
            coords_enc = coords

        # Concatenate coords and anatomy
        x = torch.cat([coords_enc, anatomy], dim=-1)

        # SIREN with FiLM conditioning
        return self.siren(x, style)
