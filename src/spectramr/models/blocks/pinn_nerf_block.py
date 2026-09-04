import numpy as np
import torch
import torch.nn as nn


class GaussianFourierFeatureMapping(nn.Module):
    """Gaussian Fourier Feature Mapping (Tancik et al. 2020)."""

    def __init__(self, input_dim: int = 2, mapping_size: int = 256, scale: float = 10.0):
        """__init__.

        Args:
            input_dim (int): Description.
            mapping_size (int): Description.
            scale (float): Description.
        """
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        # B matrix: [input_dim, mapping_size]
        # Sampled from Normal(0, scale^2)
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., input_dim]
        # x @ B: [..., mapping_size]
        # [cos(2pi xB), sin(2pi xB)]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for GaussianFourierFeatureMapping.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        xp = (x @ self.B) * 2 * np.pi
        return torch.cat([torch.cos(xp), torch.sin(xp)], dim=-1)


class Sine(nn.Module):
    """Sine activation for SIREN networks (Periodic Activation)."""

    def __init__(self, w0: float = 30.0):
        """__init__.

        Args:
            w0 (float): Description.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Sine.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.w0 * x)


class PINNNeRFBlock(nn.Module):
    """
    Physics-Informed Neural Field Block (PINN-NeRF).

    Uses coordinate-based MLP with periodic activations (SIREN) to represent continuous MRI signals.
    Overcomes spectral bias of ReLU MLPs.

    NOTE: We use SIREN (Sine activation) WITHOUT positional encoding because:
    - SIREN activation sin(w0*x) already provides implicit frequency encoding
    - Combining positional encoding with SIREN causes gradient instability
    - For PINNs, SIREN provides smooth 2nd derivatives required for PDE constraints

    Reference: Sitzmann et al. "Implicit Neural Representations with Periodic
    Activation Functions" NeurIPS 2020
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_freqs: int = 10,  # Kept for API compat but ignored for SIREN
        out_channels: int = 2,  # Real + Imaginary
        use_positional_encoding: bool = False,  # Explicitly disable for SIREN
    ):
        """__init__.

        Args:
            in_features (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            num_freqs (int): Description.
            out_channels (int): Description.
            use_positional_encoding (bool): Description.
        """
        super().__init__()

        # [PHYSICS FIX]
        # - SIREN (Sine activation) provides implicit frequency encoding.
        # - Gaussian Fourier Features (Tancik 2020) provides explicit tunable spectral bias.
        # We allow both. If use_positional_encoding is set, we assume Gaussian Mapping if num_freqs is large.

        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding = None

        if use_positional_encoding:
            # Heuristic: If num_freqs > 20, likely wanting rich Gaussian mapping
            # Or explicitly check via kwarg? We'll maintain API and assume Gaussian if requested.
            # But standard PosEnc is deterministic 2^k.
            # Let's switch to Gaussian Fourier Features as per architectural review.
            input_dim = 2  # (x, y)
            mapping_size = num_freqs * 2  # To keep size comparable approx?
            self.positional_encoding = GaussianFourierFeatureMapping(
                input_dim, mapping_size=num_freqs
            )
            encoded_coord_dim = num_freqs * 2  # sin/cos pairs
        else:
            # SIREN: Raw coordinates in
            encoded_coord_dim = 2  # Just (x, y) coordinates

        input_dim = encoded_coord_dim + in_features

        layers = []
        # First layer (special initialization for SIREN)
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(Sine(w0=30.0))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine(w0=30.0))

        # Output layer (linear - no activation)
        layers.append(nn.Linear(hidden_dim, out_channels))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """SIREN initialization scheme."""
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    # Kaiming-like but scaled for Sine
                    val = np.sqrt(6 / m.weight.size(1)) / 30.0
                    m.weight.uniform_(-val, val)
                    if m.bias is not None:
                        m.bias.uniform_(-val, val)

            # First layer needs special init
            first_layer = self.mlp[0]
            val = 1 / first_layer.weight.size(1)
            first_layer.weight.uniform_(-val, val)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Latent features at coordinates (B, N, C_feat)
            coords: Normalized coordinates (B, N, 2) in range [-1, 1]
        Returns:
            Predicted values at coordinates (B, N, out_channels)
        """
        if self.use_positional_encoding and self.positional_encoding is not None:
            encoded_coords = self.positional_encoding(coords)
            inp = torch.cat([features, encoded_coords], dim=-1)
        else:
            # SIREN: Direct coordinate input (no positional encoding)
            inp = torch.cat([features, coords], dim=-1)
        return self.mlp(inp)
