"""TenF-INR: Tensor-Factorized Implicit Neural Representation.

Efficient continuous representation for dynamic (4D) MRI via
low-rank tensor decomposition of spatial and temporal components.

Key Features:
- Separate spatial and temporal basis INRs
- Low-rank CP/Tucker decomposition
- Arbitrary space-time super-resolution
- Memory efficient for long sequences

Reference:
- TenF-INR: Tensor Factorized Implicit Neural Representations for Dynamic MRI
"""

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class FourierFeatures(nn.Module):
    """Fourier feature mapping for overcoming spectral bias.

    γ(v) = [cos(2π B v), sin(2π B v)]

    Args:
        in_dim: Input coordinate dimension
        num_frequencies: Number of frequency bands
        scale: Frequency scale
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int = 128,
        scale: float = 10.0,
    ):
        """__init__.

        Args:
            in_dim (int): Description.
            num_frequencies (int): Description.
            scale (float): Description.
        """
        super().__init__()

        # Random Fourier features (Gaussian)
        B = torch.randn(in_dim, num_frequencies) * scale
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        """out_dim.

        Returns:
            int: Description.
        """
        return self.B.shape[1] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map coordinates to Fourier features.

        Args:
            x: [..., in_dim] coordinates

        Returns:
            [..., out_dim] Fourier features

        forward method for FourierFeatures.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_proj = 2 * torch.pi * (x @ self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class SpatialBasisINR(nn.Module):
    """Spatial basis function network.

    Learns R spatial basis functions u_r(x, y) that capture
    the anatomical structure independent of time.

    Args:
        coord_dim: Spatial coordinate dimension (2 or 3)
        rank: Number of basis functions
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
    """

    def __init__(
        self,
        coord_dim: int = 2,
        rank: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_frequencies: int = 128,
    ):
        """__init__.

        Args:
            coord_dim (int): Description.
            rank (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            num_frequencies (int): Description.
        """
        super().__init__()

        self.rank = rank

        # Fourier encoding for spatial coordinates
        self.fourier = FourierFeatures(coord_dim, num_frequencies)

        # MLP backbone
        layers = []
        in_dim = self.fourier.out_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else rank
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate spatial basis at coordinates.

        Args:
            coords: [N, 2] or [N, 3] spatial coordinates

        Returns:
            [N, R] basis function values

        forward method for SpatialBasisINR.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = self.fourier(coords)
        return self.mlp(features)


class TemporalBasisINR(nn.Module):
    """Temporal basis function network.

    Learns R temporal coefficient functions v_r(t) that capture
    the dynamics (e.g., cardiac cycle, breathing).

    Args:
        rank: Number of basis functions
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        periodic: Whether to use periodic encoding for cardiac/respiratory
    """

    def __init__(
        self,
        rank: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        periodic: bool = True,
        period: float = 1.0,
    ):
        """__init__.

        Args:
            rank (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            periodic (bool): Description.
            period (float): Description.
        """
        super().__init__()

        self.rank = rank
        self.periodic = periodic
        self.period = period

        # Temporal encoding
        if periodic:
            # Periodic encoding for cardiac/respiratory cycles
            self.register_buffer(
                "frequencies",
                torch.arange(1, 17).float(),  # Harmonics
            )
            in_dim = 32 + 1  # sin/cos for 16 harmonics + raw t
        else:
            self.fourier = FourierFeatures(1, 64)
            in_dim = self.fourier.out_dim

        # MLP
        layers = []
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else rank
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.Tanh())  # Smooth temporal evolution
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Encode time coordinate."""
        if self.periodic:
            # Periodic encoding: [t, sin(2πft/T), cos(2πft/T)] for each frequency f
            t_scaled = t * self.frequencies.unsqueeze(0) / self.period
            sins = torch.sin(2 * torch.pi * t_scaled)
            coss = torch.cos(2 * torch.pi * t_scaled)
            return torch.cat([t, sins, coss], dim=-1)
        else:
            return self.fourier(t)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate temporal coefficients at time t.

        Args:
            t: [T, 1] time coordinates

        Returns:
            [T, R] temporal coefficients

        forward method for TemporalBasisINR.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t_encoded = self.encode_time(t)
        return self.mlp(t_encoded)


@register_model(name="tenf_inr", training_mode="reconstruction")
class TensorFactorizedINR(nn.Module, IGenerator):
    """TenF-INR: Tensor-Factorized Implicit Neural Representation.

    Decomposes a 4D dynamic MRI signal as:
        I(x, y, t) ≈ Σ_r u_r(x, y) · v_r(t)

    This low-rank decomposition enables:
    - Efficient representation of long sequences
    - Arbitrary spatiotemporal super-resolution
    - Temporal interpolation beyond acquired frames

    Args:
        spatial_dim: Spatial coordinate dimension (2 for 2D, 3 for 3D)
        rank: Tensor rank (number of basis functions)
        spatial_hidden: Hidden dim for spatial network
        temporal_hidden: Hidden dim for temporal network
        periodic: Use periodic encoding for temporal (cardiac)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dim: int = 2,
        rank: int = 32,
        spatial_hidden: int = 256,
        temporal_hidden: int = 128,
        spatial_layers: int = 4,
        temporal_layers: int = 3,
        periodic: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            spatial_dim (int): Description.
            rank (int): Description.
            spatial_hidden (int): Description.
            temporal_hidden (int): Description.
            spatial_layers (int): Description.
            temporal_layers (int): Description.
            periodic (bool): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.rank = rank

        # Spatial basis network
        self.spatial_net = SpatialBasisINR(
            coord_dim=spatial_dim,
            rank=rank * out_channels,  # Separate basis per channel
            hidden_dim=spatial_hidden,
            num_layers=spatial_layers,
        )

        # Temporal basis network
        self.temporal_net = TemporalBasisINR(
            rank=rank,
            hidden_dim=temporal_hidden,
            num_layers=temporal_layers,
            periodic=periodic,
        )

    def forward(
        self,
        spatial_coords: torch.Tensor,
        temporal_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate signal at spatiotemporal coordinates.

        Args:
            spatial_coords: [N, 2] or [N, 3] spatial coordinates in [-1, 1]
            temporal_coords: [T, 1] temporal coordinates in [0, 1]

        Returns:
            [T, N, C] signal values

        forward method for TensorFactorizedINR.

        Executes PyTorch tensor operations.

        Args:
            spatial_coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            temporal_coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Spatial basis: [N, R*C]
        u = self.spatial_net(spatial_coords)
        u = u.view(-1, self.out_channels, self.rank)  # [N, C, R]

        # Temporal coefficients: [T, R]
        v = self.temporal_net(temporal_coords)

        # Tensor contraction: I(x,y,t) = Σ_r u_r(x,y) · v_r(t)
        # [T, R] x [N, C, R] -> [T, N, C]
        signal = torch.einsum("tr,ncr->tnc", v, u)

        return signal

    def render_volume(
        self,
        H: int,
        W: int,
        T: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Render full 4D volume at specified resolution.

        Args:
            H: Height
            W: Width
            T: Number of temporal frames
            device: Target device

        Returns:
            [T, C, H, W] rendered dynamic volume
        """
        if device is None:
            device = next(self.parameters()).device

        # Create coordinate grids
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        t = torch.linspace(0, 1, T, device=device)

        # Spatial grid
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        spatial_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [H*W, 2]

        # Temporal grid
        temporal_coords = t.unsqueeze(-1)  # [T, 1]

        # Evaluate
        signal = self.forward(spatial_coords, temporal_coords)  # [T, H*W, C]

        # Reshape
        volume = signal.view(T, H, W, self.out_channels)
        volume = volume.permute(0, 3, 1, 2)  # [T, C, H, W]

        return volume

    def temporal_interpolate(
        self,
        t_new: torch.Tensor,
        spatial_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate to new time points (super-resolution in time).

        Args:
            t_new: [T_new, 1] new time coordinates
            spatial_coords: [N, 2] spatial coordinates

        Returns:
            [T_new, N, C] interpolated signal
        """
        return self.forward(spatial_coords, t_new)

    def fit_from_kspace(
        self,
        kspace: torch.Tensor,
        k_coords: torch.Tensor,
        t_indices: torch.Tensor,
        num_iters: int = 1000,
        lr: float = 1e-3,
    ) -> list[float]:
        """Fit INR directly from k-space measurements.

        Args:
            kspace: [N_meas] complex k-space measurements
            k_coords: [N_meas, 2] k-space coordinates
            t_indices: [N_meas] temporal frame indices
            num_iters: Optimization iterations
            lr: Learning rate

        Returns:
            Loss history
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []
        for it in range(num_iters):
            optimizer.zero_grad()

            # Get unique times
            unique_t = t_indices.unique()
            T = len(unique_t)

            # Render at measurement coordinates
            # This requires implementing NUFFT-based forward model
            # For now, placeholder for supervised fitting

            loss = torch.tensor(0.0)  # Placeholder

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        return losses

    def generate(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Generate interface for compatibility.

        Args:
            x: [B, C, H, W] input (ignored for INR, used for shape)

        Returns:
            [B, T, C, H, W] or [B, C, H, W] generated output
        """
        B, C, H, W = x.shape
        T = kwargs.get("num_frames", 20)

        # Render volume
        volume = self.render_volume(H, W, T, x.device)

        # Add batch dimension
        return volume.unsqueeze(0).expand(B, -1, -1, -1, -1)


class CPTensorINR(TensorFactorizedINR):
    """CP (Canonical Polyadic) Decomposition variant of TenF-INR.

    Adds explicit regularization on the tensor rank.
    """

    def __init__(self, *args, l2_reg: float = 0.01, **kwargs):
        """__init__.

        Args:
            l2_reg (float): Description.
        """
        super().__init__(*args, **kwargs)
        self.l2_reg = l2_reg

    def regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization on factor magnitudes."""
        reg = 0.0
        for param in self.parameters():
            reg = reg + (param**2).sum()
        return self.l2_reg * reg


__all__ = [
    "CPTensorINR",
    "FourierFeatures",
    "SpatialBasisINR",
    "TemporalBasisINR",
    "TensorFactorizedINR",
]
