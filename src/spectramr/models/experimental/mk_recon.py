import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model

# --- 1. Global Context Engine: K-Space Mamba ---
# Graceful fallback if mamba_ssm is not installed
try:
    from mamba_ssm import Mamba
except ImportError:

    class Mamba(nn.Module):
        """Mock Mamba for architectural validation without CUDA kernels."""

        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            """__init__.

            Args:
                d_model (Any): Description.
                d_state (Any): Description.
                d_conv (Any): Description.
                expand (Any): Description.
            """
            super().__init__()
            self.in_proj = nn.Linear(d_model, d_model * expand * 2)
            self.out_proj = nn.Linear(d_model * expand, d_model)
            self.act = nn.SiLU()

        def forward(self, x):
            # Simple residual MLP behavior to simulate 'processing'
            # x: [Batch, Seq, Dim]
            """forward.

                Args:
                    x (Any): Description.
                Returns:
                    Any: Description.

            forward method for Mamba.

            Executes PyTorch tensor operations.

            Args:
                x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

            Returns:
                torch.Tensor: Output tensor with shape matching the operation.

            Hardware/Device Context:
                Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
            x_in = x
            x = self.in_proj(x)
            x = self.act(x)
            x, _ = x.chunk(2, dim=-1)  # Split for gate/value
            x = self.out_proj(x)
            return x


class KSpaceMambaBlock(nn.Module):
    """
    Paradigm 2: State Space Model for Global K-Space Trajectories.
    Treats k-space readout as a continuous time-series sequence.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        """__init__.

        Args:
            dim (Any): Description.
            d_state (Any): Description.
            d_conv (Any): Description.
            expand (Any): Description.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, kspace_seq):
        """
        Input: (Batch, Readout_Length, Channels)
        Output: (Batch, Readout_Length, Channels)

        forward method for KSpaceMambaBlock.

        Executes PyTorch tensor operations.

        Args:
            kspace_seq (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = kspace_seq
        x = self.norm(kspace_seq)
        x = self.mamba(x)
        return residual + x


# --- 2. Physics Engine: Kolmogorov-Arnold Networks (KAN) ---


class KANLinear(nn.Module):
    """
    Paradigm 7: KAN Layer.
    Learns activation functions on edges (weights) using B-Splines.
    Superior for approximating physics functions (exponential decay, sinusoids).
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        _grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """__init__.

        Args:
            in_features (Any): Description.
            out_features (Any): Description.
            grid_size (Any): Description.
            spline_order (Any): Description.
            scale_noise (Any): Description.
            scale_base (Any): Description.
            scale_spline (Any): Description.
            _grid_eps (Any): Description.
            grid_range (Any): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # Scaling factors for training stability
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        # B-Spline grid (fixed, but can be made learnable)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(
                        self.grid_size + self.spline_order,
                        self.in_features,
                        self.out_features,
                    )
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_((self.scale_spline * noise).permute(2, 1, 0))

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-Spline bases for input x.
        x: (Batch, In_Features)
        Returns: (Batch, In_Features, Grid_Size + Spline_Order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid  # (In, Grid+Order+1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[
                :, :, :-1
            ] + (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]

        return bases

    def forward(self, x):
        # Flatten spatial dims if input is (B, C, H, W) -> (B, H*W, C) handled outside or reshape here
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KANLinear.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        original_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)

        # 1. Base Linear Path (Silicon)
        base_output = F.linear(F.silu(x), self.base_weight)

        # 2. Spline Path (Physics)
        # Normalize input to grid range [-1, 1] for spline evaluation using tanh-like function
        x_norm = torch.tanh(x)
        bases = self.b_splines(x_norm)  # (Batch, In, Coeffs)

        # Compute Spline activation: sum(basis * weight)
        # weight: (Out, In, Coeffs)
        # Output: (Batch, Out)
        spline_output = torch.einsum("bij,oij->bo", bases, self.spline_weight)

        output = base_output + spline_output

        # Reshape back if needed
        if len(original_shape) > 2:
            output = output.reshape(original_shape[:-1] + (self.out_features,))

        return output


class KANResidualBlock(nn.Module):
    """
    Residual Block replacing standard convolutions/dense layers with KANs.
    Used in the Decoder to approximate non-linear MRI signal decay.
    """

    def __init__(self, channels):
        """__init__.

        Args:
            channels (Any): Description.
        """
        super().__init__()
        self.kan1 = KANLinear(channels, channels)
        self.kan2 = KANLinear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # Expect input (Batch, ...., Channels)
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KANResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x
        x = self.norm(x)
        x = self.kan1(x)
        x = self.act(x)
        x = self.kan2(x)
        return x + residual


# --- 3. Unified Architecture: MK-Recon ---


@register_model(name="mk_recon", training_mode="reconstruction")
class MKRecon(nn.Module):
    """
    Mamba-KAN Hybrid Reconstruction Network.

    1. Encoder: K-Space Mamba (Sequences) -> Learns Global Hologram
    2. Bottleneck: Projection
    3. Decoder: Wav-KAN (Images) -> Learns Physics Decay (T1/T2)
    """

    def __init__(
        self,
        in_channels=2,  # Real/Imag
        out_channels=1,  # Magnitude
        seq_len=256 * 256,  # Total k-space points (flattened)
        embed_dim=64,
        mamba_depth=4,
        kan_depth=4,
        img_size=256,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            seq_len (Any): Description.
            embed_dim (Any): Description.
            mamba_depth (Any): Description.
            kan_depth (Any): Description.
            img_size (Any): Description.
        """
        super().__init__()
        self.img_size = img_size
        self.seq_len = seq_len

        # --- Encoder (K-Space Domain) ---
        # Project raw complex k-space points to embedding
        self.kspace_embed = nn.Linear(in_channels, embed_dim)

        # Stack of Mamba blocks for global sequence modeling
        self.encoder = nn.ModuleList([KSpaceMambaBlock(embed_dim) for _ in range(mamba_depth)])

        # --- Domain Transform ---
        # In a real MRI net, this might include an IFFT layer.
        # Here we simulate the domain shift via projection for end-to-end learning.
        self.domain_norm = nn.LayerNorm(embed_dim)

        # --- Decoder (Image Domain) ---
        # Uses KAN (Kolmogorov-Arnold Network) for physics-informed function approximation
        self.decoder = nn.ModuleList([KANResidualBlock(embed_dim) for _ in range(kan_depth)])

        # Final projection to image space
        self.final_proj = KANLinear(embed_dim, out_channels)

    def forward(self, kspace):
        """
        kspace: (Batch, Readout_Len, 2) -> Raw (Real, Imag) sequence

        forward method for MKRecon.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, C = kspace.shape

        # 1. Encode Global K-Space
        x = self.kspace_embed(kspace)  # (B, L, Embed)

        for layer in self.encoder:
            x = layer(x)

        x = self.domain_norm(x)

        # 2. Decode Physics (Image Domain)
        # Note: In a full reconstruction, we would reshape/IFFT here.
        # Assuming L = H*W for this architecture demonstration.

        for layer in self.decoder:
            x = layer(x)

        # 3. Final Projection
        # Map features to pixel intensity using Spline approximation
        out = self.final_proj(x)

        # Reshape to Image
        out = out.view(B, 1, self.img_size, self.img_size)
        return out
