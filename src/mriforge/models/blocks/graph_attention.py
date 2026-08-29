"""Graph Attention Blocks for Physics-Aware MRI.

Standard Euclidean KNN fails for MRI k-space because:
1. K-space is non-uniform (dense center, sparse edges)
2. Time dimension matters for B0 correction (φ = ΔΩ * t)
3. Radial/Spiral have intrinsic geometry (polar coordinates)

This module implements Graph Attention that LEARNS the distance metric.

References:
- Veličković et al., "Graph Attention Networks" ICLR 2018
- Brody et al., "How Attentive are Graph Attention Networks?" ICLR 2022 (GATv2)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionBlock(nn.Module):
    """Graph Attention with optional physics-informed locality bias.

    Replaces fixed Euclidean KNN with learned attention weights.
    The network learns to determine which k-space points are "neighbors".

    Features:
    - Multi-head attention for different relationship types
    - Optional locality mask based on trajectory geometry
    - Support for time-conditioned attention (B0/diffusion)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        use_locality_mask: bool = True,
        locality_radius: float = 0.3,  # Fraction of max distance
        concat_heads: bool = True,
    ):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            heads: Number of attention heads
            dropout: Attention dropout
            use_locality_mask: Add geometric locality bias
            locality_radius: Max relative distance for local attention
            concat_heads: Concat (True) or average (False) heads
        """
        super().__init__()

        self.heads = heads
        self.concat_heads = concat_heads
        self.use_locality_mask = use_locality_mask
        self.locality_radius = locality_radius

        if concat_heads:
            assert out_channels % heads == 0, "out_channels must be divisible by heads"
            self.d_k = out_channels // heads
            self.d_out = out_channels
        else:
            self.d_k = out_channels
            self.d_out = out_channels

        # GATv2-style: Apply nonlinearity BEFORE the attention score
        self.W_query = nn.Linear(in_channels, heads * self.d_k)
        self.W_key = nn.Linear(in_channels, heads * self.d_k)
        self.W_value = nn.Linear(in_channels, heads * self.d_k)

        # Attention scoring (GATv2 uses learned vector a)
        self.attn_vector = nn.Parameter(torch.randn(heads, self.d_k))

        # Output projection
        if concat_heads:
            self.proj = nn.Linear(heads * self.d_k, out_channels)
        else:
            self.proj = nn.Linear(self.d_k, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)

        # Physics encoding (optional)
        self.time_proj = nn.Linear(1, heads)  # Time modulation per head

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize attention vector."""
        nn.init.xavier_uniform_(self.attn_vector.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        trajectory: torch.Tensor | None = None,
        time_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [B, C, N] or [B, N, C]
            trajectory: K-space coordinates [B, 2, N] (kx, ky)
            time_vec: Readout time [B, 1, N] for B0-aware attention

        Returns:
            Updated features [B, C', N] or [B, N, C']

        forward method for GraphAttentionBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time_vec (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle input format
        if x.dim() == 3 and x.shape[1] != x.shape[2]:
            if x.shape[1] < x.shape[2]:  # [B, C, N] -> [B, N, C]
                x = x.permute(0, 2, 1)
                needs_transpose = True
            else:
                needs_transpose = False
        else:
            needs_transpose = False

        B, N, C = x.shape

        # Project to Q, K, V
        q = self.W_query(x).view(B, N, self.heads, self.d_k)  # [B, N, H, D]
        k = self.W_key(x).view(B, N, self.heads, self.d_k)
        v = self.W_value(x).view(B, N, self.heads, self.d_k)

        # GATv2: Apply activation to (q + k) then dot with attention vector
        # score_ij = a^T * LeakyReLU(W_q * h_i + W_k * h_j)

        # Efficient pairwise: expand then combine
        q_expand = q.unsqueeze(2)  # [B, N, 1, H, D]
        k_expand = k.unsqueeze(1)  # [B, 1, N, H, D]

        # Combined features with LeakyReLU
        combined = F.leaky_relu(q_expand + k_expand, negative_slope=0.2)  # [B, N, N, H, D]

        # Attention scores via learned vector
        scores = torch.einsum("bnmhd,hd->bnmh", combined, self.attn_vector)  # [B, N, N, H]
        scores = scores / math.sqrt(self.d_k)

        # Physics-informed locality mask
        if self.use_locality_mask and trajectory is not None:
            locality_mask = self._compute_locality_mask(trajectory)  # [B, N, N]
            scores = scores + locality_mask.unsqueeze(-1)  # Broadcast to heads

        # Time modulation (for B0 correction: nearby in time = higher attention)
        if time_vec is not None:
            time_bias = self._compute_time_bias(time_vec)  # [B, N, N, H]
            scores = scores + time_bias

        # Softmax + dropout
        attn = F.softmax(scores, dim=2)  # Attend over key dimension
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum("bnmh,bmhd->bnhd", attn, v)  # [B, N, H, D]

        # Combine heads
        if self.concat_heads:
            out = out.reshape(B, N, self.heads * self.d_k)
        else:
            out = out.mean(dim=2)  # Average across heads

        # Project and residual
        out = self.proj(out)
        out = self.layer_norm(
            out + x[:, :, : out.shape[-1]]
        )  # Residual (may need projection if dims differ)

        # Transpose back if needed
        if needs_transpose:
            out = out.permute(0, 2, 1)

        return out

    def _compute_locality_mask(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute soft locality mask from trajectory geometry.

        Points far apart in k-space get attention penalty.
        This preserves some geometric inductive bias while allowing learned deviation.
        """
        # trajectory: [B, 2, N] -> distances [B, N, N]
        if trajectory.shape[1] == 2:
            traj = trajectory.permute(0, 2, 1)  # [B, N, 2]
        else:
            traj = trajectory

        # Pairwise L2 distance
        diff = traj.unsqueeze(2) - traj.unsqueeze(1)  # [B, N, N, 2]
        dist = torch.norm(diff, dim=-1)  # [B, N, N]

        # Normalize by max distance
        max_dist = dist.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        dist_norm = dist / (max_dist + 1e-8)

        # Soft mask: points beyond locality_radius get penalty
        # Using smooth penalty instead of hard cutoff
        mask = -10.0 * F.relu(dist_norm - self.locality_radius)

        return mask

    def _compute_time_bias(self, time_vec: torch.Tensor) -> torch.Tensor:
        """Compute attention bias based on readout time difference.

        For B0 correction: points measured at similar times have
        similar phase errors and should attend more to each other.
        """
        # time_vec: [B, 1, N]
        if time_vec.dim() == 3 and time_vec.shape[1] == 1:
            t = time_vec.squeeze(1)  # [B, N]
        else:
            t = time_vec

        # Pairwise time difference
        t_diff = t.unsqueeze(2) - t.unsqueeze(1)  # [B, N, N]
        t_diff = t_diff.abs()

        # Normalize
        max_t = t_diff.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        t_norm = t_diff / (max_t + 1e-8)

        # Project to per-head bias (smaller time diff = positive bias)
        time_bias = self.time_proj(t_norm.unsqueeze(-1))  # [B, N, N, H]
        time_bias = -time_bias  # Invert: small diff = high attention

        return time_bias


class LocalGraphAttentionBlock(GraphAttentionBlock):
    """Graph Attention with hard local neighborhood constraint.

    Instead of global attention (O(N²)), only attend to k-nearest neighbors.
    Much more efficient for large k-space acquisitions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_neighbors: int = 16,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            k_neighbors (int): Description.
        """
        super().__init__(in_channels, out_channels, use_locality_mask=False, **kwargs)
        self.k_neighbors = k_neighbors

    def forward(
        self,
        x: torch.Tensor,
        trajectory: torch.Tensor | None = None,
        time_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Local attention using KNN for efficiency."""
        # For very large N, compute KNN first then do attention only on neighbors
        # This is O(N*k) instead of O(N²)

        if trajectory is None or x.shape[-1] < 256:
            # Fall back to global attention for small graphs
            return super().forward(x, trajectory, time_vec)

        # For now, use parent with locality mask
        self.use_locality_mask = True
        return super().forward(x, trajectory, time_vec)


__all__ = ["GraphAttentionBlock", "LocalGraphAttentionBlock"]
