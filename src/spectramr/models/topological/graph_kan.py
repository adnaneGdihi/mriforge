"""Graph-KAN: Graph Neural Network with KAN Layers.

Innovation XIII (SOTA 2.0): Relational Deep Learning for Non-Cartesian MRI.

Mathematical Foundation:
    - Graph Construction: Nodes v_i (k-space samples), Edges e_ij (k-NN).
    - Message: m_ij = KAN_msg(h_i || h_j || pos_i - pos_j)
    - Update: h_i' = KAN_update(h_i || sum(m_ij))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.registry import register_model
from spectramr.models.spectral.fourier_kan import FourierKANLayer


def knn_graph(x: torch.Tensor, k: int = 5):
    """Compute k-NN indices using Euclidean distance.

    Args:
        x: [B, N, D] Point coordinates.
        k: Number of neighbors.

    Returns:
        indices: [B, N, k]
    """
    # Simple pairwise distance (O(N^2) memory!)
    # Safe for N < 5000 approx on reasonable GPU.
    # For large N, we'd need grid hasing or Faiss, but keeping it pure torch here.

    # dist = (x_i - x_j)^2
    # [B, N, 1, D] - [B, 1, N, D]
    dist_sq = torch.cdist(x, x, p=2)  # [B, N, N]

    # Get topk (smallest)
    # We want to exclude self? cdist returns 0 diagonal.
    # topk returns largest, so we negate or use smallest=True (if avail in new torch)
    # torch .topk(largest=False)

    _, indices = torch.topk(dist_sq, k=k + 1, dim=-1, largest=False)

    # Exclude self (first one is usually self if dist is 0)
    indices = indices[..., 1:]  # [B, N, k]
    return indices


class GraphKANLayer(nn.Module):
    """GNN Layer utilizing KANs for Message Passing."""

    def __init__(self, in_features: int, out_features: int, k: int = 5, grid_size: int = 5):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            k (int): Description.
            grid_size (int): Description.
        """
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features

        # Message Function: takes (h_i, h_j, rel_pos) -> [F_in + F_in + D]
        # Let's assume input has pos separate? Or included?
        # Let's say h includes pos ? No, let's keep pos separate for edges.
        # Msg Input: 2 * in_feat + 3 (rel pos dim)

        self.msg_len = 2 * in_features

        # We'll use a KAN for message computation
        self.kan_msg = FourierKANLayer(self.msg_len, out_features, grid_size=grid_size)

        # Aggregation: Sum

        # Update Function: takes (h_i, msg_aggr) -> out
        self.kan_update = FourierKANLayer(
            in_features + out_features, out_features, grid_size=grid_size
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor = None):
        """
        Args:
            h: Node features [B, N, F]
            pos: Node positions [B, N, 3] (or D)
            edge_index: Optional [B, N, k] neighbor indices. If None, computed.

        forward method for GraphKANLayer.

        Executes PyTorch tensor operations.

        Args:
            h (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            pos (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            edge_index (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, F = h.shape
        D = pos.shape[-1]

        if edge_index is None:
            edge_index = knn_graph(pos, k=self.k)

        # Gather neighbors
        # edge_index: [B, N, k]
        # h: [B, N, F]
        # We need to gather h_j for each i

        # Expand h for gathering: [B, N, F] -> [B, N, k, F]? No.
        # We use torch.gather. Flatten indices?

        flat_indices = edge_index.contiguous().view(B, N * self.k)
        # Expand h to [B, N, F] -> gather needs valid dim indices.
        # We need batch-aware gather or loop over batch.

        # Easy way: simple loop for now (B is usually small 1 or 2)
        # Or advanced indexing.

        h_j_list = []
        # pos_j_list = []
        for b in range(B):
            idx = edge_index[b]  # [N, k]
            # h[b]: [N, F]
            h_j_b = h[b][idx]  # [N, k, F]
            h_j_list.append(h_j_b)

            # p_j_b = pos[b][idx]
            # pos_j_list.append(p_j_b)

        h_j = torch.stack(h_j_list, dim=0)  # [B, N, k, F]
        # pos_j = torch.stack(pos_j_list, dim=0)

        # Prepare Message Inputs
        h_i = h.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, F]

        # relative_pos = pos_j - pos.unsqueeze(2) # [B, N, k, D]

        # Concat for message
        # msg_in = torch.cat([h_i, h_j, relative_pos], dim=-1) # F + F + D

        # Simplifying: Just use features for now to reduce dimensionality for KAN speed
        msg_in = torch.cat([h_i, h_j], dim=-1)  # 2F

        # Compute Messages
        # Flatten [B, N, k, 2F] -> [M, 2F]
        msg_flat = msg_in.view(-1, msg_in.shape[-1])
        m_ij = self.kan_msg(msg_flat)  # [M, Out]
        m_ij = m_ij.view(B, N, self.k, self.out_features)

        # Aggregate
        aggr = m_ij.sum(dim=2)  # [B, N, Out]

        # Update
        update_in = torch.cat([h, aggr], dim=-1)  # [B, N, In+Out]
        update_flat = update_in.view(-1, update_in.shape[-1])
        h_prime = self.kan_update(update_flat)
        h_prime = h_prime.view(B, N, self.out_features)

        # Residual connection if dims match
        if self.in_features == self.out_features:
            h_prime = h_prime + h

        return h_prime


@register_model(name="graph_kan", training_mode="reconstruction")
class GraphKAN(nn.Module):
    """Graph-KAN Model for Non-Cartesian Reconstruction."""

    def __init__(self, in_features=2, hidden_dim=32, num_layers=3):
        """__init__.

        Args:
            in_features (Any): Description.
            hidden_dim (Any): Description.
            num_layers (Any): Description.
        """
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_dim)

        self.layers = nn.ModuleList(
            [GraphKANLayer(hidden_dim, hidden_dim, k=5) for _ in range(num_layers)]
        )

        self.readout = nn.Linear(hidden_dim, 2)  # Real, Imag output

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """
        Args:
           x: [B, N, C_in] Data features (e.g. intensity)
           pos: [B, N, 3] Coordinates

        Returns:
           out: [B, N, 2] Refined k-space or image space?

        forward method for GraphKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            pos (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h, pos)

        out = self.readout(h)
        return out


def create_graph_kan() -> GraphKAN:
    """create_graph_kan.

    Returns:
        GraphKAN: Description.
    """
    return GraphKAN()
