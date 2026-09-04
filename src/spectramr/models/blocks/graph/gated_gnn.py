"""Gated graph convolution block.

Implements the gated graph sequence neural network of Li *et al.* (ICLR
2016, "Gated Graph Sequence Neural Networks"), specialised to k-space
graphs. Each vertex ``v`` carries a hidden state ``h_v^{(t)} in R^D``
updated for ``T`` steps by::

    m_v^{(t)} = sum_{u in N(v)} W_{e_uv} h_u^{(t)}
    h_v^{(t+1)} = GRU(h_v^{(t)}, m_v^{(t)})

with ``W_{e_uv} in R^{D x D}`` an edge-type-specific weight matrix and a
standard GRU cell as the recurrent update.

The scatter-add aggregation is implemented with ``index_add_`` so the
block carries no ``torch_scatter`` dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GatedGraphConv", "scatter_add"]


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum rows of ``src`` into ``dim_size`` buckets indexed by ``index``.

    Dependency-free replacement for ``torch_scatter.scatter_add`` along
    dim 0. ``src`` has shape ``[E, D]`` and ``index`` shape ``[E]``; the
    result has shape ``[dim_size, D]`` where row ``i`` is the sum of all
    ``src`` rows whose ``index`` equals ``i``.

    Args:
        src: Per-edge message tensor of shape ``[E, D]``.
        index: Destination-vertex index per edge, shape ``[E]``.
        dim_size: Number of output buckets (number of vertices).

    Returns:
        Aggregated tensor of shape ``[dim_size, D]``.
    """
    out = torch.zeros(dim_size, src.size(-1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


class GatedGraphConv(nn.Module):
    """Gated graph convolution with edge-type-specific message weights.

    Args:
        hidden_dim: Vertex hidden-state dimension ``D``.
        num_edge_types: Number of distinct edge types; one ``D x D``
            weight matrix is learned per type.
        num_steps: Number of message-passing steps ``T``.
    """

    def __init__(self, hidden_dim: int = 64, num_edge_types: int = 4, num_steps: int = 5) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_edge_types < 1:
            raise ValueError(f"num_edge_types must be >= 1, got {num_edge_types}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_steps = num_steps

        # One [D, D] message matrix per edge type, stacked as [E_t, D, D].
        self.weight = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``T`` message-passing steps over the graph.

        Args:
            h: Vertex hidden states of shape ``[N, D]``.
            edge_index: Edge endpoints of shape ``[2, E]`` (row 0 =
                source, row 1 = destination).
            edge_type: Per-edge type id of shape ``[E]`` with values in
                ``[0, num_edge_types)``.

        Returns:
            Updated vertex states of shape ``[N, D]``.
        """
        if h.dim() != 2 or h.size(-1) != self.hidden_dim:
            raise ValueError(f"h must be [N, {self.hidden_dim}], got {tuple(h.shape)}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")

        num_nodes = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        for _ in range(self.num_steps):
            # Gather per-edge weight matrices: [E, D, D].
            w_e = self.weight[edge_type]
            # Source states gathered per edge: [E, D].
            h_src = h[src]
            # Edge messages W_e h_u: [E, D].
            messages = torch.einsum("eij,ej->ei", w_e, h_src)
            # Aggregate into destination vertices: [N, D].
            m = scatter_add(messages, dst, dim_size=num_nodes)
            # GRU update.
            h = self.gru(m, h)

        return h
