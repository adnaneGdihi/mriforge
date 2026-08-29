"""Gated-GNN k-space reconstruction generator.

Registered as ``gated_gnn`` (kspace -> image reconstruction). The model
treats each spatial location of the input as a graph vertex, embeds it
into a hidden state, runs gated graph message passing
(:class:`mriforge.models.blocks.graph.gated_gnn.GatedGraphConv`), and
reads out a per-vertex correction that is reshaped back to an image.

For a constructible default the graph is a 4-/8-connected grid built
internally from the input spatial dims, with edge type derived from the
relative offset (radial / angular displacement class). This lets
``forward(x)`` operate on a dense tensor without an external
``non_cartesian_graph`` transform; when such a transform is wired in,
a precomputed ``edge_index`` / ``edge_type`` can be passed instead.

The reusable message-passing block lives in ``models/blocks/graph/`` —
this file only composes it (CLAUDE.md canonical-homes rule).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.blocks.graph.gated_gnn import GatedGraphConv
from mriforge.models.registry import register_model

__all__ = ["GatedGNNReconstructor"]


# Eight grid neighbour offsets grouped into edge types by displacement
# class. With num_edge_types=4 the mapping is:
#   0 -> horizontal, 1 -> vertical, 2 -> main diagonal, 3 -> anti-diagonal.
_NEIGHBOUR_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 0),  # right        (horizontal)
    (0, -1, 0),  # left         (horizontal)
    (1, 0, 1),  # down         (vertical)
    (-1, 0, 1),  # up           (vertical)
    (1, 1, 2),  # down-right   (main diagonal)
    (-1, -1, 2),  # up-left      (main diagonal)
    (1, -1, 3),  # down-left    (anti-diagonal)
    (-1, 1, 3),  # up-right     (anti-diagonal)
)


@register_model(
    name="gated_gnn",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class GatedGNNReconstructor(nn.Module):
    """Gated graph neural network for grid-graph k-space reconstruction.

    Args:
        in_channels: Number of input channels per vertex.
        out_channels: Number of output image channels.
        hidden_dim: Vertex hidden-state dimension.
        num_edge_types: Number of edge types in the grid graph (4 -> the
            four displacement classes; must match the offset table).
        num_steps: Number of message-passing steps.
        connectivity: ``4`` for axial neighbours only, ``8`` to include
            diagonals.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 64,
        num_edge_types: int = 4,
        num_steps: int = 5,
        connectivity: int = 8,
    ) -> None:
        super().__init__()
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_steps = num_steps
        self.connectivity = connectivity

        self.encoder = nn.Linear(in_channels, hidden_dim)
        self.gnn = GatedGraphConv(
            hidden_dim=hidden_dim,
            num_edge_types=num_edge_types,
            num_steps=num_steps,
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )

        # Cache for the grid topology, keyed by (H, W). Built lazily on
        # the correct device during forward.
        self._edge_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_grid_graph(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (edge_index, edge_type) for an HxW grid graph.

        Vertices are indexed row-major: ``v = r * width + c``. Edges
        connect each vertex to its in-bounds grid neighbours; the edge
        type is the displacement class from the offset table, clamped to
        ``num_edge_types``.
        """
        key = (height, width)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached

        offsets = _NEIGHBOUR_OFFSETS if self.connectivity == 8 else _NEIGHBOUR_OFFSETS[:4]

        src_list: list[int] = []
        dst_list: list[int] = []
        type_list: list[int] = []
        for r in range(height):
            for c in range(width):
                v = r * width + c
                for dr, dc, etype in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        u = nr * width + nc
                        src_list.append(u)  # message flows source u -> dst v
                        dst_list.append(v)
                        type_list.append(min(etype, self.num_edge_types - 1))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        edge_type = torch.tensor(type_list, dtype=torch.long, device=device)
        self._edge_cache[key] = (edge_index, edge_type)
        return edge_index, edge_type

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct an image via gated graph message passing.

        Args:
            x: Input tensor of shape ``[B, in_channels, H, W]``.
            edge_index: Optional precomputed edge endpoints ``[2, E]``
                (e.g. from a ``non_cartesian_graph`` transform). When
                omitted, a grid graph is built from ``H, W``.
            edge_type: Optional precomputed per-edge type ``[E]``. Must
                be provided together with ``edge_index``.

        Returns:
            Reconstructed image of shape ``[B, out_channels, H, W]``.
        """
        if x.dim() != 4:
            raise ValueError(f"expected [B, C, H, W] input, got {tuple(x.shape)}")
        if (edge_index is None) != (edge_type is None):
            raise ValueError("edge_index and edge_type must both be provided or both omitted")

        batch, _, height, width = x.shape
        num_nodes = height * width

        if edge_index is None:
            edge_index, edge_type = self._build_grid_graph(height, width, x.device)

        outputs: list[torch.Tensor] = []
        for b in range(batch):
            # [C, H, W] -> [N, C] node features (row-major).
            feats = x[b].permute(1, 2, 0).reshape(num_nodes, self.in_channels)
            h = self.encoder(feats)
            h = self.gnn(h, edge_index, edge_type)
            corr = self.readout(h)  # [N, out_channels]
            img = corr.reshape(height, width, self.out_channels).permute(2, 0, 1)
            outputs.append(img)

        return torch.stack(outputs, dim=0)
