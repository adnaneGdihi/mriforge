"""Graph-neural-network building blocks for k-space reconstruction.

Canonical home for reusable graph message-passing layers consumed by
graph-based reconstruction generators. The blocks here are pure
``nn.Module`` building blocks (CLAUDE.md canonical-homes rule): the
registered generator wrappers live in ``models/generators/``.
"""

from mriforge.models.blocks.graph.gated_gnn import GatedGraphConv, scatter_add

__all__ = ["GatedGraphConv", "scatter_add"]
