"""Adapter blocks for shape alignment, conditioning, and the universal paradigm.

These are plain ``nn.Module`` building blocks (not ``@register_model`` —
blocks in ``models/blocks/`` are imported by path, never registered).

Two families live here:

- **Shape / conditioning adapters** (``shape_adapters``): channel/spatial
  projectors, FiLM conditioners, latent reshapers. Previously lived in the
  flat module ``models/blocks/adapters.py``; re-exported here so existing
  ``from spectramr.models.blocks.adapters import TimeFiLM2D`` imports keep
  working after the file became a package.
- **LoRA adapters** (``lora``, PR-7): low-rank trainable updates on a frozen
  base layer, the differentiator for one-model-many-regimes foundation
  reconstruction.
"""

from __future__ import annotations

from spectramr.models.blocks.adapters.lora import LoRAAdapter, inject_lora
from spectramr.models.blocks.adapters.shape_adapters import (
    ChannelProjector2D,
    ChannelProjector3D,
    ContextProjector2D,
    ContextProjector3D,
    DecoderAdapter3D,
    LatentGridSpec,
    LatentReshape2D,
    LatentReshape3D,
    SpatialResizer2D,
    SpatialResizer3D,
    TimeFiLM2D,
    TimeFiLM3D,
)

__all__ = [
    "ChannelProjector2D",
    "ChannelProjector3D",
    "ContextProjector2D",
    "ContextProjector3D",
    "DecoderAdapter3D",
    "LatentGridSpec",
    "LatentReshape2D",
    "LatentReshape3D",
    "LoRAAdapter",
    "SpatialResizer2D",
    "SpatialResizer3D",
    "TimeFiLM2D",
    "TimeFiLM3D",
    "inject_lora",
    "shape_adapters",
]
