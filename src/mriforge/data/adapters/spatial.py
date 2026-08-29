"""Spatial-rank adapters: 3D↔2D slicers.

Phase 3 ships these as the first concrete adapter pair. Stage 1
(``slice_3d_to_2d`` on ``pre_model``) flattens depth into batch so a
2D Conv model can consume slices; ``gather_2d_to_3d`` reconstitutes
volumes on the way back. The two together let a 3D-data, 2D-model
experiment run *with the user's explicit consent*, with side-effects
surfaced on the Spec Card.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.data.adapters.registry import register_adapter


@register_adapter(
    name="slice_3d_to_2d",
    bridges_from={"spatial_dims": 3, "domain": "image"},
    bridges_to={"spatial_dims": 2, "domain": "image"},
    insertion_points=("pre_model",),
    invertible=True,
    side_effects=("batch_multiplied_by_depth", "loses_inter_slice_context"),
)
class Slice3Dto2D(nn.Module):
    """Flatten ``[B, C, D, H, W]`` → ``[B*D, C, H, W]``.

    The depth axis is folded into batch. The companion adapter
    :class:`Gather2Dto3D` reverses this on the model output.

    The default ``axis=2`` assumes the standard depth-first layout
    ``[B, C, D, H, W]`` (MONAI convention). For depth-last layouts
    ``[B, C, H, W, D]`` pass ``axis=-1`` (or ``axis=4``).
    """

    def __init__(self, axis: int = 2):
        super().__init__()
        if axis not in (-1, 2, 3, 4):
            raise ValueError(f"slice_3d_to_2d expects depth axis in (-1, 2, 3, 4); got {axis}.")
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"slice_3d_to_2d expects a 5-D tensor [B, C, *spatial]; got shape {tuple(x.shape)}."
            )
        # Move depth to position 2 if it isn't already, then fold into batch.
        ax = self.axis if self.axis >= 0 else x.dim() + self.axis
        if ax != 2:
            x = x.movedim(ax, 2)
        b, c, d, h, w = x.shape
        return x.reshape(b * d, c, h, w)


@register_adapter(
    name="flatten_spatial_to_1d",
    bridges_from={"spatial_dims": 2, "domain": "image"},
    bridges_to={"spatial_dims": 1, "domain": "image"},
    insertion_points=("pre_model",),
    invertible=False,
    side_effects=("flattens_spatial_to_sequence", "loses_2d_locality"),
)
class FlattenSpatialTo1D(nn.Module):
    """Reshape ``[B, C, H, W]`` → ``[B, C, H*W]`` for 1-D temporal/sequence models.

    Used by HRF / score-on-tangent-space models that operate on a 1-D
    sequence axis but consume data delivered as 2-D patches by the
    dataset (e.g., flattening a 2-D HRF parameter grid into a 1-D
    sequence). The companion :class:`Unflatten1DtoSpatial`
    (``unflatten_1d_to_spatial``, ``post_model``) reverses the reshape
    on the way back when ``H, W`` are known at runtime.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x  # already [B, C, L]
        if x.dim() == 4:
            b, c, h, w = x.shape
            return x.reshape(b, c, h * w)
        raise ValueError(
            f"flatten_spatial_to_1d expects a 3-D or 4-D tensor; got shape {tuple(x.shape)}."
        )


@register_adapter(
    name="unflatten_1d_to_spatial",
    bridges_from={"spatial_dims": 1, "domain": "image"},
    bridges_to={"spatial_dims": 2, "domain": "image"},
    insertion_points=("post_model",),
    invertible=False,
    side_effects=("requires_known_h_w_at_runtime",),
)
class Unflatten1DtoSpatial(nn.Module):
    """Inverse of :class:`FlattenSpatialTo1D`: ``[B, C, H*W]`` → ``[B, C, H, W]``.

    Used on the ``post_model`` hook to reshape a 1-D sequence output
    back to a 2-D spatial grid when the model accepts ``[B, C, L]`` but
    the downstream loss / metric expects ``[B, C, H, W]``. The strategy
    must pass ``h`` and ``w`` at call time — the adapter cannot infer
    them from ``L`` alone (the audit-time capability check is
    shape-only; the runtime contract lives in the strategy).

    F-UNFLATTEN-1D / 2026-05-20 — added as the companion to
    ``flatten_spatial_to_1d`` (referenced in its docstring). Completes
    the rank-bridging adapter set alongside ``slice_3d_to_2d`` /
    ``gather_2d_to_3d``.
    """

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"unflatten_1d_to_spatial expects a 3-D tensor [B, C, L]; "
                f"got shape {tuple(x.shape)}."
            )
        b, c, length = x.shape
        if length != h * w:
            raise ValueError(
                f"unflatten_1d_to_spatial: sequence length {length} != "
                f"h*w ({h}*{w}={h * w}). Did you pass the wrong H, W?"
            )
        return x.reshape(b, c, h, w)


@register_adapter(
    name="gather_2d_to_3d",
    bridges_from={"spatial_dims": 2, "domain": "image"},
    bridges_to={"spatial_dims": 3, "domain": "image"},
    insertion_points=("post_model",),
    invertible=True,
    side_effects=("requires_known_depth_at_runtime",),
)
class Gather2Dto3D(nn.Module):
    """Inverse of :class:`Slice3Dto2D`: ``[B*D, C, H, W]`` → ``[B, C, D, H, W]``.

    The strategy must pass ``depth`` at call time (the adapter cannot
    infer it from B*D alone). At audit time only the capability shape
    is checked; the runtime contract lives in the strategy that wires
    the chain.
    """

    def __init__(self, axis: int = 2):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"gather_2d_to_3d expects a 4-D tensor [B*D, C, H, W]; got shape {tuple(x.shape)}."
            )
        bd, c, h, w = x.shape
        if bd % depth != 0:
            raise ValueError(f"gather_2d_to_3d: leading dim {bd} not divisible by depth {depth}.")
        b = bd // depth
        out = x.reshape(b, depth, c, h, w).movedim(1, 2)  # [B, C, D, H, W]
        return out


__all__ = [
    "FlattenSpatialTo1D",
    "Gather2Dto3D",
    "Slice3Dto2D",
    "Unflatten1DtoSpatial",
]
