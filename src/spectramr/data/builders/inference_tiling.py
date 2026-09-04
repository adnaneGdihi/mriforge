"""Sliding-window tiling for 3D inference (Phase 2 of the data-layer plan).

Wraps :class:`torchio.data.GridSampler` and
:class:`torchio.data.GridAggregator` so the rest of the codebase
talks to a small, paradigm-agnostic API:

.. code-block:: python

    handle = build_grid_inference_handle(
        subject=subject,
        patch_size=(128, 128, 1),
        patch_overlap=(16, 16, 0),
        aggregator="hann",
    )
    for batch in handle.loader:
        pred_tile = model(batch["data"][tio.DATA])
        handle.aggregator.add_batch(pred_tile, locations=batch[tio.LOCATION])
    full_volume = handle.aggregator.get_output_tensor()

This module deliberately handles ONLY the spatial sampling; paradigm
strategies (GAN, reconstruction, etc.) decide whether to call the
model once per tile or run an iterative reverse loop per tile
(diffusion — currently NOT supported, see
:attr:`BaseInferenceStrategy.supports_grid_tiling`).

The wrapper does the small but error-prone bookkeeping:

- Auto-derive ``patch_overlap`` when not given (defaults to 1/8 of
  the patch size, rounded down to even).
- Validate that ``patch_overlap < patch_size`` per axis (TorchIO
  silently fails when overlap == patch_size).
- Translate ``aggregator="hann"`` → TorchIO's
  ``overlap_mode="hann"`` for smooth tile borders.
- Refuse 4D temporal patches at this layer (Phase 4c is the
  temporal-aware variant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


AggregatorKind = Literal["average", "hann", "crop"]


@dataclass(frozen=True)
class GridInferenceHandle:
    """Bundles the per-tile sampler with the spatial-aggregator that
    reassembles model outputs into the full volume.

    Attributes:
        sampler: the underlying ``tio.data.GridSampler`` (iterable of
            tio.Subject patches with attached ``LOCATION``).
        aggregator: the matching ``tio.data.GridAggregator`` —
            strategy code calls ``add_batch(prediction, location)``
            per tile then ``get_output_tensor()`` at the end.
        patch_size: resolved (after defaults) spatial patch shape.
        patch_overlap: resolved overlap shape (one per axis).
        aggregator_mode: which TorchIO overlap mode is in use.
    """

    sampler: object  # tio.data.GridSampler — kept loose to avoid hard import at module level
    aggregator: object  # tio.data.GridAggregator
    patch_size: tuple[int, int, int]
    patch_overlap: tuple[int, int, int]
    aggregator_mode: str


def _auto_overlap(patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """Default ``patch_overlap`` heuristic: 1/8 of patch size,
    rounded down to even, clamped at min 0.

    Even values are important — ``tio.GridSampler`` requires
    overlap to be divisible by 2 along axes where it's non-zero.
    """
    out: list[int] = []
    for s in patch_size:
        guess = max(0, s // 8)
        # Round down to even
        if guess % 2 == 1:
            guess -= 1
        out.append(guess)
    return tuple(out)  # type: ignore[return-value]


def _validate_overlap(
    patch_size: tuple[int, int, int],
    patch_overlap: tuple[int, int, int],
) -> None:
    """Each axis: ``0 <= overlap < patch_size`` and overlap is even."""
    if len(patch_size) != 3 or len(patch_overlap) != 3:
        raise ValueError(
            f"patch_size and patch_overlap must both be 3-tuples; got "
            f"patch_size={patch_size!r}, patch_overlap={patch_overlap!r}"
        )
    for axis, (s, o) in enumerate(zip(patch_size, patch_overlap)):
        if o < 0:
            raise ValueError(f"patch_overlap[{axis}] = {o} is negative; overlap must be >= 0.")
        if o >= s:
            raise ValueError(
                f"patch_overlap[{axis}] = {o} must be strictly less than "
                f"patch_size[{axis}] = {s}. TorchIO GridSampler silently "
                "produces no patches when overlap == patch_size."
            )
        if o % 2 != 0:
            raise ValueError(
                f"patch_overlap[{axis}] = {o} must be even. TorchIO "
                "requires even overlap for symmetric tiling."
            )


def _aggregator_overlap_mode(kind: AggregatorKind) -> str:
    """Map our aggregator-name enum onto TorchIO's ``overlap_mode``.

    - ``"average"`` → ``"average"`` (simple mean of overlapping tiles)
    - ``"hann"``    → ``"hann"`` (cosine-window weighted mean — smooth borders)
    - ``"crop"``    → ``"crop"`` (drop tile borders entirely; no overlap usage)
    """
    if kind not in ("average", "hann", "crop"):
        raise ValueError(
            f"Unknown aggregator kind {kind!r}; expected one of 'average', 'hann', 'crop'."
        )
    return kind


def build_grid_inference_handle(
    subject,
    patch_size: tuple[int, int, int],
    patch_overlap: tuple[int, int, int] | None = None,
    aggregator: AggregatorKind = "hann",
    padding_mode: str | float | None = None,
) -> GridInferenceHandle:
    """Construct a ``GridInferenceHandle`` for a single subject volume.

    Args:
        subject: ``tio.Subject`` whose intensity images will be tiled.
        patch_size: 3-tuple ``(H, W, D)`` spatial patch shape. ``D=1``
            yields 2D tiling (still valid).
        patch_overlap: 3-tuple per-axis overlap. ``None`` → auto-derive
            via :func:`_auto_overlap`. Must be even and strictly less
            than the corresponding ``patch_size`` entry.
        aggregator: how to combine overlapping tile predictions.
            ``"hann"`` is recommended for smooth borders;
            ``"average"`` is simpler and exact for non-overlapping;
            ``"crop"`` discards tile margins.
        padding_mode: optional padding for ``tio.GridSampler`` when
            ``patch_size`` doesn't divide the volume evenly. ``None``
            uses TorchIO's default (no padding — extra tile at the
            edge with the same size).

    Returns:
        A frozen :class:`GridInferenceHandle` carrying the sampler,
        aggregator, and resolved shape metadata.

    Raises:
        ValueError: if ``patch_size``/``patch_overlap`` are malformed.

    Note:
        4D temporal patches (e.g. ``(H, W, D, T)``) are out of scope
        here — Phase 4c provides ``temporal_grid`` sampling for cine.
    """
    # Import TorchIO lazily so this module is importable in test
    # environments without TorchIO installed (used by signature tests).
    import torchio as tio

    if len(patch_size) != 3:
        raise ValueError(
            f"build_grid_inference_handle: 3D patch_size required, got "
            f"{patch_size!r}. For temporal/4D, see Phase 4c "
            "(``temporal_grid`` sampler)."
        )

    resolved_overlap = patch_overlap if patch_overlap is not None else _auto_overlap(patch_size)
    _validate_overlap(patch_size, resolved_overlap)
    overlap_mode = _aggregator_overlap_mode(aggregator)

    sampler = tio.data.GridSampler(
        subject=subject,
        patch_size=patch_size,
        patch_overlap=resolved_overlap,
        padding_mode=padding_mode,
    )
    agg = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode)

    return GridInferenceHandle(
        sampler=sampler,
        aggregator=agg,
        patch_size=tuple(patch_size),  # type: ignore[arg-type]
        patch_overlap=tuple(resolved_overlap),  # type: ignore[arg-type]
        aggregator_mode=overlap_mode,
    )


__all__ = [
    "AggregatorKind",
    "GridInferenceHandle",
    "build_grid_inference_handle",
]
