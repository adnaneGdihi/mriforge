"""Phase-2 audit helpers: cross-check inference-mode YAML against paradigm support.

When a YAML declares ``data.modes.infer.sampler.type: grid`` (sliding-
window tiling), the chosen training paradigm's inference strategy MUST
opt in via ``BaseInferenceStrategy.supports_grid_tiling = True``.
Diffusion paradigms (latent / cold) and others with iterative reverse
loops do NOT — their tile-then-aggregate composition is non-trivial
and intentionally deferred.

This module exposes a single helper that the audit ladder
(``src/cli/app.py audit``) and the data-loader director both consult.
Both call ``check_grid_tiling_supported`` before invoking
``build_grid_inference_handle`` so an incompatible combo fails at
config-load (CLAUDE.md #9) rather than producing wrong outputs at
inference time.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def check_grid_tiling_supported(
    strategy_class: type[Any],
    mode_sampler_type: str | None,
    *,
    strategy_name: str | None = None,
) -> None:
    """Raise ``ValueError`` if ``mode_sampler_type == "grid"`` but the
    inference-strategy class doesn't set ``supports_grid_tiling = True``.

    No-op for non-grid samplers (``"full"``, ``"uniform"``, etc.) and
    for ``mode_sampler_type=None`` (legacy YAMLs that don't declare
    a mode block).

    Args:
        strategy_class: a subclass of
            :class:`spectramr.infrastructure.inference.base_inference_strategy.BaseInferenceStrategy`.
            Class object, NOT an instance — the ``supports_grid_tiling``
            attribute is a class-level flag.
        mode_sampler_type: value of ``data.modes.infer.sampler.type``.
            ``"grid"`` triggers the check; anything else returns ``None``.
        strategy_name: optional override for the error message (used
            when ``strategy_class`` is a runtime-resolved object whose
            ``__name__`` isn't user-meaningful).

    Raises:
        ValueError: if grid tiling is requested but unsupported. The
            message names the strategy class and points at the
            override mechanism (set ``supports_grid_tiling = True``
            on the subclass AND implement per-tile inference logic
            that correctly composes with TorchIO's ``GridAggregator``).
    """
    if mode_sampler_type != "grid":
        return

    supports = getattr(strategy_class, "supports_grid_tiling", False)
    if supports:
        return

    name = strategy_name or getattr(strategy_class, "__name__", repr(strategy_class))
    raise ValueError(
        f"data.modes.infer.sampler.type='grid' is incompatible with "
        f"inference strategy {name!r} — it does not declare "
        "supports_grid_tiling=True. This usually means the paradigm has "
        "an iterative reverse sampler (diffusion variants) or position-"
        "aware encoding that cannot be naively tiled.\n\n"
        "Resolutions:\n"
        f"  (a) Use a different sampler: set data.modes.infer.sampler.type "
        "to 'full' (whole-volume) or 'uniform' (random patches) instead.\n"
        f"  (b) If you've implemented per-tile inference for {name}, set "
        "``supports_grid_tiling: bool = True`` on the class.\n\n"
        "See TODO/audit/data_layer_unification_plan.md §5 Phase 2 for "
        "the per-strategy compatibility table."
    )


def list_grid_tiling_compatible_strategies(
    strategy_classes: list[type[Any]],
) -> list[str]:
    """Return the subset of ``strategy_classes`` that opt into tiling.

    Used by the audit ladder to print a compatibility table when a user
    asks ``--describe-tiling-support``.
    """
    return [
        getattr(c, "__name__", repr(c))
        for c in strategy_classes
        if getattr(c, "supports_grid_tiling", False)
    ]


__all__ = [
    "check_grid_tiling_supported",
    "list_grid_tiling_compatible_strategies",
]
