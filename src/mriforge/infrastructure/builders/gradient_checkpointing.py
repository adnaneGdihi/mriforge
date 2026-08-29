"""Post-construction gradient-checkpointing wrapper, shared by builder and probe.

Split out of :mod:`generator_kwargs` in the Wave 0 exit-criterion work (#1400).
It never belonged there on concern: ``generator_kwargs`` answers *what kwargs
does this generator get*, while this mutates a generator that already exists.

Shared so that every path doing a real backward pass -- training, and the audit
probe -- exercises the same memory/recompute behaviour. An arm that checkpoints
in training but not under the probe is probed on a model training never builds.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_gradient_checkpointing(generator: Any, config: Any) -> None:
    """Apply gradient checkpointing to ``generator`` when the arm enables it.

    Shared so that a path doing a real backward pass -- training, and the audit
    probe -- exercises the same memory/recompute behaviour. An arm that
    checkpoints in training but not under the probe is probed on a model
    training never builds.

    Extracted verbatim, including the broad ``except`` that downgrades a
    wrapping failure to a warning. That swallow is a known defect, tracked in
    #1333; preserving it here keeps this extraction behaviour-identical, so the
    fix lands in one place rather than riding along with a refactor.
    """
    try:
        enabled = config.optimization.gradient.enable_checkpointing
    except AttributeError:
        return
    if not enabled:
        return

    if hasattr(generator, "set_grad_checkpointing"):
        generator.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing enabled natively via set_grad_checkpointing(True)")
        return

    logger.info("Generic torch.utils.checkpoint applying to generator components...")
    # Import stays OUTSIDE the try: a missing profiling module is a broken
    # install, which must propagate, not be downgraded to a warning.
    from mriforge.models.profiling.advanced_profiling import GradientCheckpointing

    try:
        GradientCheckpointing.apply_checkpointing(generator, checkpoint_ratio=1.0)
    except Exception as chkp_err:  # preserved verbatim from the builder
        logger.warning(f"Failed to apply generic gradient checkpointing: {chkp_err}")


__all__ = ["apply_gradient_checkpointing"]
