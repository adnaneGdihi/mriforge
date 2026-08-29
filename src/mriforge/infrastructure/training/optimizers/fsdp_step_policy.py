"""Step policy for FSDP-wrapped models.

Differs from :class:`AMPPolicy` in exactly one respect, and it is not cosmetic.

``torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`` computes the
norm over the parameters *this rank holds*. Under FSDP that is a **shard**, so
each rank computes a different norm and scales its gradients by a different
factor. The result is not the globally-clipped update the config asked for; it
is a per-rank-inconsistent one, and because every rank still steps successfully
it surfaces as training instability rather than as an error.

FSDP exposes ``model.clip_grad_norm_(max_norm)`` for this: it all-reduces the
squared norms before scaling, so every rank applies the same factor.

The whole reason this is a separate policy object rather than an ``if isinstance
(model, FSDP)`` inside ``AMPPolicy`` is that the choice belongs to the
parallelism strategy, which already knows how the model was wrapped -- the step
path should not have to re-derive it.
"""

from __future__ import annotations

import logging

import torch

from mriforge.infrastructure.training.optimizers.amp_policy import AMPPolicy

logger = logging.getLogger(__name__)

__all__ = ["FSDPStepPolicy"]


class FSDPStepPolicy(AMPPolicy):
    """:class:`AMPPolicy` with a sharding-correct gradient clip."""

    def clip_gradients(self, model: torch.nn.Module, max_norm: float) -> None:
        """Clip by the GLOBAL gradient norm across all shards.

        Falls back to the unsharded path only when the module genuinely does not
        expose FSDP's method -- e.g. a policy attached to an arm whose model was
        not wrapped after all. That is a real, reportable inconsistency rather
        than a silent degradation, so it warns.
        """
        clip = getattr(model, "clip_grad_norm_", None)
        if callable(clip):
            clip(max_norm)
            return

        logger.warning(
            "FSDPStepPolicy is clipping %s, which does not expose "
            "clip_grad_norm_ -- it is not FSDP-wrapped. Falling back to a "
            "parameter-local norm, which is correct ONLY if this rank holds the "
            "whole model.",
            type(model).__name__,
        )
        super().clip_gradients(model, max_norm)
