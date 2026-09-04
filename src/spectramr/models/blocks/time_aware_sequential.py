"""Time-Aware Sequential Container.

Provides a shared ``nn.Sequential`` subclass that passes optional time/condition
embeddings through modules whose ``forward`` signature accepts an ``emb``
parameter.  This eliminates duplicate definitions scattered across generator
files.

Usage::

    from spectramr.models.blocks.time_aware_sequential import TimeAwareSequential

    block = TimeAwareSequential(
        ConvBlock(64, 128),
        TimeInjectionBlock(128),
        ConvBlock(128, 128),
    )
    out = block(x, emb=time_embedding)
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn


class TimeAwareSequential(nn.Sequential):
    """Sequential container that passes time/condition embeddings through compatible modules.

    During ``forward``, each child module is inspected at registration time.
    If it accepts a second positional parameter (typically ``emb``) it receives
    the embedding; otherwise it is called with ``x`` only.

    This is the **Single Source of Truth** implementation — all generators
    should import from this module instead of re-defining their own.
    """

    def __init__(self, *args: nn.Module) -> None:
        """__init__."""
        super().__init__(*args)
        # Pre-compute which children accept an embedding argument
        self._emb_capable: dict[int, bool] = {}
        for idx, module in enumerate(self):
            self._emb_capable[idx] = self._accepts_emb(module)

    @staticmethod
    def _accepts_emb(module: nn.Module) -> bool:
        """Check whether *module.forward* accepts a second positional arg."""
        try:
            sig = inspect.signature(module.forward)
            params = list(sig.parameters.values())
            # params[0] is usually 'self' (already bound) or 'x'.
            # We look for a second parameter that is not **kwargs / *args.
            positional_kinds = {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            positional = [p for p in params if p.kind in positional_kinds]
            # If there are ≥ 2 positional params, the module can receive emb
            return len(positional) >= 2
        except (ValueError, TypeError):
            return False

    def forward(self, x: torch.Tensor, emb: Any = None) -> torch.Tensor:
        """Forward pass, routing ``emb`` to compatible children.

        Args:
            x: Input tensor.
            emb: Optional time/condition embedding tensor.

        Returns:
            Output tensor after all children.
        """
        for idx, module in enumerate(self):
            if emb is not None and self._emb_capable.get(idx, False):
                x = module(x, emb)
            else:
                x = module(x)
        return x
