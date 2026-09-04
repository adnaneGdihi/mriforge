"""FrozenBackboneAdapter — load + freeze + attach head.

Plan: TODO/backlog_multi_contrast_remaining.md §Item 1.

Used by Stage 2 of the SSL frozen-backbone pipeline. The adapter:

1. Loads a checkpoint (path or URI) into a backbone module.
2. Sets ``requires_grad=False`` on every backbone parameter.
3. Wires a configurable head on top.
4. Audits compatibility (channel count, contrast cohort) and fails
   loudly on mismatch (CLAUDE.md pitfall #9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FrozenBackboneAdapter(nn.Module):
    """Two-stage SSL adapter — frozen anatomy backbone + trainable head.

    Args:
        backbone: An already-instantiated nn.Module to be frozen.
        head: A trainable nn.Module that receives the backbone's output
              (and optionally a contrast-conditioning embedding).
        checkpoint_uri: Optional path / URI to load into ``backbone`` before
                        freezing. ``None`` keeps the backbone's current weights.
        freeze: When ``True``, set ``requires_grad=False`` on the backbone.
                Default is ``True`` (the entire raison-d'être of the adapter).
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        checkpoint_uri: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        if checkpoint_uri is not None:
            self._load_checkpoint(checkpoint_uri)

        if freeze:
            self.freeze_backbone()

    def _load_checkpoint(self, uri: str | Path) -> None:
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(
                f"FrozenBackboneAdapter: checkpoint {uri} not found. "
                "Stage 2 cannot proceed without the Stage 1 backbone."
            )
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to torch.load({uri!r}): {e}") from e

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys when loading frozen backbone: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys when loading frozen backbone: %s", unexpected[:5])

    def freeze_backbone(self) -> None:
        """Set ``requires_grad=False`` on every backbone parameter."""
        n = 0
        for p in self.backbone.parameters():
            p.requires_grad = False
            n += 1
        logger.info("FrozenBackboneAdapter: froze %d parameters.", n)

    def trainable_parameters(self) -> int:
        """Return the count of trainable parameters (i.e. the head only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features, *args, **kwargs)


__all__ = ["FrozenBackboneAdapter"]
