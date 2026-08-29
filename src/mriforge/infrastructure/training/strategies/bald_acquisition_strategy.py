"""BALD — Bayesian Active Learning by Disagreement (PR-25 / Part-I G).

Closed-loop active acquisition. The strategy maintains a running mask
of acquired k-space lines, and every ``retrain_every`` gradient steps
uses a posterior-disagreement score to pick the next ``query_batch``
lines from the unsampled set.

Reference: Houlsby et al. "Bayesian Active Learning for Classification
and Preference Learning" (arXiv:1112.5745); Sanchez et al. "Scalable
learning-based sampling optimization" (2020).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class BALDAcquisitionStrategy(ReconstructionTrainingStrategy):
    """Active-acquisition strategy with MC-dropout posterior."""

    #: Declared rather than inherited — see :class:`PILOTStrategy` for why.
    #: BALD *chooses which k-space lines to acquire next* from a posterior
    #: disagreement score, which is acquisition design in the closed-loop sense,
    #: and reconstructs from what it has acquired.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.STRUCTURAL}),
        tasks=frozenset({Task.RECONSTRUCTION, Task.ACQUISITION_DESIGN}),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active = self.config.acquisition.active
        if not self._active.enabled:
            logger.warning(
                "BALDAcquisitionStrategy instantiated but acquisition.active.enabled is False — "
                "the active loop is dormant; falling back to pure reconstruction."
            )
        self._acquired_mask: torch.Tensor | None = None
        self._step_count: int = 0

    def _initialize_seed_mask(self, w: int) -> torch.Tensor:
        """Acquire low-frequency centre lines as the seed mask."""
        n_lf = max(1, int(self._active.seed_mask_lf_fraction * w))
        mask = torch.zeros(w, device=self.device, dtype=torch.bool)
        centre = w // 2
        half = n_lf // 2
        mask[centre - half : centre + half + 1] = True
        return mask

    @torch.no_grad()
    def _bald_score(self, k_input: torch.Tensor) -> torch.Tensor:
        """Predictive disagreement across MC-dropout samples per k-space line.

        Cost-aware utility (Sanchez 2020 Eq. 6): score = variance / cost,
        where cost = 1 + alpha * |k|² so high-frequency lines pay more.
        """
        if getattr(self.env, "generator", None) is None:
            return torch.zeros(k_input.shape[-1], device=self.device)

        model = self.env.generator
        model.train()
        n_samples = 8
        preds = [model(k_input) for _ in range(n_samples)]
        model.eval()

        stacked = torch.stack(preds, dim=0)
        per_line_var = stacked.var(dim=0).mean(dim=tuple(range(stacked.dim() - 2)))

        # Cost-aware utility: |k|^2 penalty + acquisition-time prior.
        W = per_line_var.shape[0]
        k = torch.arange(W, device=per_line_var.device).float() - W / 2
        cost = 1.0 + 1e-3 * k.pow(2)
        return per_line_var / cost

    def _maybe_acquire_lines(self, batch: Any) -> None:
        """Run the BALD inner loop if it's time and the strategy is enabled."""
        self._step_count += 1
        if not self._active.enabled:
            return
        if self._step_count % self._active.retrain_every:
            return

        kspace = batch.get("kspace") if isinstance(batch, dict) else None
        if kspace is None:
            return
        w = kspace.shape[-1]
        if self._acquired_mask is None:
            self._acquired_mask = self._initialize_seed_mask(w)

        # Budget check.
        if self._acquired_mask.float().mean().item() >= self._active.target_rate:
            return

        scores = self._bald_score(kspace)
        scores = scores.masked_fill(self._acquired_mask, float("-inf"))
        topk = scores.topk(self._active.query_batch).indices
        self._acquired_mask[topk] = True
        logger.debug(
            "BALD acquired %d lines (total %d / %d budget)",
            self._active.query_batch,
            int(self._acquired_mask.sum().item()),
            int(self._active.target_rate * w),
        )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Run the BALD loop before forward pass — the mask the
        # backbone sees is therefore the union of seed + queried lines.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        self._maybe_acquire_lines(batch)
        return super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
