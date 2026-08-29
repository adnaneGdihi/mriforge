"""Unit tests for ``dtn2s_strategy`` -- the J-invariance mask must leave context.

Paired with ``src/mriforge/infrastructure/training/strategies/dtn2s_strategy.py``,
which had no paired test until #1028. That absence is part of the defect's
history: the mask blocked 100% of voxels at every window size and nothing at the
strategy level ever asserted otherwise.
"""

from __future__ import annotations

import types

import torch




class TestMaskLeavesVisibleContext:
    """The masked input must retain context for the model to read (#1028).

    Before the fix ``build_dtn2s_mask`` blocked 100% of voxels at every window,
    so ``s_a_masked`` was all zeros and the arm trained a map from a constant.
    These assertions are on the STRATEGY, because the mask is voxel-indexed
    while ``s_a`` is position-indexed -- a correct mask applied in the wrong
    index space is just as degenerate, and only this level catches that.
    """

    @staticmethod
    def _strategy(recv: int):
        from mriforge.config.schemas.training.strategy_knobs_2026_08 import (
            DTN2STrainingConfigSchema,
        )
        from mriforge.infrastructure.training.strategies.dtn2s_strategy import (
            DTN2SStrategy,
        )

        strat = DTN2SStrategy.__new__(DTN2SStrategy)
        strat._phi_a = strat._phi_b = strat._mask = None
        cfg = types.SimpleNamespace(
            training=types.SimpleNamespace(
                dtn2s=DTN2STrainingConfigSchema(receptive_window=recv)
            )
        )
        strat.config = cfg
        return strat

    def test_masked_sequence_is_not_all_zeros(self) -> None:
        strat = self._strategy(recv=4)
        x = torch.randn(1, 1, 16, 16)
        mask = strat._resolve_mask(x)
        s_a = strat._phi_a.linearize(x).clone()
        s_a[..., mask] = 0.0
        assert s_a.abs().sum() > 0, "whole input blanked -- model trains on zeros"
        assert mask.sum() > 0, "nothing blocked at all -- J-invariance unenforced"

    def test_declared_receptive_window_is_honoured(self) -> None:
        """A YAML value must reach the mask; it used to be discarded (#376)."""
        x = torch.randn(1, 1, 16, 16)
        narrow = self._strategy(recv=1)._resolve_mask(x).sum().item()
        wide = self._strategy(recv=8)._resolve_mask(x).sum().item()
        assert wide > narrow, (
            f"receptive_window is inert: recv=1 blocked {narrow}, recv=8 blocked {wide}"
        )
