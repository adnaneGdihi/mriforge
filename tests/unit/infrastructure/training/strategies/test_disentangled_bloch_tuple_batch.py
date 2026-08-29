"""Regression: _synthesize_via_bloch must tolerate the (input, target) TUPLE
that validation_step passes as ``batch``.

2026-05-31 audit: for ``output_mode: tissue_params`` arms, validation_step
builds ``batch = (input_batch, target_batch)`` and hands it to
``_synthesize_via_bloch`` -> ``_synthesize_via_bloch_fp32`` does
``batch.get("TR", ...)`` -> ``AttributeError: 'tuple' object has no attribute
'get'``. The wrapper now coerces a non-dict batch to ``{}`` (defaults apply).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


class TestResolveAcqParamNoSilentDefault:
    """F4: acquisition timing params must come from the model prediction or the
    batch — never a hardcoded GRE default (pitfall #9, silent fallback)."""

    def test_prefers_model_prediction(self):
        v = DisentangledTrainingStrategy._resolve_acq_param(
            {"predicted_tr": torch.tensor(1500.0)}, {"TR": 8.0}, "predicted_tr", "TR"
        )
        assert float(v) == 1500.0

    def test_falls_back_to_batch_metadata(self):
        v = DisentangledTrainingStrategy._resolve_acq_param(
            {}, {"TR": 2000.0}, "predicted_tr", "TR"
        )
        assert float(v) == 2000.0

    def test_raises_when_neither_source_present(self):
        # The old code silently returned 8.0 here; it must now fail loud.
        with pytest.raises(ValueError, match="predicted_tr"):
            DisentangledTrainingStrategy._resolve_acq_param(
                {}, {}, "predicted_tr", "TR"
            )


def test_synthesize_via_bloch_coerces_tuple_batch(monkeypatch):
    s = object.__new__(DisentangledTrainingStrategy)
    received = {}

    def _fake_fp32(self, tissue_params, batch, source_key):
        received["batch"] = batch
        return torch.zeros(1)

    monkeypatch.setattr(
        DisentangledTrainingStrategy, "_synthesize_via_bloch_fp32", _fake_fp32
    )
    # validation_step builds this 2-tuple; the old code crashed on batch.get(...)
    out = s._synthesize_via_bloch(
        {}, (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4)), "a"
    )
    assert isinstance(out, torch.Tensor)
    assert isinstance(received["batch"], dict)  # tuple coerced to {}
