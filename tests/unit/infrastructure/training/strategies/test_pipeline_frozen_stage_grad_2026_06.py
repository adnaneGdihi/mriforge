"""Regression: a frozen pipeline stage must not crash the training step.

The frozen-stage path ran the stage under ``torch.inference_mode()`` and then
called ``output.detach().requires_grad_(True)`` to re-arm gradient flow for the
downstream trainable stage. Setting ``requires_grad=True`` on an inference-mode
tensor (outside InferenceMode) raises::

    RuntimeError: Setting requires_grad=True on inference tensor outside
    InferenceMode is not allowed.

so any config with a frozen stage-1 + trainable stage-2 (the canonical use of
this strategy) crashed on the first training call. The frozen forward must run
under ``torch.no_grad()`` (whose outputs are ordinary tensors) instead.
"""

from __future__ import annotations

import types

import torch
from torch import nn

from mriforge.infrastructure.training.strategies.pipeline_strategy import (
    MultiTrainingStrategy,
)


def _frozen_stage_cfg():
    return types.SimpleNamespace(
        training_state=types.SimpleNamespace(freeze_all=True, inject_lora=False)
    )


def test_frozen_stage_forward_rearms_grad_without_crash():
    s = object.__new__(MultiTrainingStrategy)
    s.stage_order = ["s1"]
    s.multi_stages = {"s1": nn.Conv2d(1, 1, 1)}
    s.stage_configs = {"s1": _frozen_stage_cfg()}
    s._all_unfrozen = False

    x = torch.randn(1, 1, 8, 8)

    # Pre-fix: raises RuntimeError (requires_grad_ on inference tensor).
    out, intermediates = s._execute_sequential(x, training=True)

    assert out.requires_grad, (
        "frozen-stage output must re-arm grad so the downstream trainable "
        "stage receives gradients"
    )
    assert "s1" in intermediates
