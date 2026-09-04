"""Regression: the step executor hand-rolled an fp16 autocast, defeating the AMP guard.

2026-06-11 infrastructure audit. ``StepExecutor.execute_step`` (then ``Trainer``)
unconditionally entered ``torch.autocast(..., dtype=torch.float16,
enabled=amp_helper.enabled)`` around the forward, so the
``MixedPrecisionIntegrationHelper`` policy that disables fp16 autocast for
complex/k-space flows was unreachable from the canonical training path. The
executor now routes through ``amp_helper.get_autocast_context()``.

(The engine module/class was renamed ``trainer.py``/``Trainer`` →
``step_executor.py``/``StepExecutor`` in 2026-06; this test was repointed +
renamed to ``test_step_executor_autocast_2026_06.py`` — #131 missed it because
it *imports* the module, outside the src-scoped rename sweep.)
"""

from __future__ import annotations

import contextlib
import inspect

import torch

from spectramr.infrastructure.training import step_executor as step_executor_mod
from spectramr.infrastructure.training.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionIntegrationHelper,
)


def test_step_executor_routes_autocast_through_amp_helper():
    src = inspect.getsource(step_executor_mod.StepExecutor.execute_step)
    assert "self.amp_helper.get_autocast_context()" in src
    # The hand-rolled fp16 autocast must be gone.
    assert "dtype=torch.float16" not in src


def test_complex_fp16_helper_disables_autocast():
    cfg = MixedPrecisionConfig(
        enabled=True, precision="fp16", enable_complex_support=True
    )
    helper = MixedPrecisionIntegrationHelper(cfg)
    ctx = helper.get_autocast_context()
    # Complex + fp16 → autocast disabled (a plain nullcontext), NOT an enabled
    # fp16 autocast region.
    assert isinstance(ctx, contextlib.nullcontext)


def test_noncomplex_fp16_helper_returns_real_autocast():
    cfg = MixedPrecisionConfig(
        enabled=True, precision="fp16", enable_complex_support=False
    )
    helper = MixedPrecisionIntegrationHelper(cfg)
    ctx = helper.get_autocast_context()
    assert isinstance(ctx, torch.amp.autocast)
