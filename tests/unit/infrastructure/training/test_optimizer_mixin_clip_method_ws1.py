"""WS1-schemas-misc-03: ``optimization.gradient.clip.method`` must reach the
optimizer stepper. The stepper used to hard-default to "norm", silently ignoring
the configured method (pitfall #15)."""

from __future__ import annotations

from types import SimpleNamespace

from tests.utils.config_block_stub import block_stub

from spectramr.infrastructure.training.strategies.mixins.optimizer import OptimizerMixin


def _mixin(clip_method: str) -> OptimizerMixin:
    obj = OptimizerMixin()
    obj.amp_policy = SimpleNamespace(max_grad_norm=1.0, enable_gradient_clipping=True)
    # Production reads `optimization.gradient.clip.method`; a flat
    # `gradient_clip_method` here predates the decomposition and the reader never
    # saw it. block_stub routes the legacy kwarg to its canonical home off the
    # REAL schema, so the shape cannot drift from what the reader walks.
    obj.config = SimpleNamespace(
        optimization=block_stub("optimization", gradient_clip_method=clip_method)
    )
    return obj


def test_clip_method_value_is_threaded_to_stepper() -> None:
    stepper = _mixin("value")._build_optimizer_stepper(logging_service=None)
    assert stepper.gradient_clip_method == "value"


def test_clip_method_norm_is_threaded_to_stepper() -> None:
    stepper = _mixin("norm")._build_optimizer_stepper(logging_service=None)
    assert stepper.gradient_clip_method == "norm"
