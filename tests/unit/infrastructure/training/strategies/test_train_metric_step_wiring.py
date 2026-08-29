"""Regression guard: train-metric throttle must read the LIVE loop iteration.

Background (pitfall #16). ``MetricsMixin._compute_training_metrics`` throttles the
per-step reconstruction-quality metrics (SSIM / PSNR / MAE) with::

    if current_step % train_metric_interval != 0:
        return metrics

Several strategies used to feed it ``getattr(self.env, "step", 0)``. But
``TrainingEnvironment`` is ``frozen=True`` and never carries a live ``step``, so
that read is *always* 0 — and ``0 % interval == 0`` for every interval, defeating
the throttle: the (host-syncing) metrics were recomputed and logged on EVERY step
instead of once per ``train_metric_interval``.

``gan.py`` / ``diffusion.py`` / the disentangled strategies were migrated to the
``loop_state`` seam; ``vae.py`` and ``reconstruction.py`` were missed. Both now
reuse the live ``iteration`` they already resolve from ``kwargs`` for their loss
schedules. These tests pin that wiring so a revert to the frozen ``env.step`` form
fails loudly, and verify the throttle contract the fix depends on.

Sequel, and a qualification of the contract asserted below: fixing the step to be
live flipped SHORT runs from always-on to NEVER. Over iterations 1..40 with the
schema-default interval of 100 the satisfying set is empty, so
``experiment_11_attention_none`` computed no train metric at all and every
``train_*`` column in ``training_metrics.csv`` stayed blank while the loss columns
filled normally. ``_compute_training_metrics`` therefore ALSO fires on the first
and final iteration now, mirroring the CSV row gate in ``training_loop.py``. A
non-multiple step is still throttled -- as asserted here -- unless it is one of
those two ends. See ``TestComputeTrainingMetrics`` in
``tests/unit/infrastructure/training/strategies/mixins/test_metrics_mixin_expansion.py``.
"""

from __future__ import annotations

import inspect
import re

import pytest

from mriforge.infrastructure.training.strategies import reconstruction as _recon_mod
from mriforge.infrastructure.training.strategies import vae as _vae_mod


def _loss_impl_source(module: object) -> str:
    """Return the source of the module's ``_compute_losses_impl`` method."""
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        fn = obj.__dict__.get("_compute_losses_impl")
        if fn is not None:
            return inspect.getsource(fn)
    raise AssertionError(f"no _compute_losses_impl found in {module.__name__}")


@pytest.mark.unit
@pytest.mark.parametrize("module", [_recon_mod, _vae_mod], ids=["reconstruction", "vae"])
def test_train_metrics_fed_live_iteration_not_frozen_env_step(module: object) -> None:
    src = _loss_impl_source(module)

    # The metrics call must be present and fed the live iteration.
    call = re.search(r"_compute_training_metrics\((.*?)\)", src, re.DOTALL)
    assert call is not None, "no _compute_training_metrics call site found"
    assert "current_step=iteration" in call.group(1), (
        "train-metric throttle is not fed the live loop iteration; a frozen "
        "step would recompute SSIM/PSNR/MAE every step (pitfall #16)"
    )

    # And the frozen-env.step antipattern must not have crept back in.
    assert 'getattr(self.env, "step"' not in src, (
        "reintroduced the frozen self.env.step read for the metric throttle"
    )


@pytest.mark.unit
def test_metrics_throttle_honours_current_step() -> None:
    """The mechanism the fix relies on: a non-multiple step is throttled to {}.

    Constructs the mixin via ``__new__`` (no heavy strategy init) and stitches on
    only what ``_compute_training_metrics`` touches.
    """
    import torch

    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        MetricsMixin,
    )

    mixin = MetricsMixin.__new__(MetricsMixin)

    class _SpyComputer:
        def __init__(self) -> None:
            self.calls = 0

        def compute(self, pred, target):
            self.calls += 1
            return {"ssim": 1.0}

    spy = _SpyComputer()
    # ``training_metrics_computer`` is a lazy property backed by ``_training_computer``.
    mixin._training_computer = spy  # type: ignore[attr-defined]
    mixin._apply_metric_transforms = lambda p, t, c: (p, t)  # type: ignore[attr-defined]

    config = type(
        "Cfg",
        (),
        {"metrics": type("M", (), {"enable_tracking": True, "train_metric_interval": 100})()},
    )()
    # The lazy property reads ``self.config`` before returning the cached computer.
    mixin.config = config  # type: ignore[attr-defined]
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)

    # A non-multiple step is throttled (no compute); the frozen-0 bug made this
    # branch unreachable because env.step was always 0.
    assert mixin._compute_training_metrics(pred, target, config, current_step=50) == {}
    assert spy.calls == 0

    # A multiple step computes and prefixes the metric.
    out = mixin._compute_training_metrics(pred, target, config, current_step=100)
    assert out == {"train_ssim": 1.0}
    assert spy.calls == 1
