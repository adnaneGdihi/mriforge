"""AdversarialMixin: the per-step metrics accessor must not sync (#707).

This was the clearest of the three -- an explicit `.detach().item()` per key,
directly beneath closures that store on-device with a "no sync: NN#9" comment.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.strategies.mixins.adversarial import (  # noqa: E402
    AdversarialMixin,
)


def test_get_last_metrics_does_not_sync_the_gpu(no_gpu_sync):
    no_gpu_sync(AdversarialMixin.get_last_metrics)


def test_get_last_metrics_on_a_strategy_that_published_nothing(empty_metrics_is_safe):
    empty_metrics_is_safe(AdversarialMixin.get_last_metrics)
