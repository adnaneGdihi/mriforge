"""Regression: early stopping treated a flat metric as continual improvement.

2026-06-11 infrastructure audit. The comparison was non-strict (``x <= best``),
so with the default ``min_delta=0.0`` an exactly-flat validation metric — the
signature of identity-collapse / DC-blob runs — reset the patience counter on
every check and early stopping never fired. The comparison is now strict.
"""

from __future__ import annotations

from mriforge.config.schemas.early_stopping import EarlyStoppingConfigSchema
from mriforge.infrastructure.services.early_stopping import (
    EarlyStoppingService,
    MetricMode,
)


def _service(mode, metric="val_loss"):
    """Build a service for ``mode``, monitoring a metric that agrees with it.

    ``metric`` became a parameter when ``EarlyStoppingConfigSchema`` grew
    ``_mode_must_match_the_metric_direction`` (6dbd12c7b). The hardcoded
    ``val_loss`` is lower-is-better, so the MAX-mode case below was building a
    config the schema now refuses outright -- and the patience behaviour that
    test exists to check was never reached at all.

    The metric moves with the mode rather than the guard being relaxed: the
    pairing really was wrong, and the values that case feeds (30.0, i.e. a PSNR)
    have always read as a higher-is-better metric anyway.
    """
    cfg = EarlyStoppingConfigSchema(
        enabled=True,
        metric=metric,
        patience=3,
        patience_min_iterations=None,
        mode=mode,
        min_delta=0.0,
    )
    return EarlyStoppingService(cfg)


def test_flat_metric_triggers_early_stop_min_mode():
    svc = _service(MetricMode.MIN)
    svc.update(0.5, iteration=1)  # first value: improvement
    assert svc.wait_count == 0
    for it in range(2, 6):
        svc.update(0.5, iteration=it)  # perfectly flat
    assert svc.wait_count >= 3
    assert svc.should_stop() is True


def test_flat_metric_triggers_early_stop_max_mode():
    svc = _service(MetricMode.MAX, metric="val_psnr")
    svc.update(30.0, iteration=1)
    for it in range(2, 6):
        svc.update(30.0, iteration=it)
    assert svc.should_stop() is True


def test_genuine_improvement_resets_patience():
    svc = _service(MetricMode.MIN)
    svc.update(1.0, iteration=1)
    svc.update(0.9, iteration=2)  # strict improvement
    assert svc.wait_count == 0
    assert svc.should_stop() is False
