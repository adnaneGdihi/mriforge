"""Regression: MetricsTracker hardening (2026-06-11 infrastructure audit).

* ``compute_image_metrics`` substituted ``0.0`` for a failed metric — a
  catastrophic value for higher-is-better metrics that corrupts running averages
  and best-checkpoint selection. It now omits the failed metric.
* ``get_best_metrics`` inferred best-direction from a 2-keyword substring guess,
  so lower-better metrics without ``loss``/``error`` in the name (``nmse``,
  ``rmse``, ``lpips``, ``fid``) reported their WORST value as "best". Direction
  now comes from the ``METRIC_HIGHER_IS_BETTER`` SSOT.
"""

from __future__ import annotations

from spectramr.infrastructure.services import metrics_tracker as mt_mod
from spectramr.infrastructure.services.metrics_tracker import (
    MetricsTracker,
    _metric_higher_is_better,
)


def test_direction_helper_matches_ssot():
    assert _metric_higher_is_better("psnr") is True
    assert _metric_higher_is_better("ssim") is True
    for lower in ("nmse", "rmse", "lpips", "fid", "mae"):
        assert _metric_higher_is_better(lower) is False, lower
    # non-metric loss column falls back to the heuristic
    assert _metric_higher_is_better("generator_loss") is False


def test_get_best_metrics_uses_ssot_direction(tmp_path):
    tracker = MetricsTracker(output_dir=str(tmp_path))
    tracker.training_history = [
        {"nmse": 0.5}, {"nmse": 0.1}, {"nmse": 0.9},
        {"psnr": 25.0}, {"psnr": 30.0},
    ]
    best = tracker.get_best_metrics()
    # nmse is lower-better → best is the minimum, not the max.
    assert best["best_nmse"] == 0.1
    assert best["best_psnr"] == 30.0


def test_failed_metric_is_omitted_not_zeroed(tmp_path, monkeypatch):
    import torch

    def fake_compute(name, preds, targets, device=None):
        if name == "ssim":
            raise RuntimeError("boom")
        return 0.42

    monkeypatch.setattr(mt_mod, "compute_metric", fake_compute)
    tracker = MetricsTracker(output_dir=str(tmp_path), device="cpu")
    preds = torch.rand(1, 1, 8, 8)
    targets = torch.rand(1, 1, 8, 8)
    out = tracker.compute_image_metrics(preds, targets, metric_names=["psnr", "ssim"])
    assert "ssim" not in out  # omitted, NOT recorded as 0.0
    assert out["psnr"] == 0.42
