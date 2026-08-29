"""Regression: MetricsTracker in-memory history must be bounded, and the best
metric must survive eviction.

The per-stream ``training_history`` / ``validation_history`` / ``inference_history``
lists grew without bound — a run logging every step accumulated one dict per
call for the whole run (unbounded RSS), and ``get_best_metrics`` rescanned the
entire list on every call. They are now bounded deques, and best-metric tracking
is INCREMENTAL so a best value is not lost when its record is evicted from the
bounded window.
"""

from __future__ import annotations

from collections import deque

from mriforge.infrastructure.services.metrics_tracker import MetricsTracker


def _log_training(tracker: MetricsTracker, *, step: int, **quality) -> None:
    """Log a training record (no images → no auto quality metrics), then inject
    the given quality metrics into the record + the incremental running-best,
    mirroring what compute_image_metrics would have added."""
    tracker.log_training_metrics(
        epoch=0,
        step=step,
        model_type="model",
        learning_rate=1e-4,
        generator_loss=1.0,
        discriminator_loss=1.0,
    )
    tracker.training_history[-1].update(quality)
    tracker._update_running_best(quality)


def test_history_is_bounded_deque(tmp_path):
    tracker = MetricsTracker(output_dir=str(tmp_path), history_maxlen=5)
    assert isinstance(tracker.training_history, deque)
    assert tracker.training_history.maxlen == 5

    for i in range(20):
        _log_training(tracker, step=i, psnr=float(i))

    # Only the last 5 records are retained (bounded memory).
    assert len(tracker.training_history) == 5
    steps = [r["step"] for r in tracker.training_history]
    assert steps == [15, 16, 17, 18, 19]


def test_best_metric_survives_eviction(tmp_path):
    """The best PSNR occurred early and was evicted from the bounded window, but
    the incremental tracker must still report it."""
    tracker = MetricsTracker(output_dir=str(tmp_path), history_maxlen=3)

    _log_training(tracker, step=0, psnr=42.0)  # best, will be evicted
    for i in range(1, 10):
        _log_training(tracker, step=i, psnr=20.0 + i * 0.1)

    # The step-0 record is long gone from the window...
    assert all(r["step"] != 0 for r in tracker.training_history)
    # ...but its PSNR is still the reported best.
    best = tracker.get_best_metrics()
    assert best["best_psnr"] == 42.0


def test_lower_is_better_running_best(tmp_path):
    tracker = MetricsTracker(output_dir=str(tmp_path), history_maxlen=3)
    for i, nmse in enumerate([0.5, 0.05, 0.4, 0.3, 0.2, 0.25]):
        _log_training(tracker, step=i, nmse=nmse)
    # nmse is lower-is-better; the 0.05 (evicted) must remain the best.
    assert tracker.get_best_metrics()["best_nmse"] == 0.05
