"""Regression: ``_summarise_best_metrics_from_csv`` must use the metric-direction
SSOT, not a 6-substring guess.

The pre-fix ``higher_is_better_substrings`` was ``(psnr, ssim, accuracy, dice,
f1, iou)`` — so every other higher-is-better metric (snr, tsnr, cnr, vif, fsim,
uqi, vnr, pearson, …) got ``min`` treatment and its WORST value was reported as
"best". That summary feeds ``train_and_score`` → ablation/HPO model selection,
so the error propagates into which arm "wins". Direction now comes from
``core.metrics.metric_directions.METRIC_HIGHER_IS_BETTER``.
"""

from __future__ import annotations

from mriforge.pipelines.train import _summarise_best_metrics_from_csv


def test_summarise_best_respects_metric_registry_direction(tmp_path):
    csv_path = tmp_path / "training_metrics.csv"
    csv_path.write_text(
        "iteration,snr,cnr,vif,mse,g_total_loss\n"
        "1,10,5,0.4,0.5,2.0\n"
        "2,30,8,0.9,0.1,1.0\n"
        "3,20,6,0.7,0.3,1.5\n"
    )

    summary = _summarise_best_metrics_from_csv(str(csv_path))

    # Higher-is-better quality metrics → max
    assert summary["snr_best"] == 30.0
    assert summary["cnr_best"] == 8.0
    assert summary["vif_best"] == 0.9
    # Lower-is-better → min (loss columns fall back to the lower-default heuristic)
    assert summary["mse_best"] == 0.1
    assert summary["g_total_loss_best"] == 1.0


def test_summarise_best_handles_prefixed_columns(tmp_path):
    csv_path = tmp_path / "training_metrics.csv"
    csv_path.write_text(
        "iteration,val_psnr_4x,train_lpips\n"
        "1,25.0,0.30\n"
        "2,31.0,0.10\n"
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))
    assert summary["val_psnr_4x_best"] == 31.0  # psnr → higher
    assert summary["train_lpips_best"] == 0.10  # lpips → lower
