"""The measurement-aware metrics seam on the cold-diffusion validation path.

``nse_hall`` and ``ndcr`` have been registered with ``needs=("mask", ...)``
since the trust-functional work, but no training-validation path ever built
the ``MetricContext`` they read, so an arm that listed them got ``nan``
(cohort review 2026-09-02, rule 16: existing capability, unwired).
"""

from __future__ import annotations

import math

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy


class _Computer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def compute(self, pred, target, *, only=None, **kwargs):
        self.calls.append({"pred": pred, "target": target, "only": only, **kwargs})
        return {"nse_hall": 0.25, "ndcr": 0.01}


def _host() -> DiffusionTrainingStrategy:
    host = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
    host._select_batch_compatible_smaps = lambda batch: None  # single-coil surrogate
    return host


def _interleaved(k: torch.Tensor) -> torch.Tensor:
    """Complex ``[B,C,H,W]`` -> real/imag interleaved ``[B,2C,H,W]`` (the model's layout)."""
    return torch.cat([k.real, k.imag], dim=1)


def _batch(mask_fraction: float = 0.5):
    torch.manual_seed(0)
    b, c, h, w = 2, 1, 16, 16
    target = torch.randn(b, c, h, w, dtype=torch.complex64)
    pred = target + 0.3 * torch.randn(b, c, h, w, dtype=torch.complex64)
    lines = torch.zeros(1, 1, 1, w)
    lines[..., : int(w * mask_fraction)] = 1.0
    mask = lines.expand(1, 1, h, w)
    measured = target * mask
    return pred, target, measured, mask


def test_the_seam_builds_the_context_from_the_measurement_support() -> None:
    """The fires-test: ``only`` names the two metrics, the mask is the measured support."""
    pred, target, measured, mask = _batch()
    computer = _Computer()
    out = _host()._measurement_aware_metrics(
        _interleaved(pred), _interleaved(target), _interleaved(measured), computer
    )
    assert out == {"nse_hall": 0.25, "ndcr": 0.01}
    (call,) = computer.calls
    assert call["only"] == ("nse_hall", "ndcr")
    ctx = call["context"]
    assert isinstance(ctx, MetricContext)
    assert ctx.mask.shape == (2, 1, 16, 16)
    assert torch.equal(ctx.mask[0, 0], mask[0, 0])
    assert torch.is_complex(ctx.y_kspace) and ctx.y_kspace.shape == (2, 1, 16, 16)
    assert torch.is_complex(call["pred"]) and call["pred"].shape == (2, 1, 16, 16)


def test_the_registered_metrics_score_on_that_context() -> None:
    """End to end on the real metric classes: finite, and nse_hall inside [0, 1]."""
    from spectramr.core.metrics.nr_consistency import (
        NormalisedDataConsistencyResidual,
        NullSpaceEnergyHallucination,
    )
    from spectramr.infrastructure.physics.fft_ops import sense_adjoint

    pred, target, measured, mask = _batch()
    ctx = MetricContext(
        mask=mask.expand(2, 1, 16, 16).contiguous(), y_kspace=measured, coil_maps=None
    )
    pred_img, target_img = sense_adjoint(pred), sense_adjoint(target)
    nse = NullSpaceEnergyHallucination()(pred_img, target_img, context=ctx)
    ndcr = NormalisedDataConsistencyResidual()(pred_img, context=ctx)
    assert math.isfinite(nse) and 0.0 <= nse <= 1.0
    assert math.isfinite(ndcr) and ndcr >= 0.0


def test_without_a_context_the_metrics_are_nan() -> None:
    """Planted control: this is what every arm got before the seam existed."""
    from spectramr.core.metrics.nr_consistency import NullSpaceEnergyHallucination

    pred, target, _, _ = _batch()
    assert math.isnan(NullSpaceEnergyHallucination()(pred, target))
