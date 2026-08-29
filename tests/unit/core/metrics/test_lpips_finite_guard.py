"""LPIPS must reject non-finite inputs at the metric boundary.

Regression for the 2026-06-24 ``exp_hm_06`` / ``exp_hm_10`` crash: a diverged
model emitted NaN predictions, the LPIPS normaliser's ``torch.clamp`` passed
them through, and the failure surfaced inside torchmetrics as a misleading
*"values in range [nan, nan] … expected [-1, 1]"* error. LPIPS must now raise a
precise, finite-guard error (naming the model as the cause) **before** reaching
the VGG backbone — so the test does not need torchmetrics installed.
"""

import pytest
import torch

from mriforge.core.metrics.evaluation_metrics import LPIPS


class _ExplodingBackbone(torch.nn.Module):
    """Stand-in for the LPIPS backbone — fails loudly if ever called.

    An ``nn.Module`` so it can be assigned to the metric's registered ``lpips``
    submodule (the real torchmetrics VGG / lpips-package AlexNet backends are both
    ``nn.Module``s, so ``metric.lpips`` is a child module that rejects non-Modules).
    """

    def forward(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("LPIPS backbone must not be reached on NaN input")

    @property
    def device(self):
        return torch.device("cpu")


class TestLpipsFiniteGuard:
    @pytest.mark.unit
    def test_lpips_raises_on_nan_pred_before_backbone(self):
        metric = LPIPS(device="cpu")
        # Force the non-torchmetrics CI path to still exercise compute_metric:
        # a truthy backbone that explodes if the guard fails to short-circuit.
        metric.lpips = _ExplodingBackbone()

        preds = torch.randn(1, 1, 16, 16)
        preds[0, 0, 0, 0] = float("nan")
        target = torch.randn(1, 1, 16, 16)

        with pytest.raises(ValueError, match="non-finite"):
            metric.compute_metric(preds, target)

    @pytest.mark.unit
    def test_lpips_error_attributes_failure_to_the_model(self):
        metric = LPIPS(device="cpu")
        metric.lpips = _ExplodingBackbone()
        preds = torch.full((1, 1, 16, 16), float("inf"))
        target = torch.randn(1, 1, 16, 16)
        with pytest.raises(ValueError) as exc:
            metric.compute_metric(preds, target)
        assert "lpips" in str(exc.value)
        assert "model" in str(exc.value).lower()
