"""Brain-pathology classification downstream adapter (PR-5 / H3).

Wraps a pre-trained classifier (e.g. ResNet-18 fine-tuned on
infarct/meningioma/hemorrhage). All metric arithmetic delegates to the
SSOT modules under ``src/core/metrics/``:

- :class:`spectramr.core.metrics.classification.ClassificationEvaluator` —
  accuracy / precision / recall / F1 (macro-averaged for multi-class).
- ``detection_sensitivity`` / ``detection_specificity`` — registered in
  :mod:`spectramr.core.metrics.report_step_metrics` and resolved through the
  metrics registry. We surface a sensitivity / specificity pair per
  class via the registry, rather than re-implementing them here.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.core.metrics.classification import ClassificationEvaluator
from spectramr.core.metrics.registry import get_metric

from .segmentation_adapter import _to_magnitude


class BrainClassificationAdapter(nn.Module):
    """Frozen pathology classifier adapter.

    Routes all metric computation through the metrics SSOT —
    :class:`ClassificationEvaluator` for accuracy / precision / recall /
    F1, and the metrics registry's ``detection_sensitivity`` /
    ``detection_specificity`` for per-class sensitivity / specificity
    on a one-vs-rest binarisation.
    """

    def __init__(
        self,
        classifier: nn.Module,
        *,
        task_name: str = "brain_pathology_clf",
        num_classes: int = 3,
        average: str = "macro",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.classifier = classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False
        self._task_name = task_name
        self._num_classes = num_classes
        self._device = device or torch.device("cpu")
        self._evaluator = ClassificationEvaluator(num_classes=num_classes, average=average)
        # Per-class binary sensitivity / specificity via the SSOT registry.
        # We keep one DetectionSensitivity / DetectionSpecificity instance
        # so the adapter does no metric arithmetic of its own.
        self._sens = get_metric("detection_sensitivity", threshold=0.5, device=self._device)
        self._spec = get_metric("detection_specificity", threshold=0.5, device=self._device)

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    @torch.no_grad()
    def forward(self, reconstruction: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = _to_magnitude(reconstruction).to(self._device)
        return self.classifier(x)

    @torch.no_grad()
    def compute_metric(
        self,
        reconstruction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        logits = self.forward(reconstruction)
        gt = ground_truth.to(self._device).long()

        # SSOT: accuracy / precision / recall / F1.
        results: dict[str, Any] = dict(self._evaluator(logits, gt))

        # Per-class sensitivity / specificity via the metrics registry
        # (one-vs-rest). Recall = sensitivity in the binary case.
        if logits.dim() == 2 and logits.shape[1] == self._num_classes:
            pred = logits.argmax(dim=1)
            sens_acc = 0.0
            spec_acc = 0.0
            for c in range(self._num_classes):
                pred_c = (pred == c).float()
                gt_c = (gt == c).float()
                s = self._sens.compute_metric(pred_c, gt_c)
                sp = self._spec.compute_metric(pred_c, gt_c)
                # Skip classes with no positive / negative samples (NaN).
                if not torch.isnan(s):
                    sens_acc += float(s.item())
                if not torch.isnan(sp):
                    spec_acc += float(sp.item())
            sens_macro = sens_acc / self._num_classes
            spec_macro = spec_acc / self._num_classes
            results["sensitivity"] = sens_macro
            results["specificity"] = spec_macro
            results["balanced_accuracy"] = 0.5 * (sens_macro + spec_macro)
        results["task"] = self._task_name
        results["n_classes"] = self._num_classes
        return results
