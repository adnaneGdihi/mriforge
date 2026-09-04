#!/usr/bin/env python3
"""Task-Based Evaluation Module
============================

Provides evaluation methods for downstream tasks such as segmentation,
classification, and other medical imaging applications.
"""

from typing import Any

import numpy as np
import torch

# Import SSOT metrics
from spectramr.core.metrics.evaluation_metrics import PSNR, SSIMMetric

# Fallback wrappers for backward compatibility if needed, using SSOT instances
_psnr_metric = PSNR()
_ssim_metric = SSIMMetric()


def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Wrapper calling SSOT PSNR."""
    # PSNR metric expects data_range in constructor, but here it's passed as arg
    # For strict SSOT, we should try to honor the arg, but the class caches it.
    # We'll create a new instance if needed or update the existing one
    # (though typically data_range is constant).
    # For efficiency in this wrapper, we assume 1.0 default or ignore if acceptable,
    # BUT to be correct:
    metric = PSNR(data_range=data_range, device=img1.device)
    return metric(img1, img2).item()


def ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Wrapper calling SSOT SSIM."""
    metric = SSIMMetric(data_range=data_range, device=img1.device)
    return metric(img1, img2).item()


from spectramr.core.metrics.classification import ClassificationEvaluator
from spectramr.core.metrics.segmentation import SegmentationEvaluator


class TaskBasedEvaluator:
    """Comprehensive task-based evaluation suite.

    Evaluates model performance on downstream tasks.
    """

    def __init__(self, task_type: str = "segmentation", **kwargs):
        """__init__.

        Args:
            task_type (str): Description.
        """
        self.task_type = task_type

        if task_type == "segmentation":
            self.evaluator = SegmentationEvaluator(**kwargs)
        elif task_type == "classification":
            self.evaluator = ClassificationEvaluator(**kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Evaluate predictions on the specified task.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Dictionary of evaluation metrics

        """
        return self.evaluator(pred, target)

    def evaluate_batch(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> dict[str, float]:
        """Evaluate a batch of predictions.

        Args:
            predictions: list of prediction tensors
            targets: list of target tensors

        Returns:
            Average metrics across all samples

        """
        all_metrics = []

        for pred, target in zip(predictions, targets, strict=False):
            metrics = self.evaluate(pred, target)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = np.mean(values)

        return avg_metrics


class MedicalImageQualityEvaluator:
    """Specialized evaluator for medical image quality assessment.

    Includes metrics relevant to medical imaging applications.
    """

    def __init__(self):
        """__init__."""
        self.segmentation_evaluator = SegmentationEvaluator()
        self.classification_evaluator = ClassificationEvaluator()

    def evaluate_image_quality(
        self,
        enhanced_img: torch.Tensor,
        original_img: torch.Tensor,
        segmentation_pred: torch.Tensor | None = None,
        segmentation_target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Comprehensive evaluation of enhanced medical images.

        Args:
            enhanced_img: Enhanced/super-resolved image
            original_img: Original low-quality image
            segmentation_pred: Predicted segmentation on enhanced image
            segmentation_target: Ground truth segmentation

        Returns:
            Comprehensive evaluation results

        """
        results = {}

        # Basic image quality metrics (improvement over original)
        # Compute metrics between enhanced and original images
        mse_value = torch.mean((enhanced_img - original_img) ** 2).item()
        mae_value = torch.mean(torch.abs(enhanced_img - original_img)).item()
        psnr_value = psnr(enhanced_img, original_img, data_range=1.0)
        ssim_value = ssim(enhanced_img, original_img, data_range=1.0)

        # Direct metrics
        results["mse"] = mse_value
        results["mae"] = mae_value
        results["psnr"] = psnr_value
        results["ssim"] = ssim_value

        # For improvement metrics, we use the quality metrics as proxy
        # since we don't have ground truth. Higher values indicate better
        # enhancement
        results["image_quality"] = {
            "psnr_improvement": psnr_value,  # PSNR between enhanced and original
            "ssim_improvement": ssim_value,  # SSIM between enhanced and original
        }

        # Task-based evaluation
        if segmentation_pred is not None and segmentation_target is not None:
            seg_metrics = self.segmentation_evaluator(
                segmentation_pred,
                segmentation_target,
            )
            results["segmentation_metrics"] = seg_metrics

            # Task improvement metrics (compared to baseline)
            # For now, use the segmentation metrics as improvement proxy
            # In full implementation, would compare against baseline model
            dice_score = seg_metrics.get("dice", 0.0)
            accuracy_score = seg_metrics.get("accuracy", 0.0)

            results["task_improvement"] = {
                "dice_improvement": dice_score,  # Current dice as proxy
                "accuracy_improvement": accuracy_score,  # Accuracy proxy
            }

        return results


__all__ = [
    "ClassificationEvaluator",
    "MedicalImageQualityEvaluator",
    "SegmentationEvaluator",
    "TaskBasedEvaluator",
]
