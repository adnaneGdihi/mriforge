"""Segmentation Metrics Module.

Provides evaluation metrics for segmentation tasks, including Dice score
Jaccard index (IoU), and pixel-level accuracy/precision/recall.
"""

import torch

try:  # optional in degraded cluster envs (a REQUIRED dep in pyproject)
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        jaccard_score,
        precision_score,
        recall_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    accuracy_score = f1_score = jaccard_score = None  # type: ignore[assignment]
    precision_score = recall_score = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


def _require_sklearn() -> None:
    """Raise a clear error when a metric is USED without scikit-learn.

    scikit-learn is a REQUIRED dependency in pyproject; this guard only
    fires in a degraded cluster env. Raising at use-time (not import-time)
    keeps the module importable so walk-discovery still fires the rest of
    the metric registrations.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "This segmentation metric requires scikit-learn (pip install scikit-learn)."
        )


def dice_score(y_true, y_pred, average="binary"):
    """Compute Dice score (F1 score for segmentation)."""
    _require_sklearn()
    if average == "binary":
        return f1_score(y_true, y_pred, average="binary")
    if average == "macro":
        return f1_score(y_true, y_pred, average="macro")
    return f1_score(y_true, y_pred, average=average)


class SegmentationEvaluator:
    """Evaluator for segmentation tasks.

    Computes Dice score, Jaccard index, and other segmentation metrics.

    .. math::

        Dice = \\frac{2 |X \\cap Y|}{|X| + |Y|}

        Jaccard = \\frac{|X \\cap Y|}{|X \\cup Y|} = \\frac{Dice}{2 - Dice}
    """

    def __init__(self, num_classes: int = 2, ignore_index: int | None = -1):
        """__init__.

        Args:
            num_classes (int): Number of segmentation classes
                (binary → 2; otherwise > 2 enables ``*_macro`` outputs).
            ignore_index (int | None): Value in ``target`` to exclude
                from metric computation. Pass ``None`` to disable
                filtering altogether. Defaults to ``-1`` (the common
                PyTorch convention for "ignore"). Negative ints are
                valid and **do** filter; this is the canonical signal
                used by ``nn.CrossEntropyLoss`` for the same purpose.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Evaluate segmentation predictions.

        Args:
            pred: Predicted segmentation masks [B, C, H, W] or [B, H, W]
            target: Ground truth masks [B, H, W]

        Returns:
            Dictionary of evaluation metrics

        """
        _require_sklearn()

        # Convert logits to predictions if needed
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)

        # Flatten predictions and targets
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # Remove ignored pixels. ``ignore_index`` matches the PyTorch
        # convention (any int — including negatives — is a valid sentinel
        # value to drop from the metric). Pass ``None`` to disable.
        if self.ignore_index is not None:
            valid_mask = target_flat != self.ignore_index
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]

        # Convert to numpy for sklearn metrics
        pred_np = pred_flat.cpu().numpy()
        target_np = target_flat.cpu().numpy()

        results = {}

        # Overall accuracy
        results["accuracy"] = accuracy_score(target_np, pred_np)

        # Per-class metrics
        if self.num_classes > 2:
            results["precision_macro"] = precision_score(
                target_np,
                pred_np,
                average="macro",
                zero_division=0,
            )
            results["recall_macro"] = recall_score(
                target_np,
                pred_np,
                average="macro",
                zero_division=0,
            )
            results["f1_macro"] = f1_score(
                target_np,
                pred_np,
                average="macro",
                zero_division=0,
            )
        else:
            results["precision"] = precision_score(target_np, pred_np, zero_division=0)
            results["recall"] = recall_score(target_np, pred_np, zero_division=0)
            results["f1"] = f1_score(target_np, pred_np, zero_division=0)

        # Dice score (F1 score for segmentation)
        if self.num_classes == 2:
            results["dice"] = dice_score(target_np, pred_np)
            results["jaccard"] = jaccard_score(target_np, pred_np)
        else:
            results["dice_macro"] = dice_score(target_np, pred_np, average="macro")
            results["jaccard_macro"] = jaccard_score(
                target_np,
                pred_np,
                average="macro",
            )

        return results
