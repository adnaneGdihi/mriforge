"""Classification Metrics Module.

Implements standard classification evaluation metrics such as
accuracy, precision, recall, and F1-score across various averaging strategies.
"""

import torch

try:  # optional in degraded cluster envs (a REQUIRED dep in pyproject)
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    accuracy_score = f1_score = precision_score = recall_score = None  # type: ignore[assignment]
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
            "This classification metric requires scikit-learn (pip install scikit-learn)."
        )


class ClassificationEvaluator:
    """Evaluator for classification tasks.

    Computes accuracy, precision, recall, and F1 scores.

    .. math::

        Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}

        Precision = \\frac{TP}{TP + FP}

        Recall = \\frac{TP}{TP + FN}

        F1 = 2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall}
    """

    def __init__(self, num_classes: int = 2, average: str = "macro"):
        """__init__.

        Args:
            num_classes (int): Description.
            average (str): Description.
        """
        self.num_classes = num_classes
        valid_averages = ["micro", "macro", "weighted", "binary"]
        # No-silent-fallback (CLAUDE.md #9): an unknown averaging strategy must
        # raise rather than silently degrade to 'macro' (which would report a
        # different, plausible-looking score for a typo'd config).
        if average not in valid_averages:
            raise ValueError(f"Unknown average {average!r}; expected one of {valid_averages}.")
        self.average = average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Evaluate classification predictions.

        Args:
            pred: Predicted logits [B, C] or class indices [B]
            target: Ground truth labels [B]

        Returns:
            Dictionary of evaluation metrics

        """
        _require_sklearn()

        # Convert logits to predictions if needed
        if pred.dim() == 2:
            pred = torch.argmax(pred, dim=1)

        # Convert to numpy
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        results = {}

        # Basic metrics
        results["accuracy"] = accuracy_score(target_np, pred_np)

        if self.num_classes > 2:
            results["precision"] = precision_score(
                target_np,
                pred_np,
                average=self.average,
                zero_division=0,
            )
            results["recall"] = recall_score(
                target_np,
                pred_np,
                average=self.average,
                zero_division=0,
            )
            results["f1"] = f1_score(
                target_np,
                pred_np,
                average=self.average,
                zero_division=0,
            )
        else:
            results["precision"] = precision_score(target_np, pred_np, zero_division=0)
            results["recall"] = recall_score(target_np, pred_np, zero_division=0)
            results["f1"] = f1_score(target_np, pred_np, zero_division=0)

        return results
