"""Re-export module so ``training.strategy_class`` paths pointing at
``spectramr.infrastructure.training.strategies.validation_badge_strategy.ValidationBadgeStrategy``
resolve. Implementation lives in :mod:`conformal_calibration_strategy`.
"""

from .conformal_calibration_strategy import ValidationBadgeStrategy

__all__ = ["ValidationBadgeStrategy"]
