"""Stein-discrepancy utilities for distributional goodness-of-fit testing.

Used by Idea 7 (KSD defensibility audit). See ``ksd.py`` for the
kernelised-Stein-discrepancy U-statistic estimator with optional
wild-bootstrap p-values.
"""

from .ksd import KSDResult, kernelised_stein_discrepancy, ksd_p_value

__all__ = ["KSDResult", "kernelised_stein_discrepancy", "ksd_p_value"]
