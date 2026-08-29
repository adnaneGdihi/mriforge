r"""Calibrated Hallucination Detection (CHD) — ULF-PR-21 / R2.

Implements the calibrated hallucination detector from
``TODO/backlog_ulf_radiologist_acceptance_roadmap.md`` §R2.

Mathematics
-----------
**Theorem R2 (Calibrated FPR).** Let :math:`\hat\tau_h` be the
empirical :math:`(1-\beta)`-quantile of detector scores on :math:`n`
non-hallucinated calibration pixels. With probability
:math:`\ge 1 - \delta` over the calibration draw,

.. math::

    \Pr\bigl[h_\phi(\hat x)(\mathbf r) > \hat\tau_h
        \mid H(\mathbf r) = 0\bigr]
    \;\le\; \beta + \sqrt{\frac{\ln(2/\delta)}{2 n}}.

The bound follows from the Dvoretzky-Kiefer-Wolfowitz inequality
applied to the empirical CDF of the calibration scores.

Usage
-----
.. code-block:: python

    chd = CalibratedHallucinationDetector(beta=0.05, delta=1e-3)
    chd.fit(non_hallucinated_calibration_scores)  # 1-D tensor
    is_hallucination = chd.predict(test_scores)   # bool tensor
    fpr_upper_bound = chd.fpr_upper_bound

The calibrator only consumes *non*-hallucinated pixels' scores —
those are the H = 0 samples needed to bound the false-positive rate.
"""

from __future__ import annotations

import torch

# ``dkw_slack`` is twelve lines of pure ``math`` and one of its consumers lives in
# ``core/`` (trajectory_metrics), so its home is ``core/`` — importing it upward from
# here was a non-negotiable 5 violation invisible to the ``^``-anchored layering grep
# (#1183). Re-exported below so every existing ``from ...chd import dkw_slack`` and
# ``from mriforge.infrastructure.calibration import dkw_slack`` keeps working against
# the single implementation (non-negotiable 17).
from mriforge.core.metrics.dkw import dkw_slack


class CalibratedHallucinationDetector:
    """Threshold a per-pixel hallucination-detector score with a calibrated FPR.

    Args:
        beta: Target false-positive rate ``β ∈ (0, 1)``.
        delta: Confidence ``δ ∈ (0, 1)`` of the bound.

    Raises:
        ValueError: ``β`` or ``δ`` outside ``(0, 1)``.
    """

    def __init__(self, beta: float, delta: float = 1e-3) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must lie in (0, 1); got {beta}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {delta}")
        self._beta = float(beta)
        self._delta = float(delta)
        self._threshold: float | None = None
        self._n_calibration: int = 0

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        """Target false-positive rate."""
        return self._beta

    @property
    def delta(self) -> float:
        """Confidence parameter of the bound."""
        return self._delta

    @property
    def is_fitted(self) -> bool:
        """``True`` after :py:meth:`fit` has been called."""
        return self._threshold is not None

    @property
    def threshold(self) -> float:
        """The calibrated detector threshold ``hat τ_h`` (raises if unfit)."""
        if self._threshold is None:
            raise RuntimeError(
                "CHD has not been fitted; call fit() with non-hallucinated "
                "calibration scores first."
            )
        return self._threshold

    @property
    def n_calibration(self) -> int:
        """Calibration sample count."""
        return self._n_calibration

    @property
    def fpr_upper_bound(self) -> float:
        """``β + √(ln(2/δ) / (2 n))`` — the calibrated false-positive bound."""
        if self._threshold is None:
            raise RuntimeError("CHD has not been fitted; bound is undefined.")
        return self._beta + dkw_slack(self._n_calibration, self._delta)

    @property
    def bound_is_tight(self) -> bool:
        """``True`` iff the DKW slack is at most ``β`` (audit gate condition)."""
        slack = dkw_slack(self._n_calibration, self._delta)
        return slack <= self._beta

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        non_hallucinated_scores: torch.Tensor,
    ) -> CalibratedHallucinationDetector:
        """Compute the empirical ``(1-β)``-quantile of the calibration scores.

        Args:
            non_hallucinated_scores: 1-D real tensor of detector
                outputs at pixels with ground-truth label ``H = 0``.

        Returns:
            Self (fluent chain).

        Raises:
            ValueError: empty calibration set.
        """
        if non_hallucinated_scores.dim() == 0:
            raise ValueError("non_hallucinated_scores must be at least 1-D")
        flat = non_hallucinated_scores.flatten().double()
        n = flat.numel()
        if n == 0:
            raise ValueError("Calibration set is empty.")

        # Empirical (1-β)-quantile.
        rank = 1.0 - self._beta
        rank = min(max(rank, 0.0), 1.0)
        self._threshold = float(torch.quantile(flat, rank).item())
        self._n_calibration = int(n)
        return self

    def predict(
        self,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Boolean mask: ``scores > τ_h`` (per-pixel).

        Raises:
            RuntimeError: when called before :py:meth:`fit`.
        """
        return scores > self.threshold

    def empirical_fpr(
        self,
        non_hallucinated_test_scores: torch.Tensor,
    ) -> float:
        """Compute empirical FPR on a fresh test split (sanity check)."""
        return float(self.predict(non_hallucinated_test_scores).float().mean().item())


__all__ = [
    "CalibratedHallucinationDetector",
    "dkw_slack",
]
