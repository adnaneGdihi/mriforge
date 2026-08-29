r"""Split-conformal calibrator (PR-3 / H1).

Implements the standard split-conformal procedure:

1. Compute non-conformity scores on a held-out calibration set.
2. Take the empirical :math:`\lceil (n + 1)(1 - \alpha) \rceil / n`
   quantile of those scores. (The :math:`(n+1)/n` correction is what
   gives the *finite-sample* coverage guarantee.)
3. At test time, return ``(y_hat, y_hat - q, y_hat + q)``.

Theorem (Vovk-Gammerman-Shafer 2005). When the calibration scores are
exchangeable with the test scores (typical i.i.d. assumption),

.. math::

    \mathbb{P}(y \in [\hat y - q,\ \hat y + q]) \geq 1 - \alpha

with no further distributional assumptions on either the model or the
data. The prediction band is finite-sample valid; no asymptotics
required.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class IConformalCalibrator(Protocol):
    """Calibrator protocol used by reporting / strategy layers.

    Anyone wrapping an alternative method (CQR, full-conformal,
    cross-conformal, jackknife+) only needs to satisfy this surface.
    """

    @property
    def alpha(self) -> float:
        """Mis-coverage rate (lower → wider bands)."""
        ...

    @property
    def is_fitted(self) -> bool:
        """``True`` after :py:meth:`fit` has been called."""
        ...

    def fit(self, calibration_pairs: Iterable[Any]) -> None:
        """Compute the empirical quantile from a calibration set."""
        ...

    def predict(self, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(y_hat, lower, upper)`` for a fresh prediction."""
        ...


# Type alias: the score function consumes ``(prediction, target) -> per-element score``.
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ConformalCalibrator:
    """Standard (Vovk-Gammerman-Shafer) split-conformal calibrator.

    Args:
        score_fn: Non-conformity score; defaults to absolute residual.
        alpha: Mis-coverage rate ``α ∈ (0, 1)``. Smaller → wider bands.

    Raises:
        ValueError: when ``alpha`` is outside ``(0, 1)``.
    """

    def __init__(
        self,
        score_fn: ScoreFn | None = None,
        alpha: float = 0.1,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha must lie in (0, 1), got {alpha!r}. "
                "(Common choices: 0.05 → 95% coverage, 0.1 → 90%.)"
            )
        # Defer importing the default to module body to avoid a circular
        # import if a caller monkey-patches `scores`.
        if score_fn is None:
            from .scores import absolute_residual

            score_fn = absolute_residual
        self._score_fn: ScoreFn = score_fn
        self._alpha = float(alpha)
        self._quantile: float | None = None
        self._n_calibration: int = 0

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        """Mis-coverage rate."""
        return self._alpha

    @property
    def is_fitted(self) -> bool:
        """``True`` after :py:meth:`fit` has been called."""
        return self._quantile is not None

    @property
    def quantile(self) -> float:
        """The fitted band radius (raises if unfit)."""
        if self._quantile is None:
            raise RuntimeError("ConformalCalibrator has not been fitted. Call fit(...) first.")
        return self._quantile

    @property
    def n_calibration(self) -> int:
        """Number of calibration *examples* (post-flatten) used by the last ``fit``."""
        return self._n_calibration

    def fit(
        self,
        calibration_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> ConformalCalibrator:
        """Compute the empirical quantile from ``(prediction, target)`` pairs.

        Args:
            calibration_pairs: Iterable yielding ``(prediction, target)``
                tensor pairs. Predictions and targets are flattened
                across batch + spatial dims, so a single pair may
                contribute many calibration samples.

        Returns:
            Self (so the call chains: ``calibrator.fit(loader).predict(x)``).

        Raises:
            ValueError: when no calibration samples are provided.
        """
        scores: list[torch.Tensor] = []
        for pred, target in calibration_pairs:
            scores.append(self._score_fn(pred, target).flatten())
        if not scores:
            raise ValueError(
                "Calibration set is empty. fit() needs at least one (prediction, target) pair."
            )
        all_scores = torch.cat(scores, dim=0).double()
        n = all_scores.numel()
        self._n_calibration = int(n)
        # Finite-sample split-conformal band radius is the k-th *order
        # statistic* of the calibration scores, with k = ceil((n+1)(1-α))
        # (Vovk-Gammerman-Shafer; Lei et al. 2018). Two corrections over
        # the previous implementation:
        #   1. (Coverage-critical.) When k > n — which happens whenever
        #      n < 1/α - 1, e.g. n=5, α=0.1 — no finite band achieves 1-α
        #      coverage. The honest radius is +∞. The old code clamped the
        #      rank to 1.0 and returned the max score, advertising a
        #      guarantee the band does not hold (pitfalls #9/#20, a
        #      degenerate silent fallback). This is the anti-conservative
        #      bug the +∞ branch fixes.
        #   2. (Definitional.) Use the exact nearest-rank order statistic,
        #      not torch.quantile. torch.quantile linearly interpolates at
        #      fractional index k/n·(n-1) ≥ k-1, so it returns a value
        #      between the k-th and (k+1)-th order statistics — mildly
        #      conservative, but not the textbook split-conformal estimator.
        #      sorted[k-1] is the canonical choice.
        k = math.ceil((n + 1) * (1.0 - self._alpha))
        if k > n:
            self._quantile = float("inf")
        else:
            sorted_scores, _ = torch.sort(all_scores)
            self._quantile = float(sorted_scores[k - 1].item())
        return self

    def predict(
        self,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(y_hat, lower, upper)`` for a fresh prediction.

        Args:
            prediction: A point-estimate tensor.

        Returns:
            ``(y_hat, lower, upper)`` where ``lower = y_hat - q`` and
            ``upper = y_hat + q`` for the fitted band radius ``q``.

        Raises:
            RuntimeError: when called before :py:meth:`fit`.
        """
        q = self.quantile  # raises if unfit
        return prediction, prediction - q, prediction + q

    def empirical_coverage(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Fraction of ``target`` lying within the band — a sanity check.

        Should be ``≥ 1 - α`` on a fresh test split (marginal coverage).
        """
        _, lower, upper = self.predict(prediction)
        inside = (target >= lower) & (target <= upper)
        return float(inside.float().mean().item())


__all__ = ["ConformalCalibrator", "IConformalCalibrator"]
