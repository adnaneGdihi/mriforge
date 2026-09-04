"""Welford's online algorithm for efficient single-pass statistics computation.

This module provides memory-efficient computation of running statistics (mean,
variance, std) without materializing all data at once or requiring multiple passes.

Key advantage: O(N) memory and O(N) time complexity instead of O(4N).
"""

from __future__ import annotations

import torch


class WelfordStatistics:
    """Compute mean and variance using Welford's online algorithm.

    This approach computes statistics in a single pass with O(1) memory overhead,
    avoiding the need to store all data or recompute statistics multiple times.

    Mathematical foundation:
    - Uses parallel update formulas for mean and M2 (sum of squared differences)
    - Numerically stable even for large datasets
    - Variance = M2 / (count - 1) for unbiased estimate
    - Can be parallelized across batches by aggregating counts, means, and M2s

    References:
        Welford, B. P. (1962). "Note on a Method for Calculating Corrected
        Sums of Squares and Products." Technometrics, 4(3), 419–420.
    """

    def __init__(self) -> None:
        """Initialize Welford statistics accumulator."""
        self.count = 0.0
        self.mean: torch.Tensor | None = None
        self.M2: torch.Tensor | None = None

    def update(self, x: torch.Tensor) -> None:
        """Update statistics with new batch of data.

        Args:
            x: Batch of data with shape [batch_size, ...].
               Mean is computed over batch dimension.

        """
        if x.numel() == 0:
            return

        batch_count = float(x.shape[0])

        if self.count == 0:
            # First batch - initialize accumulators
            self.count = batch_count
            self.mean = x.mean(dim=0).clone()
            # M2 is sum of squared deviations from mean
            self.M2 = ((x - self.mean) ** 2).sum(dim=0)
        else:
            # Update using Welford's online algorithm
            # Process samples sequentially within the batch
            for sample in x:
                self.count += 1
                assert self.mean is not None and self.M2 is not None
                delta = sample - self.mean
                self.mean = self.mean + delta / self.count
                delta2 = sample - self.mean
                self.M2 = self.M2 + delta * delta2

    def mean_value(self) -> torch.Tensor | None:
        """Return the running mean.

        Returns:
            Mean tensor or None if no data has been accumulated.

        """
        return self.mean.clone() if self.mean is not None else None

    def variance(self, unbiased: bool = True) -> torch.Tensor | None:
        """Return the running variance.

        Args:
            unbiased: If True, compute unbiased variance (divide by n-1).
                     If False, compute biased variance (divide by n).

        Returns:
            Variance tensor or None if insufficient data.

        """
        if self.M2 is None or self.count < (2 if unbiased else 1):
            return None

        if unbiased:
            return self.M2 / (self.count - 1)
        else:
            return self.M2 / self.count

    def std(self, unbiased: bool = True) -> torch.Tensor | None:
        """Return the running standard deviation.

        Args:
            unbiased: If True, use unbiased variance estimate.
                     If False, use biased variance estimate.

        Returns:
            Standard deviation tensor or None if insufficient data.

        """
        var = self.variance(unbiased=unbiased)
        return torch.sqrt(var) if var is not None else None

    def count_value(self) -> float:
        """Return number of samples accumulated."""
        return self.count

    def aggregate(self, other: WelfordStatistics) -> None:
        """Aggregate statistics from another WelfordStatistics instance.

        This enables parallel computation across multiple workers.

        Args:
            other: Another WelfordStatistics instance to aggregate.

        """
        if other.count == 0:
            return

        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.clone() if other.mean is not None else None
            self.M2 = other.M2.clone() if other.M2 is not None else None
            return

        # Parallel aggregation formula (Chan et al., 1979)
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        delta = other.mean - self.mean
        self.mean = self.mean + delta * other.count / (self.count + other.count)

        delta2 = other.mean - self.mean
        self.M2 = (
            self.M2
            + other.M2
            + delta * delta2 * self.count * other.count / (self.count + other.count)
        )
        self.count += other.count


__all__ = ["WelfordStatistics"]
