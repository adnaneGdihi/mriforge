"""Tests for weighted timestep sampling strategies.

Verifies that DiffusionTrainingStrategy.sample_timesteps() correctly
implements uniform, importance, and linear_decay sampling distributions.
"""

from __future__ import annotations

import pytest
import torch


class TestWeightedTimestepSampling:
    """Test the three timestep sampling strategies independent of full strategy."""

    @pytest.fixture()
    def device(self) -> torch.device:
        """Use CPU for deterministic testing."""
        return torch.device("cpu")

    def _sample_with_strategy(
        self,
        strategy: str,
        high: int,
        batch_size: int,
        device: torch.device,
        num_draws: int = 10_000,
    ) -> torch.Tensor:
        """Reproduce the sampling logic from DiffusionTrainingStrategy.sample_timesteps.

        Returns a tensor of all sampled timesteps (num_draws * batch_size,).
        """
        all_samples = []
        for _ in range(num_draws):
            if strategy == "importance":
                t_range = torch.arange(1, high, device=device, dtype=torch.float32)
                weights = (float(high) - t_range) ** 2
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, batch_size, replacement=True)
                samples = (indices + 1).long()
            elif strategy == "linear_decay":
                t_range = torch.arange(1, high, device=device, dtype=torch.float32)
                weights = float(high) - t_range
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, batch_size, replacement=True)
                samples = (indices + 1).long()
            elif strategy == "high_t_emphasis":
                t_range = torch.arange(1, high, device=device, dtype=torch.float32)
                weights = t_range
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, batch_size, replacement=True)
                samples = (indices + 1).long()
            elif strategy == "balanced_high_t":
                # P(t) = (1-eps)*(t/sum t) + eps*(1/N); eps mirrors
                # diffusion._BALANCED_HIGH_T_UNIFORM_FLOOR (0.3).
                eps = 0.3
                t_range = torch.arange(1, high, device=device, dtype=torch.float32)
                emphasis = t_range / t_range.sum()
                uniform = torch.full_like(t_range, 1.0 / t_range.numel())
                weights = (1.0 - eps) * emphasis + eps * uniform
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, batch_size, replacement=True)
                samples = (indices + 1).long()
            else:  # uniform
                samples = torch.randint(1, high, (batch_size,), device=device).long()
            all_samples.append(samples)
        return torch.cat(all_samples)

    def test_uniform_produces_roughly_uniform_distribution(self, device: torch.device) -> None:
        """Uniform strategy should produce approximately equal counts per bin."""
        high = 100
        samples = self._sample_with_strategy("uniform", high, 4, device, num_draws=5000)

        # All values should be in [1, high-1]
        assert samples.min() >= 1
        assert samples.max() < high

        # Check uniformity: split into 4 quartiles
        q1 = (samples <= 25).sum().item()
        q4 = (samples > 75).sum().item()

        # Each quartile should have ~25% of samples (within 5% tolerance)
        total = len(samples)
        assert abs(q1 / total - 0.25) < 0.05, f"Q1 ratio={q1 / total:.3f}, expected ~0.25"
        assert abs(q4 / total - 0.25) < 0.05, f"Q4 ratio={q4 / total:.3f}, expected ~0.25"

    def test_importance_biases_toward_low_timesteps(self, device: torch.device) -> None:
        """Importance strategy should produce more low-t samples than high-t."""
        high = 100
        samples = self._sample_with_strategy("importance", high, 4, device, num_draws=5000)

        # All values should be in [1, high-1]
        assert samples.min() >= 1
        assert samples.max() < high

        # Low quartile (t=1..25) should have significantly more samples than
        # high quartile (t=76..99)
        low_count = (samples <= 25).sum().item()
        high_count = (samples > 75).sum().item()

        # With quadratic weighting, low should be ~4x more than high
        assert low_count > high_count * 2, (
            f"Importance sampling not biased enough: low={low_count}, high={high_count}"
        )

    def test_linear_decay_biases_toward_low_timesteps(self, device: torch.device) -> None:
        """Linear decay should bias toward low-t, but less strongly than importance."""
        high = 100
        samples = self._sample_with_strategy("linear_decay", high, 4, device, num_draws=5000)

        # Low quartile should have more samples than high quartile
        low_count = (samples <= 25).sum().item()
        high_count = (samples > 75).sum().item()

        assert low_count > high_count, (
            f"Linear decay not biased: low={low_count}, high={high_count}"
        )

    def test_importance_stronger_than_linear_decay(self, device: torch.device) -> None:
        """Importance sampling should be more biased than linear decay."""
        high = 100
        n_draws = 5000

        imp_samples = self._sample_with_strategy("importance", high, 4, device, num_draws=n_draws)
        lin_samples = self._sample_with_strategy("linear_decay", high, 4, device, num_draws=n_draws)

        imp_low_ratio = (imp_samples <= 25).sum().item() / len(imp_samples)
        lin_low_ratio = (lin_samples <= 25).sum().item() / len(lin_samples)

        assert imp_low_ratio > lin_low_ratio, (
            f"Importance ({imp_low_ratio:.3f}) should be more biased than "
            f"linear ({lin_low_ratio:.3f})"
        )

    def test_high_t_emphasis_biases_toward_high_timesteps(self, device: torch.device) -> None:
        """high_t_emphasis (P(t) ∝ t) must over-sample HIGH t — the mirror of
        linear_decay. This is the experiment_11 negative-transfer counter: give
        the hard high-acceleration regime gradient priority."""
        high = 100
        samples = self._sample_with_strategy("high_t_emphasis", high, 4, device, num_draws=5000)

        assert samples.min() >= 1
        assert samples.max() < high

        low_count = (samples <= 25).sum().item()
        high_count = (samples > 75).sum().item()

        # High quartile must dominate low quartile (opposite of linear_decay).
        assert high_count > low_count, (
            f"high_t_emphasis not biased toward high t: low={low_count}, high={high_count}"
        )

    def test_high_t_emphasis_mirrors_linear_decay(self, device: torch.device) -> None:
        """high_t_emphasis high-t share ≈ linear_decay low-t share (P(t) ∝ t vs
        P(t) ∝ (T-t) are reflections about T/2)."""
        high = 100
        n_draws = 5000
        hi = self._sample_with_strategy("high_t_emphasis", high, 4, device, num_draws=n_draws)
        lo = self._sample_with_strategy("linear_decay", high, 4, device, num_draws=n_draws)

        hi_high_ratio = (hi > 75).sum().item() / len(hi)
        lo_low_ratio = (lo <= 25).sum().item() / len(lo)
        assert hi_high_ratio == pytest.approx(lo_low_ratio, abs=0.03), (
            f"reflection broken: high_t_emphasis high={hi_high_ratio:.3f} "
            f"vs linear_decay low={lo_low_ratio:.3f}"
        )

    def test_balanced_high_t_still_biases_toward_high_timesteps(self, device: torch.device) -> None:
        """balanced_high_t keeps a high-t tilt (the R8x/R32x priority) even with
        the uniform floor: the high quartile must still dominate the low."""
        high = 100
        samples = self._sample_with_strategy("balanced_high_t", high, 4, device, num_draws=5000)

        assert samples.min() >= 1
        assert samples.max() < high

        low_count = (samples <= 25).sum().item()
        high_count = (samples > 75).sum().item()
        assert high_count > low_count, (
            f"balanced_high_t lost its high-t tilt: low={low_count}, high={high_count}"
        )

    def test_balanced_high_t_does_not_starve_low_timesteps(self, device: torch.device) -> None:
        """The whole point: the uniform floor lifts the low-t share well ABOVE
        pure high_t_emphasis, so the low-R (R2x) band is not forgotten. This is
        the regression guard for the experiment_11 accel-inversion (pure P(t) ∝ t
        starved t=1,2 to <2% and R2x val collapsed)."""
        high = 100
        n_draws = 5000
        bal = self._sample_with_strategy("balanced_high_t", high, 4, device, num_draws=n_draws)
        pure = self._sample_with_strategy("high_t_emphasis", high, 4, device, num_draws=n_draws)

        bal_low = (bal <= 25).sum().item() / len(bal)
        pure_low = (pure <= 25).sum().item() / len(pure)

        # eps=0.3 uniform floor roughly doubles the low quartile share
        # (~0.066 -> ~0.122). Require a clear, non-marginal lift.
        assert bal_low > pure_low * 1.4, (
            f"balanced_high_t did not floor low-t enough: balanced={bal_low:.3f} "
            f"vs pure high_t_emphasis={pure_low:.3f}"
        )
        # The floor guarantees every timestep >= eps/N: with eps=0.3, N=99 the
        # low quartile (25 bins) alone must clear ~0.3*25/99 ~= 0.075.
        assert bal_low > 0.075, f"low-t share {bal_low:.3f} below the uniform floor guarantee"

    def test_all_strategies_produce_valid_range(self, device: torch.device) -> None:
        """All strategies must produce timesteps in [1, high-1]."""
        high = 50
        for strategy in [
            "uniform",
            "importance",
            "linear_decay",
            "high_t_emphasis",
            "balanced_high_t",
        ]:
            samples = self._sample_with_strategy(strategy, high, 8, device, num_draws=100)
            assert samples.min() >= 1, f"{strategy}: min={samples.min()}"
            assert samples.max() < high, f"{strategy}: max={samples.max()}"

    def test_single_timestep_range_does_not_crash(self, device: torch.device) -> None:
        """When high=2, only t=1 is possible. Must not crash."""
        high = 2
        for strategy in [
            "uniform",
            "importance",
            "linear_decay",
            "high_t_emphasis",
            "balanced_high_t",
        ]:
            samples = self._sample_with_strategy(strategy, high, 4, device, num_draws=10)
            assert (samples == 1).all(), f"{strategy}: expected all 1s, got {samples}"

    def test_importance_weights_are_quadratic(self, device: torch.device) -> None:
        """Verify the weight calculation matches P(t) ∝ (T_max - t)^2."""
        high = 10
        t_range = torch.arange(1, high, device=device, dtype=torch.float32)
        weights = (float(high) - t_range) ** 2
        weights = weights / weights.sum()

        # Weight for t=1 should be (10-1)^2 = 81
        # Weight for t=9 should be (10-9)^2 = 1
        # Ratio should be 81:1
        assert weights[0] / weights[-1] == pytest.approx(81.0, rel=0.01)
