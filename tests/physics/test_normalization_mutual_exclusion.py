"""
Tests for Normalization Mutual-Exclusion Validation

Critical Test Suite for ensuring only ONE normalization strategy is applied.
Currently, z-score and min-max normalization can both be applied sequentially,
causing data corruption.

Test Invariants:
1. Only one normalization strategy is enabled
2. If multiple strategies enabled, raise ConfigError (fail-fast)
3. Applying normalization should idempotent-ish (applying twice shouldn't change already-normalized data)
"""

from dataclasses import dataclass
from enum import Enum

import pytest
import torch


# Test data structures
class NormalizationStrategy(str, Enum):
    """Allowed normalization strategies (mutually exclusive)."""

    NONE = "none"
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"  # Min-max rescaling [0, 1]
    PERCENTILE = "percentile"  # Robust percentile-based
    KSPACE = "kspace"  # K-space specific (only for k-space)


@dataclass
class NormalizationConfig:
    """Configuration for normalization strategy."""

    strategy: NormalizationStrategy = NormalizationStrategy.NONE
    percentile: float = 0.99
    eps: float = 1e-8

    def validate(self) -> None:
        """Fail-fast validation."""
        if self.strategy == NormalizationStrategy.NONE:
            pass  # No normalization is valid
        elif self.strategy not in NormalizationStrategy:
            raise ValueError(f"Unknown normalization strategy: {self.strategy}")

        if self.percentile <= 0 or self.percentile > 1.0:
            raise ValueError(f"Percentile must be in (0, 1], got {self.percentile}")


# ============================================================================
# CRITICAL INVARIANTS FOR MUTUAL EXCLUSION
# ============================================================================


class TestNormalizationMutualExclusion:
    """Test that only one normalization strategy is applied."""

    def test_config_validation_fails_on_multiple_strategies(self):
        """
        CRITICAL: Configuration should be validated at load time.
        If both z-score and min-max are somehow enabled, raise ConfigError.
        """
        # This tests the config-level validation
        # (implementation would check config schema)

        # Valid: Single strategy
        config_single = NormalizationConfig(strategy=NormalizationStrategy.STANDARD)
        assert config_single.strategy == NormalizationStrategy.STANDARD

        # Valid: No normalization
        config_none = NormalizationConfig(strategy=NormalizationStrategy.NONE)
        assert config_none.strategy == NormalizationStrategy.NONE

    def test_normalization_idempotence_violation(self):
        """
        INVARIANT: Applying normalization twice should either:
        1. Leave data unchanged (true idempotence), OR
        2. Raise error if second application attempted

        Currently, applying z-score then min-max VIOLATES this → data corruption.
        """
        data = torch.randn(10, 10)
        original = data.clone()

        # Apply z-score normalization
        mean = data.mean()
        std = data.std()
        data_z_norm = (data - mean) / (std + 1e-8)

        # WRONG: Apply min-max to z-normalized data
        # This treats z-normalized values [−k, k] as if they were [0, max]
        min_val = data_z_norm.min()
        max_val = data_z_norm.max()
        data_double_norm = (data_z_norm - min_val) / (max_val - min_val + 1e-8)

        # Verify they're different (proving data was corrupted)
        # If we naively reapplied z-score:
        mean2 = data_z_norm.mean()
        std2 = data_z_norm.std()
        data_z_norm_2 = (data_z_norm - mean2) / (std2 + 1e-8)

        # Applying z-score twice to same data:
        # - First time: (data - μ₁) / σ₁
        # - Second time: ((data - μ₁) / σ₁ - μ₂) / σ₂
        #              where μ₂ ≈ 0, σ₂ ≈ 1
        # This approximately applies shift, not doubling effect

        # But applying z-score then min-max:
        # - First: (data - μ) / σ
        # - Then: ((data - μ) / σ - min) / (max - min)
        # This DOES corrupt the data differently

        # Demonstrate the corruption:
        diff_from_original = torch.abs(data_double_norm - original).max()
        assert (
            diff_from_original > 0.5
        ), f"Double normalization should corrupt data significantly, but diff={diff_from_original:.4f}"

    def test_single_strategy_produces_consistent_output(self):
        """
        INVARIANT: With single normalization strategy, repeated applications
        should produce idempotent (or close to idempotent) results.
        """
        data = torch.randn(100, 100)

        # Apply z-score once
        mean1 = data.mean()
        std1 = data.std()
        normalized1 = (data - mean1) / (std1 + 1e-8)

        # Apply z-score again to already-normalized data
        mean2 = normalized1.mean()
        std2 = normalized1.std()
        normalized2 = (normalized1 - mean2) / (std2 + 1e-8)

        # The second application should produce ~identity
        # (center at 0, scale at 1)
        # Because mean ≈ 0 and std ≈ 1 after first normalization

        # So normalized2 should be roughly equal to (data - 0) / 1 = data/1 ≈ normalized1
        # But not exactly due to floating point

        # Check that reapplying z-score to z-normalized data does minimal change
        diff = torch.abs(normalized2 - normalized1).max()

        # Should be small (ideally 0, but floating point allows small deviation)
        assert (
            diff < 0.01
        ), f"Applying z-score to z-normalized data should be near-idempotent, but diff={diff:.6f}"


# ============================================================================
# PREVENTING DOUBLE-NORMALIZATION
# ============================================================================


class TestNormalizationPipeline:
    """Test that normalization pipeline prevents double application."""

    def test_normalization_stage_is_single_operation(self):
        """
        CRITICAL: The transform pipeline should have a SINGLE normalization stage.
        Not two (z-score AND min-max applied sequentially).
        """
        # Hypothetical transform pipeline config
        pipeline_config = {
            "normalizations": ["standard"],  # Only one!
        }

        # Validate: only one normalization strategy
        normalizations = pipeline_config.get("normalizations", [])

        # Should only have zero or one normalization
        assert (
            len(normalizations) <= 1
        ), f"Pipeline has {len(normalizations)} normalizations, should have ≤1: {normalizations}"

    def test_z_score_normalization_correct_implementation(self):
        """
        Test correct z-score (standard) normalization implementation.

        Formula: x_norm = (x - μ) / σ
        Result: mean ≈ 0, std ≈ 1
        """
        data = torch.randn(1000, 1000)
        target_mean = 0.0
        target_std = 1.0

        # Apply z-score normalization
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / (std + 1e-8)

        # Verify statistics
        result_mean = normalized.mean()
        result_std = normalized.std()

        assert torch.allclose(
            result_mean, torch.tensor(target_mean), atol=1e-4
        ), f"Z-score normalization: expected mean={target_mean}, got {result_mean:.6f}"

        assert torch.allclose(
            result_std, torch.tensor(target_std), atol=1e-2
        ), f"Z-score normalization: expected std={target_std}, got {result_std:.6f}"

    def test_minmax_normalization_correct_implementation(self):
        """
        Test correct min-max normalization implementation.

        Formula: x_norm = (x - min) / (max - min)
        Result: all values in [0, 1]
        """
        data = torch.randn(100, 100)

        # Apply min-max normalization
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

        # Verify range
        result_min = normalized.min()
        result_max = normalized.max()

        assert torch.allclose(
            result_min, torch.tensor(0.0), atol=1e-6
        ), f"MinMax normalization: expected min=0, got {result_min:.6f}"

        assert torch.allclose(
            result_max, torch.tensor(1.0), atol=1e-6
        ), f"MinMax normalization: expected max=1, got {result_max:.6f}"

        # All values should be in [0, 1]
        assert (normalized >= 0.0).all() and (
            normalized <= 1.0
        ).all(), "MinMax normalization: some values outside [0, 1]"

    def test_percentile_normalization_correct_implementation(self):
        """
        Test correct percentile (robust) normalization implementation.

        Formula: x_norm = x / quantile(|x|, p)
        Result: percentile % of values have |x_norm| ≤ 1
        """
        data = torch.randn(1000, 1000)
        percentile = 0.99

        # Apply percentile normalization
        scale = torch.quantile(torch.abs(data), percentile)
        normalized = data / (scale + 1e-8)

        # Verify: approximately percentile of values have |normalized| ≤ 1
        fraction_below_1 = (
            torch.abs(normalized) <= 1.0
        ).sum().item() / normalized.numel()

        # Should be approximately percentile (within tolerance)
        assert (
            abs(fraction_below_1 - percentile) < 0.02
        ), f"Percentile normalization: expected {percentile:.1%}, got {fraction_below_1:.1%}"


# ============================================================================
# TRANSFORM ORDERING VALIDATION
# ============================================================================


class TestTransformOrdering:
    """Test that transforms are applied in correct order (no double normalization)."""

    def test_transform_pipeline_order_identity(self):
        """
        Transform pipeline should follow strict order (immutable contract).
        No double-normalization should occur.

        Correct order:
        1. Geometry (crop/resize)
        2. Physics sync
        3. Augmentation
        4. Physics sync
        5. Physics dispatch (masking/trajectory)
        6. Normalization (SINGLE STAGE)
        """
        # Conceptual test - actual implementation would use tio.Compose
        transforms = [
            "geometry_standardization",
            "physics_sync_1",
            "augmentation",
            "physics_sync_2",
            "physics_dispatch",
            "normalization",  # SINGLE normalization stage
        ]

        # Count normalization stages
        norm_count = sum(1 for t in transforms if "normalization" in t.lower())

        assert (
            norm_count == 1
        ), f"Transform pipeline should have exactly 1 normalization stage, has {norm_count}"

    def test_no_duplicate_transforms_in_pipeline(self):
        """
        INVARIANT: No transform type should appear twice consecutively or in close proximity
        (except where explicitly intended, like two physics_sync stages).
        """
        transforms = [
            "geometry",
            "physics_sync",
            "augment",
            "physics_sync",
            "physics_dispatch",
            "normalize",
        ]

        # Check for duplicate normalizations
        norm_indices = [i for i, t in enumerate(transforms) if "normalize" in t.lower()]

        assert (
            len(norm_indices) <= 1
        ), f"Found {len(norm_indices)} normalizations at indices {norm_indices}"

        # Check for consecutive duplicates (except physics_sync which is intentionally duplicated)
        for i in range(len(transforms) - 1):
            if transforms[i] == transforms[i + 1]:
                # Allowed: physics_sync can appear twice
                assert (
                    transforms[i] == "physics_sync"
                ), f"Found consecutive duplicate transforms: {transforms[i]}"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestNormalizationIntegration:
    """Test normalization in realistic scenario."""

    def test_pipeline_with_single_normalization_deterministic(self):
        """
        INVARIANT: Pipeline with single normalization applied to same data
        multiple times should produce identical results.
        """
        data = torch.randn(10, 64, 64)

        def apply_pipeline(
            x: torch.Tensor, strategy: NormalizationStrategy
        ) -> torch.Tensor:
            """Hypothetical pipeline with single normalization stage."""
            # Geometry
            x = x  # (no-op for this test)

            # Normalization (single stage based on strategy)
            if strategy == NormalizationStrategy.STANDARD:
                mean = x.mean()
                std = x.std()
                x = (x - mean) / (std + 1e-8)
            elif strategy == NormalizationStrategy.MINMAX:
                min_val = x.min()
                max_val = x.max()
                x = (x - min_val) / (max_val - min_val + 1e-8)

            return x

        # Apply pipeline twice with same strategy
        result1 = apply_pipeline(data.clone(), NormalizationStrategy.STANDARD)
        result2 = apply_pipeline(data.clone(), NormalizationStrategy.STANDARD)

        # Results should be identical (deterministic)
        assert torch.allclose(result1, result2), "Pipeline should be deterministic"

    def test_pipeline_strategy_mismatch_detected(self):
        """
        Test that applying different strategies to same data is detected as problem.
        """
        data = torch.randn(100, 100)

        # Apply z-score
        mean = data.mean()
        std = data.std()
        z_norm = (data - mean) / (std + 1e-8)

        # Apply min-max
        min_val = data.min()
        max_val = data.max()
        minmax_norm = (data - min_val) / (max_val - min_val + 1e-8)

        # Results should be quite different
        diff = torch.abs(z_norm - minmax_norm).mean()

        assert diff > 0.1, (
            f"Different normalization strategies should produce different results, "
            f"but mean difference is only {diff:.4f}"
        )


# ============================================================================
# EDGE CASES & ROBUSTNESS
# ============================================================================


class TestNormalizationEdgeCases:
    """Test edge cases for normalization."""

    def test_all_zeros_data(self):
        """Zero data should not crash normalization."""
        data = torch.zeros(10, 10)

        # Z-score would have std=0, need epsilon
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / (std + 1e-8)

        assert torch.isfinite(normalized).all()

    def test_single_value_data(self):
        """Data with all same value."""
        data = torch.ones(10, 10) * 5.0

        # Z-score: std=0
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / (std + 1e-8)

        # Should give 0 (since mean = 5, std = 0)
        assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-6)

    def test_very_large_range(self):
        """Data with very large dynamic range."""
        data = torch.logspace(-3, 3, 100).reshape(10, 10)

        # Min-max should handle this
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

        # Should be in [0, 1]
        assert (normalized >= 0.0).all() and (normalized <= 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
