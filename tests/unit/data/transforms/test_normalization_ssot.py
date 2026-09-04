"""Unit tests for normalization SSOT (Single Source of Truth).

Tests cover:
- NormalizationStrategy enum parsing
- NormalizationConfig construction
- compute_magnitude for complex, stacked, and real tensors
- normalize_percentile with various parameters
- normalize_zscore
- normalize_minmax
- normalize_tensor (strategy dispatcher)
- denormalize_percentile (roundtrip)
- KSpaceNormalizationTransform (TorchIO wrapper)
"""

import torch
import torchio as tio

from spectramr.data.transforms.normalization import (
    KSpaceNormalizationTransform,
    NormalizationConfig,
    NormalizationStrategy,
    compute_magnitude,
    denormalize_percentile,
    normalize_minmax,
    normalize_percentile,
    normalize_tensor,
    normalize_zscore,
)

# ---------------------------------------------------------------------------
# NormalizationStrategy
# ---------------------------------------------------------------------------


class TestNormalizationStrategy:
    """Tests for NormalizationStrategy enum."""

    def test_from_string_valid(self):
        assert (
            NormalizationStrategy.from_string("percentile")
            is NormalizationStrategy.PERCENTILE
        )
        assert (
            NormalizationStrategy.from_string("zscore") is NormalizationStrategy.ZSCORE
        )
        assert (
            NormalizationStrategy.from_string("minmax") is NormalizationStrategy.MINMAX
        )
        assert NormalizationStrategy.from_string("none") is NormalizationStrategy.NONE

    def test_from_string_case_insensitive(self):
        assert (
            NormalizationStrategy.from_string("PERCENTILE")
            is NormalizationStrategy.PERCENTILE
        )
        assert (
            NormalizationStrategy.from_string("Zscore") is NormalizationStrategy.ZSCORE
        )

    def test_from_string_alias(self):
        assert (
            NormalizationStrategy.from_string("robust_percentile")
            is NormalizationStrategy.PERCENTILE
        )

    def test_from_string_invalid(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown normalization"):
            NormalizationStrategy.from_string("invalid")

    def test_from_string_whitespace(self):
        assert (
            NormalizationStrategy.from_string("  percentile  ")
            is NormalizationStrategy.PERCENTILE
        )


# ---------------------------------------------------------------------------
# NormalizationConfig
# ---------------------------------------------------------------------------


class TestNormalizationConfig:
    """Tests for NormalizationConfig frozen dataclass."""

    def test_defaults(self):
        config = NormalizationConfig()
        assert config.strategy is NormalizationStrategy.NONE
        assert config.percentile == 0.99
        assert config.out_range == (0.0, 1.0)
        assert config.eps == 1e-8
        assert config.clamp is True

    def test_frozen(self):
        import pytest

        config = NormalizationConfig()
        with pytest.raises(AttributeError):
            config.strategy = NormalizationStrategy.ZSCORE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_magnitude
# ---------------------------------------------------------------------------


class TestComputeMagnitude:
    """Tests for compute_magnitude."""

    def test_complex_tensor(self):
        data = torch.complex(torch.tensor([3.0]), torch.tensor([4.0]))
        mag = compute_magnitude(data)
        assert torch.allclose(mag, torch.tensor([5.0]))

    def test_stacked_real_imag(self):
        # (2, H, W) layout
        data = torch.zeros(2, 4, 4)
        data[0, 0, 0] = 3.0  # real
        data[1, 0, 0] = 4.0  # imag
        mag = compute_magnitude(data)
        # magnitude at [0,0] should be ~5.0
        assert mag.shape[0] == 1  # collapsed first dim
        assert mag[0, 0, 0].item() > 4.9

    def test_batched_stacked_real_imag(self):
        # (B, 2, H, W) layout - used in DisentangledTrainingStrategy
        data = torch.zeros(4, 2, 8, 8)
        data[0, 0, 0, 0] = 3.0  # real
        data[0, 1, 0, 0] = 4.0  # imag
        mag = compute_magnitude(data)
        assert mag.shape == (4, 1, 8, 8)
        assert mag[0, 0, 0, 0].item() > 4.9

    def test_real_fallback(self):
        data = torch.tensor([[-3.0, 2.0, -1.0]])
        mag = compute_magnitude(data)
        assert torch.allclose(mag, torch.tensor([[3.0, 2.0, 1.0]]))


# ---------------------------------------------------------------------------
# normalize_percentile
# ---------------------------------------------------------------------------


class TestNormalizePercentile:
    """Tests for normalize_percentile."""

    def test_basic_scaling(self):
        data = torch.linspace(0, 10, 101)
        normalized, scale = normalize_percentile(data, percentile=1.0, clamp=False)
        assert scale.item() > 9.9
        assert normalized.max().item() <= 1.1

    def test_clamp_to_range(self):
        data = torch.linspace(0, 10, 101)
        normalized, _ = normalize_percentile(data, percentile=0.5, clamp=True)
        assert normalized.max().item() <= 1.0
        assert normalized.min().item() >= 0.0

    def test_out_range(self):
        data = torch.linspace(0, 10, 101)
        normalized, _ = normalize_percentile(
            data, percentile=1.0, out_range=(-1.0, 1.0), clamp=True
        )
        assert normalized.min().item() >= -1.0
        assert normalized.max().item() <= 1.0

    def test_min_scale(self):
        data = torch.zeros(10)  # near-zero data
        _, scale = normalize_percentile(data, min_scale=0.05)
        assert scale.item() >= 0.05

    def test_nan_safety(self):
        data = torch.tensor([1.0, float("nan"), 3.0])
        normalized, _ = normalize_percentile(data, percentile=1.0)
        assert not torch.isnan(normalized).any()

    def test_precomputed_magnitude(self):
        data = torch.randn(2, 8, 8)
        mag = compute_magnitude(data)
        norm1, scale1 = normalize_percentile(data, magnitude=mag)
        norm2, scale2 = normalize_percentile(data)
        assert torch.allclose(scale1, scale2)


# ---------------------------------------------------------------------------
# denormalize_percentile
# ---------------------------------------------------------------------------


class TestDenormalizePercentile:
    """Tests for roundtrip normalize → denormalize."""

    def test_roundtrip(self):
        data = torch.randn(3, 16, 16).abs()  # positive data
        normalized, scale = normalize_percentile(data, percentile=0.99, clamp=False)
        recovered = denormalize_percentile(normalized, scale)
        assert torch.allclose(data, recovered, atol=1e-5)


# ---------------------------------------------------------------------------
# normalize_zscore
# ---------------------------------------------------------------------------


class TestNormalizeZscore:
    """Tests for normalize_zscore."""

    def test_zero_mean_unit_std(self):
        data = torch.randn(1000) * 5.0 + 3.0  # mean~3, std~5
        normalized, (mean, std) = normalize_zscore(data)
        assert abs(normalized.mean().item()) < 0.1
        assert abs(normalized.std().item() - 1.0) < 0.1

    def test_constant_tensor(self):
        data = torch.ones(10) * 5.0
        normalized, _ = normalize_zscore(data)
        # All values should be near zero
        assert torch.allclose(normalized, torch.zeros(10), atol=1e-4)


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------


class TestNormalizeMinmax:
    """Tests for normalize_minmax."""

    def test_default_range(self):
        data = torch.tensor([1.0, 3.0, 5.0])
        normalized, (min_val, max_val) = normalize_minmax(data)
        assert normalized.min().item() >= 0.0 - 1e-6
        assert normalized.max().item() <= 1.0 + 1e-6

    def test_custom_range(self):
        data = torch.tensor([1.0, 3.0, 5.0])
        normalized, _ = normalize_minmax(data, out_range=(-1.0, 1.0))
        assert abs(normalized.min().item() - (-1.0)) < 1e-6
        assert abs(normalized.max().item() - 1.0) < 1e-6

    def test_constant_tensor_passthrough(self):
        data = torch.ones(10)
        normalized, _ = normalize_minmax(data)
        assert torch.allclose(normalized, data)


# ---------------------------------------------------------------------------
# normalize_tensor (dispatcher)
# ---------------------------------------------------------------------------


class TestNormalizeTensorDispatcher:
    """Tests for normalize_tensor dispatcher."""

    def test_none_strategy(self):
        data = torch.randn(4, 4)
        config = NormalizationConfig(strategy=NormalizationStrategy.NONE)
        result = normalize_tensor(data, config)
        assert torch.equal(result, data)

    def test_percentile_strategy(self):
        data = torch.randn(4, 4).abs()
        config = NormalizationConfig(
            strategy=NormalizationStrategy.PERCENTILE,
            percentile=0.99,
        )
        result = normalize_tensor(data, config)
        assert result.max().item() <= 1.0

    def test_zscore_strategy(self):
        data = torch.randn(100) * 5.0 + 3.0
        config = NormalizationConfig(strategy=NormalizationStrategy.ZSCORE)
        result = normalize_tensor(data, config)
        assert abs(result.mean().item()) < 0.2

    def test_minmax_strategy(self):
        data = torch.tensor([1.0, 3.0, 5.0])
        config = NormalizationConfig(
            strategy=NormalizationStrategy.MINMAX,
            out_range=(-1.0, 1.0),
        )
        result = normalize_tensor(data, config)
        assert abs(result.min().item() - (-1.0)) < 1e-6
        assert abs(result.max().item() - 1.0) < 1e-6

    def test_invalid_strategy_raises(self):
        import pytest

        data = torch.randn(4, 4)
        config = NormalizationConfig.__new__(NormalizationConfig)
        object.__setattr__(config, "strategy", "bogus")
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_tensor(data, config)


# ---------------------------------------------------------------------------
# KSpaceNormalizationTransform (TorchIO wrapper)
# ---------------------------------------------------------------------------


class TestKSpaceNormalizationTransform:
    """Tests for KSpaceNormalizationTransform TorchIO wrapper."""

    def test_normalizes_kspace_key(self):
        transform = KSpaceNormalizationTransform(percentile=0.99)
        data = torch.randn(2, 8, 8, 1) * 100.0  # large scale
        subject = tio.Subject(kspace=tio.ScalarImage(tensor=data))
        result = transform(subject)
        # Data should be scaled down
        assert result["kspace"].data.abs().max().item() < 200.0

    def test_skips_image_key(self):
        transform = KSpaceNormalizationTransform(percentile=0.99)
        data = torch.randn(2, 8, 8, 1) * 100.0
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=data.clone()),
            kspace=tio.ScalarImage(tensor=data.clone()),
        )
        result = transform(subject)
        # image should be unchanged
        assert torch.equal(result["image"].data, data)
        # kspace should be normalized
        assert not torch.equal(result["kspace"].data, data)

    def test_handles_complex_magnitude(self):
        transform = KSpaceNormalizationTransform(percentile=0.99)
        # 2-channel (real/imag) data
        data = torch.randn(2, 16, 16, 1) * 50.0
        subject = tio.Subject(input=tio.ScalarImage(tensor=data))
        result = transform(subject)
        assert result["input"].data.shape == data.shape
