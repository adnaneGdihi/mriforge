"""Unit tests for HomomorphicBiasFieldDecomposition transform."""

import pytest
import torch
import torchio as tio


class TestHomomorphicBiasFieldDecomposition:
    """Tests for log-domain bias field decomposition."""

    @pytest.fixture
    def make_transform(self):
        from spectramr.data.transforms.homomorphic_bias_field import (
            HomomorphicBiasFieldDecomposition,
        )

        def _factory(**kwargs):
            defaults = {
                "smoothness": 8.0,
                "kernel_size": 31,
                "epsilon": 1e-6,
            }
            defaults.update(kwargs)
            return HomomorphicBiasFieldDecomposition(**defaults)

        return _factory

    @pytest.fixture
    def magnitude_subject(self):
        """Create a TorchIO subject with a magnitude image."""
        # Simulate ULF image: smooth bias × anatomy
        H, W, D = 64, 64, 1
        # Anatomy: checkerboard-like high-frequency content
        anatomy = torch.ones(1, H, W, D)
        anatomy[:, ::4, :, :] = 0.5
        anatomy[:, :, ::4, :] = 0.7

        # Bias field: smooth quadratic shading (like B1- profile)
        y = torch.linspace(-1, 1, H).view(1, H, 1, 1)
        x = torch.linspace(-1, 1, W).view(1, 1, W, 1)
        bias = 0.5 + 0.5 * (1 - x**2 - y**2).clamp(min=0.1)

        image = anatomy * bias
        return tio.Subject(image=tio.ScalarImage(tensor=image))

    def test_produces_both_keys(self, make_transform, magnitude_subject):
        """Transform should add log_bias and log_anatomy to subject."""
        transform = make_transform()
        result = transform(magnitude_subject)
        assert "log_bias" in result, "Missing log_bias key"
        assert "log_anatomy" in result, "Missing log_anatomy key"

    def test_shape_preservation(self, make_transform, magnitude_subject):
        """Decomposed components must have same spatial shape as input."""
        transform = make_transform()
        result = transform(magnitude_subject)
        orig_shape = magnitude_subject["image"].data.shape
        assert result["log_bias"].data.shape == orig_shape
        assert result["log_anatomy"].data.shape == orig_shape

    def test_reconstruction_invertibility(self, make_transform, magnitude_subject):
        """log_anatomy + log_bias should reconstruct original (in log domain)."""
        from spectramr.data.transforms.homomorphic_bias_field import (
            reconstruct_from_log_components,
        )

        transform = make_transform()
        result = transform(magnitude_subject)

        log_anatomy = result["log_anatomy"].data
        log_bias = result["log_bias"].data
        original = magnitude_subject["image"].data

        reconstructed = reconstruct_from_log_components(log_anatomy, log_bias)
        # Should reconstruct the original (up to epsilon floor)
        assert torch.allclose(reconstructed, original.abs() + transform.epsilon, atol=1e-3), (
            "Reconstruction from log components does not match original"
        )

    def test_bias_is_smooth(self, make_transform, magnitude_subject):
        """Log bias field should be smoother than log anatomy."""
        transform = make_transform(smoothness=10.0)
        result = transform(magnitude_subject)

        log_bias = result["log_bias"].data
        log_anatomy = result["log_anatomy"].data

        # Compute total variation as smoothness proxy
        def total_variation(x):
            return (
                (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean()
                + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
            )

        tv_bias = total_variation(log_bias)
        tv_anatomy = total_variation(log_anatomy)

        assert tv_bias < tv_anatomy, (
            f"Bias field (TV={tv_bias:.4f}) should be smoother than anatomy (TV={tv_anatomy:.4f})"
        )

    def test_disabled_noop(self, make_transform, magnitude_subject):
        """When disabled, no new keys should be added."""
        transform = make_transform(enabled=False)
        result = transform(magnitude_subject)
        assert "log_bias" not in result
        assert "log_anatomy" not in result

    def test_missing_key_noop(self, make_transform):
        """Missing image key should not crash."""
        transform = make_transform(image_key="image")
        subject = tio.Subject(
            kspace=tio.ScalarImage(tensor=torch.randn(1, 32, 32, 1)),
        )
        result = transform(subject)
        assert "log_bias" not in result

    def test_invalid_smoothness_raises(self, make_transform):
        """Zero or negative smoothness should raise."""
        with pytest.raises(ValueError, match="smoothness"):
            make_transform(smoothness=0.0)

    def test_invalid_kernel_size_raises(self, make_transform):
        """Even kernel size should raise."""
        with pytest.raises(ValueError, match="kernel_size"):
            make_transform(kernel_size=30)

    def test_original_image_unchanged(self, make_transform, magnitude_subject):
        """Original image key should not be modified."""
        original = magnitude_subject["image"].data.clone()
        transform = make_transform()
        result = transform(magnitude_subject)
        assert torch.allclose(result["image"].data, original)

    def test_finite_outputs(self, make_transform, magnitude_subject):
        """All outputs should be finite (no NaN/Inf from log)."""
        transform = make_transform()
        result = transform(magnitude_subject)
        assert torch.isfinite(result["log_bias"].data).all(), "log_bias contains non-finite values"
        assert torch.isfinite(result["log_anatomy"].data).all(), "log_anatomy contains non-finite"

    def test_custom_keys(self, make_transform):
        """Custom key names should work."""
        subject = tio.Subject(
            ulf_scan=tio.ScalarImage(tensor=torch.rand(1, 32, 32, 1) + 0.1),
        )
        transform = make_transform(
            image_key="ulf_scan",
            store_bias_key="b1_minus",
            store_anatomy_key="spin_density",
        )
        result = transform(subject)
        assert "b1_minus" in result
        assert "spin_density" in result
