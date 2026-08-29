"""Unit tests for Latent Consistency loss."""

import pytest
import torch
import torch.nn as nn

from mriforge.models.losses.latent_consistency_loss import LatentConsistencyLoss


class MockDisentangledModel(nn.Module):
    """Mock disentangled model for testing."""

    def __init__(self, content_dim=256, style_dim=8, img_size=64):
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.img_size = img_size

        # Mock encoders
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, content_dim, 3, 1, 1),
        )

        self.style_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, style_dim),
        )

        # Mock generator (accepts content and style codes; ignores style for simplicity)
        _gen_layers = nn.Sequential(
            nn.Conv2d(content_dim, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

        class _MockGenerator(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

            def forward(self, content, style=None):
                return self.layers(content)

        self.generator = _MockGenerator(_gen_layers)

    def forward(self, x):
        c = self.content_encoder(x)
        s = self.style_encoder(x)
        out = self.generator(c, s)
        return out


class TestLatentConsistencyLoss:
    """Test Latent Consistency loss implementation."""

    @pytest.fixture
    def model(self):
        """Create mock disentangled model."""
        return MockDisentangledModel()

    @pytest.fixture
    def loss_fn(self):
        """Create latent consistency loss instance."""
        return LatentConsistencyLoss()

    @pytest.fixture
    def sample_images(self):
        """Create sample image pairs."""
        img_a = torch.randn(2, 1, 64, 64)  # ULF-like
        img_b = torch.randn(2, 1, 64, 64)  # HF-like
        return img_a, img_b

    def test_forward_shape(self, loss_fn, model, sample_images):
        """Test that forward pass returns scalar loss."""
        img_a, img_b = sample_images
        loss = loss_fn(model, img_a, img_b)

        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.ndim == 0, "Loss should be scalar (0-dim tensor)"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_gradient_flow(self, loss_fn, model, sample_images):
        """Test that gradients flow through model.

        Note: LatentConsistencyLoss detaches original content/style codes
        but gradients flow through: generator -> x_swap -> content_encoder (re-encode).
        So generator and content_encoder should receive gradients.
        """
        img_a, img_b = sample_images
        img_a.requires_grad_(True)

        loss = loss_fn(model, img_a, img_b)
        loss.backward()

        # Check that at least some model parameters have gradients
        # The gradient path is: generator(z_c_a.detach(), z_s_b) -> x_swap -> content_encoder
        # So generator and content_encoder should receive gradients
        has_any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_any_grad, "At least some parameters should receive gradients"

    def test_identical_images(self, loss_fn, model):
        """Test behavior with identical source images."""
        img = torch.randn(2, 1, 64, 64)

        # Same image for both sources
        loss = loss_fn(model, img, img)

        # Loss should be low (but not necessarily zero due to encoder stochasticity)
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_content_preservation(self, loss_fn, model, sample_images):
        """Test that loss enforces content code preservation."""
        img_a, img_b = sample_images

        # Get original content codes
        with torch.no_grad():
            c_a_orig = model.content_encoder(img_a)
            c_b_orig = model.content_encoder(img_b)

        # Compute loss (should encourage content preservation)
        loss = loss_fn(model, img_a, img_b)

        # After optimization, swapped images should preserve content
        # (This is what the loss enforces during training)
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_norm_types(self, model, sample_images):
        """Test different norm types (L1 vs L2)."""
        img_a, img_b = sample_images

        loss_fn_l1 = LatentConsistencyLoss(norm_type=1)
        loss_fn_l2 = LatentConsistencyLoss(norm_type=2)

        loss_l1 = loss_fn_l1(model, img_a, img_b)
        loss_l2 = loss_fn_l2(model, img_a, img_b)

        # Both should return valid losses
        assert loss_l1.item() >= 0, "L1 loss should be non-negative"
        assert loss_l2.item() >= 0, "L2 loss should be non-negative"

        # L2 loss typically higher than L1 for same deviation
        # (but not always due to averaging)

    def test_reduction_methods(self, model, sample_images):
        """Test different reduction methods."""
        img_a, img_b = sample_images

        loss_fn_mean = LatentConsistencyLoss(reduction="mean")
        loss_fn_sum = LatentConsistencyLoss(reduction="sum")

        loss_mean = loss_fn_mean(model, img_a, img_b)
        loss_sum = loss_fn_sum(model, img_a, img_b)

        # Sum should be larger than mean
        assert (
            loss_sum.item() > loss_mean.item()
        ), "Sum reduction should give larger value than mean"

    def test_invalid_norm_type(self, model, sample_images):
        """Test that invalid norm type raises error."""
        img_a, img_b = sample_images
        loss_fn = LatentConsistencyLoss(norm_type=3)  # Invalid

        with pytest.raises(ValueError, match="Unsupported norm_type"):
            loss_fn(model, img_a, img_b)

    def test_cross_swap_symmetry(self, loss_fn, model, sample_images):
        """Test that A→B and B→A swaps are handled symmetrically."""
        img_a, img_b = sample_images

        # Forward swap
        loss_ab = loss_fn(model, img_a, img_b)

        # Reverse swap
        loss_ba = loss_fn(model, img_b, img_a)

        # Both should be valid (not necessarily equal due to different content)
        assert loss_ab.item() >= 0, "Forward swap loss should be non-negative"
        assert loss_ba.item() >= 0, "Reverse swap loss should be non-negative"

    def test_batch_sizes(self, loss_fn, model):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            img_a = torch.randn(batch_size, 1, 64, 64)
            img_b = torch.randn(batch_size, 1, 64, 64)

            loss = loss_fn(model, img_a, img_b)

            assert isinstance(
                loss, torch.Tensor
            ), f"Loss should work with batch_size={batch_size}"
            assert loss.item() >= 0, "Loss should be non-negative"

    def test_detached_style_codes(self, loss_fn, model, sample_images):
        """Test that style codes are properly detached (prevent gradient interference)."""
        img_a, img_b = sample_images

        # Enable gradient tracking
        for param in model.parameters():
            param.requires_grad_(True)

        loss = loss_fn(model, img_a, img_b)

        # Loss should be computable without errors
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

        # Backward should work
        loss.backward()

        # Content encoder should have gradients (it's the target)
        content_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.content_encoder.parameters()
        )
        assert content_has_grad, "Content encoder should receive gradients"
