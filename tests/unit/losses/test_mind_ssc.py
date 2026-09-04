"""Unit tests for MIND-SSC loss."""

import pytest
import torch

from spectramr.models.losses.mind_ssc_loss import MINDSSCLoss


class TestMINDSSCLoss:
    """Test MIND-SSC loss implementation."""

    @pytest.fixture
    def loss_fn(self):
        """Create MIND-SSC loss instance."""
        return MINDSSCLoss(patch_size=7, search_radius=3, gaussian_sigma=1.0)

    @pytest.fixture
    def sample_images(self):
        """Create sample image tensors."""
        batch_size = 2
        channels = 1
        height = 64
        width = 64

        # Create two similar but slightly different images
        img1 = torch.randn(batch_size, channels, height, width)
        img2 = img1 + 0.1 * torch.randn_like(img1)  # Add small noise

        return img1, img2

    def test_forward_shape(self, loss_fn, sample_images):
        """Test that forward pass returns scalar loss."""
        img1, img2 = sample_images
        loss = loss_fn(img1, img2)

        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.ndim == 0, "Loss should be scalar (0-dim tensor)"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_intensity_invariance(self, loss_fn):
        """Test that MIND-SSC is invariant to intensity transformations."""
        # Create base image
        img1 = torch.randn(1, 1, 64, 64)

        # Apply affine intensity transformation to img2
        # T(x) = a*x + b (where structure is preserved)
        a = 2.0
        b = 0.5
        img2 = a * img1 + b

        # Compute MIND-SSC loss
        loss = loss_fn(img1, img2)

        # Loss should be small (not exactly 0 due to numerical precision and boundary effects)
        # but significantly smaller than L1 loss would be
        l1_loss = torch.nn.functional.l1_loss(img1, img2)

        assert (
            loss.item() < l1_loss.item()
        ), "MIND-SSC should be more robust to intensity changes than L1"

    def test_identical_images(self, loss_fn):
        """Test that identical images have zero loss."""
        img = torch.randn(2, 1, 64, 64)
        loss = loss_fn(img, img)

        # Should be exactly 0 for identical images
        assert (
            loss.item() < 1e-6
        ), f"Loss for identical images should be ~0, got {loss.item()}"

    def test_gradient_flow(self, loss_fn, sample_images):
        """Test that gradients flow correctly."""
        img1, img2 = sample_images
        img1.requires_grad_(True)

        loss = loss_fn(img1, img2)
        loss.backward()

        assert img1.grad is not None, "Gradients should be computed"
        assert not torch.isnan(img1.grad).any(), "Gradients should not be NaN"
        assert not torch.isinf(img1.grad).any(), "Gradients should not be Inf"

    def test_multi_channel(self, loss_fn):
        """Test with multi-channel images."""
        # MIND-SSC should handle multi-channel images
        img1 = torch.randn(2, 3, 64, 64)  # RGB-like
        img2 = img1 + 0.1 * torch.randn_like(img1)

        loss = loss_fn(img1, img2)

        assert isinstance(
            loss, torch.Tensor
        ), "Loss should be computed for multi-channel"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_descriptor_computation(self, loss_fn):
        """Test MIND descriptor computation."""
        img = torch.randn(1, 1, 64, 64)

        # Access internal method
        descriptor = loss_fn._compute_mind_descriptor(img)

        # Should produce 6 directional features per channel
        expected_channels = 1 * 6  # channels * directions
        assert (
            descriptor.shape[1] == expected_channels
        ), f"Expected {expected_channels} descriptor channels, got {descriptor.shape[1]}"

        # Spatial dimensions should match (with padding)
        assert descriptor.shape[2] == img.shape[2], "Height should match"
        assert descriptor.shape[3] == img.shape[3], "Width should match"

    def test_self_similarity_normalization(self):
        """Test that self-similarity normalization is applied."""
        loss_fn_with_ss = MINDSSCLoss(use_self_similarity=True)
        loss_fn_without_ss = MINDSSCLoss(use_self_similarity=False)

        img1 = torch.randn(1, 1, 64, 64)
        img2 = img1 + 0.1 * torch.randn_like(img1)

        loss_with = loss_fn_with_ss(img1, img2)
        loss_without = loss_fn_without_ss(img1, img2)

        # Losses should be different (self-similarity affects the descriptor)
        assert (
            abs(loss_with.item() - loss_without.item()) > 1e-6
        ), "Self-similarity should affect loss value"

    @pytest.mark.parametrize("patch_size", [3, 5, 7, 9])
    def test_patch_sizes(self, patch_size):
        """Test different patch sizes."""
        loss_fn = MINDSSCLoss(patch_size=patch_size)
        img1 = torch.randn(1, 1, 64, 64)
        img2 = img1 + 0.1 * torch.randn_like(img1)

        loss = loss_fn(img1, img2)

        assert isinstance(
            loss, torch.Tensor
        ), f"Loss should work with patch_size={patch_size}"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_batch_consistency(self, loss_fn):
        """Test that batch computation is consistent with individual computation."""
        # Create batch of 4 images
        batch_img1 = torch.randn(4, 1, 64, 64)
        batch_img2 = batch_img1 + 0.1 * torch.randn_like(batch_img1)

        # Compute on full batch
        batch_loss = loss_fn(batch_img1, batch_img2)

        # Compute individual losses and average
        individual_losses = []
        for i in range(4):
            loss = loss_fn(batch_img1[i : i + 1], batch_img2[i : i + 1])
            individual_losses.append(loss.item())

        avg_individual = sum(individual_losses) / len(individual_losses)

        # Should be approximately equal (MSE averages over batch)
        assert (
            abs(batch_loss.item() - avg_individual) < 1e-4
        ), "Batch loss should equal average of individual losses"
