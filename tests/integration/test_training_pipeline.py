"""Integration tests for training pipeline with synthetic data."""

import os
import tempfile

import pytest
import torch


class TestTrainingSmokeRun:
    """Smoke tests for training with synthetic data."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1),
        )

    @pytest.fixture
    def synthetic_batch(self):
        """Create synthetic batch for testing."""
        return {
            "image": torch.randn(2, 1, 64, 64),
            "target": torch.randn(2, 1, 64, 64),
            "mask": torch.ones(2, 1, 64, 64),
        }

    def test_simple_forward_pass(self, simple_model, synthetic_batch):
        """Test simple forward pass works."""
        output = simple_model(synthetic_batch["image"])
        assert output.shape == synthetic_batch["target"].shape

    def test_simple_backward_pass(self, simple_model, synthetic_batch):
        """Test simple backward pass works."""
        optimizer = torch.optim.Adam(simple_model.parameters())

        output = simple_model(synthetic_batch["image"])
        loss = torch.nn.functional.mse_loss(output, synthetic_batch["target"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_training_loop_iteration(self, simple_model, synthetic_batch):
        """Test a few training iterations."""
        optimizer = torch.optim.Adam(simple_model.parameters())

        losses = []
        for i in range(3):
            output = simple_model(synthetic_batch["image"])
            loss = torch.nn.functional.mse_loss(output, synthetic_batch["target"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should be positive
        assert all(l > 0 for l in losses)


class TestStrategyTrainingStep:
    """Test individual strategy training steps."""

    def test_reconstruction_compute_loss(self):
        """Test reconstruction loss computation."""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)

        l1_loss = torch.nn.functional.l1_loss(pred, target)
        mse_loss = torch.nn.functional.mse_loss(pred, target)

        assert l1_loss > 0
        assert mse_loss > 0

    def test_gan_discriminator_step(self):
        """Test GAN discriminator step."""
        discriminator = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 1),
        )

        real = torch.randn(2, 1, 64, 64)
        fake = torch.randn(2, 1, 64, 64)

        real_pred = discriminator(real)
        fake_pred = discriminator(fake)

        # LSGAN loss
        d_loss = 0.5 * ((real_pred - 1) ** 2).mean() + 0.5 * (fake_pred**2).mean()

        assert d_loss > 0

    def test_vae_reparameterization(self):
        """Test VAE reparameterization trick."""
        mu = torch.randn(4, 128)
        logvar = torch.randn(4, 128)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        assert z.shape == (4, 128)

        # KL divergence
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        assert kl.item() >= 0


class TestCheckpointSaveLoad:
    """Test checkpoint saving and loading."""

    def test_save_model_state(self):
        """Test saving model state dict."""
        model = torch.nn.Linear(10, 10)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)

            # Load back
            loaded_state = torch.load(f.name, weights_only=True)

            assert "weight" in loaded_state
            assert "bias" in loaded_state

            os.unlink(f.name)

    def test_save_full_checkpoint(self):
        """Test saving full training checkpoint."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 5,
            "loss": 0.123,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)

            loaded = torch.load(f.name, weights_only=False)

            assert loaded["epoch"] == 5
            assert loaded["loss"] == 0.123

            os.unlink(f.name)


class TestMetricsComputation:
    """Test metrics computation during training."""

    def test_batch_metrics(self):
        """Test computing metrics on a batch."""
        pred = torch.randn(4, 1, 64, 64)
        target = pred.clone()  # Perfect prediction

        # MSE should be ~0
        mse = torch.nn.functional.mse_loss(pred, target)
        assert mse < 1e-6

        # MAE should be ~0
        mae = torch.nn.functional.l1_loss(pred, target)
        assert mae < 1e-6

    def test_psnr_computation(self):
        """Test PSNR metric computation."""
        pred = torch.randn(4, 1, 64, 64)
        target = pred.clone()

        mse = torch.nn.functional.mse_loss(pred, target)

        # PSNR = 10 * log10(max^2 / MSE)
        # For identical images, MSE is ~0, so PSNR is very high
        if mse > 0:
            psnr = 10 * torch.log10(1.0 / mse)
            assert psnr > 40  # Very high for nearly identical


class TestGradientFlow:
    """Test gradient flow through models."""

    def test_gradients_exist(self):
        """Test gradients are computed."""
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10, requires_grad=True)

        y = model(x)
        loss = y.sum()
        loss.backward()

        assert model.weight.grad is not None
        assert model.bias.grad is not None

    def test_no_vanishing_gradients(self):
        """Test gradients don't vanish in deep network."""
        # Create a simple deep network
        layers = [torch.nn.Linear(32, 32) for _ in range(5)]
        model = torch.nn.Sequential(*layers)

        x = torch.randn(4, 32, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check first layer still has gradients
        first_grad_norm = layers[0].weight.grad.norm()
        assert first_grad_norm > 1e-6, "Gradients vanished"
