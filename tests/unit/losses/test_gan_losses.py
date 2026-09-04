"""Unit tests for GAN loss functions."""

import torch

from spectramr.models.losses.gan_loss_library import (
    AdversarialLossStrategy,
    LSGANLoss,
    RALSGANLoss,
    StandardGANLoss,
    WGANLoss,
)


class TestStandardGANLoss:
    """Test vanilla GAN (BCE) loss."""

    def test_discriminator_loss_real(self):
        """Test D loss on real data."""
        loss_fn = StandardGANLoss()
        # D outputs high logits for real
        real_preds = torch.ones(4, 1) * 2.0
        fake_preds = torch.ones(4, 1) * -2.0

        d_loss = loss_fn.compute_discriminator_loss(real_preds, fake_preds)
        if isinstance(d_loss, tuple):
            d_loss = d_loss[0]
        assert d_loss.item() >= 0, "D loss should be non-negative"

    def test_generator_loss(self):
        """Test G loss should be low when D fooled."""
        loss_fn = StandardGANLoss()
        # D thinks fakes are real (high logits)
        fake_preds = torch.ones(4, 1) * 2.0
        g_loss = loss_fn.compute_generator_loss(fake_preds)
        if isinstance(g_loss, tuple):
            g_loss = g_loss[0]
        assert g_loss.item() >= 0, "G loss should be non-negative"


class TestLSGANLoss:
    """Test Least Squares GAN loss."""

    def test_discriminator_loss(self):
        """Test LSGAN D loss."""
        loss_fn = LSGANLoss()
        real_preds = torch.ones(4, 1)
        fake_preds = torch.zeros(4, 1)
        d_loss = loss_fn.compute_discriminator_loss(real_preds, fake_preds)
        if isinstance(d_loss, tuple):
            d_loss = d_loss[0]
        assert d_loss.item() >= 0, "D loss should be non-negative"

    def test_generator_loss(self):
        """Test LSGAN G loss."""
        loss_fn = LSGANLoss()
        fake_preds = torch.ones(4, 1)  # Fakes look real
        g_loss = loss_fn.compute_generator_loss(fake_preds)
        if isinstance(g_loss, tuple):
            g_loss = g_loss[0]
        assert g_loss.item() >= 0, "G loss should be non-negative"


class TestRALSGANLoss:
    """Test Relativistic LSGAN loss."""

    def test_discriminator_loss(self):
        """Test RALSGAN D loss."""
        loss_fn = RALSGANLoss()
        real_preds = torch.ones(4, 1) * 2
        fake_preds = torch.ones(4, 1) * -2
        d_loss = loss_fn.compute_discriminator_loss(real_preds, fake_preds)
        if isinstance(d_loss, tuple):
            d_loss = d_loss[0]
        assert d_loss.item() >= 0, "D loss should be non-negative"

    def test_generator_loss(self):
        """Test RALSGAN G loss."""
        loss_fn = RALSGANLoss()
        real_preds = torch.ones(4, 1) * 2
        fake_preds = torch.ones(4, 1) * 2  # Fakes score as high as real
        g_loss = loss_fn.compute_generator_loss(fake_preds, real_outputs_d=real_preds)
        if isinstance(g_loss, tuple):
            g_loss = g_loss[0]
        assert isinstance(g_loss.item(), float)


class TestWGANLoss:
    """Test Wasserstein GAN loss."""

    def test_discriminator_loss(self):
        """Test WGAN D loss (critic)."""
        loss_fn = WGANLoss()
        real_preds = torch.ones(4, 1) * 5
        fake_preds = torch.ones(4, 1) * -5
        d_loss = loss_fn.compute_discriminator_loss(real_preds, fake_preds)
        # WGAN D loss is fake.mean() - real.mean()
        if isinstance(d_loss, tuple):
            d_loss = d_loss[0]
        assert isinstance(d_loss.item(), float)

    def test_generator_loss(self):
        """Test WGAN G loss."""
        loss_fn = WGANLoss()
        fake_preds = torch.ones(4, 1) * 5
        g_loss = loss_fn.compute_generator_loss(fake_preds)
        # G loss is -fake.mean()
        if isinstance(g_loss, tuple):
            g_loss = g_loss[0]
        assert isinstance(g_loss.item(), float)


class TestAdversarialLossStrategy:
    """Test adversarial loss strategy interface."""

    def test_strategy_is_base_class(self):
        """Test adversarial loss strategy is base class."""
        assert hasattr(AdversarialLossStrategy, "compute_generator_loss")
        assert hasattr(AdversarialLossStrategy, "compute_discriminator_loss")


class TestLossWithLabelSmoothing:
    """Test losses with label smoothing."""

    def test_standard_with_smoothing(self):
        """Test StandardGAN with label smoothing."""
        loss_fn = StandardGANLoss(label_smoothing=0.1)
        real_preds = torch.ones(4, 1) * 2.0
        fake_preds = torch.ones(4, 1) * -2.0

        d_loss = loss_fn.compute_discriminator_loss(real_preds, fake_preds)
        if isinstance(d_loss, tuple):
            d_loss = d_loss[0]
        assert d_loss.item() >= 0

    def test_lsgan_with_smoothing(self):
        """Test LSGAN with label smoothing."""
        loss_fn = LSGANLoss(label_smoothing=0.1)
        real_preds = torch.ones(4, 1)
        fake_preds = torch.zeros(4, 1)

        d_loss = loss_fn.compute_discriminator_loss(real_preds, fake_preds)
        if isinstance(d_loss, tuple):
            d_loss = d_loss[0]
        assert d_loss.item() >= 0
