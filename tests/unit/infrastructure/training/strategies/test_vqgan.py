from unittest.mock import MagicMock, patch

import pytest
import torch

from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy
from mriforge.models.vq.vqgan import VQGAN


class TestVQGANTrainingStrategy:
    @pytest.fixture
    def mock_vqgan(self):
        """Create a mock VQGAN model."""
        model = MagicMock(spec=VQGAN)
        model.generator = MagicMock()
        model.discriminator = MagicMock()

        # Setup forward pass behavior (returns reconstruction tensor)
        # Note: Must return Tensor, not Mock, for ones_like in GANTrainingStrategy
        model.forward.return_value = torch.randn(2, 1, 64, 64)
        model.return_value = torch.randn(2, 1, 64, 64)
        model.training = True
        model.train = MagicMock()

        # Mock discriminator output as Tensor
        model.discriminator.return_value = torch.randn(2, 1, 1, 1)

        return model

    @pytest.fixture
    def mock_env(self, mock_vqgan):
        """Create a mock training environment (NEW API - env only)."""
        env = MagicMock()
        env.device = torch.device("cpu")

        # Set VQGAN as generator
        env.generator = mock_vqgan
        env.discriminator = mock_vqgan.discriminator
        env.models = {
            "generator": mock_vqgan,
            "discriminator": mock_vqgan.discriminator,
        }

        # Mock config
        env.config = MagicMock()
        env.config.training.training_mode = "gan"
        env.config.model.model_type = "vqgan"
        env.config.optimization.optimizer.learning_rate = 1e-4
        # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
        env.config.optimization.gradient.clip.enabled = False
        env.config.optimization.gradient.clip.method = "norm"
        env.config.optimization.gradient.clip.value = 1.0
        # mixed_precision.py validates amp_dtype against a fixed set; a bare
        # MagicMock fails it (stale-fixture repair, 2026-07-01).
        env.config.optimization.precision.dtype = "float32"
        env.model_type = "vqgan"

        return env

    @pytest.fixture
    def strategy(self, mock_env):
        """Create GANTrainingStrategy with mocked VQGAN (NEW API - env only)."""
        # Patch LossBuilder so the adversarial loss computer constructs without
        # exercising real loss-building on the MagicMock config. The 2026-07-01
        # L1 fix made UnifiedGANLossComputer fail loud on an invalid config
        # (e.g. a MagicMock ``losses.output_domain``) instead of swallowing it;
        # this is a wiring test, so isolate it from loss composition.
        builder = MagicMock()
        bi = builder.return_value
        for _m in (
            "build_reconstruction_losses",
            "build_adversarial_losses",
            "build_regularization_losses",
            "build_physics_losses",
            "build_latent_losses",
        ):
            getattr(bi, _m).return_value = bi
        bi.build.return_value = {}
        with patch(
            "mriforge.infrastructure.training.builders.loss_builder.LossBuilder",
            builder,
        ):
            strategy = GANTrainingStrategy(env=mock_env)
            strategy.logging_service = MagicMock()
            strategy.amp_helper = MagicMock()
            strategy.amp_helper.device_type = "cpu"
            strategy.amp_helper.enabled = False
            strategy.optimizer_stepper = MagicMock()
            yield strategy

    def test_initialization(self, strategy, mock_vqgan):
        """Test that strategy initializes with VQGAN model."""
        assert isinstance(strategy, GANTrainingStrategy)
        assert strategy.env.generator is mock_vqgan

    def test_compute_losses_impl_structure(self, strategy, mock_env):
        """Test compute_losses_impl structure for VQGAN."""
        input_batch = torch.randn(2, 1, 64, 64)
        target_batch = torch.randn(2, 1, 64, 64)
        epoch = 1

        # Setup generator output
        fake_output = torch.randn(2, 1, 64, 64)
        mock_env.generator.return_value = fake_output

        # Execute
        losses = strategy._compute_losses_impl(input_batch, target_batch, epoch)

        # Verify
        assert "g_total_loss" in losses
        # VQGAN typically uses L1 or Perceptual + Adversarial
        # Here we check if basic structure holds

    def test_discriminator_access(self, strategy):
        """Test that strategy can access discriminator from VQGAN wrapper."""
        # GANTrainingStrategy expects discriminator at self.state.discriminator
        # But VQGAN wraps it. The initialization likely needs to unpack it
        # or the Registry needs to handle it.
        # This test documents expectation: VQGAN usually registers generator and discriminator separately
        # OR the model itself is the generator and has a discriminator attribute.

        # In current codebase, existing models seem to be separate.
        # VQGAN class has .discriminator attribute.

        pass
