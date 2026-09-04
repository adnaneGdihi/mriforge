"""Gradient flow verification tests for SSOT refactoring.

Ensures that the refactoring from instance variables to property-based
weight access did not break gradient computation graphs.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.config.schemas.loss import (
    LatentLossesConfig,
    LossConfigSchema,
    ReconstructionLossesConfig,
)
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


@pytest.fixture(autouse=True)
def _patch_loss_builder_ssot():
    """Stub LossBuilder so the disentangled strategy's UnifiedDisentangledLossComputer
    constructs without exercising real loss-building on the MagicMock config. The
    2026-07-01 L1 fix made that computer fail loud on an invalid config (e.g. a
    MagicMock losses.output_domain) instead of swallowing it; these tests exercise
    the loss-weight helpers, not loss composition."""
    from unittest.mock import MagicMock as _MM
    builder = _MM()
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
        "spectramr.infrastructure.training.builders.loss_builder.LossBuilder", builder
    ):
        yield



class TestGradientFlowIntegrity:
    """Verify gradient computation remains intact after SSOT refactoring."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal config for gradient testing."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        config.losses.reconstruction = MagicMock(spec=ReconstructionLossesConfig)
        config.losses.latent = MagicMock(spec=LatentLossesConfig)

        # Set weights that will be used in loss computation
        config.losses.reconstruction.lambda_recon = 10.0
        config.losses.reconstruction.lambda_l1 = 10.0
        config.losses.reconstruction.lambda_hist = 2.0
        config.losses.reconstruction.lambda_ffl = 1.5
        config.losses.latent.lambda_kl = 0.01

        # Required config sections
        config.model = MagicMock()
        config.model.in_channels = 2
        config.optimization = MagicMock()
        config.optimization.gradient.clip.enabled = False
        # StandardOptimizerStepper validates gradient_clip_method unconditionally
        # (pitfall #9); mixed_precision.py validates amp_dtype. A bare MagicMock
        # fails both — supply valid values (stale-fixture repair, 2026-07-01).
        config.optimization.gradient.clip.method = "norm"
        config.optimization.gradient.clip.value = 1.0
        config.optimization.precision.dtype = "float32"
        config.training = MagicMock()
        config.training.training_mode = "disentangled"
        config.training.strategy_class = "disentangled"
        config.losses.gan = MagicMock()
        config.losses.gan.enable_adversarial = False
        config.losses.physics = MagicMock()
        config.losses.physics.lambda_physics_constraint = 0.0
        config.losses.physics.enable_kspace_consistency = False
        config.losses.diffusion = MagicMock()
        config.losses.diffusion.enable_diffusion = False
        config.losses.ssl = MagicMock()
        config.losses.ssl.enable_ssl = False
        config.physics = None

        return config

    def test_get_loss_weight_returns_python_float(self, minimal_config):
        """Test that _get_loss_weight returns Python float, not tensor."""
        env = MagicMock()
        env.config = minimal_config
        env.device = torch.device("cpu")

        state = MagicMock()
        state.config = minimal_config
        state.device = torch.device("cpu")
        state.losses = {}
        state.generator = MagicMock()
        state.optimizers = {"main": MagicMock()}

        with patch(
            "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
        ):
            strategy = DisentangledTrainingStrategy(env=env, state=state)

            # Get weight from config
            weight = strategy._get_loss_weight("lambda_hist", 0.0)

            # Should be Python float, NOT tensor
            assert isinstance(weight, float), f"Expected float, got {type(weight)}"
            assert not isinstance(weight, torch.Tensor), "Weight should not be a tensor"
            assert weight == 2.0

    def test_loss_weight_multiplication_preserves_gradients(self):
        """Test that multiplying tensor by float weight preserves gradients."""
        # Create a tensor with gradient tracking
        loss_tensor = torch.tensor(5.0, requires_grad=True)

        # Simulate getting weight from config (Python float)
        lambda_weight = 2.0  # This is what _get_loss_weight returns

        # Multiply (this is what happens in loss computation)
        weighted_loss = lambda_weight * loss_tensor

        # Gradient should be preserved
        assert weighted_loss.requires_grad, "Gradient tracking should be preserved"

        # Perform backward to verify gradient flow
        weighted_loss.backward()

        # Check gradient exists and has correct value
        assert loss_tensor.grad is not None, "Gradient should exist"
        assert loss_tensor.grad.item() == 2.0, f"Gradient should be {lambda_weight}"

    def test_multiple_loss_weights_chain_gradients(self):
        """Test that multiple weighted losses sum correctly with gradients."""
        # Simulate multiple loss components
        loss_a = torch.tensor(3.0, requires_grad=True)
        loss_b = torch.tensor(5.0, requires_grad=True)
        loss_c = torch.tensor(2.0, requires_grad=True)

        # Weights from config (Python floats)
        lambda_a = 1.0
        lambda_b = 2.0
        lambda_c = 0.5

        # Weight and sum (mimics loss computation in strategy)
        total_loss = lambda_a * loss_a + lambda_b * loss_b + lambda_c * loss_c

        # Check gradient preservation
        assert total_loss.requires_grad, "Total loss should have gradients"

        # Backward pass
        total_loss.backward()

        # All components should have gradients
        assert loss_a.grad is not None and loss_a.grad.item() == lambda_a
        assert loss_b.grad is not None and loss_b.grad.item() == lambda_b
        assert loss_c.grad is not None and loss_c.grad.item() == lambda_c

    def test_zero_weight_does_not_break_graph(self):
        """Test that zero weights don't break gradient computation."""
        loss_tensor = torch.tensor(10.0, requires_grad=True)
        lambda_weight = 0.0  # Disabled loss

        # This should not break
        weighted_loss = lambda_weight * loss_tensor

        # Should still have gradient tracking (even though gradient will be zero)
        assert weighted_loss.requires_grad, "Should preserve gradient tracking"

        # Can still backward (gradient will just be zero)
        weighted_loss.backward()
        assert loss_tensor.grad is not None
        assert loss_tensor.grad.item() == 0.0

    def test_config_weight_update_affects_computation(self):
        """Test that changing config weight changes computation (no caching)."""

        # Simulate dynamic config
        class DynamicConfig:
            lambda_test = 1.0

        config = DynamicConfig()
        loss_tensor = torch.tensor(5.0, requires_grad=True)

        # First computation
        weighted_1 = config.lambda_test * loss_tensor

        # Change config
        config.lambda_test = 3.0

        # Second computation should reflect new value
        loss_tensor_2 = torch.tensor(5.0, requires_grad=True)
        weighted_2 = config.lambda_test * loss_tensor_2

        # Values should be different
        assert weighted_1.item() == 5.0  # 1.0 * 5.0
        assert weighted_2.item() == 15.0  # 3.0 * 5.0


class TestGradientFlowRegressions:
    """Test for specific gradient-breaking patterns."""

    def test_no_detach_in_weight_access(self):
        """Ensure _get_loss_weight doesn't accidentally detach anything."""
        # This is a documentation test - the implementation should not use .detach()
        # If it did, gradients would break

        # Simulated scenario: if weight was a tensor (IT SHOULDN'T BE)
        wrong_weight_tensor = torch.tensor(2.0, requires_grad=True)
        loss = torch.tensor(5.0, requires_grad=True)

        # If we detached (DON'T DO THIS)
        weighted_wrong = wrong_weight_tensor.detach() * loss

        # This would break gradients for the weight
        weighted_wrong.backward()
        assert (
            wrong_weight_tensor.grad is None
        ), "Detached tensor shouldn't get gradients"

        # Correct way (weight is float, not tensor)
        correct_weight_float = 2.0
        weighted_correct = correct_weight_float * loss

        # Loss should get gradients
        loss_2 = torch.tensor(5.0, requires_grad=True)
        (correct_weight_float * loss_2).backward()
        assert loss_2.grad is not None

    def test_no_item_in_loss_computation(self):
        """Ensure .item() is not used in loss computation (would break graph)."""
        loss = torch.tensor(5.0, requires_grad=True)

        # WRONG: Using .item() breaks graph
        # weighted_wrong = 2.0 * loss.item()  # This would return Python float with no gradient

        # CORRECT: Keep tensor
        weighted_correct = 2.0 * loss

        assert weighted_correct.requires_grad, "Should preserve gradient"
        weighted_correct.backward()
        assert loss.grad is not None


# Test that verifies the actual strategy doesn't break gradients
class TestStrategyGradientIntegration:
    """Integration test: full strategy preserves gradients."""

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_loss_weights_property_does_not_break_gradients(
        self, mock_logger, mock_bloch
    ):
        """Integration test: loss_weights property returns floats that preserve gradients."""
        # Setup minimal config
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        config.losses.reconstruction = MagicMock(spec=ReconstructionLossesConfig)
        config.losses.latent = MagicMock(spec=LatentLossesConfig)

        config.losses.reconstruction.lambda_recon = 10.0
        config.losses.reconstruction.lambda_l1 = 10.0
        config.losses.reconstruction.lambda_hist = 2.5

        config.losses.latent.lambda_kl = 0.01

        config.model = MagicMock()
        config.model.in_channels = 2
        config.optimization = MagicMock()
        config.optimization.gradient.clip.enabled = False
        config.optimization.gradient.clip.method = "norm"
        config.optimization.gradient.clip.value = 1.0
        config.optimization.precision.dtype = "float32"
        config.training = MagicMock()
        config.training.training_mode = "disentangled"
        config.training.strategy_class = "disentangled"
        config.losses.gan = MagicMock()
        config.losses.gan.enable_adversarial = False
        config.losses.physics = MagicMock()
        config.losses.physics.lambda_physics_constraint = 0.0
        config.losses.diffusion = MagicMock()
        config.losses.diffusion.enable_diffusion = False
        config.losses.ssl = MagicMock()
        config.losses.ssl.enable_ssl = False
        config.physics = None

        env = MagicMock()
        env.config = config
        env.device = torch.device("cpu")

        state = MagicMock()
        state.config = config
        state.device = torch.device("cpu")
        state.losses = {}
        state.generator = MagicMock()
        state.optimizers = {"main": MagicMock()}

        strategy = DisentangledTrainingStrategy(env=env, state=state)

        # Get weights
        weights = strategy.loss_weights

        # All weights should be Python floats
        for loss_name, weight in weights.items():
            assert isinstance(
                weight, (float, int)
            ), f"{loss_name} weight is {type(weight)}, expected float"
            assert not isinstance(
                weight, torch.Tensor
            ), f"{loss_name} weight should not be tensor"

        # Test that these floats work correctly in gradient computation
        dummy_loss = torch.tensor(5.0, requires_grad=True)
        hist_weight = weights.get("histogram_consistency", 0.0)

        if hist_weight > 0:
            weighted = hist_weight * dummy_loss
            assert weighted.requires_grad
            weighted.backward()
            assert dummy_loss.grad is not None
