"""Unit tests for PhysicsDrivenTrainingStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.training.strategies.physics_driven_strategy import (
    PhysicsDrivenTrainingStrategy,
)


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    gen = MagicMock()
    gen.return_value = torch.randn(2, 1, 64, 64)
    gen.training = True
    gen.train = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    # Mock config
    env.config = MagicMock()
    env.config.physics.b0_simulation.enabled = True
    env.config.physics.b0_simulation.probability = 1.0
    env.config.physics.b0_simulation.max_hz = 100.0
    env.config.physics.b0_simulation.style = "smooth"
    env.config.physics.field_strength = 3.0
    env.config.physics.pinn.enabled = False
    env.config.physics.data_consistency.enabled = True
    env.config.physics.data_consistency.method = "hard"
    env.config.physics.data_consistency.train_noise_level = 0.01
    env.config.physics.data_consistency.eval_noise_level = 0.005
    env.config.physics.data_consistency.noise_type = "gaussian"

    env.config.losses.reconstruction.lambda_l1 = 1.0  # Base recon loss
    env.config.losses.reconstruction.warmup_iterations = 0
    env.config.losses.physics.lambda_bloch_residual = 0.5
    env.config.optimization.optimizer.learning_rate = 1e-4
    env.config.optimization.gradient.clip.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    env.config.optimization.precision.dtype = "float32"
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    env.config.model.model_type = "physics_driven"
    env.config.model.target_domain = "image"
    env.model_type = "physics_driven"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate PhysicsDrivenTrainingStrategy (NEW API - env only)."""
    with patch("spectramr.infrastructure.physics.field_simulation.B0MapSimulator") as MockSim:
        MockSim.return_value.generate_batch.return_value = torch.randn(2, 1, 64, 64)
        MockSim.return_value.max_hz = 100.0
        strategy = PhysicsDrivenTrainingStrategy(
            env=mock_env, logging_service=MagicMock()
        )
        return strategy


def test_initialization(strategy):
    """Test initialization and component setup."""
    assert strategy is not None
    assert strategy._b0_simulator is not None
    assert strategy._field_strength == 3.0


def test_synthesize_b0_map(strategy, mock_env):
    """Test on-the-fly B0 map synthesis."""
    b0_map = strategy.synthesize_b0_map(2, (64, 64), torch.device("cpu"))
    assert b0_map is not None
    assert b0_map.shape == (2, 1, 64, 64)

    # Test disable via probability
    mock_env.config.physics.b0_simulation.probability = 0.0
    with patch("torch.rand", return_value=torch.tensor(0.9)):  # > 0.0
        b0_map = strategy.synthesize_b0_map(2, (64, 64), torch.device("cpu"))
        assert b0_map is None


def test_compute_losses_impl_structure(strategy, mock_env):
    """Test _compute_losses_impl with Bloch residual."""
    input_batch = torch.randn(2, 3, 64, 64)
    target_batch = torch.randn(2, 3, 64, 64)

    # Mock generator returning M_pred in 5D [B, T, 3, H, W]
    # For this test, we make it 5D to trigger the Bloch loss logic
    mock_env.generator.return_value = torch.randn(2, 5, 3, 64, 64)

    # We also need to update target_batch to match if the loss computer expects it
    # Actually, ReconstructionTrainingStrategy will call generator again
    target_batch_5d = torch.randn(2, 5, 3, 64, 64)

    # Mock batch with T1, T2 maps
    full_batch = {
        "T1": torch.randn(2, 1, 64, 64),
        "T2": torch.randn(2, 1, 64, 64),
        "PD": torch.randn(2, 1, 64, 64),
    }

    with patch("spectramr.models.losses.physics_losses.BlochResidualLoss") as MockLoss:
        # Account for BlochResidualLoss().to(device) pattern
        MockLoss.return_value.to.return_value.return_value = torch.tensor(0.5)

        losses = strategy._compute_losses_impl(
            input_batch, target_batch_5d, epoch=0, batch=full_batch
        )

        assert "g_total_loss" in losses
        assert "bloch_loss" in losses
        assert float(losses["bloch_loss"]) == 0.5


def test_generate_predictions_fp32_enforcement(strategy, mock_env):
    """Test FP32 enforcement during generation."""
    lr_image = torch.randn(2, 1, 64, 64, dtype=torch.float16)  # FP16 input
    forward_kwargs = {}
    batch_context = {"use_dc": False}

    # Since generator is already a MagicMock, we can just check its call
    strategy._generate_predictions(lr_image, forward_kwargs, batch_context)

    # Check that generator was called with float32
    args, kwargs = mock_env.generator.call_args
    assert args[0].dtype == torch.float32


def test_generate_predictions_wires_b0_map(strategy, mock_env):
    """F2: physics.b0_simulation.enabled now feeds a synthesised b0_map into the
    forward call. Previously ``synthesize_b0_map`` had zero callers — the enabled
    simulator produced nothing (facade, pitfall #16)."""
    lr_image = torch.randn(2, 1, 64, 64)
    strategy._generate_predictions(lr_image, {}, {"use_dc": False})
    _, kwargs = mock_env.generator.call_args
    assert "b0_map" in kwargs  # off-resonance field offered to the forward model
    assert kwargs["b0_map"].shape == (2, 1, 64, 64)


def test_generate_predictions_no_b0_map_when_simulator_absent(strategy, mock_env):
    """With no simulator (b0_simulation disabled), no b0_map is injected."""
    strategy._b0_simulator = None
    strategy._generate_predictions(torch.randn(2, 1, 64, 64), {}, {"use_dc": False})
    _, kwargs = mock_env.generator.call_args
    assert "b0_map" not in kwargs
