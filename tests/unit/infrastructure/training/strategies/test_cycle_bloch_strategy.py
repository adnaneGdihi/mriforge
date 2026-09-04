"""Unit tests for CycleBlochStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.training.strategies.cycle_bloch_strategy import (
    CycleBlochStrategy,
    ParameterEstimator,
)


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator/discriminator
    gen = MagicMock()
    gen.return_value = torch.randn(2, 1, 64, 64)  # fake_hf
    gen.training = True
    gen.train = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    disc = MagicMock()
    disc.return_value = torch.randn(2, 1, 64, 64)  # pred_fake
    disc.training = True
    disc.train = MagicMock()
    env.discriminator = disc
    env.models["discriminator"] = disc

    # Mock optimizers
    env.opt_g = MagicMock()
    env.opt_d = MagicMock()
    env.optimizers = {"opt_g": env.opt_g, "opt_d": env.opt_d}

    # Config — must give concrete values for the SSOT loss-weight read so the
    # strategy doesn't materialise lambda_cycle as a MagicMock attribute.
    env.config = MagicMock()
    env.config.training.loss_weights = {"cycle_bloch": 5.0, "adv": 2.0}
    env.config.optimization.optimizer.learning_rate = 1e-4
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    env.config.optimization.gradient.clip.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    env.config.optimization.precision.dtype = "float32"
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    env.config.model.model_type = "cycle_bloch"
    env.config.losses.gan.lambda_cycle_bloch = 5.0
    env.config.losses.gan.lambda_cycle_adv = 2.0
    env.model_type = "cycle_bloch"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate CycleBlochStrategy (NEW API - env only)."""
    with patch("spectramr.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        return CycleBlochStrategy(env=mock_env)


def test_initialization(strategy):
    """Test strategy initialization.

    The ParameterEstimator is created lazily on first forward pass to
    match the actual generator output-channel count (which a
    ``ChannelAdapter`` may rewrite). At construction time
    ``strategy.estimator`` is therefore ``None`` — it materialises
    after the first call to ``_get_estimator``.
    """
    assert strategy is not None
    assert strategy.estimator is None
    assert strategy.lambda_cycle == 5.0
    assert strategy.lambda_adv == 2.0
    # Lazy materialisation works.
    dummy = torch.zeros(1, 1, 4, 4)
    est = strategy._get_estimator(dummy)
    assert isinstance(est, ParameterEstimator)


def test_train_step(strategy, mock_env):
    """Test train_step execution and loss computation."""
    input_batch = torch.randn(2, 1, 64, 64, requires_grad=True)  # ULF
    target_batch = torch.randn(2, 1, 64, 64, requires_grad=True)  # HF

    # Estimator is lazy-init; materialise it once so we can patch its forward.
    strategy._get_estimator(target_batch)

    # Mock estimator output (PD, T1, T2)
    with patch.object(
        strategy.estimator,
        "forward",
        return_value=torch.randn(2, 3, 64, 64, requires_grad=True),
    ) as mock_est:
        # Mock BlochSimulator
        with patch(
            "spectramr.infrastructure.training.strategies.cycle_bloch_strategy.BlochSimulator.spin_echo"
        ) as mock_bloch:
            mock_bloch.return_value = torch.randn(
                2, 1, 64, 64, requires_grad=True
            )  # rec_ulf

            # Mock criteria to return scalar tensors with gradients
            strategy.criterion_gan = MagicMock(
                return_value=torch.tensor(1.0, requires_grad=True)
            )
            strategy.criterion_cycle = MagicMock(
                return_value=torch.tensor(1.0, requires_grad=True)
            )

            # Mock discriminator
            mock_env.discriminator.return_value = torch.randn(2, 1, requires_grad=True)

            steps = strategy.train_step(
                batch=None,
                epoch=0,
                iteration=0,
                input_batch=input_batch,
                target_batch=target_batch,
            )

            # Execute closures to get metrics
            for step in steps:
                step["closure"]()

            metrics = strategy.get_last_metrics()

            assert isinstance(metrics, dict)
            assert "loss_g" in metrics
            assert "loss_cycle" in metrics
            assert "loss_d" in metrics

            # Verify flow
            mock_est.assert_called()
            mock_bloch.assert_called()


def test_train_step_missing_discriminator(strategy, mock_env):
    """Test error when discriminator is missing."""
    mock_env.discriminator = None
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)

    with pytest.raises(ValueError, match="CycleBlochStrategy requires a discriminator"):
        strategy.train_step(None, 0, input_batch=input_batch, target_batch=input_batch)
