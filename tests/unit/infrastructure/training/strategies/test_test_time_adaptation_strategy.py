"""Unit tests for TttAdaptationStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.strategies.test_time_adaptation_strategy import (
    TttAdaptationStrategy,
)

# Shared patch target for BaseTrainingStrategy.__init__ LossBuilder dependency
_LOSS_BUILDER_PATCH = "spectramr.infrastructure.training.builders.loss_builder.LossBuilder"


def _mock_loss_builder():
    """Create a MagicMock LossBuilder whose chained .build() returns {}."""
    mock_lb = MagicMock()
    builder_instance = MagicMock()
    builder_instance.build.return_value = {}
    mock_lb.return_value = builder_instance
    return mock_lb


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Generator
    gen = MagicMock()
    gen.return_value = torch.randn(2, 1, 64, 64)
    gen.training = True
    gen.train = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    # FFT Transformer
    env.fft_transformer = MagicMock()
    env.fft_transformer.forward.return_value = torch.randn(2, 1, 64, 64)

    # Optimizer
    opt_g = MagicMock()
    env.opt_g = opt_g
    env.optimizers = {"opt_g": opt_g}

    # Config. These tests exercise the LEGACY ``training.adaptation_config``
    # fallback, so the canonical ``training.ttt`` block must be absent — a bare
    # MagicMock would auto-vivify ``ttt`` and shadow the fallback the source
    # reads from (test_time_adaptation_strategy.py:144-152).
    env.config = MagicMock()
    env.config.training.ttt = None
    env.config.training.adaptation_config.adaptation_steps = 3
    env.config.training.adaptation_config.adaptation_lr = 1e-4
    env.config.optimization.optimizer.learning_rate = 1e-4
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    env.config.optimization.gradient.clip.enabled = False
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    # mixed_precision.py validates amp_dtype against a fixed set; a bare
    # MagicMock fails it. Pin a valid value (stale-fixture repair, 2026-07-01).
    env.config.optimization.precision.dtype = "float32"
    env.config.model.model_type = "ttt_adaptation"
    env.model_type = "ttt_adaptation"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate TttAdaptationStrategy (NEW API - env only)."""
    with patch(_LOSS_BUILDER_PATCH, new_callable=_mock_loss_builder):
        return TttAdaptationStrategy(env=mock_env)


def test_initialization(strategy):
    """Test initialization with config."""
    assert strategy.num_adaptation_steps == 3
    assert strategy.consistency_weight == 1.0


def test_initialization_missing_config(mock_env):
    """Test error when adaptation config is missing."""
    del mock_env.config.training.adaptation_config
    with pytest.raises(ConfigurationError):
        with patch(_LOSS_BUILDER_PATCH, new_callable=_mock_loss_builder):
            TttAdaptationStrategy(env=mock_env)


def test_train_step_loop(strategy, mock_env):
    """Test train_step performs adaptation loop."""
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)

    # Mock data consistency inputs via kwargs
    mask = torch.ones(2, 1, 64, 64)
    kspace = torch.randn(2, 1, 64, 64)

    # Mock amp helper and context for adaptation loop
    strategy.amp_helper = MagicMock()
    strategy.amp_helper.get_autocast_context.return_value = MagicMock()
    strategy.amp_helper.get_autocast_context.return_value.__enter__.return_value = None
    strategy.amp_helper.get_autocast_context.return_value.__exit__.return_value = None
    strategy.amp_policy = MagicMock()

    # Mock context's fft_transformer. The real one is passed a 4-D [B, C, H, W]
    # batch here, which the volume-shaped uncentered primitives cannot take -- see
    # the comment on the strategy's own call site. The mocked method is ``fftnc``
    # because #1350 corrected this call from the uncentered ``fft2`` to the centred
    # entry point; mocking the retired name would let a MagicMock reach torch.norm.
    strategy.context = MagicMock()
    strategy.context.fft_transformer.fftnc.return_value = torch.randn(2, 1, 64, 64)

    steps = strategy.train_step(
        batch=None,
        epoch=0,
        input_batch=input_batch,
        target_batch=target_batch,
        mask=mask,
        kspace=kspace,
    )
    loss = steps[0]["closure"]()
    metrics = strategy.get_last_metrics()

    assert "g_total_loss" in metrics
    # TTT does 3 adaptation steps: 2 intermediate (opt_g.step) + 1 final (handled by Trainer)
    assert mock_env.opt_g.step.call_count >= 2
    mock_env.generator.assert_called()


def test_compute_losses_impl_returns_losses(strategy):
    """Test that _compute_losses_impl returns a dict with g_total_loss."""
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)

    # Mock amp_helper scaler for adaptation loop
    strategy.amp_helper = MagicMock()

    losses = strategy._compute_losses_impl(input_batch, target_batch, 0)

    assert isinstance(losses, dict)
    assert "g_total_loss" in losses


def test_ttt_closure_computes_once_per_step_and_reports_final(strategy, mock_env):
    """Regression (2026-07-01): the TTT closure must compute losses exactly
    ``num_adaptation_steps`` times and report the FINAL step's metrics.

    The old code stored ``_last_step_metrics = {k: v.item() ...}`` on EVERY inner
    adaptation step — a per-step GPU-sync storm — only for the final-step store
    to overwrite it. Removing the dead intermediate store must NOT change the
    observable metric (still the final step) nor the number of loss computations.
    """
    strategy.amp_helper = MagicMock()
    strategy.amp_helper.get_autocast_context.return_value = MagicMock()
    strategy.amp_helper.get_autocast_context.return_value.__enter__.return_value = None
    strategy.amp_helper.get_autocast_context.return_value.__exit__.return_value = None
    strategy.amp_policy = MagicMock()
    strategy.context = MagicMock()

    seen_steps: list[int] = []

    def _fake_losses(inp, tgt, epoch, step=0, **kw):  # noqa: ANN001
        seen_steps.append(step)
        # loss must require grad for the inner loop's .backward().
        return {"g_total_loss": torch.zeros((), requires_grad=True) + float(step)}

    strategy._compute_losses_impl = _fake_losses

    steps = strategy.train_step(
        batch=None,
        epoch=0,
        input_batch=torch.randn(2, 1, 8, 8),
        target_batch=torch.randn(2, 1, 8, 8),
    )
    final_loss = steps[0]["closure"]()

    # num_adaptation_steps == 3 → steps 0,1 (inner) + step 2 (final), no extras.
    assert seen_steps == [0, 1, 2], seen_steps
    metrics = strategy.get_last_metrics()
    # Reported metric is the FINAL step's value (2.0), not an intermediate one.
    assert metrics["g_total_loss"] == pytest.approx(2.0)
    assert float(final_loss.detach()) == pytest.approx(2.0)
