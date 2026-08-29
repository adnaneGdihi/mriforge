"""Unit tests for B0MappingStrategy."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from tests.utils.config_block_stub import block_stub
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.b0_mapping_strategy import B0MappingStrategy


class MockConfig:
    """Mock config that returns numeric defaults and supports nested access."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        # For lambda_* return 0.0 (loss weights)
        if name.startswith("lambda_"):
            return 0.0
        # For enable_* return False (feature flags)
        if name.startswith("enable_"):
            return False
        # For *_weight return 0.0 (regularization weights)
        if name.endswith("_weight"):
            return 0.0
        # For *_steps return 1 (gradient accumulation, etc.)
        if "steps" in name:
            return 1
        # For everything else, return a new MockConfig (supports infinite nesting)
        return MockConfig()

    def __gt__(self, other):
        """Support > comparison (always False for weight comparisons)."""
        return False

    def __lt__(self, other):
        """Support < comparison."""
        return False

    def __ge__(self, other):
        """Support >= comparison."""
        return False

    def __le__(self, other):
        """Support <= comparison."""
        return False

    def __eq__(self, other):
        """Support == comparison."""
        return other == 0 or other is False or other is None

    def __bool__(self):
        """Support boolean evaluation (False for disabled features)."""
        return False

    def __int__(self):
        """Support int() conversion (for gradient_accumulation_steps, etc.)."""
        return 1

    def __float__(self):
        """Support float() conversion."""
        return 0.0

    def __index__(self):
        """Support indexing operations."""
        return 1


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator (B0 estimator) to return corrected image and flow
    gen = MagicMock()
    gen.return_value = (
        torch.randn(2, 1, 64, 64),  # corrected
        torch.randn(2, 2, 64, 64),  # flow
    )
    gen.training = True
    gen.train = MagicMock()
    gen.eval = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    # Config using MockConfig that returns proper defaults
    config = MockConfig()
    config.losses = MockConfig()
    config.losses.reconstruction = MockConfig(enable_l1=False, lambda_l1=0.0)
    config.losses.registration = MockConfig(lambda_sim=1.0, lambda_smooth=1.5)
    # The SHARED stub. Production reads `optimization.gradient.clip.method`
    # (standard_optimizer_stepper.py:30, `method not in _valid_clip`), and a
    # bare MockConfig fabricates a fresh MockConfig for `gradient` -- which is
    # unhashable, so the membership test raises rather than the value being
    # wrong. block_stub routes the flat legacy kwargs from RENAMES.
    # The three nested assignments this replaces were no-ops: MockConfig's
    # __getattr__ returns a NEW MockConfig per access, so each one wrote to a
    # throwaway object and the reader still saw a fabricated one.
    config.optimization = block_stub(
        "optimization",
        learning_rate=1e-4,
        enable_gradient_clipping=False,
        use_amp=False,
        gradient_clip_method="norm",
        gradient_clip_value=1.0,
    )
    config.model = MockConfig(model_type="b0_mapping", domain="image")
    config.physics = MockConfig()
    config.physics.kspace = MockConfig(enable_kspace_recon=False)
    # `logging.log_interval` folded to `logging.intervals.log`, which the
    # training loop reads in a modulo -- a fabricated MockConfig there is a
    # TypeError, not a wrong interval.
    config.logging = block_stub("logging", log_interval=10)

    env.config = config
    env.model_type = "b0_mapping"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate B0MappingStrategy (NEW API - env only)."""
    with patch("mriforge.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        return B0MappingStrategy(env=mock_env)


def test_initialization(strategy):
    """Test that strategy initializes correctly with losses."""
    assert strategy is not None
    assert isinstance(strategy.lncc_loss, nn.Module)
    assert isinstance(strategy.smooth_loss, nn.Module)


def test_compute_losses_impl_structure(strategy, mock_env):
    """Test _compute_losses_impl returns correct dictionary keys."""
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)

    losses = strategy._compute_losses_impl(input_batch, target_batch, epoch=0)

    assert isinstance(losses, dict)
    assert "g_total_loss" in losses
    assert "sim_loss" in losses
    assert "smooth_loss" in losses

    # Check generator call arguments
    mock_env.generator.assert_called_with(input_batch, target_batch)


def test_compute_losses_impl_missing_input(strategy):
    """Test that missing inputs raise ValueError."""
    with pytest.raises(ValueError, match="Batch must contain"):
        strategy._compute_losses_impl(None, torch.randn(1, 1, 32, 32), epoch=0)


def test_docstring_is_honest_about_being_registration_not_b0_estimation():
    """F1: the class must not claim to estimate a Hz B0 field — it is a
    deformable image registration (output `flow` is a displacement field)."""
    doc = (B0MappingStrategy.__doc__ or "").lower()
    assert "registration" in doc
    # Must explicitly disclaim being a B0 field estimator.
    assert "not" in doc and "b0" in doc
    assert "displacement" in doc
    # Must point at the real B0-fit strategy.
    assert "multiechob0fitstrategy" in doc.replace(" ", "")


def test_validation_step(strategy, mock_env):
    """Test validation step execution."""
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)

    results = strategy.validation_step(input_batch, target_batch)

    assert isinstance(results, dict)
    assert "val_loss" in results
    assert "val_lncc" in results

    # Check eval mode context
    # Note: We can't easily check context manager behavior on mocks without complex setup,
    # but we can verify generator usage.
    mock_env.generator.assert_called()


def test_validation_logging(strategy, mock_env):
    """Test validation image logging."""
    strategy.logging_service = MagicMock()
    strategy.logging_service.step = 100  # Multiple of 10

    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)

    strategy.validation_step(input_batch, target_batch)

    # Verify log_images calls
    assert strategy.logging_service.log_images.call_count >= 2
    strategy.logging_service.log_images.assert_any_call("val/corrected", ANY, 100)
    strategy.logging_service.log_images.assert_any_call("val/flow", ANY, 100)
