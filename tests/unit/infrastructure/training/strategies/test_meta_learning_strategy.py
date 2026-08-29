"""Unit tests for Meta-Learning training strategy.

Tests meta-learning/few-shot adaptation capabilities.
"""

import inspect
from unittest.mock import MagicMock

import pytest

from tests.utils.config_block_stub import block_stub
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies import (
    meta_learning_strategy as _mls,
)
from mriforge.infrastructure.training.strategies.meta_learning_strategy import (
    MetaLearningTrainingStrategy,
)


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


class SimpleMetaLearner(nn.Module):
    """Minimal meta-learning model for testing.

    Implements ``adapt_to_domain`` so the strategy's meta-aware
    contract is satisfied; otherwise the strategy raises (audit B2 fix
    — silent fallback to standard training has been removed).
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Accept and ignore meta-learning kwargs (``domain_id``,
        # ``adaptation_mode``) so the strategy's call sites don't fail
        # the kwarg-shape check.
        x = torch.relu(self.conv1(x))
        return self.conv2(x)

    def adapt_to_domain(self, *args, **kwargs) -> dict[str, float]:
        """No-op meta adaptation hook returning the stats shape the
        strategy expects (``{"final_loss": float, ...}``)."""
        return {"final_loss": 0.0}


class _NonMetaModel(nn.Module):
    """Minimal model WITHOUT ``adapt_to_domain`` to exercise the fail-loud path."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture
def mock_env() -> MagicMock:
    """Create mock training environment for meta-learning (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator
    gen = SimpleMetaLearner()
    gen.eval()
    env.generator = gen
    env.models = {"generator": gen}

    # Config using MockConfig
    config = MockConfig()

    config.training = MockConfig(training_mode="meta_learning", log_interval=100)

    # The SHARED stub. Production reads `optimization.gradient.clip.method`
    # (standard_optimizer_stepper.py:30, `method not in _valid_clip`), and a
    # bare MockConfig fabricates a fresh MockConfig for `gradient` -- which is
    # unhashable, so the membership test raises rather than the value being
    # wrong. block_stub routes the flat legacy kwargs from RENAMES.
    config.optimization = block_stub(
        "optimization",
        use_amp=False,
        enable_gradient_clipping=False,
        learning_rate=1e-3,
        gradient_clip_method="norm",
        gradient_clip_value=1.0,
    )

    # Meta-learning specific config
    config.meta_learning = MockConfig(
        num_inner_steps=5, inner_lr=1e-2, adaptation_strategy="maml"
    )

    config.objectives = MockConfig()
    config.objectives.reconstruction = MockConfig(lambda_l1=1.0)

    config.losses = MockConfig()
    config.losses.reconstruction = MockConfig(enable_l1=False, lambda_l1=0.0)

    config.model = MockConfig(model_type="meta_learning", domain="image")
    config.physics = MockConfig()
    config.physics.kspace = MockConfig(enable_kspace_recon=False)
    # `logging.log_interval` folded to `logging.intervals.log`, which the
    # training loop reads in a modulo -- a fabricated MockConfig there is a
    # TypeError, not a wrong interval.
    config.logging = block_stub("logging", log_interval=10)

    env.config = config
    env.model_type = "meta_learning"

    return env


class TestMetaLearningStrategyInitialization:
    """Test meta-learning strategy initialization."""

    def test_initialization_success(self, mock_env):
        """Test successful initialization."""
        strategy = MetaLearningTrainingStrategy(env=mock_env)

        assert strategy is not None
        assert strategy.env == mock_env

    def test_requires_meta_config(self, mock_env):
        """Test that meta-learning requires specific config."""
        strategy = MetaLearningTrainingStrategy(env=mock_env)

        # Verify meta-learning config
        assert hasattr(strategy.env.config, "meta_learning")
        assert hasattr(strategy.env.config.meta_learning, "num_inner_steps")
        assert strategy.env.config.meta_learning.num_inner_steps > 0


class TestMetaLearningLossComputation:
    """Test meta-learning loss computation."""

    def test_compute_losses_returns_dict(self, mock_env):
        """Test that _compute_losses_impl returns dict of losses."""
        strategy = MetaLearningTrainingStrategy(env=mock_env)

        # Create support and query batches
        batch_size = 4
        input_batch = torch.randn(batch_size, 1, 32, 32)
        target_batch = torch.randn(batch_size, 1, 32, 32)

        # Compute losses
        losses = strategy._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=0
        )

        # Verify structure
        assert isinstance(losses, dict)
        assert "g_total_loss" in losses
        assert isinstance(losses["g_total_loss"], torch.Tensor)


class TestMetaLearningAdaptation:
    """Test meta-learning adaptation mechanics."""

    def test_inner_loop_steps_configured(self, mock_env):
        """Test that inner adaptation steps are configured."""
        strategy = MetaLearningTrainingStrategy(env=mock_env)

        num_inner_steps = strategy.env.config.meta_learning.num_inner_steps
        assert num_inner_steps > 0
        assert isinstance(num_inner_steps, int)

    def test_meta_hyperparams_read_from_production_homes(self, mock_env):
        """``_meta_hyperparams`` must read the knobs from the REAL config homes
        used by ``exp_meta_varnet.yaml`` — ``model.model_kwargs`` (meta_lr_inner /
        inner_steps) and ``training.meta`` (first_order) — not only a root
        ``meta_learning`` block. A root-only reader would silently fall back to
        defaults against the production config (CLAUDE.md pitfall #15)."""
        strategy = MetaLearningTrainingStrategy(env=mock_env)
        strategy.env.config = MockConfig(
            meta_learning=None,
            model=MockConfig(
                model_kwargs={"meta_lr_inner": 0.05, "inner_steps": 3}
            ),
            training=MockConfig(meta=MockConfig(first_order=True)),
        )

        inner_lr, inner_steps, first_order = strategy._meta_hyperparams()
        assert inner_lr == 0.05
        assert inner_steps == 3
        assert first_order is True


class TestMetaLearningFailLoud:
    """Regression tests for the audit-B2 fail-loud behaviour."""

    def test_runs_on_plain_model_via_functional_call(self, mock_env):
        """A model WITHOUT ``adapt_to_domain`` now trains via the functional_call
        MAML — no TypeError, and the meta loss carries gradient to the base
        params. (Replaces the old fail-loud-on-missing-adapt_to_domain contract:
        NO registered model implements adapt_to_domain, so requiring it made the
        whole paradigm untrainable — 2026-05-31 audit.)"""
        mock_env.generator = _NonMetaModel()
        mock_env.models = {"generator": mock_env.generator}

        strategy = MetaLearningTrainingStrategy(env=mock_env)
        out = strategy._compute_losses_impl(
            input_batch=torch.randn(4, 1, 8, 8),
            target_batch=torch.randn(4, 1, 8, 8),
            epoch=0,
        )
        assert isinstance(out, dict)
        assert out["g_total_loss"].requires_grad

    def test_raises_on_batch_size_one(self, mock_env):
        """batch_size < 2 cannot be split into support/query — must raise."""
        strategy = MetaLearningTrainingStrategy(env=mock_env)
        with pytest.raises(ValueError, match="batch_size >= 2"):
            strategy._compute_losses_impl(
                input_batch=torch.randn(1, 1, 8, 8),
                target_batch=torch.randn(1, 1, 8, 8),
                epoch=0,
            )

class TestNoStandardStepFallback:
    """Regression: the meta-learning strategy must NOT carry a
    `_standard_step` fallback method (CLAUDE.md pitfall #9 -- no silent
    fallbacks). The real path is `_compute_losses_impl` (the MAML inner/outer
    loop); a `_standard_step` shim would let the strategy silently degrade to
    plain reconstruction. This test fails on the pre-fix code (where
    `_standard_step` was defined) and passes once it is removed.
    """

    @staticmethod
    def _strategy_class():
        classes = [
            obj
            for _, obj in inspect.getmembers(_mls, inspect.isclass)
            if obj.__module__ == _mls.__name__
        ]
        assert len(classes) == 1, (
            "expected exactly one strategy class in meta_learning_strategy"
        )
        return classes[0]

    def test_standard_step_fallback_removed(self):
        """No `_standard_step` attribute anywhere on the class/MRO."""
        cls = self._strategy_class()
        assert not hasattr(cls, "_standard_step"), (
            "meta-learning strategy must not expose a `_standard_step` "
            "fallback (forbidden silent-degradation path, pitfall #9)"
        )

    def test_real_loss_path_present(self):
        """The canonical loss-path methods remain wired after the dead
        fallback was removed (`_forward` is a local closure inside
        `_compute_losses_impl`, so it is intentionally NOT a class attr)."""
        cls = self._strategy_class()
        assert hasattr(cls, "_compute_losses_impl")
        assert hasattr(cls, "_inner_loss")
        assert hasattr(cls, "_meta_hyperparams")
