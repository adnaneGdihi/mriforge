"""Unit tests for NoiseToNoiseStrategy.

The N2N strategy was refactored to route loss computation through
``UnifiedReconstructionLossComputer`` (SSOT). Two behavioural
consequences the tests below verify:

1. The strategy no longer owns its own ``criterion`` attribute —
   loss computation lives in the shared loss computer.
2. Missing or insufficient ``image_reps`` raises ``ValueError`` rather
   than silently returning a 0-loss (CLAUDE.md pitfall #9 — no silent
   fallbacks).
"""

from unittest.mock import MagicMock, PropertyMock

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.n2n_strategy import NoiseToNoiseStrategy
from mriforge.models.losses.computers import UnifiedReconstructionLossComputer


@pytest.fixture
def mock_model():
    """Mock generator returning a (B, C, H, W) tensor."""
    model = MagicMock(spec=nn.Module)
    model.training = True
    model.return_value = torch.randn(2, 1, 64, 64)
    return model


@pytest.fixture
def mock_optimizer():
    """Mock optimizer."""
    return MagicMock()


@pytest.fixture
def strategy(mock_model, mock_optimizer):
    """Instantiate NoiseToNoiseStrategy with a mocked TrainingEnvironment."""
    from mriforge.infrastructure.training.builders.environment import TrainingEnvironment

    mock_env = MagicMock(spec=TrainingEnvironment)
    mock_env.models = {"generator": mock_model}
    mock_env.optimizers = {"main": mock_optimizer}
    mock_env.losses = {}
    mock_env.device = torch.device("cpu")
    mock_env.config = MagicMock()

    mock_env.config.optimization = MagicMock()
    mock_env.config.optimization.gradient.accumulation_steps = 1
    mock_env.config.optimization.gradient.clip.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    mock_env.config.optimization.precision.dtype = "float32"
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    mock_env.config.optimization.gradient.clip.method = "norm"
    mock_env.config.optimization.gradient.clip.value = 1.0

    mock_env.config.logging = MagicMock()
    mock_env.config.logging.log_gradients = False
    mock_env.config.logging.log_interval = 10
    # Disable debug-snapshot saving so test runs don't leak
    # ``MagicMock/mock.config.training.output_dir/<id>/`` directories
    # into the repo root. The snapshot code uses the (stringified)
    # mock config path to build its output directory.
    #
    # Nested spelling is load-bearing. `_resolve_config` reads
    # `logging.snapshots`, which a MagicMock auto-creates with truthy,
    # int-able leaves -- so the retired flat keys below resolved to
    # `enabled=True, save_images=True, save_json=True`, the exact opposite of
    # what this block asks for. Nothing went red because nothing asserted it;
    # `save_debug_snapshot`'s `run_dir` type guard was what still stopped the
    # leak. Pinned by `test_fixture_actually_disables_snapshots`.
    mock_env.config.logging.snapshots.enabled = False
    mock_env.config.logging.snapshots.max_calls = 1
    mock_env.config.logging.snapshots.save_images = False
    mock_env.config.logging.snapshots.save_json = False
    mock_env.config.logging.snapshots.interval_steps = 0

    type(mock_env).generator = PropertyMock(return_value=mock_model)
    type(mock_env).opt_g = PropertyMock(return_value=mock_optimizer)

    return NoiseToNoiseStrategy(env=mock_env)


def test_fixture_actually_disables_snapshots(strategy):
    """The fixture's snapshot off-switch must really be off.

    Asked of the resolver rather than of the attribute the fixture just set --
    reading back our own assignment is what let this rot unnoticed. After the
    block decomposition the flat keys stopped being read, and ``_resolve_config``
    fell through to the MagicMock's auto-vivified ``logging.snapshots``, whose
    leaves are truthy and int-able: the fixture resolved to snapshots ENABLED,
    with images and JSON on. No test failed, because none asked. What still
    prevented the ``MagicMock/`` directory leak this fixture was written to
    stop was the unrelated ``run_dir`` type guard in ``save_debug_snapshot``.
    """
    from mriforge.infrastructure.training.debug_snapshot import _resolve_config

    resolved = _resolve_config(strategy.config.logging)
    assert resolved.enabled is False
    assert resolved.save_images is False
    assert resolved.save_json is False


def test_initialization(strategy):
    """N2N owns a UnifiedReconstructionLossComputer (SSOT loss path)."""
    assert strategy is not None
    assert isinstance(strategy.loss_computer, UnifiedReconstructionLossComputer)


def test_unpack_batch_uses_two_distinct_repetitions(strategy):
    """Input/target are two DISTINCT repetitions drawn from image_reps.

    The source picks two distinct reps at random each step (audit 2026-06: N2N
    only needs two independent noisy draws, and fixing 0/1 wasted extra reps and
    biased the pairing). Assert membership + distinctness, NOT fixed indices —
    a fixed-index assert would be non-deterministically wrong.
    """
    reps = torch.randn(2, 3, 1, 64, 64)
    batch = {"image_reps": reps}
    input_batch, target_batch = strategy._unpack_batch(batch)
    assert input_batch.shape == (2, 1, 64, 64)
    assert target_batch.shape == (2, 1, 64, 64)
    in_idx = [k for k in range(reps.shape[1]) if torch.equal(input_batch, reps[:, k])]
    tg_idx = [k for k in range(reps.shape[1]) if torch.equal(target_batch, reps[:, k])]
    assert len(in_idx) == 1 and len(tg_idx) == 1  # each is exactly one real rep
    assert in_idx[0] != tg_idx[0]  # and they are two different reps


def test_compute_losses_raises_on_missing_reps(strategy):
    """No input/target → ValueError (pitfall #9: no silent fallbacks)."""
    with pytest.raises(ValueError, match="Noise2Noise strategy requires"):
        strategy._compute_losses_impl(
            input_batch=None, target_batch=None, epoch=0
        )


def test_compute_losses_raises_on_insufficient_reps(strategy):
    """Only one repetition → ``_unpack_batch`` itself raises loudly.

    Previously single-rep data fell through to the generic mixin path,
    returned ``(None, None)``, and only failed later with a misleading
    "requires both" message. The 2026-06 audit fix raises at the right
    layer with an explicit single-rep diagnostic.
    """
    reps = torch.randn(2, 1, 1, 64, 64)  # only 1 rep
    batch = {"image_reps": reps}
    with pytest.raises(ValueError, match="at least 2"):
        strategy._unpack_batch(batch)
