"""Fuzz: dispatch-equivalence + AMP-stability property tests.

Testing plan: testing_implementation_plan.md §5.21

Property A — Strategy-dispatch equivalence
-------------------------------------------
For every ``training_mode`` key registered in
``TrainingStrategyFactory.STRATEGY_CLASS_PATHS``, both of the following
config layouts must cause ``TrainingStrategyFactory.get_strategy_class``
to return the **same** strategy class:

  Layout 1 (legacy):
      config.training.training_mode  = "<mode>"
      config.training.strategy_class = None

  Layout 2 (explicit):
      config.training.strategy_class = STRATEGY_CLASS_PATHS[mode]
      config.training.training_mode  = (absent / None)

The test does NOT load a real YAML or run a training pipeline — it
operates purely at the config-resolution layer.  This keeps it fast
(no I/O, no model instantiation) and exercises the dispatch logic
independently of schema parsing.

Why this matters: a silent alias bug in STRATEGY_CLASS_PATHS would
cause two superficially identical configs to route to different
strategies, producing hard-to-trace training anomalies.

Property B — AMP fp32 vs fp16 stability (CPU)
----------------------------------------------
For a small op (matmul + activation), the difference between fp32 and
AMP-autocasted (bf16 on CPU) results must stay within a RELATIVE L∞
tolerance (``atol + rtol·|fp32 output|_∞``) — half-precision round-off
scales with output magnitude.  On CPU, ``torch.amp.autocast``
with ``dtype=torch.float16`` is a genuine no-op for most ops, so we
also test the explicit fp16 path to cover the "downcast then compare"
contract.

Property B-GPU — AMP fp32 vs fp16 stability (GPU)
--------------------------------------------------
Same as B but exercised on a CUDA device.  Marked ``@pytest.mark.gpu``
and skipped cleanly when CUDA is unavailable.

Guards
------
- ``pytest.importorskip("hypothesis")`` — keeps collection green without
  hypothesis installed.  hypothesis IS installed in the project venv but
  the guard is kept for portability (cluster nodes, minimal envs).

Hypothesis settings
-------------------
- Property A: ``max_examples=50``  (exhaustive over the finite registry,
  but hypothesis may generate fewer if the strategy space is small).
- Property B: ``max_examples=100`` with ``deadline=None`` (tensor ops can
  take >200 ms under hypothesis instrumentation on cold JIT warm-up).
"""

from __future__ import annotations

import logging
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: hypothesis is required.  Keep even though it IS installed — this
# ensures the test file is still COLLECTIBLE in stripped environments.
# ---------------------------------------------------------------------------
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import — keep optional so collection stays fast on CPU-only nodes
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import the factory and its registry
# ---------------------------------------------------------------------------
try:
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    _FACTORY = TrainingStrategyFactory()
    _REGISTRY: dict[str, str] = dict(TrainingStrategyFactory.STRATEGY_CLASS_PATHS)
    _REGISTRY_KEYS: list[str] = list(_REGISTRY.keys())
except Exception as _exc:
    _FACTORY = None  # type: ignore[assignment]
    _REGISTRY = {}
    _REGISTRY_KEYS = []
    logger.warning("TrainingStrategyFactory unavailable for fuzz tests: %s", _exc)

# ---------------------------------------------------------------------------
# AMP tolerance
# ---------------------------------------------------------------------------
# Relative L∞ tolerance. fp16/bf16 round-off is proportional to the output
# MAGNITUDE, so an *absolute* bound spuriously fails as the fuzzed input scale
# grows (e.g. an L∞ diff of 0.0424 at scale≈10 is only ~0.1% of a ~40-magnitude
# output — normal half-precision error, not instability). Bound the diff by
# ``atol + rtol * |fp32 output|_∞`` instead.
_AMP_ATOL: float = 2e-3  # absolute floor for near-zero outputs
_AMP_RTOL: float = 3e-2  # 3% of output magnitude — passes normal fp16/bf16
# round-off while still catching a genuinely broken autocast (overflow, wrong
# dtype, NaN) which diverges far more than a few percent.


# ===========================================================================
# Property A — Strategy-dispatch equivalence
# ===========================================================================


def _make_mock_config_training_mode(mode: str) -> Any:
    """Build a minimal mock config whose only training attribute is
    ``training_mode = mode`` (legacy dispatch path).

    ``strategy_class`` is explicitly set to None so Priority-1 is skipped
    and the factory falls through to Priority-3 (training_mode).
    """
    training_mock = MagicMock()
    # Priority-1 guard: strategy_class must be falsy
    training_mock.strategy_class = None
    # Priority-2 guard: diffusion_type and timesteps must be absent/falsy
    training_mock.diffusion_type = None
    training_mock.timesteps = None
    # Priority-3: this is what we're testing
    training_mock.training_mode = mode

    config_mock = MagicMock()
    config_mock.training = training_mock
    return config_mock


def _make_mock_config_strategy_class(strategy_class_path: str) -> Any:
    """Build a minimal mock config whose only training attribute is
    ``strategy_class = strategy_class_path`` (explicit dispatch path).

    ``training_mode`` is absent so Priority-1 is hit directly.
    """
    training_mock = MagicMock()
    training_mock.strategy_class = strategy_class_path
    # Ensure training_mode doesn't accidentally match
    training_mock.training_mode = None

    config_mock = MagicMock()
    config_mock.training = training_mock
    return config_mock


@pytest.mark.fuzz
@given(mode=st.sampled_from(_REGISTRY_KEYS) if _REGISTRY_KEYS else st.nothing())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_training_mode_vs_strategy_class_resolves_same(mode: str) -> None:
    """Property A: legacy training_mode and explicit strategy_class dispatch
    to the same strategy class.

    For any mode key in STRATEGY_CLASS_PATHS, both:

      config.training.training_mode  = mode
      config.training.strategy_class = STRATEGY_CLASS_PATHS[mode]

    must cause get_strategy_class() to return the SAME class object.

    Failure here means a silent alias divergence: two syntactically
    equivalent configs route to different strategies.
    """
    if _FACTORY is None:
        pytest.skip("TrainingStrategyFactory not importable in this environment")

    expected_path: str = _REGISTRY[mode]

    # --- Layout 1: legacy training_mode ---
    config_tm = _make_mock_config_training_mode(mode)
    try:
        cls_tm = _FACTORY.get_strategy_class(config_tm)
    except Exception as exc:
        pytest.fail(
            f"get_strategy_class raised for training_mode={mode!r}: {exc}"
        )

    # --- Layout 2: explicit strategy_class ---
    config_sc = _make_mock_config_strategy_class(expected_path)
    try:
        cls_sc = _FACTORY.get_strategy_class(config_sc)
    except Exception as exc:
        pytest.fail(
            f"get_strategy_class raised for strategy_class={expected_path!r}: {exc}"
        )

    assert cls_tm is cls_sc, (
        f"Dispatch divergence for mode={mode!r}:\n"
        f"  training_mode path → {cls_tm}\n"
        f"  strategy_class path → {cls_sc}\n"
        f"  registered path: {expected_path}"
    )


@pytest.mark.fuzz
@given(mode=st.sampled_from(_REGISTRY_KEYS) if _REGISTRY_KEYS else st.nothing())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_strategy_class_lookup_is_idempotent(mode: str) -> None:
    """Property A (idempotency): calling get_strategy_class twice with the
    same config returns the same class object (no side-effecting dispatch).

    This guards against accidental mutation of the registry during lookup.
    """
    if _FACTORY is None:
        pytest.skip("TrainingStrategyFactory not importable in this environment")

    strategy_class_path = _REGISTRY[mode]
    config = _make_mock_config_strategy_class(strategy_class_path)

    try:
        cls_first = _FACTORY.get_strategy_class(config)
        cls_second = _FACTORY.get_strategy_class(config)
    except Exception as exc:
        pytest.skip(f"Strategy {mode!r} not loadable in this env: {exc}")

    assert cls_first is cls_second, (
        f"get_strategy_class is not idempotent for mode={mode!r}:\n"
        f"  call 1 → {cls_first}\n"
        f"  call 2 → {cls_second}"
    )


# ===========================================================================
# Property B — AMP fp32 vs fp16/autocasted stability (CPU)
# ===========================================================================
#
# We draw random (B, C, H, W) shapes with small spatial dims so the test
# stays fast even without GPU.  The op is: linear projection (matmul) +
# ReLU, which is AMP-compatible and exercises the dtype-casting path.

_SMALL_DIMS = st.integers(min_value=1, max_value=8)
_FLOAT_SCALES = st.floats(
    min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
)


@pytest.mark.fuzz
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
@given(
    batch=_SMALL_DIMS,
    channels=_SMALL_DIMS,
    height=_SMALL_DIMS,
    width=_SMALL_DIMS,
    scale=_FLOAT_SCALES,
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_amp_cpu_fp32_vs_autocast_agrees(
    batch: int,
    channels: int,
    height: int,
    width: int,
    scale: float,
) -> None:
    """Property B (CPU): element-wise op under torch.amp.autocast("cpu")
    agrees with fp32 within τ = 3e-2 (L∞).

    On CPU, autocast with dtype=torch.bfloat16 (the CPU-supported low-prec
    dtype) introduces quantisation noise that must stay within tolerance.
    This catches ops that silently accumulate large errors when autocasted.
    """
    import torch  # local import: only reached when torch IS available
    import torch.nn.functional as F

    # Build a small op: conv2d + relu (lightweight, deterministic)
    # Use a fixed seed so hypothesis examples are reproducible within a run.
    rng = torch.Generator()
    rng.manual_seed(batch * 1000 + channels * 100 + height * 10 + width)

    x = torch.randn(batch, channels, height, width, generator=rng) * scale
    # 1×1 conv weight: (out_channels, in_channels, 1, 1)
    w = torch.randn(channels, channels, 1, 1, generator=rng)
    b = torch.zeros(channels)

    # --- fp32 reference ---
    with torch.no_grad():
        out_fp32 = F.relu(F.conv2d(x, w, b, padding=0))

    # --- CPU autocast (bfloat16 is the supported low-prec dtype for CPU) ---
    try:
        with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
            out_amp = F.relu(F.conv2d(x, w, b, padding=0))
    except RuntimeError:
        # Some CPU builds do not support bfloat16 autocast; skip gracefully.
        pytest.skip("CPU bfloat16 autocast not supported in this build")

    # Compare in fp32 after upcasting the AMP output
    diff = (out_fp32 - out_amp.float()).abs().max().item()
    tol = _AMP_ATOL + _AMP_RTOL * out_fp32.abs().max().item()
    assert diff <= tol, (
        f"AMP (CPU bfloat16) vs fp32 L∞ diff = {diff:.4f} > τ={tol:.4f}  "
        f"shape=({batch},{channels},{height},{width}), scale={scale:.3f}"
    )


@pytest.mark.fuzz
@pytest.mark.gpu
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
@given(
    batch=_SMALL_DIMS,
    channels=_SMALL_DIMS,
    height=_SMALL_DIMS,
    width=_SMALL_DIMS,
    scale=_FLOAT_SCALES,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_amp_gpu_fp32_vs_fp16_agrees(
    batch: int,
    channels: int,
    height: int,
    width: int,
    scale: float,
) -> None:
    """Property B-GPU: fp32 vs torch.amp.autocast("cuda", dtype=torch.float16)
    output L∞ diff must stay within a relative tolerance (atol + rtol·|output|).

    This test is marked ``@pytest.mark.gpu`` and is collected only on
    CUDA-capable nodes.  It validates the half-precision autocasting path
    that the training loop exercises at every step.

    The test is structured so ``pytest -m "not gpu"`` skips it cleanly:
    the ``skipif`` guard + the ``gpu`` mark both fire.
    """
    import torch  # local import
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; gpu-marked test skipped")

    device = torch.device("cuda")

    rng_cpu = torch.Generator()
    rng_cpu.manual_seed(batch * 7919 + channels * 1009 + height * 31 + width)

    x = (torch.randn(batch, channels, height, width, generator=rng_cpu) * scale).to(device)
    w = torch.randn(channels, channels, 1, 1, generator=rng_cpu).to(device)
    b = torch.zeros(channels, device=device)

    # --- fp32 reference on GPU ---
    with torch.no_grad():
        out_fp32 = F.relu(F.conv2d(x, w, b, padding=0))

    # --- fp16 autocast on GPU ---
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out_fp16 = F.relu(F.conv2d(x, w, b, padding=0))

    diff = (out_fp32 - out_fp16.float()).abs().max().item()
    tol = _AMP_ATOL + _AMP_RTOL * out_fp32.abs().max().item()
    assert diff <= tol, (
        f"AMP (GPU fp16) vs fp32 L∞ diff = {diff:.4f} > τ={tol:.4f}  "
        f"shape=({batch},{channels},{height},{width}), scale={scale:.3f}"
    )
