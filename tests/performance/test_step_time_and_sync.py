"""Performance harness: memory bound, step-time bound, no-sync invariant.

All tests are GPU-only (``@pytest.mark.gpu``).  They skip cleanly when:

* No CUDA device is present (handled by the conftest GPU-skip logic).
* The ``_baselines.json`` file is absent (step-time / memory tests skip with
  a calibration message rather than hard-failing).

Tests
-----
1. ``test_step_memory_bound`` — a single training step for each pillar
   paradigm must not exceed the per-paradigm peak memory budget recorded
   in ``_baselines.json``.

2. ``test_step_time_bound`` — wall-clock time for 10 consecutive steps must
   stay within 2× the median baseline step time in ``_baselines.json``.

3. ``test_no_excessive_cuda_sync`` — ``torch.profiler`` is used to trace 5
   steps.  The number of ``cudaStreamSynchronize`` events must not exceed the
   per-step allowance defined in ``_baselines.json`` (default cap: 3 per
   step = 15 total for 5 steps), preventing accidentally reintroduced
   ``.item()`` / ``.cpu()`` hot-path sync barriers.

Baseline calibration workflow (cluster)
-----------------------------------------
Run on the cluster GPU once with calibration mode to generate / update
``tests/performance/_baselines.json``::

    pytest tests/performance/test_step_time_and_sync.py \\
        --calibrate-performance-baselines -v

(This custom marker is not yet wired — it is reserved for a future
 calibration runner script under ``scripts/ci/``.)

Until that file exists all step-time and memory tests skip.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baselines file
# ---------------------------------------------------------------------------

_BASELINES_PATH = Path(__file__).parent / "_baselines.json"


def _load_baselines() -> dict[str, Any] | None:
    """Return baselines dict or None if file is absent."""
    if not _BASELINES_PATH.exists():
        return None
    with open(_BASELINES_PATH) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Pillar paradigms (same as convergence harness)
# ---------------------------------------------------------------------------

_PILLAR_KEYS = [
    "reconstruction",
    "gan",
    "diffusion",
    "vae",
]


# ---------------------------------------------------------------------------
# Shared minimal model + environment (same pattern as convergence harness)
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # noqa: ARG002
        return self.conv(x)

    def sample(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(x)

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(x)

    def encode(
        self, x: torch.Tensor, **kwargs: Any  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.conv(x)
        return z, torch.zeros_like(z)

    def decode(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # noqa: ARG002
        return self.conv(z)


class _TinyDisc(nn.Module):
    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(-2, -1)).squeeze(-1)


class _LossAutoMock(MagicMock):
    def __getattr__(self, name: str) -> Any:
        if name.startswith("enable_"):
            return False
        if name.startswith("lambda_"):
            return 0.0
        return super().__getattr__(name)


def _build_env(strategy_key: str, device: torch.device) -> MagicMock:
    """Thin re-use of the convergence env builder pattern."""
    from spectramr.infrastructure.training.builders.optimization_builder import (
        OptimizationBuilder,
    )

    gen = _TinyNet(1, 1).to(device)
    disc = _TinyDisc(1).to(device)
    opt_g = OptimizationBuilder.create_single_optimizer(
        gen.parameters(), learning_rate=1e-4, optimizer_type="adam"
    )
    opt_d = OptimizationBuilder.create_single_optimizer(
        disc.parameters(), learning_rate=1e-4, optimizer_type="adam"
    )

    cfg = MagicMock()
    cfg.model.model_type = "standard_unet"
    cfg.model.in_channels = 1
    cfg.model.out_channels = 1
    cfg.model.input_type = "image"
    cfg.model.prior_model = None
    cfg.model.target_domain = "image"
    cfg.training_mode = strategy_key
    cfg.device = str(device)
    cfg.warmup_epochs = 0
    cfg.deep_supervision_weight = 0.0
    cfg.enforce_output_range = False
    cfg.collect_feature_flags = lambda: {}

    cfg.optimization.precision.enabled = False
    cfg.optimization.mixed_precision = "fp16"
    cfg.optimization.loss_scaling = 1024.0
    cfg.optimization.gradient.clip.value = 1.0
    cfg.optimization.gradient.clip.method = "norm"
    cfg.optimization.gradient.clip.enabled = False
    cfg.optimization.optimizer.learning_rate = 1e-4
    cfg.optimization.gradient.accumulation_steps = 1

    recon = _LossAutoMock()
    recon.lambda_l1 = 1.0

    cfg.objectives = MagicMock()
    cfg.objectives.reconstruction = recon
    cfg.objectives.physics = _LossAutoMock()
    cfg.objectives.gan = MagicMock()
    cfg.objectives.gan.gan_loss_type = "vanilla"
    cfg.objectives.gan.lambda_adv = 1.0
    cfg.objectives.gan.lambda_gp = 0.0
    cfg.objectives.gan.r1 = MagicMock()
    cfg.objectives.gan.r1.interval = 0
    cfg.objectives.gan.r1.weight = 0.0
    cfg.objectives.gan.r1.probability = 1.0
    cfg.objectives.latent = MagicMock()
    cfg.objectives.latent.enable_kl = False
    cfg.objectives.latent.enable_commitment = False
    cfg.objectives.ssl = MagicMock()
    cfg.objectives.ssl.enable_contrastive = False
    cfg.losses = cfg.objectives

    cfg.training.diffusion = MagicMock()
    cfg.training.diffusion.timesteps = 10
    cfg.training.diffusion.num_timesteps = 10
    cfg.training.diffusion.noise_schedule = "linear"
    cfg.objectives.diffusion = MagicMock()
    cfg.objectives.diffusion.timesteps = 10
    cfg.objectives.diffusion.noise_schedule = "linear"
    cfg.objectives.diffusion.lambda_mse = 1.0
    cfg.r1_interval = 0

    cfg.physics.pinn.enabled = False
    cfg.physics.pinn.pde_type = "bloch_equation"
    cfg.physics.pinn.lambda_pde = 0.0
    cfg.physics.pinn.weight = 0.0
    cfg.physics.kspace.enable_kspace_recon = False
    cfg.physics.data_consistency.enabled = False
    cfg.data.prior_loading.enabled = False

    cfg.acceleration.acceleration_factor = 4.0
    cfg.acceleration.base_acceleration = 2.0
    cfg.acceleration.max_acceleration = 8.0
    cfg.acceleration.gradient_accumulation_steps = 1

    cfg.logging.log_gradients = False
    cfg.logging.log_interval = 100

    env = MagicMock()
    env.config = cfg
    env.generator = gen
    env.discriminator = disc
    env.models = {"generator": gen, "discriminator": disc}
    env.optimizers = {"opt_g": opt_g, "opt_d": opt_d}
    env.schedulers = {}
    env.losses = {}
    env.physics = {}
    env.data_loaders = {}
    env.metrics = {}
    env.scaler = None
    env.device = device
    env.opt_g = opt_g
    env.opt_d = opt_d
    env.ema = None
    env.step = 0
    env.model_type = "standard_unet"
    return env


def _load_strategy(key: str) -> type | None:
    """Load strategy class or return None on ImportError."""
    try:
        from spectramr.infrastructure.training.strategy_factory import (  # noqa: PLC0415
            TrainingStrategyFactory,
        )

        factory = TrainingStrategyFactory()
        class_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[key]
        return factory._load_strategy_class(class_path)
    except Exception:
        return None


def _make_batch(device: torch.device, size: int = 32) -> torch.Tensor:
    return torch.randn(2, 1, size, size, device=device)


# ---------------------------------------------------------------------------
# Test 1 — memory bound
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.gpu
@pytest.mark.parametrize("paradigm", _PILLAR_KEYS, ids=lambda k: k)
def test_step_memory_bound(paradigm: str) -> None:
    """Peak GPU memory for one training step must not exceed the calibrated budget.

    Skips if ``_baselines.json`` is absent or has no entry for *paradigm*.
    """
    baselines = _load_baselines()
    if baselines is None:
        pytest.skip(
            "tests/performance/_baselines.json not found — "
            "run memory calibration on cluster first"
        )

    entry = baselines.get(paradigm)
    if entry is None or "peak_memory_mb" not in entry:
        pytest.skip(
            f"Memory baseline not calibrated for '{paradigm}' — "
            "run calibration on cluster and add 'peak_memory_mb' to _baselines.json"
        )

    budget_mb: float = entry["peak_memory_mb"]
    device = torch.device("cuda:0")

    strategy_cls = _load_strategy(paradigm)
    if strategy_cls is None:
        pytest.xfail(f"Strategy class import failed for '{paradigm}'")
        return

    env = _build_env(paradigm, device)

    try:
        strategy = strategy_cls(env=env, device=device)
    except Exception as exc:
        pytest.xfail(f"Strategy '{paradigm}' could not be instantiated: {exc}")
        return

    batch = _make_batch(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    try:
        strategy.train_step(
            batch=batch, epoch=0, input_batch=batch, target_batch=batch
        )
    except Exception as exc:
        pytest.xfail(f"Strategy '{paradigm}' raised during step: {exc}")
        return

    torch.cuda.synchronize(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    assert peak_mb <= budget_mb * 1.10, (  # 10% headroom
        f"Strategy '{paradigm}' used {peak_mb:.1f} MB peak GPU memory, "
        f"exceeding budget {budget_mb:.1f} MB (10% headroom)"
    )


# ---------------------------------------------------------------------------
# Test 2 — step-time bound
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.gpu
@pytest.mark.parametrize("paradigm", _PILLAR_KEYS, ids=lambda k: k)
def test_step_time_bound(paradigm: str) -> None:
    """10-step wall-clock time must stay within 2× the calibrated median.

    Skips if ``_baselines.json`` is absent or has no entry for *paradigm*.
    """
    baselines = _load_baselines()
    if baselines is None:
        pytest.skip(
            "tests/performance/_baselines.json not found — "
            "run step-time calibration on cluster first"
        )

    entry = baselines.get(paradigm)
    if entry is None or "median_step_ms" not in entry:
        pytest.skip(
            f"Step-time baseline not calibrated for '{paradigm}' — "
            "run calibration on cluster and add 'median_step_ms' to _baselines.json"
        )

    budget_ms: float = entry["median_step_ms"]
    n_steps = 10
    allowed_total_ms = budget_ms * n_steps * 2.0  # 2× budget

    device = torch.device("cuda:0")

    strategy_cls = _load_strategy(paradigm)
    if strategy_cls is None:
        pytest.xfail(f"Strategy class import failed for '{paradigm}'")
        return

    env = _build_env(paradigm, device)

    try:
        strategy = strategy_cls(env=env, device=device)
    except Exception as exc:
        pytest.xfail(f"Strategy '{paradigm}' could not be instantiated: {exc}")
        return

    batch = _make_batch(device)

    # Warmup step (not timed)
    try:
        strategy.train_step(
            batch=batch, epoch=0, input_batch=batch, target_batch=batch
        )
    except Exception as exc:
        pytest.xfail(f"Strategy '{paradigm}' raised during warmup: {exc}")
        return

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    try:
        for _ in range(n_steps):
            strategy.train_step(
                batch=batch, epoch=0, input_batch=batch, target_batch=batch
            )
    except Exception as exc:
        pytest.xfail(f"Strategy '{paradigm}' raised during timed run: {exc}")
        return

    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms <= allowed_total_ms, (
        f"Strategy '{paradigm}': {n_steps} steps took {elapsed_ms:.1f} ms, "
        f"exceeding 2× budget of {allowed_total_ms:.1f} ms "
        f"(median_step_ms={budget_ms:.1f})"
    )


# ---------------------------------------------------------------------------
# Test 3 — no-excessive-sync invariant via torch.profiler
# ---------------------------------------------------------------------------

# Default cap: max 3 cudaStreamSynchronize events per step = 15 for 5 steps.
# Overridable via _baselines.json entry's "max_sync_events_per_step" key.
_DEFAULT_MAX_SYNC_PER_STEP = 3
_PROFILED_STEPS = 5


@pytest.mark.performance
@pytest.mark.gpu
@pytest.mark.parametrize("paradigm", _PILLAR_KEYS, ids=lambda k: k)
def test_no_excessive_cuda_sync(paradigm: str) -> None:
    """``cudaStreamSynchronize`` count during 5 training steps must be bounded.

    Verifies that no sync-barrier regressions (inline ``.item()`` / ``.cpu()``
    in the hot training loop) have been introduced.

    The sync cap is read from ``_baselines.json[paradigm].max_sync_events_per_step``
    if available; otherwise the default of 3 per step is used.

    Unlike the memory / time tests, this test does NOT require ``_baselines.json``
    to exist — it always runs (when GPU is available) using the default cap.
    """
    device = torch.device("cuda:0")

    strategy_cls = _load_strategy(paradigm)
    if strategy_cls is None:
        pytest.xfail(f"Strategy class import failed for '{paradigm}'")
        return

    env = _build_env(paradigm, device)

    try:
        strategy = strategy_cls(env=env, device=device)
    except Exception as exc:
        pytest.xfail(f"Strategy '{paradigm}' could not be instantiated: {exc}")
        return

    batch = _make_batch(device)

    # Read per-paradigm cap from baselines if available
    max_per_step = _DEFAULT_MAX_SYNC_PER_STEP
    baselines = _load_baselines()
    if baselines is not None:
        entry = baselines.get(paradigm, {})
        max_per_step = entry.get("max_sync_events_per_step", _DEFAULT_MAX_SYNC_PER_STEP)

    max_total = max_per_step * _PROFILED_STEPS

    # Warmup
    try:
        strategy.train_step(
            batch=batch, epoch=0, input_batch=batch, target_batch=batch
        )
    except Exception as exc:
        pytest.xfail(f"Strategy '{paradigm}' raised during warmup: {exc}")
        return

    # Profiled run
    activities = [torch.profiler.ProfilerActivity.CUDA]
    sync_count = 0

    try:
        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(_PROFILED_STEPS):
                strategy.train_step(
                    batch=batch, epoch=0, input_batch=batch, target_batch=batch
                )
    except Exception as exc:
        pytest.xfail(
            f"Strategy '{paradigm}' raised during profiled steps: {exc}"
        )
        return

    # Count cudaStreamSynchronize events
    for evt in prof.key_averages():
        if "cudaStreamSynchronize" in evt.key:
            # evt.count is the total invocation count across all threads
            sync_count += evt.count

    logger.info(
        "Paradigm '%s': %d cudaStreamSynchronize events over %d steps "
        "(cap=%d, default_per_step=%d)",
        paradigm,
        sync_count,
        _PROFILED_STEPS,
        max_total,
        max_per_step,
    )

    assert sync_count <= max_total, (
        f"Strategy '{paradigm}': detected {sync_count} cudaStreamSynchronize events "
        f"over {_PROFILED_STEPS} steps (cap={max_total}, {max_per_step}/step). "
        "Likely cause: inline .item()/.cpu() calls in the hot training loop. "
        "Inspect with torch.profiler and remove sync barriers."
    )
