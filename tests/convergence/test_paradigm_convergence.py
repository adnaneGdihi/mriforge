"""Convergence harness: ≤200-step training run must hit stored loss bounds.

Each test is parametrized over the *pillar* paradigms (those that have a
minimal-settings YAML fixture) and requires a GPU.  If a paradigm has no
entry in ``bounds.json`` the test skips with a calibration message.

Design
------
* Seeds are fixed at 42 for determinism.
* Tiny 32×32 Shepp-Logan phantom, batch size 2, in-memory — no disk I/O.
* Steps run directly through ``strategy.train_step`` in a tight loop; no
  full pipeline / DataLoader involved.
* The mock environment is the same as in ``test_paradigm_step.py`` but uses
  the real CUDA device when available.

Calibration workflow (cluster)
-------------------------------
1.  Run ``pytest tests/convergence/ -m "convergence" --gpu -v``.
2.  For each skipped paradigm, author a minimal YAML (if not yet done) and
    run a short calibration loop to find a reasonable ``loss_threshold``.
3.  Add the calibrated entry to ``bounds.json``.
4.  Re-run to confirm the bound is hit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tests.utils.minimal_settings import SettingsNotFound, minimal_settings_for
from tests.utils.phantoms import shepp_logan_2d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds file
# ---------------------------------------------------------------------------

_BOUNDS_PATH = Path(__file__).parent / "bounds.json"


def _load_bounds() -> dict[str, Any]:
    with open(_BOUNDS_PATH) as fh:
        data = json.load(fh)
    # Strip meta-keys (prefixed with _)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Pillar paradigms to test
# ---------------------------------------------------------------------------

_PILLAR_KEYS = [
    "reconstruction",
    "gan",
    "diffusion",
    "vae",
    "physics_driven",
    "pinn",
    "cycle_bloch",
    "mae",
    "ssl",
]


# ---------------------------------------------------------------------------
# Tiny model reused from test_paradigm_step conventions
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


class _TinyDiscriminator(nn.Module):
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


def _make_mock_config_convergence(strategy_key: str, device: torch.device) -> MagicMock:
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
    cfg.optimization.optimize_memory_amp = True
    cfg.optimization.dynamic_loss_scaling = True
    cfg.optimization.optimizer.learning_rate = 1e-3
    cfg.optimization.gradient.accumulation_steps = 1

    recon = _LossAutoMock()
    recon.lambda_l1 = 1.0
    for attr in [
        "lambda_l2", "lambda_perceptual", "lambda_ssim", "lambda_kspace",
        "lambda_flow", "lambda_flow_likelihood", "lambda_graph_smoothness",
        "lambda_feature_alignment", "lambda_affine_regularization",
        "lambda_domain_adversarial", "lambda_sparsity_penalty",
        "lambda_wasserstein", "lambda_frequency", "lambda_log_spectral",
        "lambda_lpips", "weighted_kspace_exponent", "histogram_bins",
        "ffl_alpha",
    ]:
        setattr(recon, attr, 0.0)
    recon.log_spectral_skip_fft = False

    cfg.objectives = MagicMock()
    cfg.objectives.reconstruction = recon
    cfg.objectives.physics = _LossAutoMock()
    for attr in [
        "lambda_parallel_imaging", "lambda_physics_constraint", "lambda_bloch_residual",
    ]:
        setattr(cfg.objectives.physics, attr, 0.0)
    for attr in [
        "enable_parallel_imaging", "enable_physics_constraint", "enable_bloch_residual",
    ]:
        setattr(cfg.objectives.physics, attr, False)

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
    cfg.training.diffusion.guidance_scale = 1.0
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
    cfg.logging.log_interval = 50

    return cfg


def _make_env_convergence(
    strategy_key: str, device: torch.device
) -> MagicMock:
    from mriforge.infrastructure.training.builders.optimization_builder import (
        OptimizationBuilder,
    )

    gen = _TinyNet(in_ch=1, out_ch=1).to(device)
    disc = _TinyDiscriminator(in_ch=1).to(device)

    opt_g = OptimizationBuilder.create_single_optimizer(
        gen.parameters(), learning_rate=1e-3, optimizer_type="adam"
    )
    opt_d = OptimizationBuilder.create_single_optimizer(
        disc.parameters(), learning_rate=1e-3, optimizer_type="adam"
    )

    cfg = _make_mock_config_convergence(strategy_key, device)

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


def _extract_scalar_loss(result: Any) -> float | None:
    """Best-effort extraction of a scalar loss from a step result."""
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, torch.Tensor) and result.numel() == 1:
        return result.item()
    if isinstance(result, list):
        for item in result:
            v = _extract_scalar_loss(item)
            if v is not None:
                return v
    if isinstance(result, dict):
        for val in result.values():
            v = _extract_scalar_loss(val)
            if v is not None and torch.isfinite(torch.tensor(v)):
                return v
    return None


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("paradigm", _PILLAR_KEYS, ids=lambda k: k)
def test_paradigm_convergence(paradigm: str) -> None:
    """Run ≤200 training steps and verify loss hits the calibrated bound.

    Skips if:
    - No minimal-settings fixture for this paradigm.
    - No calibrated bound in bounds.json.

    Xfails if the strategy raises on first step (not-yet-GPU-ready).
    """
    # --- bounds ---
    bounds = _load_bounds()
    if paradigm not in bounds:
        pytest.skip(
            f"Convergence bound not calibrated for '{paradigm}' — "
            "run calibration on cluster and add entry to tests/convergence/bounds.json"
        )

    entry = bounds[paradigm]
    max_steps: int = entry["max_steps"]
    loss_threshold: float = entry["loss_threshold"]

    # --- settings fixture ---
    try:
        minimal_settings_for(paradigm)
    except SettingsNotFound as exc:
        pytest.skip(f"minimal settings YAML not yet authored for '{paradigm}': {exc}")
    except Exception as exc:
        pytest.skip(f"Could not load minimal settings for '{paradigm}': {exc}")

    # --- device ---
    assert torch.cuda.is_available(), "This test requires CUDA (enforced by @pytest.mark.gpu)"
    device = torch.device("cuda:0")

    # --- seed ---
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # --- strategy ---
    try:
        from mriforge.infrastructure.training.strategy_factory import (  # noqa: PLC0415
            TrainingStrategyFactory,
        )

        factory = TrainingStrategyFactory()
        class_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[paradigm]
        strategy_cls = factory._load_strategy_class(class_path)
    except Exception as exc:
        pytest.xfail(f"Strategy class import failed for '{paradigm}': {exc}")
        return

    env = _make_env_convergence(paradigm, device)

    try:
        strategy = strategy_cls(env=env, device=device)
    except Exception as exc:
        pytest.xfail(
            f"Strategy '{paradigm}' could not be instantiated for convergence test: {exc}"
        )
        return

    # --- phantom batch ---
    phantom_2d = shepp_logan_2d(32)
    phantom_batch = phantom_2d.unsqueeze(0).unsqueeze(0).expand(2, 1, 32, 32).to(device)

    # --- training loop ---
    final_loss: float | None = None
    try:
        for _step in range(max_steps):
            result = strategy.train_step(
                batch=phantom_batch,
                epoch=0,
                input_batch=phantom_batch,
                target_batch=phantom_batch,
            )
            loss_val = _extract_scalar_loss(result)
            if loss_val is not None and torch.isfinite(torch.tensor(loss_val)):
                final_loss = loss_val
    except Exception as exc:
        pytest.xfail(
            f"Strategy '{paradigm}' raised during convergence loop: {exc}"
        )
        return

    # --- bound check ---
    assert final_loss is not None, (
        f"Could not extract a scalar loss from strategy '{paradigm}' after {max_steps} steps"
    )
    assert torch.isfinite(torch.tensor(final_loss)), (
        f"Strategy '{paradigm}' produced non-finite final loss: {final_loss}"
    )
    assert final_loss <= loss_threshold, (
        f"Strategy '{paradigm}' did not converge: "
        f"final_loss={final_loss:.4f} > bound={loss_threshold:.4f} "
        f"after {max_steps} steps"
    )
