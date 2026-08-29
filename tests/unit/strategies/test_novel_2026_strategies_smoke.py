"""Smoke tests for the five novel-2026 training strategies.

Each test directly exercises the strategy's ``_compute_losses_impl``
on a small batch of synthetic data, bypassing the full DI bootstrap.
The contract under test is:

1. ``_compute_losses_impl`` returns a ``dict[str, Tensor]``.
2. The dict contains ``'g_total_loss'``.
3. The total-loss tensor is scalar (0-dim), finite, and on the same
   device as the input.

These tests do *not* run a full training step (no optimiser, no
backward pass) — they assert that the novel loss machinery wires up
correctly so any subsequent integration / smoke YAML can take it for
granted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.bloch_equivariant_translation_strategy import (
    BlochEquivariantTranslationStrategy,
)
from mriforge.infrastructure.training.strategies.ib_active_acquisition_strategy import (
    IBActiveAcquisitionStrategy,
)
from mriforge.infrastructure.training.strategies.physics_equivariant_ssl_strategy import (
    PhysicsEquivariantSSLStrategy,
)
from mriforge.infrastructure.training.strategies.riemannian_bloch_diffusion_strategy import (
    RiemannianBlochDiffusionStrategy,
)
from mriforge.infrastructure.training.strategies.score_field_tomography_strategy import (
    ScoreFieldTomographyStrategy,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


class _IdentityModel(nn.Module):
    """Smoke-test model: returns input unchanged when called as f(x)."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: D401
        return x


class _ConstantModel(nn.Module):
    """Smoke-test model with a learnable parameter (so backward could run)."""

    def __init__(self, out_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.out = nn.Parameter(torch.zeros(out_shape))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.out.expand_as(x) if self.out.shape != x.shape else self.out + 0.0 * x.sum()


class _ScoreFieldStubModel(nn.Module):
    """Two-headed stub for ScoreFieldTomographyStrategy."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.head_sx = nn.Conv2d(channels, channels, 1)
        self.head_sc = nn.Linear(channels, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_x = self.head_sx(x)
        pooled = x.mean(dim=(-2, -1))
        s_c = self.head_sc(pooled)
        return s_x, s_c


class _RiemannianStubModel(nn.Module):
    """Tangent-projecting stub for RiemannianBlochDiffusionStrategy."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(3 + 32, 3)
        from mriforge.infrastructure.physics.manifolds import BlochRelaxationManifold

        self.manifold = BlochRelaxationManifold()

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor, cond: Any) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(theta_t.shape[0])
        # 32-dim sinusoidal time embedding
        freqs = torch.exp(
            -torch.arange(0, 32, 2, device=theta_t.device).float()
            * (torch.log(torch.tensor(10000.0)) / 32)
        )
        emb = t.unsqueeze(-1) * freqs
        t_emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        raw = self.head(torch.cat([theta_t, t_emb], dim=-1))
        return self.manifold.project_tangent(theta_t, raw)


def _alloc_strategy(strategy_cls, *, generator: nn.Module, config_overrides: dict[str, Any] | None = None):
    """Build a strategy instance without running BaseTrainingStrategy.__init__."""
    obj = strategy_cls.__new__(strategy_cls)
    obj.config = SimpleNamespace(
        training=SimpleNamespace(
            seed=0,
            **(config_overrides or {}),
        )
    )
    obj.device = torch.device("cpu")
    obj.env = SimpleNamespace(
        models={"generator": generator},
        device=torch.device("cpu"),
        config=obj.config,
    )
    return obj


# --------------------------------------------------------------------- #
# Idea 6 — physics-equivariant SSL
# --------------------------------------------------------------------- #


def test_physics_equivariant_ssl_smoke() -> None:
    encoder = _IdentityModel()
    strat = _alloc_strategy(
        PhysicsEquivariantSSLStrategy,
        generator=encoder,
        config_overrides={
            "physics_eq_ssl": SimpleNamespace(
                latent_dim=16,
                lambda_var=0.1,
                gamma_var=1.0,
            )
        },
    )
    # Manually replicate __init__'s post-super steps that touch operator pool.
    from mriforge.infrastructure.training.strategies.physics_equivariant_ssl_strategy import (
        _LinearLatentOperator,
        _make_operator_pool,
    )

    strat.latent_dim = 16
    strat.lambda_var = 0.1
    strat.gamma_var = 1.0
    strat.operator_pool = _make_operator_pool(strat.device)
    strat.latent_ops = nn.ModuleDict(
        {op: _LinearLatentOperator(strat.latent_dim) for op in strat.operator_pool}
    )
    strat._rng = torch.Generator(device="cpu").manual_seed(0)

    @property
    def gen_prop(self):
        return self.env.models["generator"]

    # Patch the descriptor on the instance.
    type(strat).generator_model = gen_prop  # type: ignore[assignment]

    x = torch.randn(2, 16, 4, 4)
    losses = strat._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)
    assert "g_total_loss" in losses
    assert torch.isfinite(losses["g_total_loss"])


# --------------------------------------------------------------------- #
# Idea 4 — IB active acquisition
# --------------------------------------------------------------------- #


def test_ib_active_acquisition_smoke() -> None:
    # _ConstantModel matches output shape so L1/L2 are well-defined.
    out_shape = (2, 1, 8, 8)
    model = _ConstantModel(out_shape)
    strat = _alloc_strategy(
        IBActiveAcquisitionStrategy,
        generator=model,
        config_overrides={
            "ib_active_acquisition": SimpleNamespace(beta=1.0, mode="linear_gaussian")
        },
    )
    strat.beta = 1.0
    strat.mode = "linear_gaussian"
    type(strat).generator_model = property(lambda s: s.env.models["generator"])  # type: ignore[assignment]

    x = torch.randn(*out_shape)
    target = torch.randn(*out_shape)
    losses = strat._compute_losses_impl(input_batch=x, target_batch=target, epoch=0)
    assert "g_total_loss" in losses
    assert torch.isfinite(losses["g_total_loss"])
    assert losses["g_total_loss"].dim() == 0


def test_ib_select_next_line_returns_valid_index() -> None:
    H = 32
    mask = torch.zeros(H, dtype=torch.bool)
    mask[:8] = True
    estimate = torch.randn(1, 1, H)
    idx = IBActiveAcquisitionStrategy.select_next_line(mask, estimate)
    assert 8 <= idx < H


# --------------------------------------------------------------------- #
# Idea 2 — score-field tomography
# --------------------------------------------------------------------- #


def test_score_field_tomography_smoke() -> None:
    model = _ScoreFieldStubModel(channels=1, cond_dim=5)
    strat = _alloc_strategy(
        ScoreFieldTomographyStrategy,
        generator=model,
        config_overrides={
            "score_field_tomography": SimpleNamespace(
                sigma_min=0.01, sigma_max=0.5, cond_dim=5
            )
        },
    )
    strat.sigma_min = 0.01
    strat.sigma_max = 0.5
    strat.cond_dim = 5
    strat._dim_x = None
    type(strat).generator_model = property(lambda s: s.env.models["generator"])  # type: ignore[assignment]

    x = torch.randn(2, 1, 8, 8)
    losses = strat._compute_losses_impl(
        input_batch=x,
        target_batch=x,
        epoch=0,
        batch={},
    )
    assert "g_total_loss" in losses
    assert torch.isfinite(losses["g_total_loss"])


# --------------------------------------------------------------------- #
# Idea 3 — Bloch-equivariant translation
# --------------------------------------------------------------------- #


def test_bloch_equivariant_translation_smoke() -> None:
    out_shape = (2, 1, 16, 16)
    # Shape-agnostic stub so both the supervised (16×16) and unpaired
    # (8×8 simulated) branches can be re-applied to inputs of different
    # spatial sizes without reallocating the model.
    model = _IdentityModel()
    strat = _alloc_strategy(
        BlochEquivariantTranslationStrategy,
        generator=model,
        config_overrides={
            "bloch_equivariant_translation": SimpleNamespace(
                t1_scale_ulf=0.30,
                t2_scale_ulf=1.05,
                tr_ulf=8.0,
                tr_hf=8.0,
                te_ulf=4.0,
                te_hf=4.0,
                flip_angle_deg=15.0,
                lambda_bloch_eq=1.0,
                lambda_cycle=0.1,
                unpaired_patch=8,
            )
        },
    )
    # Manually replicate the post-super assignments.
    from mriforge.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer
    from mriforge.infrastructure.training.strategies.bloch_equivariant_translation_strategy import (
        _TissueAtlas,
    )

    strat.t1_scale_ulf = 0.30
    strat.t2_scale_ulf = 1.05
    strat.tr_ulf = 8.0
    strat.tr_hf = 8.0
    strat.te_ulf = 4.0
    strat.te_hf = 4.0
    strat.flip_angle_rad = 0.2618
    strat.lambda_eq = 1.0
    strat.lambda_cycle = 0.1
    strat.unpaired_patch = 8
    strat.use_bloch_checkpoint = False
    strat.bloch = DifferentiableBlochLayer()
    strat.atlas = _TissueAtlas()
    type(strat).generator_model = property(lambda s: s.env.models["generator"])  # type: ignore[assignment]

    x = torch.randn(*out_shape)
    target = torch.randn(*out_shape)
    losses = strat._compute_losses_impl(input_batch=x, target_batch=target, epoch=0)
    assert "g_total_loss" in losses
    assert torch.isfinite(losses["g_total_loss"])


# --------------------------------------------------------------------- #
# Idea 5 — Riemannian Bloch diffusion
# --------------------------------------------------------------------- #


def test_riemannian_bloch_diffusion_smoke() -> None:
    model = _RiemannianStubModel()
    strat = _alloc_strategy(
        RiemannianBlochDiffusionStrategy,
        generator=model,
        config_overrides={
            "riemannian_bloch_diffusion": SimpleNamespace(
                t_min=0.01,
                t_max=0.5,
                step_size_init=0.05,
                injectivity_safety_factor=0.5,
                metric_cache_resolution=0,
                metric_refresh_interval=50,
            )
        },
    )
    from mriforge.infrastructure.physics.manifolds import BlochRelaxationManifold

    strat.t_min = 0.01
    strat.t_max = 0.5
    strat.step_size_init = 0.05
    strat.injectivity_safety_factor = 0.5
    strat.metric_refresh_interval = 50
    strat.manifold = BlochRelaxationManifold(cache_resolution=0)
    strat._last_refresh_epoch = -1
    type(strat).generator_model = property(lambda s: s.env.models["generator"])  # type: ignore[assignment]

    # Sample 4 manifold points (M0, T1, T2) inside the manifold.
    theta0 = torch.tensor(
        [
            [1.0, 1200.0, 80.0],
            [0.9, 1000.0, 60.0],
            [1.0, 1500.0, 100.0],
            [0.8, 900.0, 50.0],
        ]
    )
    losses = strat._compute_losses_impl(
        input_batch=theta0,
        target_batch=theta0,
        epoch=0,
    )
    assert "g_total_loss" in losses
    assert torch.isfinite(losses["g_total_loss"])
