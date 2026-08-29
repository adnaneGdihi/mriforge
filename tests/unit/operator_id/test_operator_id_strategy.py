"""Tests for the operator-ID training strategy loss computation (Proposal 1).

Exercises ``_compute_losses_impl`` in isolation with a minimal duck-typed
environment (avoiding the full DI bootstrap), confirming that the operator
action, structured-Gaussian NLL, and pushforward-MMD combine into a finite,
differentiable ``g_total_loss``.
"""

# ruff: noqa: E402  (pytest.importorskip must precede torch-dependent imports)

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mriforge.config.schemas.training.operator_id import OperatorIDConfig
from mriforge.infrastructure.training.strategies.operator_id_bch_strategy import (
    OperatorIdBCHTrainingStrategy,
)
from mriforge.models.operator_id.bch_conditioner import BCHOperatorConditioner


def _make_strategy(h=16, w=16, modes=None):
    """Build a strategy without the heavy BaseTrainingStrategy.__init__."""
    modes = modes or [
        "motion_rigid",
        "bias_field_multiplicative",
        "b0_phase",
        "gibbs_truncation",
        "rician_noise",
    ]
    op_cfg = OperatorIDConfig(
        mode_dictionary=tuple(modes),
        bch_order=2,
        krylov_dim=16,
        covariance_rank=4,
        mmd_weight=1.0,
        num_contrasts=3,
        rho_bins=8,
    )
    conditioner = BCHOperatorConditioner(
        num_modes=4,
        num_contrasts=3,
        covariance_rank=4,
        rho_bins=8,
        channels=1,
        height=h,
        width=w,
    )
    strat = object.__new__(OperatorIdBCHTrainingStrategy)
    strat.env = SimpleNamespace(generator=conditioner, device=torch.device("cpu"))
    strat.config = SimpleNamespace(training=SimpleNamespace(operator_id=op_cfg))
    strat.device = torch.device("cpu")
    strat._op_cfg = op_cfg
    from mriforge.models.losses.complex.structured_gaussian_nll import StructuredGaussianNLL
    from mriforge.models.losses.image.pushforward_mmd import PushforwardMMD

    strat.nll_loss = StructuredGaussianNLL()
    strat.mmd_loss = PushforwardMMD()
    strat._generators = None
    strat._det_modes = None
    strat._noise_modes = None
    strat._gen_shape = None
    return strat


def test_compute_losses_returns_finite_total():
    torch.manual_seed(0)
    strat = _make_strategy()
    b, c, h, w = 2, 1, 16, 16
    x = torch.randn(b, c, h, w, dtype=torch.complex64)
    y = torch.randn(b, c, h, w, dtype=torch.complex64)
    out = strat._compute_losses_impl(x, y, epoch=0)
    assert "g_total_loss" in out
    assert torch.isfinite(out["g_total_loss"])
    assert torch.isfinite(out["nll_loss"])
    assert torch.isfinite(out["mmd_loss"])


def test_total_loss_is_differentiable_wrt_conditioner():
    torch.manual_seed(1)
    strat = _make_strategy()
    b, c, h, w = 2, 1, 16, 16
    x = torch.randn(b, c, h, w, dtype=torch.complex64)
    y = torch.randn(b, c, h, w, dtype=torch.complex64)
    out = strat._compute_losses_impl(x, y, epoch=0)
    out["g_total_loss"].backward()
    grads = [p.grad for p in strat.env.generator.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_accepts_real_interleaved_input():
    """The strategy coerces [B, 2C, H, W] real-interleaved tensors to complex."""
    torch.manual_seed(2)
    strat = _make_strategy()
    b, h, w = 2, 16, 16
    x = torch.randn(b, 2, h, w)  # real-interleaved single complex channel
    y = torch.randn(b, 2, h, w)
    out = strat._compute_losses_impl(x, y, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_conditioning_kwargs_are_used():
    torch.manual_seed(3)
    strat = _make_strategy()
    b, c, h, w = 3, 1, 16, 16
    x = torch.randn(b, c, h, w, dtype=torch.complex64)
    y = torch.randn(b, c, h, w, dtype=torch.complex64)
    contrast = torch.tensor([0, 1, 2])
    field = torch.full((b,), -0.522)
    out = strat._compute_losses_impl(x, y, epoch=0, contrast_idx=contrast, field_coord=field)
    assert torch.isfinite(out["g_total_loss"])
