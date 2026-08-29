"""Tests for the differentiable Lipschitz spectral penalty (A-8.3 CertRob, component A)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mriforge.core.metrics.lipschitz_bound_metrics import layer_spectral_norms
from mriforge.models.losses.lipschitz_penalty import (
    LipschitzSpectralPenalty,
    _power_iteration_sigma,
)
from mriforge.models.losses.registry import create_loss, list_available


def _scaled_linear(dim: int, c: float) -> nn.Linear:
    lin = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        lin.weight.copy_(c * torch.eye(dim))
    return lin


# --------------------------------------------------------------------------- #
# Hinge behaviour: 0 iff all sigma <= target
# --------------------------------------------------------------------------- #


def test_penalty_zero_when_all_layers_contractive() -> None:
    # Both layers have sigma = 0.5 / 0.8 <= target 1.0 -> hinge is exactly 0.
    model = nn.Sequential(_scaled_linear(4, 0.5), _scaled_linear(4, 0.8))
    pen = LipschitzSpectralPenalty()(model=model)
    assert float(pen.detach()) == pytest.approx(0.0, abs=1e-5)


def test_penalty_positive_when_layer_exceeds_budget() -> None:
    # sigma = 3.0 > target 1.0 -> hinge ~ (3 - 1) = 2 (single layer).
    model = nn.Sequential(_scaled_linear(4, 3.0))
    pen = float(LipschitzSpectralPenalty()(model=model).detach())
    assert pen == pytest.approx(2.0, rel=0.05)


def test_target_sigma_knob_raises_budget() -> None:
    model = nn.Sequential(_scaled_linear(4, 2.5))
    strict = float(LipschitzSpectralPenalty(target_sigma=1.0)(model=model).detach())  # ~1.5
    loose = float(LipschitzSpectralPenalty(target_sigma=3.0)(model=model).detach())  # 0
    assert strict > loose
    assert loose == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Differentiability: minimising the penalty actually shrinks the true sigma
# --------------------------------------------------------------------------- #


def test_penalty_is_differentiable_and_reduces_true_sigma() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        lin.weight.mul_(4.0)  # large initial spectral norm
    model = nn.Sequential(lin)
    sigma0 = layer_spectral_norms(model)[0][1]  # exact SVD sigma before
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    penalty = LipschitzSpectralPenalty(n_power_iterations=3)
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = penalty(model=model)
        loss.backward()
        opt.step()
    sigma1 = layer_spectral_norms(model)[0][1]  # exact SVD sigma after
    assert sigma1 < sigma0  # penalty pulled the spectral norm down
    assert sigma1 < sigma0 * 0.9  # by a meaningful margin


# --------------------------------------------------------------------------- #
# Power-iteration estimate tracks the exact SVD
# --------------------------------------------------------------------------- #


def test_power_iteration_estimate_close_to_svd() -> None:
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    est = float(_power_iteration_sigma(w, n_iters=30, eps=1e-12))
    exact = float(torch.linalg.matrix_norm(w.double(), ord=2))
    assert est == pytest.approx(exact, rel=0.02)


def test_power_iteration_underestimates_not_overestimates() -> None:
    # The Rayleigh quotient at any vector is a LOWER bound on sigma_max, so the
    # estimate is conservative -> the penalty never over-penalises.
    torch.manual_seed(1)
    w = torch.randn(12, 20)
    est = float(_power_iteration_sigma(w, n_iters=1, eps=1e-12))
    exact = float(torch.linalg.matrix_norm(w.double(), ord=2))
    assert est <= exact * (1.0 + 1e-4)


# --------------------------------------------------------------------------- #
# Anti-facade contract (#16): raises without a model
# --------------------------------------------------------------------------- #


def test_raises_when_used_as_pred_target_loss() -> None:
    pen = LipschitzSpectralPenalty()
    with pytest.raises(ValueError, match="PARAMETER penalty"):
        pen(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))


def test_accepts_positional_module() -> None:
    model = nn.Sequential(_scaled_linear(4, 2.0))
    assert float(LipschitzSpectralPenalty()(model).detach()) == pytest.approx(1.0, rel=0.05)


def test_complex_weight_penalty_finite() -> None:
    lin = nn.Linear(4, 4, bias=False).to(torch.cfloat)
    with torch.no_grad():
        lin.weight.mul_(3.0)
    pen = float(LipschitzSpectralPenalty()(model=nn.Sequential(lin)).detach())
    assert pen > 0.0 and torch.isfinite(torch.tensor(pen))


def test_module_without_weight_layers_returns_zero_tensor() -> None:
    pen = LipschitzSpectralPenalty()(model=nn.Sequential(nn.ReLU()))
    assert isinstance(pen, torch.Tensor) and float(pen) == 0.0


def test_registered_and_constructible() -> None:
    assert "lipschitz_spectral_penalty" in list_available()
    loss = create_loss("lip_penalty", target_sigma=2.0)  # via alias
    assert isinstance(loss, LipschitzSpectralPenalty)
    assert loss.target_sigma == 2.0
