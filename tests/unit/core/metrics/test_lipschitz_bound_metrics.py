"""Oracle tests for the certified global Lipschitz-bound metrics (A-8.3 CertRob)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from spectramr.core.metrics import get_metric
from spectramr.core.metrics.lipschitz_bound_metrics import (
    ComposedSpectralNormBound,
    MaxLayerSpectralNorm,
    layer_spectral_norms,
)
from spectramr.core.metrics.registry import MetricsRegistry


def _scaled_identity_linear(dim: int, c: float) -> nn.Linear:
    """A ``dim x dim`` Linear whose weight is ``c * I`` (so its sigma_max == c)."""
    lin = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        lin.weight.copy_(c * torch.eye(dim))
    return lin


# --------------------------------------------------------------------------- #
# Oracle 1: exactness on a known linear stack
# --------------------------------------------------------------------------- #


def test_exact_product_on_scaled_identity_stack() -> None:
    # f = (c2 I)(c1 I): the composed operator norm is exactly c1*c2, and the
    # per-layer sigmas are exactly c1, c2 -> the product metric must equal c1*c2.
    c1, c2 = 2.0, 3.0
    model = nn.Sequential(_scaled_identity_linear(4, c1), _scaled_identity_linear(4, c2))
    bound = ComposedSpectralNormBound()(model=model)
    assert bound == pytest.approx(c1 * c2, rel=1e-6)


def test_layer_spectral_norms_match_known_singular_values() -> None:
    # A non-square Linear with a hand-set spectrum: sigma_max is the largest
    # diagonal entry of the constructed weight.
    w = torch.zeros(3, 5)
    w[0, 0], w[1, 1], w[2, 2] = 4.0, 1.5, 0.5
    lin = nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        lin.weight.copy_(w)
    sigmas = layer_spectral_norms(lin)
    assert len(sigmas) == 1
    assert sigmas[0][1] == pytest.approx(4.0, rel=1e-6)


def test_max_layer_spectral_norm_is_worst_layer() -> None:
    model = nn.Sequential(_scaled_identity_linear(4, 2.0), _scaled_identity_linear(4, 5.0))
    assert MaxLayerSpectralNorm()(model=model) == pytest.approx(5.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Oracle 2: the product is a genuine UPPER BOUND on empirical amplification
# --------------------------------------------------------------------------- #


def test_upper_bounds_empirical_amplification_linear_relu() -> None:
    # For a Linear+ReLU net (ReLU is 1-Lipschitz), the empirical worst-case
    # amplification ||f(y+d) - f(y)|| / ||d|| must NOT exceed the product bound.
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).eval()
    bound = ComposedSpectralNormBound()(model=model)

    y0 = torch.randn(8)
    worst = 0.0
    for _ in range(2000):
        d = torch.randn(8)
        d = d / d.norm() * 1e-2  # small perturbation in many directions
        amp = (model(y0 + d) - model(y0)).norm().item() / d.norm().item()
        worst = max(worst, amp)
    # The certificate must dominate the empirical worst case (tiny tol for FP).
    assert worst <= bound * (1.0 + 1e-6)
    # ...and be reasonably tight, not vacuous (within ~10x of the observed worst).
    assert bound <= worst * 10.0


# --------------------------------------------------------------------------- #
# Oracle 3: responds to spectral normalisation (the loss/penalty target)
# --------------------------------------------------------------------------- #


def test_spectral_normalization_reduces_bound() -> None:
    # Wrapping a high-gain layer in torch.nn.utils.spectral_norm constrains its
    # sigma to ~1, so the global bound must drop sharply.
    torch.manual_seed(0)
    big = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        big.weight.mul_(6.0)  # large spectral norm
    plain_bound = ComposedSpectralNormBound()(model=nn.Sequential(big))

    sn = nn.utils.spectral_norm(nn.Linear(8, 8, bias=False))
    with torch.no_grad():
        sn.weight_orig.copy_(big.weight)
    sn.train()
    for _ in range(100):
        sn(torch.randn(4, 8))  # each forward runs one power-iteration step on u/v
    sn_bound = ComposedSpectralNormBound()(model=nn.Sequential(sn))

    assert sn_bound < plain_bound
    assert sn_bound == pytest.approx(1.0, abs=0.05)  # converged: spectral_norm targets sigma=1


# --------------------------------------------------------------------------- #
# Contract: no-model / degenerate -> nan; model from context; registration
# --------------------------------------------------------------------------- #


def test_no_model_returns_nan() -> None:
    # A bound of 0 would falsely claim perfect contraction -> report nan instead.
    assert math.isnan(ComposedSpectralNormBound()(torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 4)))
    assert math.isnan(MaxLayerSpectralNorm()(model=None))


def test_module_without_weight_layers_returns_nan() -> None:
    assert math.isnan(ComposedSpectralNormBound()(model=nn.Sequential(nn.ReLU(), nn.GELU())))


def test_reads_model_from_reconstructor_context() -> None:
    from spectramr.core.metrics.context import MetricContext

    model = nn.Sequential(_scaled_identity_linear(4, 2.0))
    ctx = MetricContext(reconstructor=model)
    assert ComposedSpectralNormBound()(context=ctx) == pytest.approx(2.0, rel=1e-6)


def test_conv_layer_is_scanned() -> None:
    # Conv weights enter the product via the reshape-2-D surrogate (documented).
    model = nn.Sequential(nn.Conv2d(2, 4, 3, bias=False))
    bound = ComposedSpectralNormBound()(model=model)
    assert math.isfinite(bound) and bound > 0.0


def test_complex_weight_layer_supported() -> None:
    lin = nn.Linear(4, 4, bias=False).to(torch.cfloat)
    bound = ComposedSpectralNormBound()(model=nn.Sequential(lin))
    assert math.isfinite(bound) and bound > 0.0


def test_residual_skip_is_not_counted_documented_underestimate() -> None:
    # Architecture-validity boundary (A-8.3 review): for a residual block
    # f(x) = conv(x) + x, the TRUE Lipschitz is ~Lip(conv)+1, but the metric counts
    # ONLY the conv layer's sigma (the +1 skip is not a weight layer). So the product
    # is a documented UNDER-estimate on residual nets -> a regularisation target, not a
    # strict certificate. This test pins that documented behaviour.
    class _Res(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c = nn.Conv2d(2, 2, 1, bias=False)
            with torch.no_grad():
                self.c.weight.copy_(0.5 * torch.eye(2).view(2, 2, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c(x) + x  # skip not visible to the metric

    bound = ComposedSpectralNormBound()(model=_Res())
    assert bound == pytest.approx(0.5, rel=1e-5)  # conv sigma only; true Lip ~1.5


def test_registered_lower_is_better() -> None:
    for key in ("composed_spectral_norm_bound", "lipschitz_bound", "max_layer_spectral_norm"):
        assert MetricsRegistry.is_registered(key)
    assert get_metric("composed_spectral_norm_bound").higher_is_better is False
    assert get_metric("max_layer_spectral_norm").higher_is_better is False
