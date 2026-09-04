r"""Unit tests for the spectral-norm Lipschitz certificate utilities (LCAH core).

Targets ``spectramr.models.blocks.spectral_lipschitz``.

LCAH attaches a *finite, computable extrapolation certificate* to acquisition-
parameter conditioning: with the hypernetwork and target both spectral-normalised,
the worst-case prediction drift at an unseen acquisition :math:`\boldsymbol\varphi`
is bounded by :math:`L_w L_h\lVert\boldsymbol\varphi-\boldsymbol\varphi^\ast\rVert`,
where :math:`\boldsymbol\varphi^\ast` is the nearest training acquisition and the
Lipschitz constants are products of per-layer spectral norms. The certificate is
zero at a training point (consistency) and grows linearly with distance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


def test_layer_spectral_norm_matches_matrix_2norm() -> None:
    from spectramr.models.blocks.spectral_lipschitz import layer_spectral_norm

    torch.manual_seed(0)
    layer = nn.Linear(5, 3, bias=True)
    sigma = layer_spectral_norm(layer)
    expected = torch.linalg.matrix_norm(layer.weight.detach(), ord=2).item()
    assert sigma == pytest.approx(expected, rel=1e-4)


def test_spectral_norm_product_bounds_empirical_lipschitz() -> None:
    from spectramr.models.blocks.spectral_lipschitz import spectral_norm_product

    torch.manual_seed(1)
    mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    lip = spectral_norm_product(mlp)

    max_ratio = 0.0
    with torch.no_grad():
        for _ in range(200):
            a, b = torch.randn(4), torch.randn(4)
            ratio = (mlp(a) - mlp(b)).norm() / (a - b).norm().clamp_min(1e-6)
            max_ratio = max(max_ratio, float(ratio))
    assert max_ratio <= lip + 1e-4, f"empirical {max_ratio:.3f} exceeds bound {lip:.3f}"


def test_extrapolation_radius_zero_at_training_point() -> None:
    from spectramr.models.blocks.spectral_lipschitz import extrapolation_radius

    phi_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    radius = extrapolation_radius(2.0, torch.tensor([3.0, 4.0]), phi_train)
    assert float(radius) == pytest.approx(0.0, abs=1e-6)


def test_extrapolation_radius_grows_linearly_with_distance() -> None:
    from spectramr.models.blocks.spectral_lipschitz import extrapolation_radius

    phi_train = torch.tensor([[0.0, 0.0]])
    r1 = extrapolation_radius(3.0, torch.tensor([1.0, 0.0]), phi_train)
    r2 = extrapolation_radius(3.0, torch.tensor([2.0, 0.0]), phi_train)
    assert float(r1) == pytest.approx(3.0, abs=1e-5)
    assert float(r2) == pytest.approx(6.0, abs=1e-5)
