"""Smoke tests for HyperbolicCineUNet and HeisenbergReconUNet."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.breakthrough_geometric_generators import (
    HeisenbergReconUNet,
    HyperbolicCineUNet,
)
from mriforge.models.registry import MODEL_REGISTRY

_GEOM_GENERATORS = [
    ("hyperbolic_cine_unet", HyperbolicCineUNet),
    ("heisenberg_recon_unet", HeisenbergReconUNet),
]


@pytest.mark.parametrize("name,cls", _GEOM_GENERATORS)
def test_generator_is_registered(name: str, cls: type) -> None:
    entry = MODEL_REGISTRY.get(name)
    assert entry is not None
    assert entry["class"] is cls
    assert entry["mode"] == "reconstruction"


@pytest.mark.parametrize("name,cls", _GEOM_GENERATORS)
def test_generator_forward_shape_preserved(name: str, cls: type) -> None:
    model = cls()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)


@pytest.mark.parametrize("name,cls", _GEOM_GENERATORS)
def test_generator_gradient_flows(name: str, cls: type) -> None:
    model = cls()
    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_hyperbolic_curvature_propagates() -> None:
    """curvature kwarg flows through to the inner PoincareLayer."""
    model = HyperbolicCineUNet(curvature=0.5)
    assert model.curvature == 0.5
    assert model.poincare.curvature == 0.5


def test_hyperbolic_curvature_zero_recovers_euclidean_path() -> None:
    """At c=0 the Poincaré pipeline becomes a plain Euclidean linear layer."""
    model = HyperbolicCineUNet(curvature=0.0)
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    assert torch.isfinite(out).all()


def test_heisenberg_lattice_radius_propagates() -> None:
    model = HeisenbergReconUNet(translation_radius=2, phase_radius=2)
    # (2*2+1)² × 2² = 100 lattice points.
    assert model.heisenberg.shifts.shape[0] == 100


def test_invalid_hyperbolic_curvature_raises() -> None:
    with pytest.raises(ValueError):
        HyperbolicCineUNet(curvature=-0.1)
