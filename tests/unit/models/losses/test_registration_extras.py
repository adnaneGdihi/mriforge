"""Tests for extra registration regularizers.

Targets ``mriforge.models.losses.registration_extras.BendingEnergyLoss``.

Regression: the mixed second-derivative term must be a true mixed partial
``d^2 f / dx dy`` (a central first difference in each axis), not the composition
of two SECOND differences (which approximates ``d^4 f / dx^2 dy^2``). The
bilinear field ``f(x, y) = x * y`` is the decisive case — its pure second
derivatives vanish while ``d^2 f / dx dy = 1``, so a correct bending energy is
nonzero whereas the old 4th-order proxy gave exactly zero.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.losses.registration_extras import BendingEnergyLoss  # noqa: E402


def _grid_field(values: torch.Tensor) -> torch.Tensor:
    """Wrap an ``(H, W)`` field as a ``(B=1, ndim=1, H, W)`` deformation field."""
    return values.unsqueeze(0).unsqueeze(0)


def test_bilinear_field_has_nonzero_bending_energy() -> None:
    """``f = x*y``: pure 2nd derivatives are 0 but the mixed term is 1, so the
    bending energy must be clearly positive (it was ~0 with the old stencil)."""
    n = 8
    ii, jj = torch.meshgrid(
        torch.arange(n, dtype=torch.float32),
        torch.arange(n, dtype=torch.float32),
        indexing="ij",
    )
    field = _grid_field(ii * jj)

    loss = BendingEnergyLoss()(field, field)
    assert loss.item() > 0.1


def test_linear_field_has_near_zero_bending_energy() -> None:
    """``f = x + y`` has all relevant second derivatives zero — sanity floor."""
    n = 8
    ii, jj = torch.meshgrid(
        torch.arange(n, dtype=torch.float32),
        torch.arange(n, dtype=torch.float32),
        indexing="ij",
    )
    field = _grid_field(ii + jj)

    loss = BendingEnergyLoss()(field, field)
    assert loss.item() < 1e-6
