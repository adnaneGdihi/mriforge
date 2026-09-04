"""Tests for the synthetic PDE benchmark dataset.

Two reference solvers ship under
``spectramr.data.datasets.pde_synthetic`` — Burgers 1-D (Fourier
pseudo-spectral) and Darcy 2-D (5-point FD + scipy spsolve). These
tests pin numerical sanity invariants that any operator-paper user
relies on:

1. Burgers solution is bounded and not equal to the initial condition
   (the PDE actually evolved the field).
2. Darcy solution respects zero Dirichlet boundary conditions exactly.
3. Darcy solution is non-negative under positive forcing (max principle
   for elliptic operators with positive coefficients and boundary).
4. Determinism: same seed → same data.
5. Train/val seed offset produces disjoint samples.
6. Factory dispatcher accepts the two known problems and rejects others.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from spectramr.data.datasets.pde_synthetic import (  # noqa: E402
    BurgersDataset,
    DarcyDataset,
    make_pde_dataset,
)

# ── Burgers ──────────────────────────────────────────────────────────


def test_burgers_shape_and_subject_keys() -> None:
    ds = BurgersDataset(n_samples=2, resolution=64)
    assert len(ds) == 2
    s = ds[0]
    assert {"input", "target", "mri", "file_id"} <= set(s.keys())
    assert s["input"].shape == (1, 64, 1, 1)
    assert s["target"].shape == (1, 64, 1, 1)


def test_burgers_solution_actually_evolved() -> None:
    """The PDE must produce a target distinct from the initial condition."""
    ds = BurgersDataset(n_samples=1, resolution=128)
    s = ds[0]
    ic = s["input"].data
    sol = s["target"].data
    assert not torch.allclose(ic, sol, atol=1e-3)


def test_burgers_deterministic() -> None:
    ds_a = BurgersDataset(n_samples=1, resolution=64, seed_offset=0)
    ds_b = BurgersDataset(n_samples=1, resolution=64, seed_offset=0)
    assert torch.allclose(ds_a[0]["input"].data, ds_b[0]["input"].data)
    assert torch.allclose(ds_a[0]["target"].data, ds_b[0]["target"].data)


def test_burgers_seed_offset_differs() -> None:
    ds_a = BurgersDataset(n_samples=1, resolution=64, seed_offset=0)
    ds_b = BurgersDataset(n_samples=1, resolution=64, seed_offset=1_000_000)
    assert not torch.allclose(ds_a[0]["input"].data, ds_b[0]["input"].data)


# ── Darcy ────────────────────────────────────────────────────────────


def test_darcy_shape_and_subject_keys() -> None:
    pytest.importorskip("scipy")
    ds = DarcyDataset(n_samples=1, resolution=16)
    s = ds[0]
    assert s["input"].shape == (1, 16, 16, 1)
    assert s["target"].shape == (1, 16, 16, 1)


def test_darcy_dirichlet_bc_exactly_zero() -> None:
    """Boundary values of u must be exactly 0 (Dirichlet)."""
    pytest.importorskip("scipy")
    ds = DarcyDataset(n_samples=1, resolution=16)
    sol = ds[0]["target"].data.squeeze()  # (16, 16)
    assert sol[0, :].abs().max().item() < 1e-6
    assert sol[-1, :].abs().max().item() < 1e-6
    assert sol[:, 0].abs().max().item() < 1e-6
    assert sol[:, -1].abs().max().item() < 1e-6


def test_darcy_solution_nonnegative_with_positive_forcing() -> None:
    """Maximum principle: u ≥ 0 when f ≥ 0 and BC = 0."""
    pytest.importorskip("scipy")
    ds = DarcyDataset(n_samples=1, resolution=16, f_const=1.0)
    sol = ds[0]["target"].data
    assert sol.min().item() >= -1e-6  # allow rounding


# ── Factory ──────────────────────────────────────────────────────────


def test_factory_dispatch_burgers() -> None:
    ds = make_pde_dataset(problem="burgers_1d", n_samples=1, resolution=32)
    assert isinstance(ds, BurgersDataset)


def test_factory_dispatch_darcy() -> None:
    pytest.importorskip("scipy")
    ds = make_pde_dataset(problem="darcy_2d", n_samples=1, resolution=8)
    assert isinstance(ds, DarcyDataset)


def test_factory_rejects_unknown_problem() -> None:
    with pytest.raises(ValueError, match="problem"):
        make_pde_dataset(problem="navier_stokes_2d", n_samples=1, resolution=8)
