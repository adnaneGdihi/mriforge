"""Oracle tests for Fisher-optimal cross-dimensional budget (A-2.3 FOBA)."""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.fisher_optimal_budget import (
    FisherOptimalBudgetAllocator,
    d_optimal_budget,
    d_optimal_logdet,
    fisher_rate_from_jacobian,
)

# --------------------------------------------------------------------------- #
# Budget conservation + symmetry
# --------------------------------------------------------------------------- #


def test_budget_is_conserved() -> None:
    a = torch.tensor([1.0, 2.0, 0.5, 3.0])
    r = torch.tensor([1.0, 0.5, 2.0, 1.5])
    b = d_optimal_budget(a, r, budget=10.0)
    assert float(b.sum()) == pytest.approx(10.0, rel=1e-4)
    assert (b >= -1e-6).all()


def test_symmetric_dimensions_get_equal_split() -> None:
    a = torch.tensor([1.0, 1.0, 1.0, 1.0])
    r = torch.tensor([1.0, 1.0, 1.0, 1.0])
    b = d_optimal_budget(a, r, budget=8.0)
    assert torch.allclose(b, torch.full((4,), 2.0), atol=1e-3)


# --------------------------------------------------------------------------- #
# THE D-optimal oracles: water-filling beats uniform; KKT equal marginal gains
# --------------------------------------------------------------------------- #


def test_water_filling_beats_uniform_on_asymmetric_problem() -> None:
    # Dimensions differ in baseline + rate -> the D-optimal allocation must achieve
    # a strictly higher log-det than a naive equal split.
    a = torch.tensor([0.1, 5.0, 0.5, 2.0])
    r = torch.tensor([2.0, 0.3, 1.5, 0.8])
    out = FisherOptimalBudgetAllocator().allocate(a, r, budget=12.0)
    assert out["logdet"] > out["uniform_logdet"]


def test_kkt_marginal_gains_equal_across_active_dims() -> None:
    # At the optimum, r_d/(a_d + r_d b_d) is equal across dims that received budget.
    a = torch.tensor([0.2, 1.0, 0.5])
    r = torch.tensor([1.0, 0.7, 1.3])
    b = d_optimal_budget(a, r, budget=9.0)
    marginal = r / (a + r * b)
    active = b > 1e-3
    assert active.sum() >= 2
    m = marginal[active]
    assert float(m.max() - m.min()) < 1e-2  # equal water level across active dims


def test_high_baseline_dimension_gets_less_budget() -> None:
    # Two dims, equal rate; the one with far higher baseline FI should get less new
    # budget (its marginal gain is already lower).
    a = torch.tensor([0.1, 10.0])
    r = torch.tensor([1.0, 1.0])
    b = d_optimal_budget(a, r, budget=6.0)
    assert b[0] > b[1]


def test_dimension_dropped_when_marginal_too_low() -> None:
    # A dimension whose baseline FI is enormous relative to the budget gets ~0.
    a = torch.tensor([0.1, 1000.0])
    r = torch.tensor([1.0, 1.0])
    b = d_optimal_budget(a, r, budget=2.0)
    assert float(b[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(b[0]) == pytest.approx(2.0, rel=1e-2)


def test_more_budget_increases_objective() -> None:
    a = torch.tensor([0.5, 1.0, 2.0])
    r = torch.tensor([1.0, 1.0, 1.0])
    small = d_optimal_logdet(a, r, d_optimal_budget(a, r, budget=3.0))
    large = d_optimal_logdet(a, r, d_optimal_budget(a, r, budget=30.0))
    assert large > small


def test_min_per_dim_floor_respected() -> None:
    a = torch.tensor([0.1, 1000.0])  # dim 1 would otherwise get ~0
    r = torch.tensor([1.0, 1.0])
    b = d_optimal_budget(a, r, budget=4.0, min_per_dim=0.5)
    assert float(b.min()) >= 0.5 - 1e-4
    assert float(b.sum()) == pytest.approx(4.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# Fisher-rate adapter (the physics seam) + guards
# --------------------------------------------------------------------------- #


def test_fisher_rate_from_jacobian_is_diag_jtj() -> None:
    torch.manual_seed(0)
    j = torch.randn(20, 3)
    rate = fisher_rate_from_jacobian(j)
    assert torch.allclose(rate, j.pow(2).sum(dim=0), rtol=1e-5)


def test_fisher_rate_complex_jacobian_splits_real_imag() -> None:
    j = torch.randn(10, 2, dtype=torch.cfloat)
    rate = fisher_rate_from_jacobian(j)
    expected = j.real.pow(2).sum(0) + j.imag.pow(2).sum(0)
    assert torch.allclose(rate, expected, rtol=1e-5)


def test_zero_rate_dimension_raises() -> None:
    with pytest.raises(ValueError, match="rate_fisher"):
        d_optimal_budget(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0]), budget=2.0)


def test_infeasible_floor_raises() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        d_optimal_budget(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]), budget=1.0, min_per_dim=1.0)


def test_allocator_reports_marginal_gains() -> None:
    out = FisherOptimalBudgetAllocator().allocate(
        torch.tensor([0.5, 1.0]), torch.tensor([1.0, 1.0]), budget=4.0
    )
    assert "allocation" in out and "logdet" in out and "marginal_gains" in out
    assert isinstance(out["logdet"], float)
