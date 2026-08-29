"""Canary / parametrized / edge-case / expected-failure tests for adjoint_sequence.py.

Covers:
  - steady_state_se_signal
  - steady_state_ir_se_signal
  - optimise_sequence_adjoint  (short run — 5 iterations, CPU-only)
  - jacobian_rank_augmentation_test
  - SequenceOptResult (dataclass)
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scalars(t1=800.0, t2=80.0, m0=1.0):
    return (
        torch.tensor(float(t1)),
        torch.tensor(float(t2)),
        torch.tensor(float(m0)),
    )


def _phi_se():
    return {
        "tr": torch.tensor(1000.0).requires_grad_(True),
        "te": torch.tensor(20.0).requires_grad_(True),
    }


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_adjoint_sequence_imports():
    """All public symbols import cleanly."""
    from mriforge.infrastructure.physics.adjoint_sequence import (
        steady_state_se_signal,
        steady_state_ir_se_signal,
        optimise_sequence_adjoint,
        jacobian_rank_augmentation_test,
        SequenceOptResult,
    )
    assert all([
        steady_state_se_signal,
        steady_state_ir_se_signal,
        optimise_sequence_adjoint,
        jacobian_rank_augmentation_test,
        SequenceOptResult,
    ])


@pytest.mark.physics
def test_canary_se_signal_trivial():
    """SE signal with scalar inputs returns a positive scalar."""
    from mriforge.infrastructure.physics.adjoint_sequence import steady_state_se_signal
    t1, t2, m0 = _scalars()
    s = steady_state_se_signal(
        t1, t2, m0,
        tr=torch.tensor(1000.0),
        te=torch.tensor(20.0),
    )
    assert s.ndim == 0 or s.numel() == 1
    assert float(s.item()) > 0.0


# ---------------------------------------------------------------------------
# Parametrized — SE signal
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "t1, t2, tr, te",
    [
        (800.0, 80.0,  1000.0, 20.0),
        (400.0, 40.0,   500.0, 10.0),
        (1500.0, 200.0, 2000.0, 30.0),
        (100.0, 10.0,   200.0,  5.0),
    ],
    ids=["tissue-WM", "tissue-GM", "tissue-CSF", "tissue-fast"],
)
def test_se_signal_positive(t1, t2, tr, te):
    """SE signal is positive for typical tissue parameters."""
    from mriforge.infrastructure.physics.adjoint_sequence import steady_state_se_signal
    s = steady_state_se_signal(
        torch.tensor(t1), torch.tensor(t2), torch.tensor(1.0),
        tr=torch.tensor(tr), te=torch.tensor(te),
    )
    assert float(s.item()) > 0.0


@pytest.mark.physics
def test_se_signal_decreasing_with_te():
    """SE signal decreases as TE increases (more T2 decay)."""
    from mriforge.infrastructure.physics.adjoint_sequence import steady_state_se_signal
    t1, t2, m0 = _scalars()
    tr = torch.tensor(1000.0)
    s10 = steady_state_se_signal(t1, t2, m0, tr=tr, te=torch.tensor(10.0))
    s50 = steady_state_se_signal(t1, t2, m0, tr=tr, te=torch.tensor(50.0))
    assert float(s10.item()) > float(s50.item())


@pytest.mark.physics
def test_se_signal_gradient_flows():
    """Gradient of SE signal w.r.t. TR/TE is non-zero (autograd works)."""
    from mriforge.infrastructure.physics.adjoint_sequence import steady_state_se_signal
    t1, t2, m0 = _scalars()
    tr = torch.tensor(1000.0, requires_grad=True)
    te = torch.tensor(20.0, requires_grad=True)
    s = steady_state_se_signal(t1, t2, m0, tr=tr, te=te)
    s.backward()
    assert tr.grad is not None and abs(float(tr.grad.item())) > 0
    assert te.grad is not None and abs(float(te.grad.item())) > 0


# ---------------------------------------------------------------------------
# Parametrized — IR-SE signal
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "t1, ti",
    [(800.0, 200.0), (1500.0, 400.0), (300.0, 100.0)],
    ids=["WM", "CSF", "fat"],
)
def test_ir_se_signal_shape_and_finite(t1, ti):
    """IR-SE signal returns a finite scalar for various TI values."""
    from mriforge.infrastructure.physics.adjoint_sequence import steady_state_ir_se_signal
    s = steady_state_ir_se_signal(
        torch.tensor(t1),
        torch.tensor(80.0),
        torch.tensor(1.0),
        tr=torch.tensor(2000.0),
        te=torch.tensor(20.0),
        ti=torch.tensor(float(ti)),
    )
    assert not torch.isnan(s), "NaN in IR-SE signal"
    assert not torch.isinf(s), "Inf in IR-SE signal"


# ---------------------------------------------------------------------------
# optimise_sequence_adjoint — short run
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_optimise_sequence_adjoint_runs():
    """optimise_sequence_adjoint converges without error for 5 iterations."""
    from mriforge.infrastructure.physics.adjoint_sequence import (
        steady_state_se_signal,
        optimise_sequence_adjoint,
        SequenceOptResult,
    )
    t1, t2, m0 = _scalars()
    target = torch.tensor(0.5)
    result = optimise_sequence_adjoint(
        steady_state_se_signal,
        t1, t2, m0, target,
        initial_phi={"tr": torch.tensor(1000.0), "te": torch.tensor(20.0)},
        n_iters=5,
        lr=1e-2,
    )
    assert isinstance(result, SequenceOptResult)
    assert len(result.loss_history) == 5
    assert result.best_loss < float("inf")


@pytest.mark.physics
def test_optimise_with_marker_anchor():
    """Marker-anchored optimisation runs without error."""
    from mriforge.infrastructure.physics.adjoint_sequence import (
        steady_state_se_signal,
        optimise_sequence_adjoint,
    )
    t1, t2, m0 = _scalars()
    mt1, mt2, mm0 = _scalars(t1=300.0, t2=30.0, m0=0.8)
    result = optimise_sequence_adjoint(
        steady_state_se_signal,
        t1, t2, m0,
        torch.tensor(0.5),
        initial_phi={"tr": torch.tensor(1000.0), "te": torch.tensor(20.0)},
        marker_t1=mt1, marker_t2=mt2, marker_m0=mm0,
        target_marker_signal=torch.tensor(0.3),
        n_iters=5,
    )
    assert len(result.loss_history) == 5


# ---------------------------------------------------------------------------
# jacobian_rank_augmentation_test
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_rank_augmentation_test_anatomy_only():
    """Anatomy-only rank equals column rank of J_anat."""
    from mriforge.infrastructure.physics.adjoint_sequence import jacobian_rank_augmentation_test
    p = 4
    J_anat = torch.eye(p)         # full rank p
    J_mark = torch.zeros(1, p)    # zero row — linearly dependent
    r_anat, r_combined = jacobian_rank_augmentation_test(J_anat, J_mark)
    assert r_anat == p
    assert r_combined == p   # no new direction


@pytest.mark.physics
def test_rank_augmentation_strictly_increases():
    """Marker Jacobian in new direction strictly increases rank."""
    from mriforge.infrastructure.physics.adjoint_sequence import jacobian_rank_augmentation_test
    p = 3
    J_anat = torch.eye(p)
    # A direction outside the span of J_anat: effectively adds a row, but
    # eye is already p×p so we test with J_anat of rank < p
    J_anat_low = torch.zeros(p - 1, p)
    for i in range(p - 1):
        J_anat_low[i, i] = 1.0
    # Marker adds the missing direction
    J_mark = torch.zeros(1, p)
    J_mark[0, p - 1] = 1.0
    r_a, r_c = jacobian_rank_augmentation_test(J_anat_low, J_mark)
    assert r_a == p - 1
    assert r_c == p


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_se_signal_edge_very_long_tr():
    """Very long TR → signal ≈ M0 (full longitudinal recovery)."""
    from mriforge.infrastructure.physics.adjoint_sequence import steady_state_se_signal
    s = steady_state_se_signal(
        torch.tensor(800.0), torch.tensor(80.0), torch.tensor(1.0),
        tr=torch.tensor(1e6),    # ~infinite TR
        te=torch.tensor(0.1),    # negligible TE decay
    )
    # Signal should be close to M0 = 1.0
    assert abs(float(s.item()) - 1.0) < 0.01


@pytest.mark.physics
def test_ir_se_signal_edge_ti_equals_t1_ln2():
    """At TI ≈ T1*ln(2) the null point: signal ≈ 0 for long TR."""
    import math
    from mriforge.infrastructure.physics.adjoint_sequence import steady_state_ir_se_signal
    t1 = 800.0
    ti_null = t1 * math.log(2)
    s = steady_state_ir_se_signal(
        torch.tensor(t1), torch.tensor(80.0), torch.tensor(1.0),
        tr=torch.tensor(6000.0),   # long TR
        te=torch.tensor(0.1),
        ti=torch.tensor(ti_null),
    )
    assert abs(float(s.item())) < 0.05
