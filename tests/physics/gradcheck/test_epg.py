"""Tier-1 gradcheck tests for the EPG (Extended Phase Graph) simulation.

Two functions from ``spectramr.infrastructure.physics.epg`` are covered:

1. ``simulate_differentiable_epg_fse`` — simplified differentiable FSE proxy.
   All operations are standard torch ops; gradcheck is feasible on a small
   spatial grid with a short echo train.

2. ``simulate_full_epg_fse`` — full EPG with complex-typed state tensors
   (``torch.complex64`` state, ``torch.roll``, etc.).  The complex states
   prevent straightforward fp64 gradcheck (gradcheck works only on real
   tensors).  The test is written but skipped with a precise reason and TODO.

Physiological inputs:
  T1 = 1000 ms, T2 = 100 ms, ρ = 1.0
  excitation FA = 90°, refocusing FA = 180°
  echo_train_length = 4 (short for speed), te_spacing = 10 ms
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from spectramr.infrastructure.physics.epg import (
    simulate_differentiable_epg_fse,
    simulate_full_epg_fse,
)

_EPS = 1e-6
_ATOL = 1e-5
_SHAPE = (1, 1, 2, 2)  # (B, C, H, W) — small spatial grid

_T1_VAL = 1000.0
_T2_VAL = 100.0
_RHO_VAL = 1.0
_ETL = 4        # short echo train — gradcheck perturbs each element
_TE_SP = 10.0   # ms
_EX_FA = 90.0   # degrees
_REF_FA = 180.0  # degrees


def _fp64(value: float) -> torch.Tensor:
    return torch.full(_SHAPE, value, dtype=torch.float64, requires_grad=True)


# ---------------------------------------------------------------------------
# simulate_differentiable_epg_fse
# ---------------------------------------------------------------------------


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_epg_differentiable_fse_wrt_rho():
    """Gradcheck of simplified EPG w.r.t. proton density ρ."""
    rho = _fp64(_RHO_VAL)
    t1 = _fp64(_T1_VAL).detach()
    t2 = _fp64(_T2_VAL).detach()
    ex_fa = torch.full(_SHAPE, _EX_FA, dtype=torch.float64)
    ref_fa = torch.full(_SHAPE, _REF_FA, dtype=torch.float64)

    def fn(rho_in):
        return simulate_differentiable_epg_fse(
            rho=rho_in,
            t1=t1,
            t2=t2,
            excitation_fa=ex_fa,
            refocusing_fa=ref_fa,
            echo_train_length=_ETL,
            te_spacing=_TE_SP,
        )

    assert gradcheck(fn, (rho,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_epg_differentiable_fse_wrt_t1():
    """Gradcheck of simplified EPG w.r.t. T1."""
    rho = _fp64(_RHO_VAL).detach()
    t1 = _fp64(_T1_VAL)
    t2 = _fp64(_T2_VAL).detach()
    ex_fa = torch.full(_SHAPE, _EX_FA, dtype=torch.float64)
    ref_fa = torch.full(_SHAPE, _REF_FA, dtype=torch.float64)

    def fn(t1_in):
        return simulate_differentiable_epg_fse(
            rho=rho,
            t1=t1_in,
            t2=t2,
            excitation_fa=ex_fa,
            refocusing_fa=ref_fa,
            echo_train_length=_ETL,
            te_spacing=_TE_SP,
        )

    assert gradcheck(fn, (t1,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_epg_differentiable_fse_wrt_t2():
    """Gradcheck of simplified EPG w.r.t. T2."""
    rho = _fp64(_RHO_VAL).detach()
    t1 = _fp64(_T1_VAL).detach()
    t2 = _fp64(_T2_VAL)
    ex_fa = torch.full(_SHAPE, _EX_FA, dtype=torch.float64)
    ref_fa = torch.full(_SHAPE, _REF_FA, dtype=torch.float64)

    def fn(t2_in):
        return simulate_differentiable_epg_fse(
            rho=rho,
            t1=t1,
            t2=t2_in,
            excitation_fa=ex_fa,
            refocusing_fa=ref_fa,
            echo_train_length=_ETL,
            te_spacing=_TE_SP,
        )

    assert gradcheck(fn, (t2,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_epg_differentiable_fse_wrt_excitation_fa():
    """Gradcheck of simplified EPG w.r.t. excitation flip angle."""
    rho = _fp64(_RHO_VAL).detach()
    t1 = _fp64(_T1_VAL).detach()
    t2 = _fp64(_T2_VAL).detach()
    ex_fa = torch.full(_SHAPE, _EX_FA, dtype=torch.float64, requires_grad=True)
    ref_fa = torch.full(_SHAPE, _REF_FA, dtype=torch.float64)

    def fn(ex_fa_in):
        return simulate_differentiable_epg_fse(
            rho=rho,
            t1=t1,
            t2=t2,
            excitation_fa=ex_fa_in,
            refocusing_fa=ref_fa,
            echo_train_length=_ETL,
            te_spacing=_TE_SP,
        )

    assert gradcheck(fn, (ex_fa,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_epg_differentiable_fse_wrt_refocusing_fa():
    """Gradcheck of simplified EPG w.r.t. refocusing flip angle."""
    rho = _fp64(_RHO_VAL).detach()
    t1 = _fp64(_T1_VAL).detach()
    t2 = _fp64(_T2_VAL).detach()
    ex_fa = torch.full(_SHAPE, _EX_FA, dtype=torch.float64)
    ref_fa = torch.full(_SHAPE, _REF_FA, dtype=torch.float64, requires_grad=True)

    def fn(ref_fa_in):
        return simulate_differentiable_epg_fse(
            rho=rho,
            t1=t1,
            t2=t2,
            excitation_fa=ex_fa,
            refocusing_fa=ref_fa_in,
            echo_train_length=_ETL,
            te_spacing=_TE_SP,
        )

    assert gradcheck(fn, (ref_fa,), eps=_EPS, atol=_ATOL, raise_exception=True)


# ---------------------------------------------------------------------------
# simulate_full_epg_fse — skipped: complex state tensors
# ---------------------------------------------------------------------------


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
@pytest.mark.skip(
    reason=(
        "simulate_full_epg_fse initialises state tensors as torch.complex64 "
        "(F_plus, F_minus, Z) inside the function body.  torch.autograd.gradcheck "
        "requires all leaf tensors and intermediate values to be real-valued fp64; "
        "the internal complex64 states trigger a dtype mismatch that gradcheck "
        "cannot traverse.  "
        "TODO: refactor simulate_full_epg_fse to accept a dtype kwarg and "
        "propagate it to state tensor allocation (torch.zeros(..., dtype=dtype)), "
        "then re-enable this test with dtype=torch.complex128."
    )
)
def test_full_epg_fse_wrt_rho_skipped():
    """Placeholder for full EPG gradcheck — skipped until complex128 path exists.

    To enable: make ``simulate_full_epg_fse`` accept a ``dtype`` kwarg and
    pass ``dtype=torch.complex128`` for the state arrays, then call::

        gradcheck(fn, (rho,), eps=1e-6, atol=1e-5)
    """
    raise NotImplementedError("This test should be skipped by the decorator above.")
