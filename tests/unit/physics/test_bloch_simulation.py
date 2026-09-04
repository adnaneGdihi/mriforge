"""Contract tests for the analytical Bloch signal equations.

Covers the inversion-recovery doc-contract finding (WS-4 / bloch_epg cluster):
``BlochSimulator.inversion_recovery`` returns the *signed* IR signal (it crosses
zero for ``TI < T1 * ln 2``); it deliberately does NOT apply ``torch.abs``.  A
phase-insensitive magnitude image is the caller's responsibility (cf.
``MultiPhysicsBlochLayer._synthesize_inversion_recovery`` which does take abs).

These tests pin the *true* (signed) numerical contract so the equation and its
documentation stay in agreement.  They do not depend on the comment wording.

Shape convention: tissue maps are ``[B, 1, H, W]`` (or any broadcastable shape);
all units are milliseconds for relaxation/time parameters.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.biophysics.bloch import BlochSimulator


@pytest.mark.physics
def test_inversion_recovery_is_signed_not_magnitude():
    """For TI < T1*ln2 the IR signal is NEGATIVE (signed, not magnitude)."""
    b, h, w = 1, 4, 4
    m0 = torch.ones(b, 1, h, w)
    t1 = torch.full((b, 1, h, w), 1000.0)  # ms
    # Null point TI = T1 * ln 2 ≈ 693 ms; pick TI well below it -> negative.
    ti = 100.0
    tr = 1.0e6  # effectively infinite (e_tr -> 0) so sign is dominated by 1 - 2 e_ti
    sig = BlochSimulator.inversion_recovery(m0, t1, ti, tr)
    assert (sig < 0).all(), "IR signal must be signed/negative below the null point"


@pytest.mark.physics
def test_inversion_recovery_matches_signed_equation():
    """Output equals the documented signed equation S = M0 (1 - 2 e_ti + e_tr)."""
    b, h, w = 2, 3, 5
    g = torch.Generator().manual_seed(0)
    m0 = torch.rand(b, 1, h, w, generator=g).clamp(min=0.1)
    t1 = torch.rand(b, 1, h, w, generator=g) * 900.0 + 100.0
    ti, tr = 250.0, 2000.0
    sig = BlochSimulator.inversion_recovery(m0, t1, ti, tr)
    e_ti = torch.exp(torch.tensor(-ti) / t1)
    e_tr = torch.exp(torch.tensor(-tr) / t1)
    expected = m0 * (1.0 - 2.0 * e_ti + e_tr)
    assert torch.allclose(sig, expected, atol=1e-6)
    # The equation is NOT a magnitude: it must differ from its own abs here.
    assert not torch.allclose(sig, sig.abs())


@pytest.mark.physics
def test_inversion_recovery_zero_crossing_near_null_point():
    """Signal passes through ~0 at TI ≈ T1*ln2 with TR -> inf."""
    m0 = torch.ones(1, 1, 2, 2)
    t1 = torch.full((1, 1, 2, 2), 1000.0)
    ti = 1000.0 * math.log(2)  # null point
    tr = 1.0e7
    sig = BlochSimulator.inversion_recovery(m0, t1, ti, tr)
    assert torch.allclose(sig, torch.zeros_like(sig), atol=1e-3)
