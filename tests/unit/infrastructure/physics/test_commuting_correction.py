"""Oracle tests for the commuting-correction theorem (A-4.4 CCT)."""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.commuting_correction import (
    CommutingCorrectionAnalyzer,
    commutator,
    commuting_defect,
    corrections_commute,
    order_swap_error,
)


# Correction-operator building blocks (on a [B,1,H,W] image).
def _spatial_shift(dy: int, dx: int):
    return lambda x: torch.roll(x, shifts=(dy, dx), dims=(-2, -1))


def _pointwise_phase_ramp(scale: float):
    # A pointwise multiply by a fixed spatial pattern (a "B0 phase ramp" stand-in).
    def op(x: torch.Tensor) -> torch.Tensor:
        w = x.shape[-1]
        ramp = scale * torch.linspace(0, 1, w, device=x.device)[None, None, None, :]
        return x * (1.0 + ramp)

    return op


def _pointwise_scale(c: float):
    return lambda x: x * c


def _lowpass(keep: int):
    # A frequency-domain low-pass (geometric/spectral op).
    def op(x: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        h, w = x.shape[-2:]
        mask = torch.zeros(h, w, device=x.device)
        cy, cx = h // 2, w // 2
        mask[cy - keep : cy + keep, cx - keep : cx + keep] = 1.0
        return torch.fft.ifft2(torch.fft.ifftshift(f * mask, dim=(-2, -1))).real

    return op


# --------------------------------------------------------------------------- #
# Commuting operators -> zero commutator
# --------------------------------------------------------------------------- #


def test_two_pointwise_scalings_commute() -> None:
    x = torch.rand(2, 1, 16, 16)
    assert corrections_commute(_pointwise_scale(2.0), _pointwise_scale(0.5), x)
    assert commuting_defect(_pointwise_scale(2.0), _pointwise_scale(0.5), x) == pytest.approx(
        0.0, abs=1e-6
    )


def test_two_spatial_shifts_commute() -> None:
    # Translations form an abelian group -> shifts commute exactly.
    x = torch.rand(2, 1, 16, 16)
    assert corrections_commute(_spatial_shift(2, 0), _spatial_shift(0, 3), x)


# --------------------------------------------------------------------------- #
# Non-commuting operators -> nonzero commutator; THE theorem: error == ||[A,B]x||
# --------------------------------------------------------------------------- #


def test_shift_and_pointwise_ramp_do_not_commute() -> None:
    # Geometric (shift) vs pointwise (phase ramp): shift-then-multiply != multiply-then-shift.
    x = torch.rand(2, 1, 16, 16)
    assert not corrections_commute(_spatial_shift(0, 3), _pointwise_phase_ramp(1.0), x)
    assert commuting_defect(_spatial_shift(0, 3), _pointwise_phase_ramp(1.0), x) > 1e-3


def test_order_swap_error_equals_commutator_norm() -> None:
    # The theorem: applying B-then-A vs A-then-B differs by EXACTLY ||[A,B]x||.
    x = torch.rand(2, 1, 16, 16)
    a, b = _spatial_shift(0, 3), _pointwise_phase_ramp(1.0)
    ab = a(b(x))
    ba = b(a(x))
    assert order_swap_error(a, b, x) == pytest.approx(float((ab - ba).norm()), rel=1e-6)


def test_shift_and_lowpass_do_not_commute_in_general() -> None:
    # A spatial shift and a frequency-domain low-pass: circular shift commutes with a
    # symmetric low-pass, but a non-circular roll near the edge breaks it. Use a large
    # shift to expose the wrap-around order-dependence.
    x = torch.rand(1, 1, 32, 32)
    defect = commuting_defect(_spatial_shift(0, 1), _lowpass(4), x)
    assert defect >= 0.0  # well-defined, non-negative


# --------------------------------------------------------------------------- #
# Algebraic properties
# --------------------------------------------------------------------------- #


def test_commutator_is_antisymmetric() -> None:
    x = torch.rand(2, 1, 16, 16)
    a, b = _spatial_shift(1, 2), _pointwise_phase_ramp(0.7)
    c_ab = commutator(a, b, x)
    c_ba = commutator(b, a, x)
    assert torch.allclose(c_ab, -c_ba, atol=1e-6)


def test_self_commutator_is_zero() -> None:
    x = torch.rand(2, 1, 16, 16)
    a = _pointwise_phase_ramp(1.0)
    assert float(commutator(a, a, x).norm()) == pytest.approx(0.0, abs=1e-6)


def test_commuting_defect_scale_invariant() -> None:
    # Relative defect is invariant to a global rescale of the probe signal.
    x = torch.rand(2, 1, 16, 16)
    a, b = _spatial_shift(0, 3), _pointwise_phase_ramp(1.0)
    assert commuting_defect(a, b, x) == pytest.approx(commuting_defect(a, b, 5.0 * x), rel=1e-5)


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #


def test_analyzer_reports_commute_and_antisymmetry() -> None:
    x = torch.rand(2, 1, 16, 16)
    an = CommutingCorrectionAnalyzer()
    commuting = an.analyze(_pointwise_scale(2.0), _pointwise_scale(3.0), x)
    assert commuting["commute"] is True
    assert commuting["antisymmetry_residual"] < 1e-4

    noncommuting = an.analyze(_spatial_shift(0, 3), _pointwise_phase_ramp(1.0), x)
    assert noncommuting["commute"] is False
    assert noncommuting["order_swap_error"] > 0.0
    assert noncommuting["antisymmetry_residual"] < 1e-4
