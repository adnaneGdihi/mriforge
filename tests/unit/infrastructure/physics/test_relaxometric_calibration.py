"""Tests for fiducial-calibrated contrast transfer between field strengths.

The claim is that ``kappa`` is a KNOWN constant, not an estimated one, and that
it describes a genuine contrast change rather than a global scale. Both are
checked numerically here: a global scale would be absorbed by any normalisation
and would leave the network nothing to be handed.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.relaxation_priors import (  # noqa: E402
    TissueClass,
    bottomley_t1,
)
from spectramr.infrastructure.physics.relaxometric_calibration import (  # noqa: E402
    AcquisitionParams,
    measured_gain,
    relaxometric_gain,
    spgr_signal,
    tissue_gain_map,
)

ULF = AcquisitionParams(field_strength_t=0.064, tr_ms=500.0, te_ms=15.0, flip_deg=90.0)
HF = AcquisitionParams(field_strength_t=3.0, tr_ms=500.0, te_ms=15.0, flip_deg=90.0)


# ── declared acquisitions ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw",
    [
        {"field_strength_t": 0.0},
        {"tr_ms": 0.0},
        {"te_ms": -1.0},
        {"te_ms": 600.0},  # TE >= TR
        {"flip_deg": 0.0},
        {"flip_deg": 200.0},
    ],
)
def test_impossible_acquisitions_raise(kw: dict) -> None:
    base = {
        "field_strength_t": 3.0,
        "tr_ms": 500.0,
        "te_ms": 15.0,
        "flip_deg": 90.0,
    }
    with pytest.raises(ValueError):
        AcquisitionParams(**{**base, **kw})


# ── the closed form ──────────────────────────────────────────────────────────


def test_spgr_matches_the_analytic_expression() -> None:
    t1, t2 = 900.0, 80.0
    e1 = math.exp(-HF.tr_ms / t1)
    e2 = math.exp(-HF.te_ms / t2)
    expected = math.sin(math.pi / 2) * (1 - e1) / (1 - math.cos(math.pi / 2) * e1) * e2
    assert spgr_signal(t1, t2, HF) == pytest.approx(expected)


def test_longer_t1_gives_less_signal_at_fixed_tr() -> None:
    """The direction of the whole effect: T1 lengthens with field, so the same
    sequence recovers less magnetisation at 3 T than at 64 mT."""
    assert spgr_signal(1200.0, 80.0, HF) < spgr_signal(300.0, 80.0, HF)


def test_non_positive_relaxation_times_raise() -> None:
    with pytest.raises(ValueError, match="must be > 0 ms"):
        spgr_signal(0.0, 80.0, HF)


# ── the gain ─────────────────────────────────────────────────────────────────


def test_identical_fields_and_sequence_give_unit_gain() -> None:
    """The consistency check: no field difference, no contrast transfer."""
    assert relaxometric_gain(900.0, 900.0, 80.0, HF, HF) == pytest.approx(1.0)


def test_gain_is_a_contrast_change_not_a_global_scale() -> None:
    """If every tissue shared one gain it would be absorbed by normalisation.

    Over 64 mT to 3 T the transfer is dominated by CSF against parenchyma; the
    white-to-grey part is real but small, which bounds what an arm may claim.
    """
    g = tissue_gain_map(ULF, HF)
    assert g[TissueClass.CSF] / g[TissueClass.WHITE_MATTER] > 1.5
    assert 1.0 < g[TissueClass.WHITE_MATTER] / g[TissueClass.GRAY_MATTER] < 1.2


def test_gains_follow_the_bottomley_t1_dispersion() -> None:
    """Not a free parameter: T1 comes from the package's existing table, so
    this module adds no second source of relaxometry."""
    for tissue in TissueClass:
        assert bottomley_t1(3.0, tissue) > bottomley_t1(0.064, tissue)
    g = tissue_gain_map(ULF, HF)
    # CSF's T1 is nearly field-independent, so its gain is nearest unity
    assert abs(g[TissueClass.CSF] - 1.0) < abs(g[TissueClass.WHITE_MATTER] - 1.0)


def test_zero_source_signal_raises_rather_than_dividing() -> None:
    """A T1 so long that no magnetisation recovers within TR drives the source
    signal to exactly zero in float64. Degenerate, but a silent division there
    would produce an infinite gain the factored model would then apply."""
    with pytest.raises(ValueError, match="undefined"):
        relaxometric_gain(1e30, 900.0, 80.0, ULF, HF)


# ── the empirical check ──────────────────────────────────────────────────────


def test_measured_gain_recovers_a_known_constant_on_the_support() -> None:
    """Noise OUTSIDE the support must not perturb the estimate: on marker
    support the declared relaxometry holds, and nowhere else."""
    torch.manual_seed(0)
    source = torch.rand(3, 1, 32, 32) + 0.2
    support = torch.zeros(3, 1, 32, 32)
    support[..., 8:24, 8:24] = 1.0
    target = 0.437 * source + 2.0 * (1 - support) * torch.randn(3, 1, 32, 32)
    assert measured_gain(target, source, support) == pytest.approx(
        torch.full((3,), 0.437), abs=1e-5
    )


def test_measured_gain_is_least_squares_not_a_voxel_ratio() -> None:
    """A per-voxel ratio blows up where the source is near zero; the weighted
    least squares does not."""
    source = torch.tensor([[[[1.0, 1e-9]]]])
    target = torch.tensor([[[[0.5, 1e-9]]]])
    support = torch.ones_like(source)
    assert float(measured_gain(target, source, support)) == pytest.approx(0.5, abs=1e-6)


def test_measured_gain_is_differentiable() -> None:
    source = torch.rand(1, 1, 8, 8) + 0.5
    target = (0.4 * source).requires_grad_(True)
    measured_gain(target, source, torch.ones_like(source)).sum().backward()
    assert target.grad is not None and float(target.grad.abs().sum()) > 0.0


def test_measured_gain_rejects_mismatched_and_negative_inputs() -> None:
    a = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError, match="must match"):
        measured_gain(a, torch.rand(1, 1, 4, 4), torch.ones_like(a))
    with pytest.raises(ValueError, match="share batch and spatial"):
        measured_gain(a, a, torch.ones(1, 1, 4, 4))
    with pytest.raises(ValueError, match="non-negative"):
        measured_gain(a, a, -torch.ones_like(a))
