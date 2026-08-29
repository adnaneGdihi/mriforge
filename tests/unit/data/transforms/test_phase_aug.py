"""Tests for ``PhaseSimulator`` and ``apply_synthetic_phase``.

Targets ``mriforge.data.transforms.phase_aug``. This is a phase *augmentation*
(smooth random low-frequency phase, uncalibrated), NOT a physical B0/
susceptibility field simulator — used to bootstrap complex-domain training when
only magnitude images are available.

Categories:

- Unit: complex passthrough, magnitude → complex transformation
- Property: |output| ≈ input magnitude (pure phase, no amplitude change)
- Sanity-shape: parametrised matrix
- Honesty: the docstrings must not claim a physical B0 field (F5 relabel)
"""

from __future__ import annotations

import pytest
import torch

from mriforge.data.transforms import phase_aug
from mriforge.data.transforms.phase_aug import PhaseSimulator, apply_synthetic_phase


# ---------------------------------------------------------------------------
# Behavioural contract
# ---------------------------------------------------------------------------


def test_complex_input_returned_unchanged() -> None:
    """If the input is already complex, the simulator returns it as-is."""
    sim = PhaseSimulator()
    cplx = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    out = sim(cplx)
    assert torch.equal(out, cplx)


def test_magnitude_input_yields_complex_output() -> None:
    """Real magnitude input → complex output (with synthesised phase)."""
    sim = PhaseSimulator()
    mag = torch.rand(1, 1, 16, 16)
    out = sim(mag)
    assert torch.is_complex(out)
    assert out.shape == mag.shape


def test_output_magnitude_preserves_input_magnitude() -> None:
    """``|exp(j*phi)| == 1`` so |output| should equal input magnitude.

    Tolerance is loose because the smoothed phase is normalised by std,
    which involves floating-point reductions.
    """
    sim = PhaseSimulator()
    mag = torch.rand(1, 1, 16, 16) + 0.1
    out = sim(mag)
    output_mag = out.abs()
    rel_err = (output_mag - mag).abs().max().item() / mag.max().item()
    assert rel_err < 1e-4, f"magnitude mismatch rel_err = {rel_err:.2e}"


def test_output_phase_is_non_trivial() -> None:
    """The synthesised phase is not identically zero."""
    torch.manual_seed(0)
    sim = PhaseSimulator()
    mag = torch.rand(1, 1, 32, 32) + 0.1
    out = sim(mag)
    phase = torch.angle(out)
    # Phase should vary across the image (smooth, but not constant).
    assert phase.std().item() > 0.1


def test_different_seeds_produce_different_phases() -> None:
    """Stochastic — running twice produces different phase realisations."""
    sim = PhaseSimulator()
    torch.manual_seed(1)
    out1 = sim(torch.ones(1, 1, 16, 16))
    torch.manual_seed(2)
    out2 = sim(torch.ones(1, 1, 16, 16))
    assert not torch.allclose(torch.angle(out1), torch.angle(out2))


def test_kernel_size_attribute_persisted() -> None:
    """Custom kernel_size is stored on the module."""
    sim = PhaseSimulator(kernel_size=11)
    assert sim.kernel_size == 11


def test_sigma_range_attribute_persisted() -> None:
    """Custom sigma_range is stored on the module."""
    sim = PhaseSimulator(sigma_range=(0.5, 2.0))
    assert sim.sigma_range == (0.5, 2.0)


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


def test_apply_synthetic_phase_functional_interface() -> None:
    """The functional helper produces a complex output for magnitude input."""
    mag = torch.rand(1, 1, 8, 8) + 0.1
    out = apply_synthetic_phase(mag)
    assert torch.is_complex(out)
    assert out.shape == mag.shape


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 16, 16), (2, 1, 32, 32), (1, 2, 16, 16), (4, 1, 64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_shape_matrix(shape: tuple[int, int, int, int]) -> None:
    """Output shape matches input across the parametrised matrix."""
    sim = PhaseSimulator()
    mag = torch.rand(*shape) + 0.1
    out = sim(mag)
    assert out.shape == shape
    assert torch.is_complex(out)
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_input_yields_zero_output() -> None:
    """Magnitude=0 everywhere → complex output with magnitude=0 everywhere."""
    sim = PhaseSimulator()
    out = sim(torch.zeros(1, 1, 16, 16))
    assert torch.allclose(out.abs(), torch.zeros_like(out.abs()), atol=1e-7)


# ---------------------------------------------------------------------------
# Honesty (F5): must not claim a calibrated physical B0/susceptibility field
# ---------------------------------------------------------------------------


def test_module_docstring_marks_augmentation_not_physical_field() -> None:
    doc = (phase_aug.__doc__ or "") + (PhaseSimulator.__doc__ or "")
    lower = doc.lower()
    assert "augmentation" in lower
    # It must explicitly disclaim being a calibrated B0 field simulator.
    assert "not a calibrated b0" in lower or "not a b0" in lower
