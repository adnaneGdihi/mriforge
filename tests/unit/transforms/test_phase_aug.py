"""Unit tests for phase_aug transforms.

Covers:
  - PhaseSimulator: complex passthrough identity, shape preservation,
    magnitude preservation (|exp(j*phi)|=1), non-trivial phase generation,
    determinism with same seed, parametrised severity (sigma_range).
  - apply_synthetic_phase functional helper.

All tests are CPU-deterministic; no external deps beyond torch.
"""
from __future__ import annotations

import pytest
import torch

from mriforge.data.transforms.phase_aug import PhaseSimulator, apply_synthetic_phase


# ── Canary ──────────────────────────────────────────────────────────────────


def test_canary_magnitude_to_complex() -> None:
    """Minimal end-to-end: real input → complex output, same shape."""
    sim = PhaseSimulator()
    mag = torch.rand(1, 1, 16, 16) + 0.1
    out = sim(mag)
    assert torch.is_complex(out)
    assert out.shape == mag.shape


# ── Identity at neutral / zero-severity ─────────────────────────────────────


def test_complex_input_passthrough_is_identity() -> None:
    """Already-complex input is returned unchanged (zero-work path)."""
    sim = PhaseSimulator()
    cplx = torch.complex(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    out = sim(cplx)
    assert torch.equal(out, cplx)


def test_zero_magnitude_yields_zero_complex_magnitude() -> None:
    """Magnitude=0 everywhere ⇒ |output|=0 everywhere."""
    sim = PhaseSimulator()
    out = sim(torch.zeros(1, 1, 16, 16))
    assert torch.allclose(out.abs(), torch.zeros(1, 1, 16, 16), atol=1e-7)


# ── Magnitude preservation (|exp(jφ)|=1) ────────────────────────────────────


def test_output_magnitude_matches_input_magnitude() -> None:
    """M * exp(j*phi) should have |output| = M."""
    sim = PhaseSimulator()
    mag = torch.rand(1, 1, 16, 16) + 0.1
    out = sim(mag)
    rel_err = (out.abs() - mag).abs().max() / mag.max()
    assert rel_err.item() < 1e-4, f"magnitude relative error {rel_err:.2e} too large"


# ── Non-trivial phase ────────────────────────────────────────────────────────


def test_output_phase_is_non_trivial() -> None:
    """Phase varies across the image — not constant zero."""
    torch.manual_seed(0)
    sim = PhaseSimulator()
    out = sim(torch.rand(1, 1, 32, 32) + 0.1)
    assert torch.angle(out).std().item() > 0.1


def test_different_seeds_produce_different_phases() -> None:
    """Stochastic draw: different seeds → different phases."""
    sim = PhaseSimulator()
    torch.manual_seed(1)
    out1 = sim(torch.ones(1, 1, 16, 16))
    torch.manual_seed(2)
    out2 = sim(torch.ones(1, 1, 16, 16))
    assert not torch.allclose(torch.angle(out1), torch.angle(out2))


# ── Determinism ──────────────────────────────────────────────────────────────


def test_determinism_same_seed() -> None:
    """Same manual seed → identical phase map."""
    sim = PhaseSimulator()
    mag = torch.ones(1, 1, 16, 16)
    torch.manual_seed(42)
    out1 = sim(mag)
    torch.manual_seed(42)
    out2 = sim(mag)
    assert torch.allclose(torch.angle(out1), torch.angle(out2), atol=1e-6)


# ── Parametrised severity ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sigma_range",
    [(0.5, 1.0), (1.0, 3.0), (2.0, 5.0)],
    ids=["tight", "default", "wide"],
)
def test_parametrised_sigma_range_preserves_shape(
    sigma_range: tuple[float, float],
) -> None:
    sim = PhaseSimulator(sigma_range=sigma_range)
    mag = torch.rand(1, 1, 16, 16) + 0.1
    out = sim(mag)
    assert out.shape == mag.shape
    assert torch.is_complex(out)
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()


# ── Shape matrix ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 16, 16), (2, 1, 32, 32), (1, 2, 16, 16)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_shape_matrix(shape: tuple[int, int, int, int]) -> None:
    sim = PhaseSimulator()
    mag = torch.rand(*shape) + 0.1
    out = sim(mag)
    assert out.shape == shape
    assert torch.is_complex(out)


# ── Constructor attributes ────────────────────────────────────────────────────


def test_kernel_size_stored_on_module() -> None:
    sim = PhaseSimulator(kernel_size=11)
    assert sim.kernel_size == 11


def test_sigma_range_stored_on_module() -> None:
    sim = PhaseSimulator(sigma_range=(0.5, 2.0))
    assert sim.sigma_range == (0.5, 2.0)


# ── Functional helper ────────────────────────────────────────────────────────


def test_apply_synthetic_phase_returns_complex_same_shape() -> None:
    mag = torch.rand(1, 1, 8, 8) + 0.1
    out = apply_synthetic_phase(mag)
    assert torch.is_complex(out)
    assert out.shape == mag.shape
