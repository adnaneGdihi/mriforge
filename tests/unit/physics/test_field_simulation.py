"""Unit tests for src/mriforge/infrastructure/physics/field_simulation.py.

Focus (doc-contract regression):
  - ``B0MapSimulator.generate_concomitant_field`` longitudinal-gradient term.
    The code uses an explicit ``/ (8 * B0)`` prefactor; the docstring header
    states ``(1/(2*B0)) * [Gz^2 * (x^2+y^2)/4]``. These are algebraically
    identical (``1/(2*B0) * 1/4 == 1/(8*B0)``). This test locks the identity so
    the code-vs-docstring equivalence stays a regression guard, and pins the
    exact analytical longitudinal coefficient (Gz-only, isocenter, z_offset=0).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.field_simulation import B0MapSimulator


@pytest.mark.physics
def test_concomitant_field_runs_and_shape() -> None:
    """Output is the documented [1, 1, H, W] field map and is finite."""
    H, W = 16, 16
    out = B0MapSimulator.generate_concomitant_field(
        shape=(H, W),
        field_strength=0.064,
        gx=5.0,
        gy=3.0,
        gz=10.0,
        z_offset=0.02,
        device="cpu",
    )
    assert out.shape == (1, 1, H, W)
    assert torch.isfinite(out).all()


@pytest.mark.physics
def test_longitudinal_coefficient_matches_docstring_and_code() -> None:
    """Gz-only, isocenter (z_offset=0): the concomitant field must equal BOTH the
    code's ``Gz^2 (x^2+y^2) / (8 B0)`` and the docstring's
    ``(1/(2 B0)) [Gz^2 (x^2+y^2) / 4]``, converted to Hz via gamma.

    Locks the docstring-vs-code algebraic identity (1/(2B0) * 1/4 == 1/(8B0)).
    """
    H, W = 24, 24
    B0 = 0.064  # Tesla (ULF, where Maxwell terms matter)
    gz_mT = 10.0  # mT/m
    fov = 0.25  # must match the implementation's hardcoded FoV
    gamma_hz = 42.576e6  # must match the implementation

    out = B0MapSimulator.generate_concomitant_field(
        shape=(H, W),
        field_strength=B0,
        gx=0.0,
        gy=0.0,
        gz=gz_mT,
        z_offset=0.0,  # kills transverse and cross terms -> longitudinal only
        device="cpu",
    )
    assert out.shape == (1, 1, H, W)

    # Rebuild the same centered grid the implementation uses.
    y = torch.linspace(-fov / 2, fov / 2, H)
    x = torch.linspace(-fov / 2, fov / 2, W)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    Gz = gz_mT * 1e-3  # mT/m -> T/m

    # Code form: Gz^2 (x^2+y^2) / (8 B0), then -> Hz.
    expected_code = ((Gz**2 * (xx**2 + yy**2)) / (8.0 * B0)) * gamma_hz
    # Docstring header form: (1/(2 B0)) * [Gz^2 (x^2+y^2)/4], then -> Hz.
    expected_doc = ((1.0 / (2.0 * B0)) * (Gz**2 * (xx**2 + yy**2) / 4.0)) * gamma_hz

    # The two algebraic forms must be identical (the doc-contract identity).
    assert torch.allclose(expected_code, expected_doc, atol=1e-6, rtol=1e-5)

    # And the actual output must match the analytical longitudinal field.
    assert torch.allclose(out[0, 0], expected_code, atol=1e-6, rtol=1e-5)


@pytest.mark.physics
def test_zero_gradients_give_zero_field() -> None:
    """All-zero gradients produce an identically-zero concomitant field."""
    out = B0MapSimulator.generate_concomitant_field(
        shape=(12, 12),
        field_strength=0.064,
        gx=0.0,
        gy=0.0,
        gz=0.0,
        z_offset=0.05,
        device="cpu",
    )
    assert torch.allclose(out, torch.zeros_like(out))
