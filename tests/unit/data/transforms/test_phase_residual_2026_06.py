"""Tests for ``PhaseResidualTransform`` (inverse-Bloch phase_residual producer).

Resolves the 2026-06 audit finding "phase-residual loss dead — no key producer":
the transform emits a ``phase_residual`` image (background-removed measured
phase) that ``InverseBlochPhaseStrategy``'s smoothness prior consumes.
"""
from __future__ import annotations

import math

import pytest
import torch

from mriforge.data.transforms.phase_residual import PhaseResidualTransform


class _Img:
    def __init__(self, data: torch.Tensor) -> None:
        self._d = data

    @property
    def data(self) -> torch.Tensor:
        return self._d

    def set_data(self, d: torch.Tensor) -> None:
        self._d = d


class _Subj(dict):
    def add_image(self, image: object, name: str) -> None:
        self[name] = image


def _apply(z: torch.Tensor, kernel_size: int = 3) -> _Subj:
    subj = _Subj({"input": _Img(z.clone())})
    PhaseResidualTransform(kernel_size=kernel_size, keys=("input",)).apply_transform(subj)
    return subj


def test_emits_phase_residual_preserving_input() -> None:
    z = torch.randn(1, 8, 8) * torch.exp(1j * torch.rand(1, 8, 8))
    out = _apply(z)
    assert "input" in out and "phase_residual" in out
    assert torch.equal(out["input"].data, z)  # input untouched
    r = out["phase_residual"].data
    assert r.shape == z.shape
    assert not torch.is_complex(r)  # residual is a real phase map
    assert r.abs().max() <= math.pi + 1e-5  # wrapped to (-pi, pi]


def test_constant_phase_has_near_zero_residual() -> None:
    # A spatially-constant phase is pure background -> residual ~ 0.
    z = torch.ones(1, 8, 8, dtype=torch.complex64) * torch.exp(1j * torch.tensor(0.7))
    r = _apply(z)["phase_residual"].data
    assert r.abs().max() < 1e-4


def test_local_phase_bump_produces_local_residual() -> None:
    phase = torch.zeros(1, 9, 9)
    phase[0, 4, 4] = 1.2  # a localized phase deviation
    z = torch.exp(1j * phase)
    r = _apply(z, kernel_size=3)["phase_residual"].data
    # Residual is large at the bump, ~0 in a far corner.
    assert r[0, 4, 4].abs() > 0.3
    assert r[0, 0, 0].abs() < 1e-3


def test_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd"):
        PhaseResidualTransform(kernel_size=4)


# --- registry wiring (D9) ---------------------------------------------------


def test_phase_residual_is_registered_under_its_config_name() -> None:
    """The transform must be reachable from ``data.processing.transforms``.

    Before the registry it existed on disk with no way to be constructed from
    any config, so the consumer chain that reads its output was dead at link 0.
    """
    from mriforge.data.transforms.registry import get_transform

    entry = get_transform("phase_residual")
    assert entry.cls is PhaseResidualTransform
    assert "phase_residual" in entry.produces
