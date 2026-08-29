"""Tests for B0 phase-shift utility.

Targets ``mriforge.infrastructure.physics.b0_utils``. Small but important
utility — applies B0-induced phase accumulation to a real-stacked
k-space signal.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.b0_utils import apply_b0_phase_shift


def test_no_time_vector_returns_signal_unchanged() -> None:
    """If ``time_vec`` is None, the input is returned untouched."""
    signal = torch.randn(2, 2, 16)
    b0 = torch.zeros(2, 1, 8, 8)
    out = apply_b0_phase_shift(signal, b0, time_vec=None)
    assert torch.equal(out, signal)


def test_zero_b0_map_yields_no_phase_shift() -> None:
    """Zero B0 → exp(0) = 1 → output identical to input within tolerance."""
    signal = torch.randn(1, 2, 16)
    b0 = torch.zeros(1, 1, 8, 8)
    time_vec = torch.linspace(0, 10, 16).unsqueeze(0)
    out = apply_b0_phase_shift(signal, b0, time_vec=time_vec)
    assert torch.allclose(out, signal, atol=1e-6)


def test_phase_shift_depends_only_on_spatial_mean() -> None:
    """Documented mean-field behavior: ΔΩ is the spatial mean of b0_map, so two
    maps with the same mean but different structure produce identical output."""
    torch.manual_seed(0)
    signal = torch.randn(2, 2, 8)
    time_vec = torch.linspace(0.0, 0.01, 8).unsqueeze(0)
    a = torch.randn(2, 1, 4, 4)
    # b has the same per-sample spatial mean as a but a different pattern.
    b = torch.randn(2, 1, 4, 4)
    b = b - b.mean(dim=(2, 3), keepdim=True) + a.mean(dim=(2, 3), keepdim=True)
    out_a = apply_b0_phase_shift(signal, a, time_vec=time_vec)
    out_b = apply_b0_phase_shift(signal, b, time_vec=time_vec)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_nonzero_b0_rotates_signal() -> None:
    """A constant non-zero B0 produces a phase rotation."""
    signal = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])  # [1, 2, 3] real-imag
    b0 = torch.full((1, 1, 4, 4), 100.0)  # Hz
    time_vec = torch.tensor([[0.001, 0.002, 0.003]])
    out = apply_b0_phase_shift(signal, b0, time_vec=time_vec, scale=1.0)
    # Output should differ from input (rotation occurred).
    assert not torch.allclose(out, signal, atol=1e-3)
    # Energy preserved (rotation is unitary).
    in_energy = (signal ** 2).sum()
    out_energy = (out ** 2).sum()
    assert torch.allclose(in_energy, out_energy, atol=1e-3)


def test_output_shape_matches_input() -> None:
    """Shape ``[B, 2, N]`` is preserved by the rotation."""
    signal = torch.randn(2, 2, 32)
    b0 = torch.full((2, 1, 8, 8), 50.0)
    time_vec = torch.linspace(0, 10, 32).unsqueeze(0).expand(2, 32)
    out = apply_b0_phase_shift(signal, b0, time_vec=time_vec)
    assert out.shape == signal.shape


def test_scale_parameter_modulates_phase_amount() -> None:
    """Larger scale → larger rotation angle (output diverges from input more)."""
    signal = torch.randn(1, 2, 16)
    b0 = torch.full((1, 1, 4, 4), 100.0)
    time_vec = torch.linspace(0, 5, 16).unsqueeze(0)
    out_small = apply_b0_phase_shift(signal, b0, time_vec=time_vec, scale=0.1)
    out_large = apply_b0_phase_shift(signal, b0, time_vec=time_vec, scale=10.0)
    diff_small = (out_small - signal).abs().mean()
    diff_large = (out_large - signal).abs().mean()
    assert diff_large > diff_small


def test_non_2_channel_input_returns_unchanged() -> None:
    """If signal is not [B, 2, N], it's returned unchanged (real-stack contract)."""
    signal = torch.randn(1, 1, 16)  # not real-stacked
    b0 = torch.zeros(1, 1, 4, 4)
    time_vec = torch.linspace(0, 5, 16).unsqueeze(0)
    out = apply_b0_phase_shift(signal, b0, time_vec=time_vec)
    assert torch.equal(out, signal)


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 8), (2, 2, 16), (4, 2, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_shape_matrix(shape: tuple[int, int, int]) -> None:
    """Shape preserved across a parametrised matrix of real-stacked signals."""
    signal = torch.randn(*shape)
    b, _, n = shape
    b0 = torch.full((b, 1, 4, 4), 50.0)
    time_vec = torch.linspace(0, 5, n).unsqueeze(0).expand(b, n)
    out = apply_b0_phase_shift(signal, b0, time_vec=time_vec)
    assert out.shape == signal.shape
    assert torch.isfinite(out).all()
