"""Unit tests for :class:`ComplexRMSNorm`.

The pure-k-space ``complex_unet`` backbone had NO inter-layer normalization
(spatial BN/GN flatten the 1/f k-space spectrum, so the norm slots were left as
Identity), which let activation magnitude drift over training into the
experiment_11 measurement-independent collapse. ``ComplexRMSNorm`` is the
k-space-appropriate replacement: it pins the GLOBAL activation scale (one scalar
per sample) while preserving both phase and the relative spectrum. These tests
assert exactly those three properties.
"""

import pytest
import torch

from spectramr.models.layers.complex_norm import ComplexRMSNorm


def _split(x: torch.Tensor):
    """Interleaved [B, 2C, H, W] -> (real, imag), matching ComplexConv2d."""
    return x[:, 0::2], x[:, 1::2]


def test_shape_preserved():
    norm = ComplexRMSNorm(complex_channels=3)
    x = torch.randn(2, 6, 8, 8)
    assert norm(x).shape == x.shape


def test_scale_invariance_pins_activation_magnitude():
    """Output global RMS is ~1 regardless of input scale — the divergence control.

    A diverging backbone feeds each block an ever-larger activation; the norm
    must map a 1x, 10x, ... 1000x input to the same bounded scale. (The reverse
    direction, a near-zero input, is intentionally floored by ``eps`` so the
    normalizer never amplifies noise to unit scale — divergence is the
    growing-magnitude direction, which is what this asserts.)
    """
    norm = ComplexRMSNorm(complex_channels=4)
    x = torch.randn(3, 8, 16, 16)
    for factor in (1.0, 10.0, 100.0, 1000.0):
        out = norm(x * factor)
        r, i = _split(out)
        rms = torch.sqrt((r * r + i * i).mean())
        assert torch.isclose(rms, torch.tensor(1.0), atol=1e-2), (factor, rms)


def test_near_zero_input_is_not_amplified():
    """The ``eps`` floor keeps a near-zero activation near zero (no noise blow-up)."""
    norm = ComplexRMSNorm(complex_channels=2, eps=1e-5)
    x = torch.randn(2, 4, 8, 8) * 1e-4
    out = norm(x)
    r, i = _split(out)
    assert torch.sqrt((r * r + i * i).mean()) < 0.5


def test_phase_preserved_at_unit_gain():
    """Normalization scales real+imag by the SAME positive scalar => phase kept."""
    norm = ComplexRMSNorm(complex_channels=2)
    x = torch.randn(2, 4, 8, 8) + 3.0  # keep |z| comfortably away from 0
    out = norm(x)
    r0, i0 = _split(x)
    r1, i1 = _split(out)
    ang_in = torch.atan2(i0, r0)
    ang_out = torch.atan2(i1, r1)
    assert torch.allclose(ang_in, ang_out, atol=1e-5)


def test_relative_spectrum_preserved_at_unit_gain():
    """Every frequency bin is scaled by the SAME per-sample scalar => the 1/f
    spectrum shape (ratios between bins) is untouched. This is why RMSNorm is
    safe here where BatchNorm/GroupNorm are not."""
    norm = ComplexRMSNorm(complex_channels=1)
    x = torch.randn(1, 2, 8, 8) + 2.0
    out = norm(x)
    r0, i0 = _split(x)
    r1, i1 = _split(out)
    mag_in = torch.sqrt(r0 * r0 + i0 * i0)
    mag_out = torch.sqrt(r1 * r1 + i1 * i1)
    ratio = mag_out / mag_in  # constant per sample if the spectrum is preserved
    assert torch.allclose(ratio, ratio.mean(), atol=1e-5)


def test_gradients_flow_to_input_and_gain():
    norm = ComplexRMSNorm(complex_channels=2)
    x = torch.randn(1, 4, 8, 8, requires_grad=True)
    norm(x).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm.gain.grad is not None and torch.isfinite(norm.gain.grad).all()


def test_wrong_channel_count_raises():
    norm = ComplexRMSNorm(complex_channels=3)
    with pytest.raises(ValueError):
        norm(torch.randn(1, 4, 8, 8))  # 4 != 2*3
