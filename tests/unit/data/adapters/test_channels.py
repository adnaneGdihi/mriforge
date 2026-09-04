"""Tests for ``src/spectramr/data/adapters/channels.py``.

Covers ``complex_to_real_imag_interleave``: a complex tensor is split
into interleaved real/imag channels, and a real (non-complex) tensor
arriving at this adapter is a misconfiguration that must raise (NN#3 /
pitfall #9) rather than silently passing through unchanged.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Force adapter registration.
import spectramr.data.adapters  # noqa: F401, E402  triggers @register_adapter
from spectramr.data.adapters.channels import ComplexToRealImagInterleave  # noqa: E402


def test_complex_to_real_imag_interleave_raises_on_real_input() -> None:
    """A real (non-complex) tensor is a misconfiguration → must raise (NN#3)."""
    x = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    with pytest.raises(
        ValueError, match="complex_to_real_imag_interleave expects a complex tensor"
    ):
        ComplexToRealImagInterleave()(x)


def test_complex_to_real_imag_interleave_complex_input_doubles_channels() -> None:
    """A complex ``[B, C, H, W]`` input yields ``[B, 2C, H, W]`` interleaved real."""
    real = torch.randn(2, 4, 8, 8)
    imag = torch.randn(2, 4, 8, 8)
    x = torch.complex(real, imag)
    out = ComplexToRealImagInterleave()(x)
    assert out.shape == (2, 8, 8, 8)
    assert out.dtype == torch.float32
    # Interleave contract: even channels are real, odd channels are imag.
    torch.testing.assert_close(out[:, 0::2], real)
    torch.testing.assert_close(out[:, 1::2], imag)
