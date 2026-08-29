"""Tests for ``DataConsistencyLayer``.

Targets ``mriforge.infrastructure.physics.data_consistency_layer``. The
classic hard-DC layer: at sampled k-space locations, the network
guess is replaced with the acquired measurement.

Categories:

- Unit: complex input round-trip preserves shape; mask=0 → output equals
  iFFT(FFT(input)) (identity); mask=1 → output equals iFFT(measured)
- Property: idempotence — applying DC twice == once
- Property: 2-channel real input also supported
- Edge: odd C-channel input fails loud (not 2-channel or complex)
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.data_consistency_layer import DataConsistencyLayer
from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default ``noise_robust=False``."""
    dc = DataConsistencyLayer()
    assert dc.noise_robust is False


# ---------------------------------------------------------------------------
# Complex input
# ---------------------------------------------------------------------------


def test_complex_input_zero_mask_returns_iFFTFFT_input() -> None:
    """With mask=0, output equals iFFT(FFT(input)) (no DC enforced)."""
    dc = DataConsistencyLayer()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.zeros(1, 1, 8, 8)
    out = dc(img, measured, mask)
    expected = ifft2c(fft2c(img))
    assert torch.allclose(out, expected, atol=1e-5)


def test_complex_input_full_mask_returns_iFFT_measured() -> None:
    """With mask=1, output equals iFFT(measured) (full overwrite)."""
    dc = DataConsistencyLayer()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.ones(1, 1, 8, 8)
    out = dc(img, measured, mask)
    expected = ifft2c(measured)
    assert torch.allclose(out, expected, atol=1e-5)


def test_complex_input_preserves_shape() -> None:
    """Output shape matches input shape."""
    dc = DataConsistencyLayer()
    img = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    measured = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    mask = torch.zeros(2, 1, 16, 16)
    out = dc(img, measured, mask)
    assert out.shape == img.shape


def test_dc_is_idempotent() -> None:
    """Applying DC twice gives the same result as applying it once."""
    dc = DataConsistencyLayer()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0  # half-mask
    once = dc(img, measured, mask)
    twice = dc(once, measured, mask)
    assert torch.allclose(once, twice, atol=1e-4)


def test_bool_mask_accepted() -> None:
    """Boolean mask is auto-converted to float."""
    dc = DataConsistencyLayer()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    measured = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    out = dc(img, measured, mask)
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# 2-channel real input
# ---------------------------------------------------------------------------


def test_real_2channel_input_preserves_shape() -> None:
    """``[B, 2, H, W]`` real input → ``[B, 2, H, W]`` output."""
    dc = DataConsistencyLayer()
    img = torch.randn(1, 2, 8, 8)
    measured = torch.randn(1, 2, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    out = dc(img, measured, mask)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Edge: odd channels
# ---------------------------------------------------------------------------


def test_odd_channel_count_raises() -> None:
    """Odd C (not complex, not 2-channel pairs) → ``ValueError``."""
    dc = DataConsistencyLayer()
    img = torch.randn(1, 3, 8, 8)
    measured = torch.randn(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="2-channel"):
        dc(img, measured, mask)


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8), (2, 1, 16, 16), (1, 1, 32, 16)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_complex_dc_shape_matrix(shape: tuple[int, ...]) -> None:
    """Complex DC handles the parametrised shape matrix."""
    dc = DataConsistencyLayer()
    img = torch.complex(torch.randn(*shape), torch.randn(*shape))
    measured = torch.complex(torch.randn(*shape), torch.randn(*shape))
    mask = torch.zeros(*shape)
    out = dc(img, measured, mask)
    assert out.shape == shape
    assert torch.is_complex(out)
