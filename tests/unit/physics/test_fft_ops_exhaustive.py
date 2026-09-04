import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import (
    FFTTransformer,
    _to_complex,
    coil_combine_rss,
    fft2c,
    ifft2c,
)


def test_to_complex():
    """Test complex tensor conversion logic."""
    # Already complex
    x_complex = torch.randn(2, 2, dtype=torch.complex64)
    assert _to_complex(x_complex) is x_complex

    # Real/imag encoding at last dim
    x_real_imag = torch.randn(2, 3, 2)
    res = _to_complex(x_real_imag)
    assert res.is_complex()
    assert res.shape == (2, 3)

    # Interleaved channel dimension [B, C, H, W] where C is even
    x_interleaved = torch.randn(2, 4, 8, 8)  # 4 channels -> 2 complex channels
    res = _to_complex(x_interleaved)
    assert res.is_complex()
    assert res.shape == (2, 2, 8, 8)

    # Strict mode on pure real tensor without matching shapes
    x_pure_real = torch.randn(2, 3, 8, 8)  # Odd channels
    with pytest.raises(ValueError):
        _to_complex(x_pure_real, strict=True)

    # Relaxed mode on pure real
    res = _to_complex(x_pure_real, strict=False)
    assert res.is_complex()
    assert torch.all(res.imag == 0)

def test_fft2c_ifft2c_roundtrip():
    """Test centered 2D FFT and IFFT maintain perfect roundtrip reconstruction."""
    x = torch.randn(2, 3, 16, 16, dtype=torch.complex64)
    kspace = fft2c(x)

    # K-space should be complex and same shape
    assert kspace.is_complex()
    assert kspace.shape == x.shape

    reconstructed = ifft2c(kspace)

    # Reconstruction should closely match original
    assert torch.allclose(x, reconstructed, atol=1e-5)

def test_fft_transformer():
    """Test the FFTTransformer wrapper class."""
    transformer = FFTTransformer(device="cpu")

    x = torch.randn(1, 2, 8, 8, dtype=torch.complex64)

    kspace = transformer.fft2c(x)
    assert kspace.shape == x.shape

    recon = transformer.ifft2c(kspace)
    assert torch.allclose(x, recon, atol=1e-5)

    # Test masked
    mask = torch.ones(1, 1, 8, 8)
    mask[..., 4:, :] = 0

    kspace_masked = transformer.fft2c_masked(x, mask)
    recon_masked = transformer.ifft2c_masked(kspace, mask)

    assert torch.allclose(kspace_masked, kspace * mask)
    # Masking in image domain after IFFT should happen before IFFT
    recon_direct = transformer.ifft2c(kspace * mask)
    assert torch.allclose(recon_masked, recon_direct)

def test_coil_combine_rss():
    """Test Root-Sum-of-Squares coil combination."""
    # B=1, C_coils=4, H=16, W=16
    coils = torch.randn(1, 4, 16, 16, dtype=torch.complex64)

    combined = coil_combine_rss(coils)

    # Output should be real magnitude [1, 1, 16, 16]
    assert not combined.is_complex()
    assert combined.shape == (1, 1, 16, 16)

    # Manual check
    mag2 = (coils.real**2 + coils.imag**2).sum(dim=1, keepdim=True)
    expected = torch.sqrt(mag2 + 1e-8)
    assert torch.allclose(combined, expected)
