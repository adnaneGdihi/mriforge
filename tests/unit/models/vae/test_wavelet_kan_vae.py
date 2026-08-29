"""Regression tests for ``mriforge.models.vae.wavelet_kan_vae``.

The registered ``vae_wavelet_kan_vae`` headline mechanism is the wavelet
KAN convolution. The module used to guard its import with ``try/except
ImportError`` that defined a MOCK plain-``nn.Conv2d`` stand-in (even
swallowing ``wavelet_type``) — and because the guarded import targeted a
name (``WavKANConv2D``) that ``wav_kan.py`` never exported, the mock path
ALWAYS fired: the registered model was a plain conv VAE from day one
(pitfall #9/#16). These tests are the mechanism-fires probe: the real
``WavKANConv2DLayer`` must be importable and present in the module tree.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.layers.kan.kan_convs.wav_kan import (  # noqa: E402
    WavKANConv2DLayer,
)
from mriforge.models.vae.wavelet_kan_vae import WaveletKANVAE  # noqa: E402


class TestWaveletKANVAEMechanismFires:
    def test_no_mock_conv_module_level_fallback(self) -> None:
        """The module must not define its own WavKANConv2D mock class."""
        import mriforge.models.vae.wavelet_kan_vae as mod

        assert not hasattr(mod, "WavKANConv2D"), (
            "mock WavKANConv2D fallback reintroduced — the real layer is "
            "WavKANConv2DLayer from kan_convs.wav_kan"
        )

    def test_encoder_and_decoder_use_real_wavelet_kan(self) -> None:
        """All four conv stages are real WavKANConv2DLayer instances."""
        vae = WaveletKANVAE(in_channels=1, latent_dim=16)
        wav_layers = [
            m for m in vae.modules() if isinstance(m, WavKANConv2DLayer)
        ]
        assert len(wav_layers) == 4
        # wavelet_type must be threaded through, not swallowed
        for layer in wav_layers:
            assert layer.wavelet_type == "daubechies"
        # And no bare nn.Conv2d imposters at the top of encoder/decoder
        assert isinstance(vae.encoder[0], WavKANConv2DLayer)
        assert isinstance(vae.decoder_layers[0], WavKANConv2DLayer)

    def test_forward_finite_and_shapes(self) -> None:
        torch.manual_seed(0)
        vae = WaveletKANVAE(in_channels=1, latent_dim=16)
        x = torch.randn(2, 1, 64, 64)
        recon, mu, logvar = vae(x)
        assert recon.shape == (2, 1, 64, 64)
        assert mu.shape == (2, 16)
        assert logvar.shape == (2, 16)
        assert torch.isfinite(recon).all()
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_unsupported_wavelet_type_raises(self) -> None:
        """The real layer rejects unknown wavelet types (no silent default)."""
        with pytest.raises((AssertionError, ValueError)):
            WavKANConv2DLayer(1, 8, kernel_size=3, padding=1, wavelet_type="db2")
