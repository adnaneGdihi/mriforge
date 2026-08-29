"""Phase-2 unit tests for NavEncoder1D."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.encoders.nav_encoder_1d import NavEncoder1D  # noqa: E402
from mriforge.models.registry import MODEL_REGISTRY  # noqa: E402


class TestRegistration:
    def test_registered(self) -> None:
        # Force the package init to fire.
        import mriforge.models.encoders  # noqa: F401
        # Direct import path triggers registration too.
        import mriforge.models.encoders.nav_encoder_1d  # noqa: F401
        assert "nav_encoder_1d" in MODEL_REGISTRY


class TestForward:
    def test_real_imag_interleaved_input(self) -> None:
        enc = NavEncoder1D(in_channels=2, nav_latent_dim=32, depth=3)
        x = torch.randn(4, 2, 64)
        z = enc(x)
        assert z.shape == (4, 32)
        # L2-normalised — unit norm per sample.
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_complex_input_split(self) -> None:
        enc = NavEncoder1D(in_channels=2, nav_latent_dim=16, depth=2)
        x = torch.randn(2, 32, dtype=torch.complex64)
        z = enc(x)
        assert z.shape == (2, 16)

    def test_gradient_flow(self) -> None:
        enc = NavEncoder1D(in_channels=2, nav_latent_dim=16, depth=2)
        x = torch.randn(2, 2, 32, requires_grad=True)
        z = enc(x)
        loss = z.pow(2).sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_wrong_channel_count_raises(self) -> None:
        enc = NavEncoder1D(in_channels=2, nav_latent_dim=8, depth=2)
        with pytest.raises(ValueError, match="channel dim"):
            enc(torch.randn(2, 4, 32))

    def test_parameter_count_nonzero(self) -> None:
        enc = NavEncoder1D(nav_latent_dim=8, depth=2)
        assert enc.get_parameter_count() > 0
