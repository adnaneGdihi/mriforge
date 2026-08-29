"""Regression tests for ``mriforge.models.gans.stylegan_kan``.

The registered ``style_kan_gan`` headline mechanism is the KAN mapping
network and the KAN AdaIN style transforms. Both used to be wrapped in
``try/except Exception`` blocks that silently substituted a plain
MLP/``nn.Linear`` (DEBUG log only), collapsing the model to a vanilla
StyleGAN — pitfall #9/#16. These tests are the mechanism-fires probe:
construction must succeed AND the real KAN components must be present in
the module tree; any construction error must propagate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.gans.stylegan_kan import (  # noqa: E402
    AdaIN,
    MappingNetwork,
    StyleKANGAN,
)
from mriforge.models.layers.kan.kan_convs.kans.kan import KAN  # noqa: E402


class TestMappingNetworkMechanismFires:
    def test_mapping_is_real_kan(self) -> None:
        """The mapping net is the KAN, not a silently substituted MLP."""
        net = MappingNetwork(z_dim=16, w_dim=16)
        assert isinstance(net.mapping, KAN)
        assert not isinstance(net.mapping, torch.nn.Sequential)

    def test_forward_finite(self) -> None:
        torch.manual_seed(0)
        net = MappingNetwork(z_dim=16, w_dim=16)
        out = net(torch.randn(2, 16))
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()


class TestAdaINMechanismFires:
    def test_style_transforms_are_real_kan(self) -> None:
        """Both style transforms are KANs, never nn.Linear fallbacks."""
        adain = AdaIN(channels=8, w_dim=16)
        assert isinstance(adain.style_scale_transform, KAN)
        assert isinstance(adain.style_shift_transform, KAN)
        assert not isinstance(adain.style_scale_transform, torch.nn.Linear)
        assert not isinstance(adain.style_shift_transform, torch.nn.Linear)

    def test_forward_finite(self) -> None:
        torch.manual_seed(0)
        adain = AdaIN(channels=8, w_dim=16)
        out = adain(torch.randn(2, 8, 4, 4), torch.randn(2, 16))
        assert out.shape == (2, 8, 4, 4)
        assert torch.isfinite(out).all()


class TestStyleKANGANMechanismFires:
    def test_full_model_tree_contains_kans(self) -> None:
        """The assembled generator carries KANs in mapping AND every AdaIN."""
        model = StyleKANGAN(out_channels=1, z_dim=32, w_dim=32, img_size=8)
        assert isinstance(model.mapping_network.mapping, KAN)
        adains = [m for m in model.modules() if isinstance(m, AdaIN)]
        assert adains, "no AdaIN blocks in the synthesis tree"
        for adain in adains:
            assert isinstance(adain.style_scale_transform, KAN)
            assert isinstance(adain.style_shift_transform, KAN)

    def test_forward_finite(self) -> None:
        torch.manual_seed(0)
        model = StyleKANGAN(out_channels=1, z_dim=32, w_dim=32, img_size=8)
        out = model(torch.randn(1, 32))
        assert out.shape == (1, 1, 8, 8)
        assert torch.isfinite(out).all()

    def test_construction_errors_propagate(self, monkeypatch) -> None:
        """A KAN construction failure must raise, not degrade (pitfall #9)."""

        def _boom(*args, **kwargs):
            raise RuntimeError("KAN construction failed")

        monkeypatch.setattr(
            "mriforge.models.gans.stylegan_kan.KAN",
            _boom,
        )
        with pytest.raises(RuntimeError, match="KAN construction failed"):
            MappingNetwork(z_dim=16, w_dim=16)
        with pytest.raises(RuntimeError, match="KAN construction failed"):
            AdaIN(channels=8, w_dim=16)
