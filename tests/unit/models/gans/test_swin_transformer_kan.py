"""Regression tests for ``spectramr.models.gans.swin_transformer_kan``.

The registered ``swin_kan_transformer`` headline mechanisms are the KAN
feed-forward and the windowed attention. Both constructors used to wrap
the real component in ``try/except Exception`` that silently degraded to
a plain MLP (KAN FFN) or a single ``nn.Linear`` (attention, with no log
at all) — pitfall #9/#16. These tests are the mechanism-fires probe:
construction must succeed AND the real components must be present in the
module tree; any construction error must propagate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.gans.swin_transformer_kan import (  # noqa: E402
    KANFeedForward,
    KanSwinBlock,
    SimpleWindowAttention,
    SwinKANGAN,
)
from spectramr.models.layers.kan.kan_convs.kans.kan import KAN  # noqa: E402


class TestKANFeedForwardMechanismFires:
    def test_ffn_head_is_real_kan(self) -> None:
        """First stage of the FFN is the KAN, not a Linear/GELU MLP."""
        ffn = KANFeedForward(dim=16, hidden_dim=32)
        assert isinstance(ffn.net[0], KAN)
        assert ffn.use_kan is True

    def test_ffn_without_layer_norm_is_real_kan(self) -> None:
        ffn = KANFeedForward(dim=16, hidden_dim=32, use_layer_norm=False)
        assert isinstance(ffn.net[0], KAN)

    def test_forward_finite(self) -> None:
        torch.manual_seed(0)
        ffn = KANFeedForward(dim=16, hidden_dim=32)
        out = ffn(torch.randn(2, 8, 16))
        assert out.shape == (2, 8, 16)
        assert torch.isfinite(out).all()

    def test_construction_errors_propagate(self, monkeypatch) -> None:
        """A KAN construction failure must raise, not degrade (pitfall #9)."""

        def _boom(*args, **kwargs):
            raise RuntimeError("KAN construction failed")

        monkeypatch.setattr(
            "spectramr.models.gans.swin_transformer_kan.KAN",
            _boom,
        )
        with pytest.raises(RuntimeError, match="KAN construction failed"):
            KANFeedForward(dim=16, hidden_dim=32)


class TestKanSwinBlockAttentionMechanismFires:
    def test_attention_is_windowed_not_linear(self) -> None:
        """The block's attention is SimpleWindowAttention, never nn.Linear."""
        block = KanSwinBlock(
            dim=16, heads=2, head_dim=8, mlp_dim=32, shifted=False, window_size=4
        )
        assert isinstance(block.attention, SimpleWindowAttention)
        assert not isinstance(block.attention, torch.nn.Linear)
        assert block.use_windowed_attention is True
        assert isinstance(block.ffn.net[0], KAN)

    def test_forward_finite(self) -> None:
        torch.manual_seed(0)
        block = KanSwinBlock(
            dim=16, heads=2, head_dim=8, mlp_dim=32, shifted=False, window_size=4
        )
        out = block(torch.randn(1, 16, 16))
        assert out.shape == (1, 16, 16)
        assert torch.isfinite(out).all()


class TestSwinKANGANMechanismFires:
    def test_generator_tree_contains_real_components(self) -> None:
        model = SwinKANGAN(
            z_dim=8,
            dim=8,
            depths=[1],
            heads=[2],
            window_size=2,
            mlp_dim=16,
            final_res=8,
            patch_size=2,
            channels=1,
        )
        attn = [m for m in model.modules() if isinstance(m, SimpleWindowAttention)]
        kans = [m for m in model.modules() if isinstance(m, KAN)]
        assert attn, "no SimpleWindowAttention in the module tree"
        assert kans, "no KAN in the module tree"

    def test_forward_finite(self) -> None:
        torch.manual_seed(0)
        model = SwinKANGAN(
            z_dim=8,
            dim=8,
            depths=[1],
            heads=[2],
            window_size=2,
            mlp_dim=16,
            final_res=8,
            patch_size=2,
            channels=1,
        )
        out = model(torch.randn(1, 8))
        assert out.shape == (1, 1, 8, 8)
        assert torch.isfinite(out).all()
