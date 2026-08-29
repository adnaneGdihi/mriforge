"""SwinMambaKAN's MambaLayer must use the official mamba_ssm kernel.

The block's novelty is Swin + KAN; its sequence mixer should be a real
selective-SSM (via :class:`MambaBlock`), not the slow pure-Python ``for t in
range(L)`` reimplementation it shipped with.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.blocks.mamba_block import MambaBlock  # noqa: E402
from mriforge.models.topological.swin_mamba_kan import MambaLayer  # noqa: E402
from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402


def test_mamba_layer_delegates_to_mamba_block(monkeypatch) -> None:
    """MambaLayer wraps the official MambaBlock (no hand-rolled SSM scan)."""
    monkeypatch.setenv("MRIFORGE_ALLOW_MAMBA_FALLBACK", "1")  # CPU/no-kernel CI
    layer = MambaLayer(d_model=16, d_state=8)
    assert isinstance(layer.mamba, MambaBlock)
    # The bespoke selective-scan params must be gone (delegated, not reimplemented).
    assert not hasattr(layer, "A_log")


@requires_cuda_for_mamba
def test_mamba_layer_preserves_sequence_shape(monkeypatch) -> None:
    monkeypatch.setenv("MRIFORGE_ALLOW_MAMBA_FALLBACK", "1")
    layer = MambaLayer(d_model=16, d_state=8)
    x = torch.randn(2, 32, 16)  # [B, L, D]
    out = layer(x)
    assert out.shape == x.shape
