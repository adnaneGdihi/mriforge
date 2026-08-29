"""Unit tests for the HDSF (Hierarchical Dilated Space-Filling) Mamba encoder.

Focus: the linearizer factory ``HDSFMambaEncoder._get_lin`` must route strict
Hilbert modes through the resolution-agnostic ``*_rect`` sibling so a non-cubic
dynamic sub-grid (the dilated global stream, or a padded non-cubic input) cannot
raise ``AssertionError`` mid-forward. Byte-identical on cubic power-of-2 blocks.

The 3D Mamba blocks need ``mamba_ssm``. Where it is ABSENT the opt-in
Gated-Conv+GRU fallback below keeps ``HDSFMambaEncoder`` constructible, so the
linearizer path — the actual subject of these tests — stays reachable.

That opt-in does NOT make a forward pass CPU-runnable where the kernel is
PRESENT: it is consulted only in ``MambaBlock``'s ``except ImportError`` branch,
and a present kernel never reaches it. The forward test therefore carries
``requires_cuda_for_mamba`` as well; see tests/utils/optional_backends.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_mamba_fallback(monkeypatch):
    """Kernel-absent boxes only; a no-op wherever ``mamba_ssm`` imports."""
    monkeypatch.setenv("MRIFORGE_ALLOW_MAMBA_FALLBACK", "1")


def _encoder():
    from mriforge.models.generators.hdsf_mamba_generators import HDSFMambaEncoder

    return HDSFMambaEncoder(
        d_model=8, num_layers=1, d_state=4, block_size=4, dilation=2
    )


def test_get_lin_remaps_strict_hilbert_to_rect() -> None:
    enc = _encoder()
    # A non-cubic shape would AssertionError under strict hilbert_3d; the remap
    # must build a hilbert_3d_rect linearizer instead.
    lin = enc._get_lin((6, 10, 12), "hilbert_3d", torch.device("cpu"))
    assert lin.mode == "hilbert_3d_rect"


@requires_cuda_for_mamba
def test_forward_non_cubic_non_pow2() -> None:
    """A non-cubic, non-power-of-2 volume must forward without crashing (the
    dilated global stream produces a non-cubic sub-grid)."""
    enc = _encoder()
    x = torch.randn(1, 8, 6, 10, 12)
    out = enc(x)
    assert out.shape == (1, 8, 6, 10, 12)
