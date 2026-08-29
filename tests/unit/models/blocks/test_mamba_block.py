"""Unit tests for MambaBlock — kernel requirement + (opt-in) GRU fallback.

The GRU fallback is **NOT equivalent to the SSM** and is disabled by default:
a missing/broken ``mamba_ssm`` must fail loud (pitfall #9) so an experiment
never silently trains a GRU under the "Mamba" label. The fallback survives only
as an explicit ``MRIFORGE_ALLOW_MAMBA_FALLBACK`` opt-in for CPU/CI wiring tests.
"""

import importlib.util

import pytest
import torch

from mriforge.models.blocks.mamba_block import MambaBlock

_MAMBA_SSM_AVAILABLE = importlib.util.find_spec("mamba_ssm") is not None


def test_mamba_ssm_importable_reflects_environment() -> None:
    """The audit helper must report the true install state of mamba_ssm."""
    from mriforge.models.blocks.mamba_block import _mamba_ssm_importable

    assert _mamba_ssm_importable() is _MAMBA_SSM_AVAILABLE


@pytest.mark.skipif(
    _MAMBA_SSM_AVAILABLE,
    reason="exercises the kernel-absent path; mamba_ssm IS installed here",
)
class TestMambaBlockRequiresKernel:
    """Default behaviour: no kernel → raise, do not silently degrade to GRU."""

    def test_raises_when_kernel_absent_and_fallback_not_allowed(self, monkeypatch):
        monkeypatch.delenv("MRIFORGE_ALLOW_MAMBA_FALLBACK", raising=False)
        with pytest.raises((ImportError, RuntimeError), match="mamba"):
            MambaBlock(d_model=16, d_state=8)

    def test_fallback_flag_allows_gru_with_warning(self, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("MRIFORGE_ALLOW_MAMBA_FALLBACK", "1")
        with caplog.at_level(logging.WARNING):
            blk = MambaBlock(d_model=16, d_state=8)
        assert blk.use_official is False
        assert any(
            "fallback" in r.message.lower() or "gru" in r.message.lower()
            for r in caplog.records
        ), "opt-in fallback must emit a loud non-equivalence warning"


@pytest.mark.skipif(
    _MAMBA_SSM_AVAILABLE,
    reason="fallback path only exists when the mamba_ssm kernel is absent",
)
class TestMambaBlockFallback:
    """Tests for the opt-in MambaBlock fallback (Gated Conv + GRU)."""

    @pytest.fixture
    def block(self, monkeypatch) -> MambaBlock:
        """Create a MambaBlock via the explicit opt-in GRU fallback."""
        monkeypatch.setenv("MRIFORGE_ALLOW_MAMBA_FALLBACK", "1")
        blk = MambaBlock(d_model=32, d_state=16, d_conv=4, expand=2)
        assert blk.use_official is False  # opt-in fallback engaged (no kernel)
        return blk

    def test_fallback_forward_cpu(self, block: MambaBlock) -> None:
        """Fallback MambaBlock produces correct shape on CPU."""
        x = torch.randn(2, 64, 32)  # [B, L, D]
        out = block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_fallback_forward_contiguity(self, block: MambaBlock) -> None:
        """Output tensor is contiguous after fallback forward pass."""
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.is_contiguous()

    def test_fallback_forward_different_seq_lengths(self, block: MambaBlock) -> None:
        """Fallback handles various sequence lengths (power-of-2 and non)."""
        for L in [16, 64, 100, 256]:
            x = torch.randn(1, L, 32)
            out = block(x)
            assert out.shape == (1, L, 32)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fallback_forward_cuda(self, block: MambaBlock) -> None:
        """Fallback MambaBlock runs on CUDA without cuDNN errors."""
        device = torch.device("cuda")
        block = block.to(device)
        x = torch.randn(2, 64, 32, device=device)
        out = block(x)
        assert out.shape == x.shape
        assert out.device == x.device

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fallback_forward_cuda_with_amp(self, block: MambaBlock) -> None:
        """Fallback MambaBlock runs correctly under AMP autocast on CUDA.

        This tests the exact scenario that caused CUDNN_STATUS_NOT_SUPPORTED.
        """
        device = torch.device("cuda")
        block = block.to(device)
        x = torch.randn(2, 64, 32, device=device)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            out = block(x)

        assert out.shape == x.shape
        assert out.device == x.device

    def test_fallback_gradient_flow(self, block: MambaBlock) -> None:
        """Gradients flow through the fallback MambaBlock."""
        x = torch.randn(2, 64, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
