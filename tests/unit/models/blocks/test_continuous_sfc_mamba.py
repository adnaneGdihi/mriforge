"""ContinuousSFCMamba kernel-backend wiring (incl. the opt-in mamba_ssm path)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import mriforge.models.blocks.triton_scan as _ts  # noqa: E402
from mriforge.models.blocks.continuous_sfc_mamba import ContinuousSFCMamba  # noqa: E402


def test_accepts_mamba_ssm_backend() -> None:
    m = ContinuousSFCMamba(dim=8, d_state=16, kernel_backend="mamba_ssm")
    assert m.kernel_backend == "mamba_ssm"


def test_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="kernel_backend"):
        ContinuousSFCMamba(dim=8, kernel_backend="cuda_sketch")


def test_mamba_ssm_forward_raises_without_kernel(monkeypatch) -> None:
    """No silent fallback: forward fails loud when the kernel is unavailable."""
    monkeypatch.setattr(_ts, "mamba_ssm_available", lambda: False)
    m = ContinuousSFCMamba(dim=8, d_state=16, use_physical_arc=False,
                           kernel_backend="mamba_ssm")
    with pytest.raises(RuntimeError, match="mamba_ssm"):
        m(torch.randn(2, 16, 8))


def test_python_backend_still_works() -> None:
    m = ContinuousSFCMamba(dim=8, d_state=16, use_physical_arc=False,
                           kernel_backend="python")
    out = m(torch.randn(2, 16, 8))
    assert out.shape == (2, 16, 8)
    assert torch.isfinite(out).all()
