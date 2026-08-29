"""Tests for the multi-backend selective scan.

Three backends are exercised:

* ``python``  — always.
* ``compile`` — always (``torch.compile`` works on CPU).
* ``triton``  — gated on ``CUDA + triton``; tests are silently
                skipped without a GPU.

The contract this file pins:

1. All available backends produce **numerically equivalent** outputs
   on the same inputs (max abs diff ≤ 1e-5 in fp32).
2. ``backend='triton'`` raises a clean ``RuntimeError`` when CUDA is
   unavailable — no silent fallback.
3. ``backend='auto'`` picks the fastest available backend.
4. Unknown backends raise ``ValueError`` at the dispatcher.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

import mriforge.models.blocks.triton_scan as _ts  # noqa: E402
from mriforge.models.blocks.triton_scan import (  # noqa: E402
    mamba_ssm_available,
    selective_scan,
    triton_available,
)


def _make_inputs(
    B: int = 2, T: int = 32, C: int = 8, S: int = 16, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    A_diag = -torch.exp(torch.randn(C, S, device=device))
    B_seq = torch.randn(B, T, S, device=device)
    delta = torch.rand(T, device=device) * 0.1
    return A_diag, B_seq, delta


def test_python_backend_baseline_runs() -> None:
    A, B, dt = _make_inputs()
    out = selective_scan(A, B, dt, backend="python")
    assert out.shape == B.shape
    assert torch.isfinite(out).all()


def test_compile_matches_python() -> None:
    A, B, dt = _make_inputs()
    y_py = selective_scan(A, B, dt, backend="python")
    y_co = selective_scan(A, B, dt, backend="compile")
    assert torch.allclose(y_py, y_co, atol=1e-5)


def test_auto_picks_python_on_cpu() -> None:
    A, B, dt = _make_inputs()
    y_auto = selective_scan(A, B, dt, backend="auto")
    y_py = selective_scan(A, B, dt, backend="python")
    assert torch.allclose(y_auto, y_py, atol=1e-6)


def test_triton_on_cpu_raises_no_silent_fallback() -> None:
    A, B, dt = _make_inputs()
    with pytest.raises(RuntimeError, match="(cuda|CUDA|Triton)"):
        selective_scan(A, B, dt, backend="triton")


def test_unknown_backend_raises() -> None:
    A, B, dt = _make_inputs()
    with pytest.raises(ValueError, match="backend"):
        selective_scan(A, B, dt, backend="cuda_sketch")


# ── mamba_ssm backend ────────────────────────────────────────────────


def test_mamba_ssm_backend_raises_when_unavailable(monkeypatch) -> None:
    """No silent fallback: requesting the kernel without the package raises."""
    monkeypatch.setattr(_ts, "mamba_ssm_available", lambda: False)
    A, B, dt = _make_inputs()
    with pytest.raises(RuntimeError, match="mamba_ssm"):
        selective_scan(A, B, dt, backend="mamba_ssm")


def test_mamba_ssm_backend_requires_cuda(monkeypatch) -> None:
    """The kernel is CUDA-only — CPU tensors must raise, not fall back."""
    monkeypatch.setattr(_ts, "mamba_ssm_available", lambda: True)
    A, B, dt = _make_inputs(device="cpu")
    with pytest.raises(RuntimeError, match="(cuda|CUDA)"):
        selective_scan(A, B, dt, backend="mamba_ssm")


def test_auto_never_selects_mamba_ssm(monkeypatch) -> None:
    """``auto`` must stay reproducible — never silently switch to the
    (numerically-divergent) mamba_ssm backend even when it is installed."""
    monkeypatch.setattr(_ts, "mamba_ssm_available", lambda: True)
    A, B, dt = _make_inputs(device="cpu")
    # If auto picked mamba_ssm it would demand CUDA and raise; instead it
    # falls to the python backend on CPU and matches it.
    y_auto = selective_scan(A, B, dt, backend="auto")
    y_py = selective_scan(A, B, dt, backend="python")
    assert torch.allclose(y_auto, y_py, atol=1e-6)


def _ref_mean_before_exp(A_diag, B_seq, delta):
    """Ground truth for the mamba_ssm backend: mean-BEFORE-exp diagonal scan."""
    A_bar = A_diag.mean(dim=0)  # (S,)
    Bn, T, S = B_seq.shape
    h = B_seq.new_zeros(Bn, S)
    out = B_seq.new_empty(Bn, T, S)
    for t in range(T):
        h = h * torch.exp(delta[t] * A_bar) + delta[t] * B_seq[:, t]
        out[:, t] = h
    return out


def _install_fake_mamba_ssm(monkeypatch) -> None:
    """Inject a pure-PyTorch ``selective_scan_fn`` faithful to mamba_ssm's
    ``selective_scan_ref`` (variable 3-D B & C), so the axis-remapping in
    :func:`_mamba_ssm_scan` is validatable on CPU without the CUDA kernel.
    """
    import sys
    import types

    def fake_selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        # u, delta: (B, D, L); A: (D, N); B, C: (B, N, L)
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        x = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
        ys = []
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
        y = torch.stack(ys, dim=2)  # (B, D, L)
        if D is not None:
            y = y + u * D.view(1, -1, 1)
        return y

    top = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = fake_selective_scan_fn
    monkeypatch.setitem(sys.modules, "mamba_ssm", top)
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", ops)
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.selective_scan_interface", iface)


def test_mamba_ssm_mapping_matches_reference_cpu(monkeypatch) -> None:
    """Validate the selective_scan_fn axis-remap on CPU via a faithful fake
    kernel — covers the layout logic the CUDA-gated test cannot run here."""
    _install_fake_mamba_ssm(monkeypatch)
    A, B, dt = _make_inputs(device="cpu")
    y = _ts._mamba_ssm_scan(A, B, dt)
    assert y.shape == B.shape
    assert torch.allclose(y, _ref_mean_before_exp(A, B, dt), atol=1e-5)


@pytest.mark.skipif(
    not (mamba_ssm_available() and torch.cuda.is_available()),
    reason="mamba_ssm CUDA kernel not available",
)
def test_mamba_ssm_matches_mean_before_exp_reference() -> None:
    """The kernel path reproduces the mean-BEFORE-exp diagonal recurrence.

    (It cannot reproduce the python/triton mean-AFTER-exp decay — that is the
    documented, deliberate difference — so it is validated against its own
    reference, not against ``backend='python'``.)
    """
    A, B, dt = _make_inputs(device="cuda")
    A_bar = A.mean(dim=0)  # (S,)
    Bn, T, S = B.shape
    h = B.new_zeros(Bn, S)
    ref = B.new_empty(Bn, T, S)
    for t in range(T):
        h = h * torch.exp(dt[t] * A_bar) + dt[t] * B[:, t]
        ref[:, t] = h
    y = selective_scan(A, B, dt, backend="mamba_ssm")
    assert torch.allclose(y, ref, atol=1e-3)


# ── CUDA-gated equivalence tests ────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton equivalence requires CUDA"
)
def test_triton_matches_python_on_cuda() -> None:
    A, B, dt = _make_inputs(device="cuda")
    y_py = selective_scan(A, B, dt, backend="python")
    y_tt = selective_scan(A, B, dt, backend="triton")
    # Allow a slightly larger tolerance because reduction order differs.
    assert torch.allclose(y_py, y_tt, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="auto=triton check requires CUDA"
)
def test_auto_picks_triton_on_cuda() -> None:
    A, B, dt = _make_inputs(device="cuda")
    if not triton_available():
        pytest.skip("triton not installed")
    # When CUDA + triton are both available, auto must dispatch to triton —
    # not silently fall back to compile/python.
    import mriforge.models.blocks.triton_scan as ts

    calls: list[str] = []
    real_triton = ts._triton_scan
    real_python = ts._python_scan
    real_compile = ts._compile_scan

    def spy_triton(*args, **kw):
        calls.append("triton")
        return real_triton(*args, **kw)

    def spy_python(*args, **kw):
        calls.append("python")
        return real_python(*args, **kw)

    def spy_compile(*args, **kw):
        calls.append("compile")
        return real_compile(*args, **kw)

    ts._triton_scan = spy_triton
    ts._python_scan = spy_python
    ts._compile_scan = spy_compile
    try:
        selective_scan(A, B, dt, backend="auto")
    finally:
        ts._triton_scan = real_triton
        ts._python_scan = real_python
        ts._compile_scan = real_compile

    assert calls == ["triton"], f"auto did not pick triton on CUDA: {calls!r}"
