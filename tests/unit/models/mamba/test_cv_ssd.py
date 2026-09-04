"""Equivalence tests for the CV-SSD readout hoist (perf, 2026-07-01).

The old scan loop ran the ``C``-contraction einsum and the ``D``-skip per
token (2L kernel launches per forward); the rework keeps the state update
loop verbatim (complex dtype, per-channel state, time-invariant decay — not
the selective-scan recurrence; the cumprod closed form is unstable) and
batches the readout as one einsum after the loop.

The old loop math is kept here verbatim as the ground-truth reference.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.mamba.cv_ssd import ComplexSSMLayer


def _reference_forward(module: ComplexSSMLayer, x: torch.Tensor) -> torch.Tensor:
    """The pre-2026-07-01 loop, verbatim."""
    if not torch.is_complex(x):
        D = x.shape[-1] // 2
        x = torch.complex(x[..., :D], x[..., D:])

    B, L, D = x.shape

    A = module._get_A_discrete()
    B_c = torch.complex(module.B_real, module.B_imag)
    C_c = torch.complex(module.C_real, module.C_imag)
    D_c = torch.complex(module.D_real, module.D_imag)

    x_B = torch.einsum("bld,dn->bln", x, B_c)

    h = torch.zeros(B, D, module.d_state, dtype=torch.cfloat, device=x.device)
    outputs = []
    for t in range(L):
        h = A.unsqueeze(0) * h + x_B[:, t : t + 1, :].transpose(1, 2)
        y = torch.einsum("bdn,dn->bd", h, C_c) + D_c * x[:, t]
        outputs.append(y)

    return torch.stack(outputs, dim=1)


@pytest.mark.unit
def test_forward_matches_old_loop_complex_input() -> None:
    torch.manual_seed(0)
    # NOTE: the state-update broadcast requires d_state == d_model (a
    # pre-existing constraint of this layer's transpose quirk).
    module = ComplexSSMLayer(d_model=6, d_state=6)
    x = torch.randn(2, 5, 6, dtype=torch.complex64)

    out = module(x)
    ref = _reference_forward(module, x)

    assert out.shape == ref.shape == (2, 5, 6)
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-7), (
        f"max abs diff {(out - ref).abs().max().item():.3e}"
    )


@pytest.mark.unit
def test_forward_matches_old_loop_real_stacked_input() -> None:
    """The real (B, L, 2D) input path must be preserved."""
    torch.manual_seed(1)
    module = ComplexSSMLayer(d_model=4, d_state=4)
    x = torch.randn(1, 6, 8)

    out = module(x)
    ref = _reference_forward(module, x)

    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-7)


@pytest.mark.unit
def test_gradients_match_old_loop() -> None:
    torch.manual_seed(2)
    module = ComplexSSMLayer(d_model=4, d_state=4)

    x_new = torch.randn(1, 5, 4, dtype=torch.complex64, requires_grad=True)
    x_ref = x_new.detach().clone().requires_grad_(True)

    module(x_new).abs().sum().backward()
    grads_new = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in module.named_parameters()
    }
    for p in module.parameters():
        p.grad = None

    _reference_forward(module, x_ref).abs().sum().backward()
    grads_ref = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in module.named_parameters()
    }
    for p in module.parameters():
        p.grad = None

    assert torch.allclose(x_new.grad, x_ref.grad, rtol=1e-5, atol=1e-7)
    for name in grads_ref:
        if grads_ref[name] is None:
            assert grads_new[name] is None, f"{name}: grad appeared only in new"
            continue
        assert grads_new[name] is not None, f"{name}: grad vanished in new"
        assert torch.allclose(
            grads_new[name], grads_ref[name], rtol=1e-5, atol=1e-7
        ), f"gradient mismatch for {name}"
