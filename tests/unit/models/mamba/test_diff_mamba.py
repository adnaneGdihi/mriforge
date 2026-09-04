"""Equivalence tests for the DiffSSM integration-loop rework (perf, 2026-07-01).

The old ``DiffSSM.forward`` built a fresh ``torch.tensor(current_t)`` per ODE
substep and ``DiffSSMDeriv`` called ``t.item()`` on it — a host→device alloc
plus a device→host sync for every substep (``L·steps_per_token`` pipeline
stalls per layer per forward). It also ran the B-projection matmul per substep
and the C/D readout linears per token.

The rework keeps the (inherently sequential — nonlinear tanh clamp) Euler loop
but: passes plain float times, projects the input sequence through B once and
interpolates the projections (B is linear, so ``B(interp(x,t)) ==
interp(B(x),t)`` in exact arithmetic), and batches the C/D readout after the
loop. Times reproduce the old ``current_t += dt`` accumulation bitwise; the
matmul reassociation makes outputs ``allclose`` rather than bitwise.

The old loop math is kept here verbatim as the ground-truth reference.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.mamba.diff_mamba import DiffSSM, DiffSSMDeriv


def _reference_forward(module: DiffSSM, x: torch.Tensor) -> torch.Tensor:
    """The pre-2026-07-01 loop, verbatim (incl. the tensor-t round-trip)."""
    B, L, _D = x.shape
    h = torch.zeros(B, module.d_state, device=x.device)
    deriv_fn = DiffSSMDeriv(module.A, module.B.weight, x)

    outputs = []
    current_t = 0.0
    steps_per_token = int(1.0 / module.dt_step)

    for i in range(L):
        y_curr = module.C(h) + module.D(x[:, i, :])
        outputs.append(y_curr)
        for _ in range(steps_per_token):
            t_tensor = torch.tensor(current_t, device=x.device)
            dh = deriv_fn(t_tensor, h)
            dh = torch.tanh(dh)
            h = h + module.dt_step * dh
            current_t += module.dt_step

    return torch.stack(outputs, dim=1)


@pytest.mark.unit
@pytest.mark.parametrize("dt_step", [1.0, 0.5, 0.25])
def test_forward_matches_old_loop(dt_step: float) -> None:
    torch.manual_seed(0)
    module = DiffSSM(d_model=6, d_state=4, dt_step=dt_step)
    x = torch.randn(2, 7, 6)

    out = module(x)
    ref = _reference_forward(module, x)

    assert out.shape == ref.shape == (2, 7, 6)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6), (
        f"max abs diff {(out - ref).abs().max().item():.3e}"
    )


@pytest.mark.unit
def test_gradients_match_old_loop() -> None:
    torch.manual_seed(1)
    module = DiffSSM(d_model=5, d_state=3, dt_step=0.5)

    x_new = torch.randn(1, 6, 5, requires_grad=True)
    x_ref = x_new.detach().clone().requires_grad_(True)

    # NOTE: ``in_proj`` is a pre-existing dead layer (defined, never used in
    # forward — old or new), so its grad is None in BOTH graphs; compare the
    # None-sets symmetrically instead of assuming every param gets a grad.
    module(x_new).sum().backward()
    grads_new = {n: p.grad for n, p in module.named_parameters()}
    grads_new = {
        n: (g.detach().clone() if g is not None else None)
        for n, g in grads_new.items()
    }
    for p in module.parameters():
        p.grad = None

    _reference_forward(module, x_ref).sum().backward()
    grads_ref = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in module.named_parameters()
    }
    for p in module.parameters():
        p.grad = None

    assert torch.allclose(x_new.grad, x_ref.grad, rtol=1e-4, atol=1e-6)
    for name in grads_ref:
        if grads_ref[name] is None:
            assert grads_new[name] is None, f"{name}: grad appeared only in new"
            continue
        assert grads_new[name] is not None, f"{name}: grad vanished in new"
        assert torch.allclose(
            grads_new[name], grads_ref[name], rtol=1e-4, atol=1e-6
        ), f"gradient mismatch for {name}"


@pytest.mark.unit
def test_deriv_accepts_float_and_tensor_t() -> None:
    """Back-compat: torchdiffeq-style callers pass scalar tensors; the hot
    path passes plain floats. Both must agree exactly."""
    torch.manual_seed(2)
    A = torch.randn(4) - 0.5
    B_matrix = torch.randn(4, 6)
    x_seq = torch.randn(2, 5, 6)
    h = torch.randn(2, 4)
    deriv = DiffSSMDeriv(A, B_matrix, x_seq)

    out_float = deriv(1.5, h)
    out_tensor = deriv(torch.tensor(1.5), h)
    assert torch.equal(out_float, out_tensor)


@pytest.mark.unit
def test_forward_contains_no_tensor_t_roundtrip() -> None:
    """Pin the perf contract: the integration loop must not rebuild scalar
    time tensors (and thus not sync via ``t.item()``) per substep."""
    import inspect

    src = inspect.getsource(DiffSSM.forward)
    assert "torch.tensor(current_t" not in src
    assert ".item()" not in src
