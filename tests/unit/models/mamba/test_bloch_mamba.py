"""Equivalence tests for the BlochMambaBlock recurrence hoist (perf, 2026-07-01).

The old ``run_recurrence`` recomputed per-forward constants EVERY token: the
Bloch relaxation factors (``exp(log_T1)`` → ``exp(-dt/T1)`` …, via
``BlochTransition.forward``) and ``A_resid.mean(dim=0)``, and launched the
``B_phys``/``B_resid`` input projections and the ``C``/``D`` readout per
token. The rework hoists the constants, batches the projections before the
loop and the readout after it; the loop keeps only the two fused
multiply-add state updates (affine Bloch dynamics — deliberately NOT routed
onto the selective-scan kernel, whose recurrence this is not).

The old loop math is kept here verbatim as the ground-truth reference. The
dead ``seq_t1`` per-step map plumbing (documented dimension mismatch; the old
loop always passed ``None`` to the transition) is preserved as dead.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.mamba.bloch_mamba import BlochMambaBlock, BlochTransition


def _reference_block_forward(
    block: BlochMambaBlock,
    x: torch.Tensor,
    T1_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """The pre-2026-07-01 forward, verbatim (per-token factor recompute)."""
    B, L, _D = x.shape
    device = x.device
    x = block.norm(x)

    def run_recurrence(seq_x, seq_t1):
        h_phys = torch.zeros(B, block.d_phys, device=device)
        h_resid = torch.zeros(B, block.d_resid, device=device)
        A_resid = torch.exp(-torch.exp(block.A_resid_log))

        outputs_scan = []
        for t in range(L):
            xt = seq_x[:, t]
            b_phys = block.B_phys(xt)
            b_resid = block.B_resid(xt)
            # Old loop always passed None maps (dead plumbing).
            h_phys_new = block.bloch(h_phys, None, None) + b_phys
            h_resid_new = (A_resid.mean(dim=0) * h_resid) + b_resid
            h_phys, h_resid = h_phys_new, h_resid_new
            h = torch.cat([h_phys, h_resid], dim=-1)
            outputs_scan.append(block.C(h) + block.D * xt)
        return torch.stack(outputs_scan, dim=1)

    y_fwd = run_recurrence(x, T1_map)
    x_bwd = torch.flip(x, dims=[1])
    y_bwd = run_recurrence(x_bwd, T1_map)
    return y_fwd + torch.flip(y_bwd, dims=[1])


@pytest.mark.unit
def test_forward_matches_old_loop() -> None:
    torch.manual_seed(0)
    block = BlochMambaBlock(d_model=6, d_state=8, physics_ratio=0.5)
    block.eval()
    x = torch.randn(2, 7, 6)

    with torch.no_grad():
        out = block(x)
        ref = _reference_block_forward(block, x)

    assert out.shape == ref.shape == (2, 7, 6)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6), (
        f"max abs diff {(out - ref).abs().max().item():.3e}"
    )


@pytest.mark.unit
def test_forward_with_maps_arg_matches_old_loop() -> None:
    """T1/T2 maps were dead plumbing in the old loop (always None reached the
    transition); passing them must still not change the output."""
    torch.manual_seed(1)
    block = BlochMambaBlock(d_model=4, d_state=8)
    block.eval()
    x = torch.randn(1, 5, 4)
    t1 = torch.rand(1, 4) * 1000

    with torch.no_grad():
        out = block(x, T1_map=t1, T2_map=t1)
        out_no_maps = block(x)
        ref = _reference_block_forward(block, x, T1_map=t1)

    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)
    assert torch.equal(out, out_no_maps)


@pytest.mark.unit
def test_gradients_match_old_loop() -> None:
    torch.manual_seed(2)
    block = BlochMambaBlock(d_model=5, d_state=6, physics_ratio=0.5)

    x_new = torch.randn(1, 6, 5, requires_grad=True)
    x_ref = x_new.detach().clone().requires_grad_(True)

    block(x_new).sum().backward()
    grads_new = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in block.named_parameters()
    }
    for p in block.parameters():
        p.grad = None

    _reference_block_forward(block, x_ref).sum().backward()
    grads_ref = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in block.named_parameters()
    }
    for p in block.parameters():
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
def test_bloch_transition_factors_match_forward_path() -> None:
    """``factors`` + ``apply_factors`` must equal the single-step ``forward``
    (the hoisted scan path and the reference path share the same math)."""
    torch.manual_seed(3)
    bloch = BlochTransition(d_model=8, dt=1.0)
    h = torch.randn(3, 8)

    E1, E2, M0 = bloch.factors(4)
    assert torch.equal(bloch.apply_factors(h, E1, E2, M0), bloch(h))
