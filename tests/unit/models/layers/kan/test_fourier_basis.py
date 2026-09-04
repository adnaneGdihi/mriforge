"""Equivalence tests for the FourierBasis vectorization (perf, 2026-07-01).

The old ``FourierBasis.forward`` looped over ``num_bases`` in Python, launching
``2 * num_bases`` elementwise kernels plus a stack per forward. The fix
broadcasts over the basis dim in one shot, keeping the multiplication grouping
``(2π·f)·x`` of the old scalar path (bitwise-identical per element) and
reproducing the exact ``[sin_0, cos_0, sin_1, cos_1, ...]`` interleave so
``kan_weights`` layouts and checkpoints stay compatible.

The old loop math is kept here as the ground-truth reference.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.layers.kan.fourier_basis import FourierBasis


def _reference_forward(module: FourierBasis, x: torch.Tensor) -> torch.Tensor:
    """The pre-2026-07-01 per-basis loop, verbatim."""
    original_shape = x.shape
    x_flat = x.view(original_shape[0], -1) if len(original_shape) > 2 else x

    basis = []
    for i in range(module.num_bases):
        freq = module.frequencies[i]
        phase = module.phases[i]
        sin_component = torch.sin(2 * math.pi * freq * x_flat + phase)
        cos_component = torch.cos(2 * math.pi * freq * x_flat + phase)
        if module.use_gabor:
            mu = module.means[i]
            sigma = module.variances[i]
            envelope = torch.exp(-0.5 * ((x_flat - mu) / sigma) ** 2)
            sin_component = sin_component * envelope
            cos_component = cos_component * envelope
        basis.extend([sin_component, cos_component])

    basis_tensor = torch.stack(basis, dim=-1)
    if len(original_shape) > 2:
        basis_tensor = basis_tensor.view(
            original_shape[0],
            original_shape[1],
            original_shape[2],
            original_shape[3],
            -1,
        )
    return basis_tensor


@pytest.mark.unit
@pytest.mark.parametrize("use_gabor", [False, True])
def test_forward_2d_matches_old_loop(use_gabor: bool) -> None:
    torch.manual_seed(0)
    module = FourierBasis(num_bases=5, use_gabor=use_gabor)
    # Non-trivial gabor params so the envelope actually varies.
    if use_gabor:
        with torch.no_grad():
            module.means.uniform_(-0.5, 0.5)
            module.variances.uniform_(0.5, 1.5)
    x = torch.randn(3, 7)

    out = module(x)
    ref = _reference_forward(module, x)

    assert out.shape == ref.shape == (3, 7, 10)  # num_bases * 2
    assert torch.equal(out, ref), "broadcast forward diverged from the old loop"


@pytest.mark.unit
@pytest.mark.parametrize("use_gabor", [False, True])
def test_forward_4d_matches_old_loop(use_gabor: bool) -> None:
    """The 4D path (flatten -> basis -> reshape tail) must be preserved."""
    torch.manual_seed(1)
    module = FourierBasis(num_bases=4, use_gabor=use_gabor)
    x = torch.randn(2, 3, 5, 6)

    out = module(x)
    ref = _reference_forward(module, x)

    assert out.shape == ref.shape == (2, 3, 5, 6, 8)
    assert torch.equal(out, ref)


@pytest.mark.unit
def test_interleave_order_sin_then_cos() -> None:
    """Pin [sin_0, cos_0, sin_1, cos_1, ...] — a phase-zero input of zeros has
    sin(0)=0 in even slots and cos(0)=1 in odd slots."""
    module = FourierBasis(num_bases=3, trainable=False)
    x = torch.zeros(1, 4)
    out = module(x)
    assert torch.equal(out[..., 0::2], torch.zeros(1, 4, 3))
    assert torch.equal(out[..., 1::2], torch.ones(1, 4, 3))


@pytest.mark.unit
def test_gradients_match_old_loop() -> None:
    torch.manual_seed(2)
    module = FourierBasis(num_bases=4, trainable=True, use_gabor=True)
    with torch.no_grad():
        module.means.uniform_(-0.5, 0.5)
        module.variances.uniform_(0.5, 1.5)

    x_new = torch.randn(2, 6, requires_grad=True)
    x_ref = x_new.detach().clone().requires_grad_(True)

    module(x_new).sum().backward()
    grads_new = {
        name: p.grad.detach().clone()
        for name, p in module.named_parameters()
    }
    for p in module.parameters():
        p.grad = None

    _reference_forward(module, x_ref).sum().backward()
    grads_ref = {
        name: p.grad.detach().clone()
        for name, p in module.named_parameters()
    }
    for p in module.parameters():
        p.grad = None

    assert torch.allclose(x_new.grad, x_ref.grad, rtol=1e-6, atol=1e-6)
    for name in grads_ref:
        assert torch.allclose(
            grads_new[name], grads_ref[name], rtol=1e-6, atol=1e-6
        ), f"gradient mismatch for {name}"
