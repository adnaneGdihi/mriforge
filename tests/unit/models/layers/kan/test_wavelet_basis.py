"""Equivalence tests for the WaveletBasis vectorization (perf, 2026-07-01).

The old ``WaveletBasis.forward`` looped ``for _i in range(num_bases)`` with an
UNUSED loop variable: the two depthwise convolutions inside had loop-invariant
inputs and weights, so the loop computed the same two tensors ``num_bases``
times (16 convs where 2 suffice at the default ``num_bases=8``). The fix
computes each conv once and tiles along the basis dim with ``repeat``, which
preserves the exact ``[scaling, wavelet] * num_bases`` interleave.

The old loop math is kept here as the ground-truth reference — forward must be
bitwise identical and grads must match (backward over the tiled tensor sums
per-copy grads exactly like the old duplicate graph nodes).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mriforge.models.layers.kan.wavelet_basis import WaveletBasis


def _reference_forward(module: WaveletBasis, x: torch.Tensor) -> torch.Tensor:
    """The pre-2026-07-01 per-basis loop, verbatim."""
    _batch, channels, _h, _w = x.shape

    scaling_expanded = module.coeffs[0:1].repeat(channels, 1, 1, 1)
    wavelet_expanded = module.coeffs[1:2].repeat(channels, 1, 1, 1)

    basis_functions = []
    for _i in range(module.num_bases):
        scaling_basis = F.conv2d(x, scaling_expanded, padding=1, groups=channels)
        wavelet_basis = F.conv2d(x, wavelet_expanded, padding=1, groups=channels)
        if scaling_basis.shape[-1] != x.shape[-1]:
            scaling_basis = scaling_basis[:, :, : x.shape[-2], : x.shape[-1]]
            wavelet_basis = wavelet_basis[:, :, : x.shape[-2], : x.shape[-1]]
        basis_functions.extend([scaling_basis, wavelet_basis])

    return torch.stack(basis_functions, dim=-1)


@pytest.mark.unit
@pytest.mark.parametrize("family", ["haar", "db2", "mexican_hat", "morlet"])
def test_forward_bitwise_matches_old_loop(family: str) -> None:
    torch.manual_seed(0)
    module = WaveletBasis(num_bases=4, wavelet_family=family)
    x = torch.randn(2, 3, 8, 8)

    out = module(x)
    ref = _reference_forward(module, x)

    # NOTE: families with >2-tap filters shrink W under padding=1 and the
    # layer's crop only fires on a W mismatch — a pre-existing quirk. The
    # equivalence contract is out == ref, whatever that shape is.
    assert out.shape == ref.shape
    assert out.shape[-1] == 2 * module.num_bases
    assert torch.equal(out, ref), "vectorized forward diverged from the old loop"


@pytest.mark.unit
def test_interleave_order_is_scaling_then_wavelet() -> None:
    """Pin the ``[s, w, s, w, ...]`` layout the kan_weights consumers rely on."""
    torch.manual_seed(1)
    module = WaveletBasis(num_bases=3, wavelet_family="haar")
    x = torch.randn(1, 2, 6, 6)
    out = module(x)

    # Every even slot equals slot 0 (scaling), every odd equals slot 1 (wavelet).
    for i in range(0, 6, 2):
        assert torch.equal(out[..., i], out[..., 0])
    for i in range(1, 6, 2):
        assert torch.equal(out[..., i], out[..., 1])
    assert not torch.equal(out[..., 0], out[..., 1])


@pytest.mark.unit
def test_gradients_match_old_loop() -> None:
    torch.manual_seed(2)
    module = WaveletBasis(num_bases=4, wavelet_family="haar", basis_trainable=True)

    x_new = torch.randn(2, 3, 8, 8, requires_grad=True)
    x_ref = x_new.detach().clone().requires_grad_(True)

    module(x_new).sum().backward()
    coeff_grad_new = module.coeffs.grad.detach().clone()
    module.coeffs.grad = None

    _reference_forward(module, x_ref).sum().backward()
    coeff_grad_ref = module.coeffs.grad.detach().clone()
    module.coeffs.grad = None

    assert torch.allclose(x_new.grad, x_ref.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(coeff_grad_new, coeff_grad_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_non_trainable_buffer_path_matches() -> None:
    torch.manual_seed(3)
    module = WaveletBasis(num_bases=2, wavelet_family="db2", basis_trainable=False)
    x = torch.randn(1, 1, 5, 5)
    assert torch.equal(module(x), _reference_forward(module, x))
