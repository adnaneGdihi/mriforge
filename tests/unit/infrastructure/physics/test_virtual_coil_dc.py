"""Unit tests for VirtualCoilDataConsistency (C1, audit 2026-05-28).

The virtual-coil DC block was previously imported by no test. It implements
one unrolled data-consistency step for the SIREN-VarNet:

    x_DC = x - alpha * S^* . F^H P^T ( P F (S . x) - y )

The invariants checked here are the ones that make a DC block correct:

* **Format / shape** — input and output are both real/imag-stacked ``[B, 2, H, W]``.
* **Consistency fixed point** — if the current estimate already reproduces the
  measured k-space (``y = P F (S x)``), the residual is zero and the block is a
  no-op. This is *the* defining property of a data-consistency update.
* **alpha=0 is identity** — with a zero step size the update term vanishes.
* **alpha is a learnable parameter** and the block is differentiable wrt it.
"""

import pytest
import torch

from mriforge.infrastructure.physics.fft_ops import fft2c
from mriforge.infrastructure.physics.virtual_coil_dc import VirtualCoilDataConsistency


def _to_stacked(complex_tensor: torch.Tensor) -> torch.Tensor:
    """[B, H, W] complex -> [B, 2, H, W] real/imag stacked."""
    return torch.stack([complex_tensor.real, complex_tensor.imag], dim=1)


@pytest.fixture
def fields():
    """A small, deterministic complex image / sensitivity / mask set."""
    torch.manual_seed(0)
    b, h, w = 2, 16, 16
    x = torch.randn(b, h, w) + 1j * torch.randn(b, h, w)
    s = torch.randn(b, h, w) + 1j * torch.randn(b, h, w)
    # A non-trivial (but not all-zero / all-one) Cartesian mask.
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, ::2, :] = 1.0
    return x, s, mask


def test_output_is_realimag_stacked_with_input_shape(fields):
    x, s, mask = fields
    block = VirtualCoilDataConsistency(init_alpha=0.5)
    y = torch.zeros_like(_to_stacked(x))
    out = block(_to_stacked(x), y, mask, _to_stacked(s))
    assert out.shape == (x.shape[0], 2, x.shape[1], x.shape[2])
    assert torch.is_floating_point(out)


def test_consistency_fixed_point_is_noop(fields):
    """If y already equals P F (S x), the DC update must leave x unchanged."""
    x, s, mask = fields
    block = VirtualCoilDataConsistency(init_alpha=0.7)

    # Construct measured k-space that the current estimate exactly reproduces.
    y_complex = mask[:, 0] * fft2c(s * x)
    y = _to_stacked(y_complex)

    out = block(_to_stacked(x), y, mask, _to_stacked(s))
    out_complex = torch.complex(out[:, 0], out[:, 1])

    # Residual is zero -> gradient term is zero -> x unchanged.
    assert torch.allclose(out_complex, x, atol=1e-5), (
        "DC step is not a no-op at the consistency fixed point"
    )


def test_alpha_zero_is_identity(fields):
    """A zero step size makes the update term vanish regardless of residual."""
    x, s, mask = fields
    block = VirtualCoilDataConsistency(init_alpha=0.0)
    # Arbitrary (non-consistent) measurement -> non-zero residual.
    y = torch.randn_like(_to_stacked(x))
    out = block(_to_stacked(x), y, mask, _to_stacked(s))
    out_complex = torch.complex(out[:, 0], out[:, 1])
    assert torch.allclose(out_complex, x, atol=1e-6)


def test_alpha_is_learnable_and_block_is_differentiable(fields):
    x, s, mask = fields
    block = VirtualCoilDataConsistency(init_alpha=0.5)
    assert block.alpha.requires_grad
    assert list(block.parameters()), "alpha must be registered as a parameter"

    y = torch.randn_like(_to_stacked(x))
    out = block(_to_stacked(x), y, mask, _to_stacked(s))
    out.pow(2).sum().backward()
    assert block.alpha.grad is not None
    assert torch.isfinite(block.alpha.grad).all()


def test_nonzero_residual_moves_estimate(fields):
    """A non-consistent measurement with alpha>0 must change the estimate."""
    x, s, mask = fields
    block = VirtualCoilDataConsistency(init_alpha=0.9)
    y = _to_stacked(mask[:, 0] * fft2c(s * x) + 1.0)  # perturb measured k-space
    out = block(_to_stacked(x), y, mask, _to_stacked(s))
    out_complex = torch.complex(out[:, 0], out[:, 1])
    assert not torch.allclose(out_complex, x, atol=1e-4)


def test_complex_input_raises(fields):
    """A complex tensor (not real/imag-stacked) must fail-fast, not be mis-parsed.

    Regression: forward() parses channel-0/1 as real/imag, so a complex
    tensor would crash or silently mis-parse. The fail-fast guard now raises.
    """
    x, s, mask = fields
    block = VirtualCoilDataConsistency(init_alpha=0.5)
    y = torch.zeros_like(_to_stacked(x))
    with pytest.raises(ValueError, match="already complex"):
        block(x, y, mask, _to_stacked(s))  # x is [B,H,W] complex, not stacked


def test_wrong_channel_count_raises(fields):
    """A non-2-channel real tensor must fail-fast (index 1 would be invalid)."""
    x, s, mask = fields
    block = VirtualCoilDataConsistency(init_alpha=0.5)
    bad = x.real.unsqueeze(1)  # [B, 1, H, W]
    y = torch.zeros_like(_to_stacked(x))
    with pytest.raises(ValueError, match="exactly 2 channels"):
        block(bad, y, mask, _to_stacked(s))
