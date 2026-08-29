"""Regression: MultiCoilForwardOperator's centered FFT must route through the
``fft2c`` / ``ifft2c`` SSOT.

The operator used raw ``torch.fft`` with a ``fftshift ∘ fft2 ∘ ifftshift``
ordering. That is numerically identical to ``fft2c`` on EVEN grids (so the
existing even-size tests don't notice), but:

* it bypasses the FP32-autocast guard ``fft_ops`` wraps every transform in,
  which prevents NaN gradients under AMP (CLAUDE.md non-negotiable #2), and
* on ODD grids its shift ordering diverges from ``fft2c`` — so an operator
  output mixed with ``fft2c`` k-space elsewhere in a model is inconsistent.

Routing the centered branch to ``fft2c`` / ``ifft2c`` keeps even-size behaviour
identical and makes odd-size behaviour match the SSOT.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.infrastructure.physics.forward_operator import MultiCoilForwardOperator


def test_centered_forward_matches_fft2c_on_odd_grid():
    op = MultiCoilForwardOperator(centered=True)  # no coils, no mask → pure FFT
    x = torch.complex(torch.randn(1, 1, 9, 9), torch.randn(1, 1, 9, 9))

    y = op.forward(x)
    assert torch.allclose(y, fft2c(x), atol=1e-5), (
        "centered forward must equal fft2c(x) (SSOT) — diverges on odd grids "
        "when using raw torch.fft with the opposite shift ordering"
    )


def test_centered_adjoint_matches_ifft2c_on_odd_grid():
    op = MultiCoilForwardOperator(centered=True)
    y = torch.complex(torch.randn(1, 1, 9, 9), torch.randn(1, 1, 9, 9))

    x = op.adjoint(y)
    assert torch.allclose(x, ifft2c(y), atol=1e-5)


def test_adjoint_round_trip_preserved_odd_grid():
    op = MultiCoilForwardOperator(centered=True)
    x = torch.complex(torch.randn(1, 1, 9, 9), torch.randn(1, 1, 9, 9))
    assert torch.allclose(op.adjoint(op.forward(x)), x, atol=1e-4)
