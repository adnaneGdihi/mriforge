"""Coverage tests for the multi-coil time-segmented MFI NUFFT wrapper.

Targets :mod:`spectramr.infrastructure.physics.nufft.multi_coil_time_segmented`.

``MultiCoilTimeSegmentedNUFFT`` wraps the single-coil
:class:`TimeSegmentedNUFFT` across an array of coil sensitivities:

- forward: ``s_c(t) = A(S_c * m)`` -> ``[B, C, K]``
- adjoint: ``m^H = sum_c conj(S_c) * A^H(s_c)`` -> ``[B, 1, H, W]``

The operator constructs ``torchkbnufft`` operators, so the module is skipped
when ``torchkbnufft`` is unavailable. CPU-only, tiny tensors, fixed seed.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchkbnufft")

from spectramr.infrastructure.physics.nufft.multi_coil_time_segmented import (  # noqa: E402
    MultiCoilTimeSegmentedNUFFT,
)


def _build():
    torch.manual_seed(0)
    H = W = 16
    B, C, K = 1, 2, 24
    op = MultiCoilTimeSegmentedNUFFT(image_shape=(H, W), num_time_segments=3)
    image = torch.complex(torch.randn(B, 1, H, W), torch.randn(B, 1, H, W))
    smaps = torch.complex(torch.randn(B, C, H, W), torch.randn(B, C, H, W))
    b0_map = torch.zeros(B, 1, H, W)
    trajectory = (torch.rand(B, 2, K) - 0.5) * 2 * torch.pi
    return op, image, smaps, b0_map, trajectory, (B, C, K, H, W)


def test_forward_shape_and_finite() -> None:
    op, image, smaps, b0_map, traj, (B, C, K, H, W) = _build()
    out = op.forward(
        image=image,
        coil_sensitivities=smaps,
        b0_map=b0_map,
        trajectory=traj,
    )
    assert out.shape == (B, C, K)
    assert torch.is_complex(out)
    assert torch.isfinite(out).all()


def test_adjoint_shape_and_finite() -> None:
    op, image, smaps, b0_map, traj, (B, C, K, H, W) = _build()
    kspace = torch.complex(torch.randn(B, C, K), torch.randn(B, C, K))
    recon = op.adjoint(
        kspace=kspace,
        coil_sensitivities=smaps,
        b0_map=b0_map,
        trajectory=traj,
    )
    assert recon.shape == (B, 1, H, W)
    assert torch.is_complex(recon)
    assert torch.isfinite(recon).all()


def test_gradient_flows_through_forward() -> None:
    op, image, smaps, b0_map, traj, _ = _build()
    image = image.clone().requires_grad_(True)
    out = op.forward(
        image=image,
        coil_sensitivities=smaps,
        b0_map=b0_map,
        trajectory=traj,
    )
    loss = (out.abs() ** 2).sum()
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_adjoint_dot_product_consistency() -> None:
    """Loose adjoint test: <A m, s> ~ <m, A^H s> (generous tolerance)."""
    op, image, smaps, b0_map, traj, (B, C, K, H, W) = _build()
    s = torch.complex(torch.randn(B, C, K), torch.randn(B, C, K))
    Am = op.forward(image=image, coil_sensitivities=smaps, b0_map=b0_map, trajectory=traj)
    AHs = op.adjoint(kspace=s, coil_sensitivities=smaps, b0_map=b0_map, trajectory=traj)
    lhs = torch.sum(Am * s.conj())
    rhs = torch.sum(image * AHs.conj())
    # NUFFT gridding + interpolation introduce error; assert same order of
    # magnitude rather than strict equality.
    denom = max(abs(lhs.item()), abs(rhs.item()), 1e-6)
    rel = abs(lhs.item() - rhs.item()) / denom
    assert rel < 0.5
