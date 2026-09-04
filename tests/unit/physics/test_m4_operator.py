"""Unit tests for the PMA-VarNet global forward operator (m4raw topology).

Covers two regression contracts hardened by the WS-4 physics review:

* ``Operator_A_m4.adjoint`` must raise a clean ``RuntimeError`` (mirroring
  ``forward``) when ``torchkbnufft`` is unavailable, rather than leaking an
  opaque ``AttributeError`` from a never-constructed ``self.adjnufft``.
* ``apply_rigid_transform`` advertises a ``theta: torch.Tensor | None``
  contract and must pass the image through unchanged when ``theta is None``.
"""

from __future__ import annotations

import pytest
import torch

import spectramr.infrastructure.physics.m4_operator as m4
from spectramr.infrastructure.physics.m4_operator import (
    Operator_A_m4,
    apply_rigid_transform,
)


def test_adjoint_raises_clean_runtimeerror_when_torchkbnufft_missing(monkeypatch):
    """adjoint() must fail-fast like forward() when the NUFFT lib is absent.

    Pre-fix the missing-library branch left self.adjnufft unset, so adjoint()
    raised AttributeError instead of the intended RuntimeError. We force the
    missing-library branch via monkeypatch so the test runs regardless of
    whether torchkbnufft is actually installed.
    """
    monkeypatch.setattr(m4, "TORCHKBNUFFT_AVAILABLE", False)
    op = Operator_A_m4(im_size=(8, 8))
    y = torch.randn(1, 4, 16, dtype=torch.complex64)
    maps = {"dK": torch.randn(1, 2, 16)}
    with pytest.raises(RuntimeError, match="TorchKbNufft required"):
        op.adjoint(y, maps)


def test_forward_raises_clean_runtimeerror_when_torchkbnufft_missing(monkeypatch):
    """forward() already guarded; lock in the symmetric fail-fast contract."""
    monkeypatch.setattr(m4, "TORCHKBNUFFT_AVAILABLE", False)
    op = Operator_A_m4(im_size=(8, 8))
    x = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    maps = {"dK": torch.randn(1, 2, 16)}
    with pytest.raises(RuntimeError, match="TorchKbNufft required"):
        op.forward(x, maps)


def test_apply_rigid_transform_none_theta_passthrough():
    """theta=None is a supported (pass-through) input per the new annotation."""
    x = torch.randn(1, 1, 4, 4)
    out = apply_rigid_transform(x, None)
    assert torch.equal(out, x)
