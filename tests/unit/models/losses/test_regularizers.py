"""Tests for k-space regularizers.

Targets ``mriforge.models.losses.regularizers.HermitianSymmetryLoss``.

Regression: for a CENTERED k-space (DC at index N/2) the Hermitian partner
of sample ``i`` is sample ``N − i`` (DC maps to itself). A plain ``flip``
gives ``N − 1 − i`` — half a sample off for even N — so a perfectly real
image would be penalised. The fix rolls by +1 after the flip; this test
asserts the loss is ≈0 on the k-space of a real image and clearly positive
on a non-Hermitian (complex) k-space.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.physics.fft_ops import fft2c  # noqa: E402
from mriforge.models.losses.regularizers import HermitianSymmetryLoss  # noqa: E402


def test_real_image_kspace_is_hermitian_symmetric() -> None:
    """k-space of a real image scores ≈0 (it is Hermitian by construction)."""
    torch.manual_seed(0)
    real_image = torch.randn(1, 1, 16, 16)
    kspace = fft2c(real_image)  # complex, centered

    loss = HermitianSymmetryLoss(normalize=True)(kspace)
    assert loss.item() < 1e-4, f"real-image k-space should be Hermitian, got {loss.item()}"


def test_non_hermitian_kspace_is_penalised() -> None:
    """A k-space with no conjugate symmetry scores clearly above zero."""
    torch.manual_seed(0)
    complex_image = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    kspace = fft2c(complex_image)

    loss = HermitianSymmetryLoss(normalize=True)(kspace)
    assert loss.item() > 1e-2


def test_hermitian_loss_is_differentiable() -> None:
    """The loss carries gradient back to the input k-space."""
    real_image = torch.randn(1, 1, 8, 8, requires_grad=True)
    kspace = fft2c(real_image)
    loss = HermitianSymmetryLoss(normalize=False)(kspace)
    loss.backward()
    assert real_image.grad is not None
