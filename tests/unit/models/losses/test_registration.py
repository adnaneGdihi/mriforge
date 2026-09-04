"""Tests for registration losses.

Targets ``spectramr.models.losses.registration.LocalCrossCorrelationLoss``.

Regression: the local variances ``Var = E[X^2] - E[X]^2`` can dip a few ULP
below zero from float round-off. The denominator ``I_var * J_var + eps`` must
not be allowed to flip sign (which would spike the loss / produce NaN on flat
regions), so each variance is clamped to be non-negative.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.registration import LocalCrossCorrelationLoss  # noqa: E402


def test_lncc_finite_on_constant_input() -> None:
    """A constant image has zero local variance — the clamped denominator must
    keep the loss finite (no NaN/Inf from a sign-flipped variance product)."""
    loss_fn = LocalCrossCorrelationLoss(kernel_size=9)
    img = torch.full((1, 1, 32, 32), 0.5)

    loss = loss_fn(img, img)
    assert torch.isfinite(loss).all()


def test_lncc_bounded_on_random_input() -> None:
    """With non-negative variances, ``cc = cov^2 / (var_I var_J)`` is in [0, 1]
    by Cauchy-Schwarz, so the negated loss stays within roughly [-1, 0]."""
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)

    loss = LocalCrossCorrelationLoss(kernel_size=9)(pred, target)
    assert torch.isfinite(loss).all()
    # Small boundary-padding slack on the [-1, 0] range.
    assert -1.05 <= loss.item() <= 0.05
