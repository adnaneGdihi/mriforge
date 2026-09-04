"""Oracle tests for equivariant motion-state resolution (A-4.3 EMR)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.motion_equivariance import (
    MotionEquivarianceResolver,
    is_motion_equivariant,
    motion_equivariance_defect,
    rotation_90,
    translation,
)


def _conv(weight: torch.Tensor):
    # A circular convolution: exactly translation-equivariant.
    k = weight.shape[-1]
    pad = k // 2

    def op(x: torch.Tensor) -> torch.Tensor:
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="circular")
        return torch.nn.functional.conv2d(xp, weight)

    return op


def _fixed_mask(mask: torch.Tensor):
    # Multiply by a FIXED spatial mask: NOT translation-equivariant.
    return lambda x: x * mask


# --------------------------------------------------------------------------- #
# Equivariant operators -> zero defect
# --------------------------------------------------------------------------- #


def test_identity_is_motion_equivariant() -> None:
    x = torch.rand(1, 1, 16, 16)
    transforms = [translation(2, 0), translation(0, 3), rotation_90(1)]
    assert motion_equivariance_defect(lambda z: z, transforms, x) == pytest.approx(0.0, abs=1e-6)


def test_circular_conv_is_translation_equivariant() -> None:
    torch.manual_seed(0)
    w = torch.randn(1, 1, 3, 3)
    x = torch.rand(1, 1, 16, 16)
    transforms = [translation(1, 0), translation(0, 2), translation(3, 3)]
    assert is_motion_equivariant(_conv(w), transforms, x, tol=1e-5)


# --------------------------------------------------------------------------- #
# Non-equivariant operator -> positive defect
# --------------------------------------------------------------------------- #


def test_fixed_mask_is_not_translation_equivariant() -> None:
    torch.manual_seed(1)
    mask = (torch.rand(1, 1, 16, 16) > 0.5).float()
    x = torch.rand(1, 1, 16, 16)
    transforms = [translation(2, 0), translation(0, 3)]
    defect = motion_equivariance_defect(_fixed_mask(mask), transforms, x)
    assert defect > 1e-2
    assert not is_motion_equivariant(_fixed_mask(mask), transforms, x)


def test_defect_nonnegative_and_scale_invariant() -> None:
    torch.manual_seed(2)
    mask = (torch.rand(1, 1, 16, 16) > 0.5).float()
    x = torch.rand(1, 1, 16, 16)
    transforms = [translation(2, 1)]
    d1 = motion_equivariance_defect(_fixed_mask(mask), transforms, x)
    d2 = motion_equivariance_defect(_fixed_mask(mask), transforms, 7.0 * x)
    assert d1 >= 0.0
    assert d1 == pytest.approx(d2, rel=1e-5)


def test_empty_transforms_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        motion_equivariance_defect(lambda z: z, [], torch.rand(1, 1, 8, 8))


# --------------------------------------------------------------------------- #
# Cert bundle
# --------------------------------------------------------------------------- #


def test_resolver_distinguishes_equivariant_from_not() -> None:
    torch.manual_seed(3)
    x = torch.rand(1, 1, 16, 16)
    transforms = [translation(1, 0), translation(0, 2)]
    res = MotionEquivarianceResolver()

    equi = res.certify(_conv(torch.randn(1, 1, 3, 3)), transforms, x)
    assert equi["equivariant"] is True
    assert len(equi["per_transform_defect"]) == 2

    mask = (torch.rand(1, 1, 16, 16) > 0.5).float()
    noneq = res.certify(_fixed_mask(mask), transforms, x)
    assert noneq["equivariant"] is False
    assert noneq["mean_defect"] > 0.0
