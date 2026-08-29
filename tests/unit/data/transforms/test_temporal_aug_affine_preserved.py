"""Regression tests for TDA-003: temporal transforms preserve the affine.

Targets ``mriforge.data.transforms.temporal_aug``.

Each of ``PhaseShiftTransform``, ``TemporalFlipTransform`` and
``FrameDropoutTransform`` rebuilds a ``tio.ScalarImage`` from a tensor
after its operation. Before the fix these constructors omitted
``affine=img.affine``, so TorchIO silently reset the affine to the
identity (``_parse_affine`` returns ``np.eye(4)`` when ``affine=None``),
breaking spatial consistency for subjects that carry a real voxel-spacing
affine. The fix passes ``affine=img.affine`` at all three sites; these
tests pin that the non-identity affine survives each transform.
"""

from __future__ import annotations

import numpy as np
import torch

from mriforge.data.transforms.temporal_aug import (
    FrameDropoutTransform,
    PhaseShiftTransform,
    TemporalFlipTransform,
)

# Non-identity affine: 2 mm isotropic voxel spacing.
_NON_IDENTITY_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float64)


def _make_subject() -> "object":
    import torchio as tio

    # [C=frames, W, H, D]; multiple frames so flip/dropout/shift are non-trivial.
    tensor = torch.rand(4, 8, 8, 1)
    return tio.Subject(
        input=tio.ScalarImage(tensor=tensor, affine=_NON_IDENTITY_AFFINE.copy()),
        target=tio.ScalarImage(tensor=tensor.clone(), affine=_NON_IDENTITY_AFFINE.copy()),
    )


def test_phase_shift_preserves_affine() -> None:
    torch.manual_seed(0)
    subject = _make_subject()
    # max_shift>0 guarantees the image is rebuilt.
    out = PhaseShiftTransform(max_shift=3, include_zero=False)(subject)
    np.testing.assert_array_equal(out["input"].affine, _NON_IDENTITY_AFFINE)
    np.testing.assert_array_equal(out["target"].affine, _NON_IDENTITY_AFFINE)


def test_temporal_flip_preserves_affine() -> None:
    torch.manual_seed(0)
    subject = _make_subject()
    # p=1.0 forces the flip branch to rebuild the images.
    out = TemporalFlipTransform(p=1.0)(subject)
    np.testing.assert_array_equal(out["input"].affine, _NON_IDENTITY_AFFINE)
    np.testing.assert_array_equal(out["target"].affine, _NON_IDENTITY_AFFINE)


def test_frame_dropout_preserves_affine() -> None:
    torch.manual_seed(0)
    subject = _make_subject()
    # dropout_rate>0 and only_input=False so both keys are rebuilt.
    out = FrameDropoutTransform(dropout_rate=0.5, only_input=False)(subject)
    np.testing.assert_array_equal(out["input"].affine, _NON_IDENTITY_AFFINE)
    np.testing.assert_array_equal(out["target"].affine, _NON_IDENTITY_AFFINE)
