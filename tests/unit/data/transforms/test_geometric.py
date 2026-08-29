"""Tests for ``EnsureSpatialConsistency`` and ``SmartGeometricStandardization``.

Targets ``mriforge.data.transforms.geometric``. These two TorchIO transforms
are the spatial-alignment SSOT used by every dataset before the
``Queue / Sampler`` because TorchIO requires identical affines across
images in a Subject.

Categories:

- Unit: identity-affine forced; correct-size pass-through; resize for
  square mismatched; centre-crop / pad for non-square
- Edge: complex-valued data round-trip
- Sanity-shape: parametrised over square/rectangular/in-place sizes
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchio as tio

from mriforge.data.transforms.geometric import (
    EnsureSpatialConsistency,
    SmartGeometricStandardization,
)


def _make_subject(*, h: int, w: int, complex_data: bool = False) -> tio.Subject:
    """Helper: a small Subject with one ``input`` ScalarImage."""
    if complex_data:
        # tio expects float — use real-stacked then we view-as-complex
        # at test sites. Here: real float for simplicity.
        data = torch.complex(torch.rand(1, h, w, 1), torch.rand(1, h, w, 1))
    else:
        data = torch.rand(1, h, w, 1)  # (C, H, W, D)
    img = tio.ScalarImage(tensor=data, affine=np.eye(4))
    return tio.Subject(input=img)


# ---------------------------------------------------------------------------
# EnsureSpatialConsistency
# ---------------------------------------------------------------------------


def test_ensure_spatial_consistency_forces_identity_affine() -> None:
    """Default ``EnsureSpatialConsistency`` rewrites every image's affine to identity."""
    subject = _make_subject(h=8, w=8)
    # Manually set a non-identity affine
    subject["input"] = tio.ScalarImage(
        tensor=subject["input"].data, affine=np.diag([2, 2, 2, 1])
    )

    transform = EnsureSpatialConsistency()
    out = transform(subject)
    assert np.allclose(out["input"].affine, np.eye(4))


def test_ensure_spatial_consistency_custom_reference_affine() -> None:
    """A custom ``reference_affine`` is propagated to all images."""
    custom = np.diag([1.5, 1.5, 1.5, 1])
    transform = EnsureSpatialConsistency(reference_affine=custom)
    subject = _make_subject(h=8, w=8)
    out = transform(subject)
    assert np.allclose(out["input"].affine, custom)


def test_ensure_spatial_consistency_preserves_data() -> None:
    """Image data is unchanged — only affine is rewritten."""
    subject = _make_subject(h=4, w=4)
    original = subject["input"].data.clone()
    transform = EnsureSpatialConsistency()
    out = transform(subject)
    assert torch.equal(out["input"].data, original)


def test_ensure_spatial_consistency_preserves_image_class() -> None:
    """``LabelMap`` instances stay as ``LabelMap`` after the transform."""
    label = tio.LabelMap(tensor=torch.zeros(1, 4, 4, 1).long(), affine=np.eye(4))
    subject = tio.Subject(seg=label)
    out = EnsureSpatialConsistency()(subject)
    assert isinstance(out["seg"], tio.LabelMap)


# ---------------------------------------------------------------------------
# SmartGeometricStandardization
# ---------------------------------------------------------------------------


def test_smart_standardization_correct_size_is_no_op() -> None:
    """When input size already matches target, data is unchanged."""
    subject = _make_subject(h=320, w=320)
    original = subject["input"].data.clone()
    transform = SmartGeometricStandardization(target_shape=(320, 320))
    out = transform(subject)
    assert torch.equal(out["input"].data, original)


def test_smart_standardization_square_resize() -> None:
    """Square but wrong-sized input is resized to the target shape."""
    subject = _make_subject(h=64, w=64)
    transform = SmartGeometricStandardization(target_shape=(32, 32))
    out = transform(subject)
    assert out["input"].data.shape == (1, 32, 32, 1)


def test_smart_standardization_non_square_centre_crops() -> None:
    """Non-square input (FastMRI-style) is centre-cropped to the target."""
    # FastMRI native: 640 (height/RO) × 368 (width/PE)
    subject = _make_subject(h=640, w=368)
    transform = SmartGeometricStandardization(target_shape=(320, 320))
    out = transform(subject)
    assert out["input"].data.shape == (1, 320, 320, 1)


def test_smart_standardization_pads_smaller_dim() -> None:
    """Non-square input with one dim < target is padded, then cropped to target."""
    subject = _make_subject(h=64, w=200)
    transform = SmartGeometricStandardization(target_shape=(128, 128))
    out = transform(subject)
    assert out["input"].data.shape == (1, 128, 128, 1)


def test_smart_standardization_skips_kspace_key() -> None:
    """``kspace`` is excluded from resizing (different native resolution)."""
    img = tio.ScalarImage(
        tensor=torch.rand(1, 640, 368, 1), affine=np.eye(4)
    )
    kspace = tio.ScalarImage(
        tensor=torch.rand(1, 640, 368, 1), affine=np.eye(4)
    )
    subject = tio.Subject(input=img, kspace=kspace)
    transform = SmartGeometricStandardization(target_shape=(320, 320))
    out = transform(subject)
    assert out["input"].data.shape == (1, 320, 320, 1)
    assert out["kspace"].data.shape == (1, 640, 368, 1)


def test_smart_standardization_complex_round_trip() -> None:
    """Complex data is preserved as complex through the transform."""
    cplx = torch.complex(torch.rand(1, 64, 64, 1), torch.rand(1, 64, 64, 1))
    img = tio.ScalarImage(tensor=cplx, affine=np.eye(4))
    subject = tio.Subject(input=img)
    transform = SmartGeometricStandardization(target_shape=(32, 32))
    out = transform(subject)
    assert torch.is_complex(out["input"].data)
    assert out["input"].data.shape == (1, 32, 32, 1)


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape, target",
    [
        ((64, 64), (32, 32)),     # square shrink
        ((640, 368), (320, 320)), # non-square crop (FastMRI)
        ((100, 200), (128, 128)), # non-square pad+crop
        ((128, 128), (128, 128)), # already correct
    ],
    ids=["64_to_32", "640x368_to_320", "100x200_to_128", "noop"],
)
def test_smart_standardization_shape_matrix(
    input_shape: tuple[int, int], target: tuple[int, int]
) -> None:
    """Output is exactly the target shape, regardless of input geometry."""
    h, w = input_shape
    subject = _make_subject(h=h, w=w)
    out = SmartGeometricStandardization(target_shape=target)(subject)
    assert out["input"].data.shape == (1, target[0], target[1], 1)


# ---------------------------------------------------------------------------
# EnsureSpatialConsistency: the Subject.__dict__ re-sync (#1213)
# ---------------------------------------------------------------------------
#
# ``tio.Subject`` is a ``dict`` subclass that MIRRORS its entries into
# ``self.__dict__``, and ``Subject.__setitem__`` is not defined — so
# ``subject[name] = new_image`` reaches ``dict.__setitem__`` and the two views
# diverge silently. ``tio.Crop.apply_transform`` (the engine behind every
# ``PatchSampler``, hence every ``tio.Queue``) builds its output *solely* from
# ``subject.__dict__``, so a desynced subject is cropped from its PRE-transform
# images. Because ``EnsureSpatialConsistency`` runs FIRST in every built chain
# and replaces EVERY image, one missing sync discarded the whole chain's output:
# a declared k-space normalization reached ``train_step`` as raw data
# (|k|max 2478 where the transform yields ~4).


def test_ensure_spatial_consistency_keeps_dict_and_mapping_aliased() -> None:
    """After the transform, ``subject[k]`` and ``subject.__dict__[k]`` are one object.

    This is the invariant whose violation made the chain's output unreachable
    through a crop. Asserted per image, by identity — equal *values* would pass
    even while the two views were separate objects about to diverge.
    """
    subject = _make_subject(h=8, w=8)
    subject.add_image(tio.ScalarImage(tensor=torch.rand(1, 8, 8, 1)), "target")

    out = EnsureSpatialConsistency()(subject)

    for name in out.get_images_names():
        assert out.__dict__[name] is out[name], (
            f"{name!r} desynced: tio.Crop reads __dict__ and would return the pre-transform image"
        )


def test_ensure_spatial_consistency_survives_a_crop() -> None:
    """The values the transform installed are the values a crop returns.

    The behavioural half of the invariant above: ``EnsureSpatialConsistency``
    replaces every image object, so before the fix a crop reproduced the input
    tensor no matter what the transform (or any later chain member) had written.
    """
    subject = _make_subject(h=8, w=8)
    marker = torch.full((1, 8, 8, 1), 0.25)

    consistent = EnsureSpatialConsistency()(subject)
    # A later chain member's edit, applied the way the real transforms do it:
    # in place on the object the MAPPING holds.
    consistent["input"].set_data(marker)

    cropped = tio.Crop((0, 0, 0, 0, 0, 0))(consistent)
    assert torch.equal(cropped["input"].data, marker)


def test_non_image_keys_survive_a_crop_after_the_transform() -> None:
    """Metadata a transform publishes is not dropped at patch extraction.

    ``tio.Crop`` deep-copies non-image entries out of ``__dict__`` too, so an
    unsynced scalar key is not merely stale — it is **absent** from the patch.
    That is how ``kspace_scale`` / ``kspace_normalized`` vanished from the batch,
    leaving the strategy's gate to read "never normalized" and compensate.
    """
    subject = _make_subject(h=8, w=8)
    subject["kspace_scale"] = torch.tensor(48.33)

    consistent = EnsureSpatialConsistency()(subject)
    cropped = tio.Crop((0, 0, 0, 0, 0, 0))(consistent)

    assert "kspace_scale" in cropped
    assert float(cropped["kspace_scale"]) == pytest.approx(48.33)
