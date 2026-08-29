"""Unit tests for the joint k-space / image rotation transform (B-1 / Idea 1).

Direct-import tests on tiny tensors. Covers shape, no-NaN, the Fourier-rotation
consistency property (k-space rotation == image rotation up to interpolation),
complex-phase preservation, deterministic draws, and the TorchIO transform.
"""

from __future__ import annotations

import numpy as np
import torch
import torchio as tio

from mriforge.data.transforms.joint_rotation import (
    JointKSpaceImageRotation,
    rotate_image,
    rotate_kspace,
)
from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c


def test_rotate_image_shape_and_finite():
    x = torch.randn(2, 1, 32, 32)
    r = rotate_image(x, 0.5)
    assert r.shape == x.shape
    assert torch.isfinite(r).all()


def test_rotate_image_zero_angle_is_identity():
    x = torch.randn(1, 1, 32, 32)
    assert torch.allclose(rotate_image(x, 0.0), x, atol=1e-5)


def test_rotate_image_complex_preserves_dtype():
    x = torch.randn(1, 2, 24, 24, dtype=torch.complex64)
    r = rotate_image(x, 0.3)
    assert r.is_complex()
    assert r.shape == x.shape
    assert torch.isfinite(r.real).all() and torch.isfinite(r.imag).all()


def test_kspace_rotation_matches_image_rotation():
    """Fourier identity: rotate(kspace) == fft2c(rotate(ifft2c(kspace))).

    Build a smooth image, rotate it directly; build its k-space and rotate that;
    the two reconstructions must agree up to bilinear-interpolation error.
    """
    torch.manual_seed(0)
    h = w = 48
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    # Smooth, band-limited image so interpolation error stays small.
    img = torch.exp(-(xx**2 + yy**2) * 4.0).view(1, 1, h, w).to(torch.complex64)
    angle = 0.4

    rotated_img = rotate_image(img, angle)
    k = fft2c(img)
    rotated_k = rotate_kspace(k, angle)
    img_from_k = ifft2c(rotated_k)

    err = (rotated_img - img_from_k).abs().mean().item()
    ref = rotated_img.abs().mean().item()
    assert err / (ref + 1e-8) < 0.05  # < 5% relative interpolation error


def test_rotate_kspace_returns_complex():
    k = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    out = rotate_kspace(k, 0.2)
    assert out.is_complex()
    assert out.shape == k.shape


def test_integer_shift_is_exact_roll():
    """SE2 integer shift uses torch.roll => exactly invertible."""
    x = torch.randn(1, 1, 16, 16)
    shifted = rotate_image(x, 0.0, shift_px=(3, -2))
    unshifted = torch.roll(shifted, shifts=(-3, 2), dims=(-2, -1))
    assert torch.allclose(unshifted, x, atol=1e-5)


def test_transform_deterministic_with_seed():
    t1 = JointKSpaceImageRotation(group="SO2", seed=11)
    t2 = JointKSpaceImageRotation(group="SO2", seed=11)
    assert t1._draw_element() == t2._draw_element()


def test_seeded_draws_are_diverse_within_an_instance():
    """A seeded instance must still vary draw-to-draw (no augmentation collapse).

    Regression: the generator was reseeded with the *same* constant on every
    call, so every sample received an identical angle/shift — the seeded
    "augmentation" applied one fixed rotation to the whole dataset. Consecutive
    draws within one instance must differ.
    """
    t = JointKSpaceImageRotation(group="SO2", seed=11)
    draws = [t._draw_element() for _ in range(5)]
    assert len(set(draws)) > 1, f"all draws identical (collapse): {draws}"


def test_seeded_draw_sequence_is_reproducible_across_instances():
    """Two instances with the same seed produce the same *sequence* of draws.

    This is the reproducibility guarantee that must survive the diversity fix:
    determinism means a repeatable sequence, not a frozen constant.
    """
    a = JointKSpaceImageRotation(group="SE2", seed=7)
    b = JointKSpaceImageRotation(group="SE2", seed=7)
    seq_a = [a._draw_element() for _ in range(5)]
    seq_b = [b._draw_element() for _ in range(5)]
    assert seq_a == seq_b


def test_transform_unknown_group_raises():
    try:
        JointKSpaceImageRotation(group="SO3")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown group")


def test_transform_bad_distribution_raises():
    try:
        JointKSpaceImageRotation(angle_distribution="gaussian")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-uniform distribution")


def test_torchio_subject_roundtrip():
    """Applying the transform to a TorchIO subject preserves shape + finiteness."""
    data = torch.randn(1, 32, 32, 1)  # [C, H, W, D]
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=data.clone(), affine=np.eye(4)),
        target=tio.ScalarImage(tensor=data.clone(), affine=np.eye(4)),
    )
    transform = JointKSpaceImageRotation(group="SO2", seed=3)
    out = transform(subject)
    assert out["input"].data.shape == data.shape
    assert torch.isfinite(out["input"].data).all()
    # input and target share the seed-driven element, so they rotate identically.
    assert torch.allclose(out["input"].data, out["target"].data, atol=1e-5)


def test_torchio_zero_angle_is_identity():
    data = torch.randn(1, 24, 24, 1)
    subject = tio.Subject(input=tio.ScalarImage(tensor=data.clone(), affine=np.eye(4)))
    # Monkeypatch the draw to a 0-angle element to verify the rotate path is a
    # true no-op at theta=0 within the TorchIO permute/reshape plumbing.
    transform = JointKSpaceImageRotation(group="SO2", seed=0)
    transform._draw_element = lambda: (0.0, (0, 0))  # type: ignore[method-assign]
    out = transform(subject)
    assert torch.allclose(out["input"].data, data, atol=1e-5)
