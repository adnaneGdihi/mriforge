"""Unit tests for the TorchIO intensity-augmentation adapters.

Pairs ``src/spectramr/data/transforms/intensity_augmentation.py``.

The adapters exist to bridge two layouts, so the layout assertions here are the
point, not boilerplate: TorchIO hands out ``[C, W, H, D]`` with no batch axis
while ``MotionBlur3D`` unpacks ``B, C, D, H, W``.  A silently transposed volume
still trains — it just trains on the wrong anatomy — so shape and content are
both checked.
"""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.intensity_augmentation import (
    RandomBrightness,
    RandomContrast,
    RandomMotionBlur,
    RandomRicianNoise,
)

# TorchIO layout is [C, W, H, D]; a 2-D slice is stored with D == 1.
SHAPE_2D = (1, 16, 16, 1)
SHAPE_3D = (1, 16, 16, 8)


def _subject(shape: tuple[int, ...], fill: float | None = None) -> tio.Subject:
    data = torch.full(shape, fill) if fill is not None else torch.rand(*shape) + 0.5
    return tio.Subject(image=tio.ScalarImage(tensor=data))


@pytest.mark.parametrize("shape", [SHAPE_2D, SHAPE_3D], ids=["2d", "3d"])
@pytest.mark.parametrize(
    "transform",
    [
        RandomBrightness((0.8, 1.2)),
        RandomContrast((0.8, 1.2)),
        RandomRicianNoise(0.05),
    ],
    ids=["brightness", "contrast", "rician"],
)
def test_shape_is_preserved(transform: tio.Transform, shape: tuple[int, ...]) -> None:
    """Every adapter is shape-preserving on both 2-D slices and 3-D volumes."""
    out = transform(_subject(shape))
    assert out.image.data.shape == torch.Size(shape)


class TestRandomBrightness:
    def test_scales_by_the_drawn_factor(self) -> None:
        """A degenerate range pins the factor, so the output is exactly k * x."""
        out = RandomBrightness((3.0, 3.0))(_subject(SHAPE_2D, fill=2.0))
        assert torch.allclose(out.image.data, torch.full(SHAPE_2D, 6.0))

    def test_label_maps_are_untouched(self) -> None:
        """``IntensityTransform`` must not rescale a segmentation mask."""
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.ones(*SHAPE_2D)),
            mask=tio.LabelMap(tensor=torch.ones(*SHAPE_2D)),
        )
        out = RandomBrightness((5.0, 5.0))(subject)
        assert float(out.image.data.mean()) == pytest.approx(5.0)
        assert float(out.mask.data.mean()) == pytest.approx(1.0)


class TestRandomContrast:
    def test_mean_is_preserved_and_spread_scales(self) -> None:
        """``(x - mean) * c + mean`` moves the spread, never the mean."""
        data = torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1)
        subject = tio.Subject(image=tio.ScalarImage(tensor=data))
        out = RandomContrast((2.0, 2.0))(subject).image.data
        assert float(out.mean()) == pytest.approx(2.0)
        assert float(out.std()) == pytest.approx(float(data.std()) * 2.0)

    def test_identity_factor_is_a_no_op(self) -> None:
        subject = _subject(SHAPE_2D)
        before = subject.image.data.clone()
        after = RandomContrast((1.0, 1.0))(subject).image.data
        assert torch.allclose(before, after, atol=1e-6)


class TestRandomRicianNoise:
    def test_zero_level_is_a_no_op(self) -> None:
        """``RicianNoise`` short-circuits at 0, so the image must be untouched."""
        subject = _subject(SHAPE_3D)
        before = subject.image.data.clone()
        after = RandomRicianNoise(0.0)(subject).image.data
        assert torch.equal(before, after)

    def test_noise_is_non_negative(self) -> None:
        """The Rician magnitude is a norm, so it cannot go negative even on
        an input that Gaussian noise would push below zero."""
        subject = _subject(SHAPE_2D, fill=0.0)
        out = RandomRicianNoise(1.0)(subject).image.data
        assert bool((out >= 0).all())
        assert float(out.abs().sum()) > 0.0


class TestRandomMotionBlur:
    def test_raises_on_a_volume_thinner_than_the_kernel(self) -> None:
        """A 2-D slice cannot carry a 5-voxel kernel; zero padding would
        silently darken it, so the adapter refuses (non-negotiable 3)."""
        with pytest.raises(ValueError, match="blur_kernel_size"):
            RandomMotionBlur(0.5, blur_kernel_size=5)(_subject(SHAPE_2D))

    def test_error_names_the_config_knob_to_change(self) -> None:
        """A raise inside a dataloader worker is only actionable if it says
        which YAML key produced it."""
        with pytest.raises(ValueError, match="enable_motion_blur"):
            RandomMotionBlur(0.5, blur_kernel_size=5)(_subject(SHAPE_2D))

    def test_blurs_a_volume_thick_enough_for_the_kernel(self) -> None:
        subject = _subject(SHAPE_3D)
        before = subject.image.data.clone()
        after = RandomMotionBlur(0.5, blur_kernel_size=5)(subject).image.data
        assert after.shape == before.shape
        assert not torch.allclose(before, after)

    def test_zero_intensity_is_a_no_op(self) -> None:
        """``MotionBlur3D`` short-circuits at 0 intensity."""
        subject = _subject(SHAPE_3D)
        before = subject.image.data.clone()
        after = RandomMotionBlur(0.0, blur_kernel_size=5)(subject).image.data
        assert torch.equal(before, after)

    def test_layout_bridge_round_trips(self) -> None:
        """The adapter permutes to ``[B, C, D, H, W]`` and back.  An anisotropic
        volume catches a transposition that a cubic one would hide."""
        shape = (1, 12, 10, 8)
        out = RandomMotionBlur(0.5, blur_kernel_size=5)(_subject(shape))
        assert out.image.data.shape == torch.Size(shape)
