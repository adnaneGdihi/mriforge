"""Tests for _ResampleToReferenceTransform.

Verifies that paired NIfTI subjects with mismatched spatial shapes
are correctly resampled to a common grid before spatial augmentations.
"""

import numpy as np
import pytest
import torch
import torchio as tio


class TestResampleToReferenceTransform:
    """Verify _ResampleToReferenceTransform handles paired NIfTI shape mismatch."""

    @pytest.fixture
    def transform(self):
        from mriforge.data.builders.torchio_transform_builder import (
            _ResampleToReferenceTransform,
        )
        return _ResampleToReferenceTransform()

    @pytest.fixture
    def mismatched_subject(self):
        """Create a Subject with mismatched spatial shapes (simulating ULF/HF pair)."""
        affine = np.eye(4)
        image_tensor = torch.randn(1, 181, 222, 196)  # ULF shape
        target_tensor = torch.randn(1, 221, 221, 144)  # HF shape (different!)
        return tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor, affine=affine),
            target=tio.ScalarImage(tensor=target_tensor, affine=affine),
        )

    @pytest.fixture
    def matched_subject(self):
        """Create a Subject where all images already match."""
        affine = np.eye(4)
        tensor = torch.randn(1, 256, 256, 1)
        return tio.Subject(
            image=tio.ScalarImage(tensor=tensor.clone(), affine=affine),
            target=tio.ScalarImage(tensor=tensor.clone(), affine=affine),
        )

    def test_resamples_mismatched_shapes(self, transform, mismatched_subject):
        """Target should be resampled to match image spatial shape."""
        result = transform(mismatched_subject)

        image_shape = result["image"].spatial_shape
        target_shape = result["target"].spatial_shape

        assert image_shape == target_shape, (
            f"Shapes still mismatch after resample: "
            f"image={image_shape}, target={target_shape}"
        )

    def test_reference_is_image_key(self, transform, mismatched_subject):
        """The 'image' key should be the reference (unchanged)."""
        original_image_shape = mismatched_subject["image"].spatial_shape
        result = transform(mismatched_subject)

        assert result["image"].spatial_shape == original_image_shape, (
            "Reference image shape was modified — it should be unchanged"
        )

    def test_noop_when_shapes_match(self, transform, matched_subject):
        """No resampling should occur when shapes already match."""
        original_data = matched_subject["target"].data.clone()
        result = transform(matched_subject)

        # Data should be identical (no interpolation artifacts)
        assert torch.allclose(result["target"].data, original_data, atol=1e-6), (
            "Data was modified even though shapes already matched"
        )

    def test_single_image_noop(self, transform):
        """Single-image subjects should pass through unchanged."""
        affine = np.eye(4)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.randn(1, 64, 64, 32), affine=affine),
        )
        result = transform(subject)
        assert result["image"].spatial_shape == (64, 64, 32)

    def test_random_affine_succeeds_after_resample(self, transform, mismatched_subject):
        """RandomAffine should work after resample (the actual bug fix)."""
        # This is the exact scenario that was failing in the 23 experiments
        resampled = transform(mismatched_subject)

        # This should NOT raise RuntimeError about spatial_shape
        augmented = tio.RandomAffine(
            degrees=(0, 0, 0, 0, -5, 5),
            scales=0,
            translation=0,
            p=1.0,
        )(resampled)

        assert augmented["image"].spatial_shape == augmented["target"].spatial_shape

    def test_preserves_channel_dimension(self, transform):
        """Channel dimension should be preserved during resample."""
        affine = np.eye(4)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.randn(2, 64, 64, 32), affine=affine),
            target=tio.ScalarImage(tensor=torch.randn(2, 48, 48, 24), affine=affine),
        )
        result = transform(subject)
        assert result["target"].data.shape[0] == 2, "Channel dim was corrupted"

    def test_3d_volume_resample(self, transform):
        """Full 3D volume resample should work correctly."""
        affine = np.eye(4)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.randn(1, 128, 128, 64), affine=affine),
            target=tio.ScalarImage(tensor=torch.randn(1, 96, 96, 48), affine=affine),
        )
        result = transform(subject)
        assert result["target"].spatial_shape == (128, 128, 64)


class TestEnsureMinimumSpatialSize:
    """Verify _EnsureMinimumSpatialSize pads volumes smaller than patch_size."""

    @pytest.fixture
    def transform_256(self):
        from mriforge.data.builders.torchio_transform_builder import (
            _EnsureMinimumSpatialSize,
        )
        return _EnsureMinimumSpatialSize((256, 256, 1))

    def test_pads_undersized_volume(self, transform_256):
        """Volume (179, 218, 196) should be padded to at least (256, 256, 196)."""
        affine = np.eye(4)
        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=torch.randn(1, 179, 218, 196), affine=affine
            ),
        )
        result = transform_256(subject)
        w, h, d = result["image"].spatial_shape
        assert w >= 256, f"W={w} still < 256 after padding"
        assert h >= 256, f"H={h} still < 256 after padding"
        assert d == 196, f"Depth was modified: {d} != 196 (should be unchanged for D=1 patch)"

    def test_noop_when_large_enough(self, transform_256):
        """Volumes already >= patch_size should not be modified."""
        affine = np.eye(4)
        tensor = torch.randn(1, 320, 320, 64)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=tensor.clone(), affine=affine),
        )
        result = transform_256(subject)
        assert result["image"].spatial_shape == (320, 320, 64)

    def test_pads_all_images_consistently(self, transform_256):
        """Both image and target should be padded to same size."""
        affine = np.eye(4)
        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=torch.randn(1, 179, 218, 196), affine=affine
            ),
            target=tio.ScalarImage(
                tensor=torch.randn(1, 179, 218, 196), affine=affine
            ),
        )
        result = transform_256(subject)
        assert result["image"].spatial_shape == result["target"].spatial_shape

    def test_exact_cluster_scenario(self, transform_256):
        """Reproduce the exact L5_batch_fetch failure from the cluster."""
        affine = np.eye(4)
        # Create ULF volume with typical cluster dimensions
        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=torch.randn(1, 179, 218, 196), affine=affine
            ),
            target=tio.ScalarImage(
                tensor=torch.randn(1, 179, 218, 196), affine=affine
            ),
        )
        result = transform_256(subject)

        # Verify TorchIO CropOrPad with patch_size (256, 256, 1) will NOT fail
        cropper = tio.CropOrPad((256, 256, 1))
        patch = cropper(result)  # This is what the sampler does
        assert patch["image"].spatial_shape == (256, 256, 1)

    def test_2d_patch_size(self):
        """Test with 2-element patch_size (auto-expand to 3D)."""
        from mriforge.data.builders.torchio_transform_builder import (
            _EnsureMinimumSpatialSize,
        )
        transform = _EnsureMinimumSpatialSize((128, 128))
        affine = np.eye(4)
        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=torch.randn(1, 64, 100, 32), affine=affine
            ),
        )
        result = transform(subject)
        w, h, d = result["image"].spatial_shape
        assert w >= 128
        assert h >= 128
        assert d == 32  # Depth unchanged
