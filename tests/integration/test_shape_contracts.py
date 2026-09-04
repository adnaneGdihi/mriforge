import pytest
import torch

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.data.collation.strategies import squeezing_collate


class TestShapeContracts:
    """Regression tests for data pipeline shape contracts.

    References:
        - docs/SHAPE_CONTRACTS.md
        - src/data/collation/strategies.py (_recursive_squeeze fix)
    """

    @pytest.fixture
    def collate_fn(self):
        """Get the squeezing collate function."""
        return squeezing_collate

    def test_2d_slice_contract(self, collate_fn):
        """
        Contract: 2D data (patch_size=[H,W,1]) must be squeezed to (N,C,H,W).

        TorchIO produces: (1, C, H, W, 1) per sample
        Collate stack:    (N, C, H, W, 1)
        Expected output:  (N, C, H, W)  <-- The critical squeezed 5th dim
        """
        batch_size = 4
        C, H, W, D = 2, 64, 64, 1

        # Simulate TorchIO image objects
        class MockImage:
            def __init__(self, data):
                self.data = data
                self.affine = torch.eye(4)

        batch = []
        for _ in range(batch_size):
            sample = {
                "kspace": MockImage(torch.randn(C, H, W, D)),
                "image": MockImage(torch.randn(1, H, W, D)),
                "location": torch.tensor([0, 0, 0]),
            }
            batch.append(sample)

        # Act
        collated = collate_fn(batch)

        # Assert - K-space
        kspace_tensor = collated["kspace"]
        assert kspace_tensor.ndim == 4, f"Expected 4D kspace, got {kspace_tensor.shape}"
        assert kspace_tensor.shape == (batch_size, C, H, W)

        # Assert - Image
        image_tensor = collated["image"]
        assert image_tensor.ndim == 4, f"Expected 4D image, got {image_tensor.shape}"
        assert image_tensor.shape == (batch_size, 1, H, W)

    def test_3d_volume_contract(self, collate_fn):
        """
        Contract: 3D data (patch_size=[W,H,D] where D>1) must REMAIN (N,C,H,W,D).

        TorchIO produces: (C, H, W, D) tensors.
        Collate stacks to: (N, C, H, W, D).
        Since D > 1, _recursive_squeeze should NOT squeeze dim -1.
        """
        batch_size = 2
        C, H, W, D = 1, 32, 32, 16  # D > 1

        class MockImage:
            def __init__(self, data):
                self.data = data
                self.affine = torch.eye(4)

        batch = []
        for _ in range(batch_size):
            sample = {
                "kspace": MockImage(torch.randn(C, H, W, D)),
                "image": MockImage(torch.randn(C, H, W, D)),
            }
            batch.append(sample)

        # Act
        collated = collate_fn(batch)

        # Assert - K-space
        kspace_tensor = collated["kspace"]
        # Should remain 5D: (N, C, H, W, D)
        assert (
            kspace_tensor.ndim == 5
        ), f"Expected 5D kspace for 3D volume, got {kspace_tensor.shape}"
        assert kspace_tensor.shape == (batch_size, C, H, W, D)

        # Assert - Image
        image_tensor = collated["image"]
        assert (
            image_tensor.ndim == 5
        ), f"Expected 5D image for 3D volume, got {image_tensor.shape}"
        assert image_tensor.shape == (batch_size, C, H, W, D)

    def test_legacy_shape_migration(self):
        """Test strict validator for patch_size migration."""
        # Legacy case 1: img_size 2D
        legacy_config = {"dataset_type": "kspace", "img_size": [128, 128]}
        cfg = DataConfigSchema(**legacy_config)
        assert cfg.sampling.patch_size == (128, 128, 1)

        # Legacy case 2: target_size 3D
        legacy_config_3d = {"dataset_type": "kspace", "target_size": [64, 64, 32]}
        cfg_3d = DataConfigSchema(**legacy_config_3d)
        assert cfg_3d.sampling.patch_size == (64, 64, 32)
