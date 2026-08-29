import torch

from mriforge.data.collation.strategies import ImageCollateStrategy


class TestImageCollateStrategy:
    def test_squeeze_depth_dim_enabled(self):
        """Test that squeeze_depth_dim=True removes singleton depth dim."""
        strategy = ImageCollateStrategy(squeeze_depth_dim=True)

        # Simulate TorchIO batch: (N, C, H, W, 1) -> List of (C, H, W, 1)
        batch = [{"image": torch.randn(2, 64, 64, 1)} for _ in range(4)]

        result = strategy.collate(batch)
        images = result["image"]

        # Should squeeze the last dim -> (4, 2, 64, 64)
        assert images.shape == (4, 2, 64, 64)
        assert images.ndim == 4

    def test_squeeze_depth_dim_disabled(self):
        """Test that squeeze_depth_dim=False preserves singleton depth dim."""
        strategy = ImageCollateStrategy(squeeze_depth_dim=False)

        # Simulate TorchIO batch: (N, C, H, W, 1)
        batch = [{"image": torch.randn(2, 64, 64, 1)} for _ in range(4)]

        result = strategy.collate(batch)
        images = result["image"]

        # Should NOT squeeze -> (4, 2, 64, 64, 1)
        assert images.shape == (4, 2, 64, 64, 1)
        assert images.ndim == 5

    def test_squeeze_ignore_3d_volume(self):
        """Test that squeeze_depth_dim=True ignores volumes with D > 1."""
        strategy = ImageCollateStrategy(squeeze_depth_dim=True)

        # Simulate 3D volume batch: (N, C, H, W, D=32)
        batch = [{"image": torch.randn(1, 64, 64, 32)} for _ in range(2)]

        result = strategy.collate(batch)
        images = result["image"]

        # Should NOT squeeze -> (2, 1, 64, 64, 32)
        assert images.shape == (2, 1, 64, 64, 32)
        assert images.ndim == 5
