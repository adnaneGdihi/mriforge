import torch

from spectramr.data.collation.strategies import ImageCollateStrategy


class TestFix5DCollation:
    """Regression tests for 5D tensor (singleton depth) collation issues."""

    # Test for squeezing_collate is skipped - function not implemented
    # def test_squeezing_collate_direct(self):
    #     """Test that squeezing_collate function handles 5D tensors correctly."""
    #     # 4D images (C, H, W, 1) -> should be stacked to (B, C, H, W, 1) then squeezed
    #     batch = [
    #         {"image": torch.randn(1, 64, 64, 1)},
    #         {"image": torch.randn(1, 64, 64, 1)},
    #     ]
    #
    #     result = squeezing_collate(batch)
    #     collated = result["image"]
    #
    #     # Check shapes
    #     print(f"Collated shape: {collated.shape}")
    #
    #     assert (
    #         collated.ndim == 4
    #     ), f"Expected 4D tensor, got {collated.ndim}D shape {collated.shape}"
    #     assert collated.shape[0] == 2

    def test_preserves_true_3d_volumes(self):
        """Test that TRUE 3D volumes (depth > 1) are NOT squeezed."""
        strategy = ImageCollateStrategy()

        # 3D volume patches (C, H, W, D) where D=16
        batch = [
            {"image": torch.randn(1, 64, 64, 16)},
            {"image": torch.randn(1, 64, 64, 16)},
        ]

        result = strategy.collate(batch)
        collated = result["image"]

        assert (
            collated.ndim == 5
        ), "True 3D volumes should remain 5D (Batch, C, H, W, D)"
        assert collated.shape[-1] == 16
