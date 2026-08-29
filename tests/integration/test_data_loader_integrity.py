"""Integration tests for data loader batch adapter with device and size integrity.

This module tests the complete data loading pipeline with focus on:
- Tensor shape consistency across data pipeline
- Device placement correctness
- Batch adapter routing for different data formats
- Multi-coil k-space handling
- TorchIO integration with device movement
"""

import pytest
import torch

from mriforge.data.batch_types import BatchAdapter, TrainingBatch
from mriforge.infrastructure.training.utils.data_adapters import TorchIOAdapter


class TestDataLoaderBatchAdapterIntegration:
    """Integration tests for batch adapter in data loading pipeline."""

    def test_batch_adapter_routing_kspace_reconstruction(self):
        """Test batch adapter routes k-space reconstruction correctly."""
        # Simulate batch from TorchIO loader
        batch_dict = {
            "input": torch.randn(4, 36, 256, 256),  # 18 coils * 2
            "target": torch.randn(4, 1, 256, 256),  # Magnitude image
            "mask": torch.ones(4, 1, 256, 256),
            "filename": "test.h5",
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Verify routing
        assert isinstance(batch, TrainingBatch)
        assert batch.input.shape[1] == 36  # k-space channels
        assert batch.target.shape[1] == 1  # magnitude channels
        assert batch.input.device == batch.target.device
        assert "filename" in batch.metadata

    def test_batch_adapter_routing_cold_diffusion(self):
        """Test batch adapter routes cold diffusion k-space correctly."""
        batch_dict = {
            "input": torch.randn(4, 36, 256, 256),  # Undersampled
            "target": torch.randn(4, 36, 256, 256),  # Full
            "mask": torch.ones(4, 1, 256, 256),
            "coils": 18,
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Verify routing
        assert batch.input.shape == torch.Size([4, 36, 256, 256])
        assert batch.target.shape == torch.Size([4, 36, 256, 256])
        assert batch.metadata["coils"] == 18

        # Verify accessing via mapped keys
        assert torch.equal(batch["input"], batch.input)
        assert torch.equal(batch["target"], batch.target)

    def test_batch_device_consistency(self):
        """Test device consistency after batch routing."""
        batch_dict = {
            "input": torch.randn(4, 2, 64, 64),
            "target": torch.randn(4, 2, 64, 64),
            "mask": torch.ones(4, 1, 64, 64),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # All tensors should be on same device
        assert batch.input.device == batch.target.device
        if batch.mask is not None:
            assert batch.target.device == batch.mask.device

    def test_batch_size_preservation_through_adapter(self):
        """Test that batch sizes are preserved through adapter."""
        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            batch_dict = {
                "input": torch.randn(batch_size, 2, 64, 64),
                "target": torch.randn(batch_size, 2, 64, 64),
            }

            batch = BatchAdapter.from_dict(batch_dict)

            assert batch.input.shape[0] == batch_size
            assert batch.target.shape[0] == batch_size

    def test_batch_shape_consistency_multimodal(self):
        """Test shape consistency for multi-modal data."""
        # Different modalities might have different spatial sizes
        batch_dict = {
            "input": torch.randn(4, 36, 256, 256),  # k-space
            "target": torch.randn(4, 36, 256, 256),  # Full k-space
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Spatial dimensions should match
        assert batch.input.shape[2:] == batch.target.shape[2:]


class TestTorchIOAdapterIntegration:
    """Integration tests for TorchIO adapter with device handling."""

    def test_torchio_format_to_batch_format_single_slice(self):
        """Test converting TorchIO format to batch format for single slice."""
        # TorchIO format: (C, H, W, D) with D=1 for single slice
        torchio_tensor = torch.randn(36, 256, 256, 1)

        batch_tensor = TorchIOAdapter.to_batch_format(
            torchio_tensor, expected_channels=36
        )

        # Should convert to (B, C, H, W)
        assert batch_tensor.ndim == 4
        assert batch_tensor.shape[1] == 36  # Channels preserved
        assert batch_tensor.shape[2:] == torch.Size([256, 256])

    def test_torchio_format_to_batch_format_volume(self):
        """Test converting TorchIO format to batch format for volume."""
        # TorchIO format: (C, H, W, D) with D > 1 for volume
        torchio_tensor = torch.randn(2, 256, 256, 18)  # 18 slices

        batch_tensor = TorchIOAdapter.to_batch_format(
            torchio_tensor, expected_channels=2
        )

        # Should convert to (B*D, C, H, W)
        assert batch_tensor.ndim == 4
        assert batch_tensor.shape[0] == 18  # Batch becomes slices
        assert batch_tensor.shape[1] == 2  # Channels preserved

    def test_torchio_adapter_device_handling(self):
        """Test device handling in TorchIO adapter."""
        torchio_tensor = torch.randn(36, 256, 256, 1, device="cpu")

        batch_tensor = TorchIOAdapter.to_batch_format(
            torchio_tensor, expected_channels=36
        )

        # Device should be preserved
        assert batch_tensor.device.type == "cpu"

    def test_torchio_adapter_channel_matching(self):
        """Test channel matching in TorchIO adapter."""
        # Prediction and target have different channels
        prediction = torch.randn(4, 2, 256, 256)  # 2-channel (complex)
        target = torch.randn(4, 1, 256, 256)  # 1-channel (magnitude)

        pred_matched, tgt_matched = TorchIOAdapter.ensure_channel_match(
            prediction, target
        )

        # Should now have same channels
        assert pred_matched.shape[1] == tgt_matched.shape[1]

    def test_batch_adapter_channel_consistency(self):
        """Test that batch adapter maintains channel consistency."""
        # Multi-coil k-space: 18 coils * 2 (real/imag) = 36 channels
        batch_dict = {
            "input": torch.randn(4, 36, 256, 256),
            "target": torch.randn(4, 36, 256, 256),
        }

        batch = BatchAdapter.from_dict(batch_dict)
        moved = batch.to("cpu")

        # Channels should be preserved
        assert moved.input.shape[1] == 36
        assert moved.target.shape[1] == 36


class TestBatchShapeIntegrity:
    """Test batch shape integrity through pipeline."""

    def test_batch_shape_before_after_device_move(self):
        """Test that batch shapes are preserved after device movement."""
        shapes = [
            (4, 2, 64, 64),
            (8, 36, 256, 256),
            (1, 1, 512, 512),
            (16, 4, 32, 32),
        ]

        for shape in shapes:
            batch = TrainingBatch(
                input=torch.randn(*shape),
                target=torch.randn(*shape),
            )

            moved = batch.to("cpu")

            assert moved.input.shape == shape
            assert moved.target.shape == shape

    def test_batch_size_consistency_across_operations(self):
        """Test batch size consistency through multiple operations."""
        batch_size = 4
        channels = 36

        batch = TrainingBatch(
            input=torch.randn(batch_size, channels, 256, 256),
            target=torch.randn(batch_size, channels, 256, 256),
        )

        # Extract batch size
        extracted_size = batch.input.shape[0]
        assert extracted_size == batch_size

        # Move and verify
        moved = batch.to("cpu")
        assert moved.input.shape[0] == batch_size

    def test_batch_spatial_size_consistency(self):
        """Test spatial size consistency between input and target."""
        spatial_sizes = [64, 128, 256, 512]

        for size in spatial_sizes:
            batch = TrainingBatch(
                input=torch.randn(4, 2, size, size),
                target=torch.randn(4, 2, size, size),
            )

            # Input and target should have same spatial dims
            assert batch.input.shape[2:] == batch.target.shape[2:]
            assert batch.input.shape[-1] == size
            assert batch.input.shape[-2] == size

    def test_batch_channel_size_consistency(self):
        """Test channel size consistency."""
        channel_sizes = [1, 2, 4, 36, 64]

        for channels in channel_sizes:
            batch = TrainingBatch(
                input=torch.randn(4, channels, 256, 256),
                target=torch.randn(4, channels, 256, 256),
            )

            assert batch.input.shape[1] == channels
            assert batch.target.shape[1] == channels


class TestDeviceIntegrityAcrossPipeline:
    """Test device integrity across full pipeline."""

    def test_device_consistency_dict_to_batch_to_device(self):
        """Test device consistency from dict to batch to device move."""
        batch_dict = {
            "input": torch.randn(4, 36, 256, 256),
            "target": torch.randn(4, 36, 256, 256),
            "mask": torch.ones(4, 1, 256, 256),
        }

        # Initial device
        initial_device = batch_dict["input"].device

        # Through adapter
        batch = BatchAdapter.from_dict(batch_dict)
        assert batch.input.device == initial_device

        # After device move
        moved = batch.to("cpu")
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"
        assert moved.mask.device.type == "cpu"

    def test_device_movement_multicoil_batch(self):
        """Test device movement for multi-coil batch."""
        num_coils = 18
        num_channels = num_coils * 2

        batch = TrainingBatch(
            input=torch.randn(4, num_channels, 256, 256),
            target=torch.randn(4, num_channels, 256, 256),
            mask=torch.ones(4, 1, 256, 256),
        )

        # All should be on CPU initially
        assert batch.input.device.type == "cpu"

        # Move again (should still be CPU)
        moved = batch.to("cpu")
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"
        assert moved.mask.device.type == "cpu"

    def test_nonblocking_transfer_integrity(self):
        """Test that non-blocking transfer maintains integrity."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
            mask=torch.ones(4, 1, 256, 256),
        )

        # Non-blocking transfer should work
        moved = batch.to("cpu", non_blocking=True)

        # All should be on CPU
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"
        assert moved.mask.device.type == "cpu"


class TestDataTypeIntegrity:
    """Test data type integrity through pipeline."""

    def test_dtype_preservation_through_adapter(self):
        """Test that dtypes are preserved through adapter."""
        batch_dict = {
            "input": torch.randn(4, 2, 256, 256, dtype=torch.float32),
            "target": torch.randn(4, 2, 256, 256, dtype=torch.float64),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        assert batch.input.dtype == torch.float32
        assert batch.target.dtype == torch.float64

    def test_dtype_preservation_through_device_move(self):
        """Test that dtypes are preserved through device movement."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, dtype=torch.complex64),
            target=torch.randn(4, 2, 256, 256, dtype=torch.float32),
        )

        moved = batch.to("cpu")

        assert moved.input.dtype == torch.complex64
        assert moved.target.dtype == torch.float32

    def test_mixed_dtype_batch(self):
        """Test batch with mixed dtypes."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, dtype=torch.float32),
            target=torch.randn(4, 2, 256, 256, dtype=torch.float32),
            mask=torch.ones(4, 1, 256, 256, dtype=torch.bool),
        )

        moved = batch.to("cpu")

        # All dtypes should be preserved
        assert moved.input.dtype == torch.float32
        assert moved.target.dtype == torch.float32
        assert moved.mask.dtype == torch.bool


class TestMetadataIntegrity:
    """Test metadata handling and integrity."""

    def test_metadata_preservation_through_adapter(self):
        """Test that metadata is preserved through adapter."""
        batch_dict = {
            "input": torch.randn(4, 2, 256, 256),
            "target": torch.randn(4, 2, 256, 256),
            "filename": "test.h5",
            "slice_idx": 10,
            "acceleration": 4.0,
            "coils": 18,
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # All metadata should be preserved
        assert batch.metadata["filename"] == "test.h5"
        assert batch.metadata["slice_idx"] == 10
        assert batch.metadata["acceleration"] == 4.0
        assert batch.metadata["coils"] == 18

    def test_metadata_preservation_through_device_move(self):
        """Test that metadata is preserved through device movement."""
        metadata = {
            "filename": "test.h5",
            "acquisition_date": "2024-01-01",
            "params": {"field_strength": 3.0, "seq_type": "FLAIR"},
        }

        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
            metadata=metadata,
        )

        moved = batch.to("cpu")

        # Metadata should be identical
        assert moved.metadata == batch.metadata
        assert moved.metadata["acquisition_date"] == "2024-01-01"


class TestBatchAdapterErrorHandling:
    """Test error handling in batch adapter."""

    def test_missing_required_fields(self):
        """Test that missing required fields raise error."""
        batch_dict = {
            "unknown_field": torch.randn(4, 2, 256, 256),
        }

        with pytest.raises(ValueError):
            BatchAdapter.from_dict(batch_dict)

    def test_none_tensors_handled(self):
        """Test handling of None tensors."""
        batch_dict = {
            "input": torch.randn(4, 2, 256, 256),
            "target": torch.randn(4, 2, 256, 256),  # Valid target
        }

        # BatchAdapter from_dict should work with valid tensors
        batch = BatchAdapter.from_dict(batch_dict)
        assert batch.input is not None
        assert batch.target is not None

    def test_empty_metadata_dict(self):
        """Test batch adapter with empty metadata."""
        batch_dict = {
            "input": torch.randn(4, 2, 256, 256),
            "target": torch.randn(4, 2, 256, 256),
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Should have empty metadata, not error
        assert batch.metadata == {}


class TestColdDiffusionDataFlow:
    """Test complete data flow for cold diffusion training."""

    def test_cold_diffusion_batch_routing(self):
        """Test complete cold diffusion batch routing."""
        # Simulate real cold diffusion data
        batch_dict = {
            "input": torch.randn(4, 36, 256, 256),  # Undersampled
            "target": torch.randn(4, 36, 256, 256),  # Full
            "mask": torch.ones(4, 1, 256, 256),
            "filename": "test.h5",
            "coils": 18,
            "acceleration": 4.0,
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Verify routing
        assert torch.equal(batch.input, batch_dict["input"])
        assert torch.equal(batch.target, batch_dict["target"])
        assert torch.equal(batch.mask, batch_dict["mask"])

        # Verify shapes
        assert batch.input.shape == batch.target.shape
        assert batch.input.shape[1] == 36

        # Verify metadata
        assert batch.metadata["coils"] == 18
        assert batch.metadata["acceleration"] == 4.0

        # Verify device movement
        moved = batch.to("cpu")
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"

    def test_cold_diffusion_multi_batch_consistency(self):
        """Test consistency across multiple cold diffusion batches."""
        batches = []

        for batch_idx in range(4):
            batch_dict = {
                "input": torch.randn(4, 36, 256, 256),
                "target": torch.randn(4, 36, 256, 256),
                "mask": torch.ones(4, 1, 256, 256),
                "batch_idx": batch_idx,
            }

            batch = BatchAdapter.from_dict(batch_dict)
            batches.append(batch)

        # All batches should have consistent shapes
        for batch in batches:
            assert batch.input.shape == torch.Size([4, 36, 256, 256])
            assert batch.target.shape == torch.Size([4, 36, 256, 256])
            assert batch.mask.shape == torch.Size([4, 1, 256, 256])
