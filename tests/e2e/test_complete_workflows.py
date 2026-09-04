"""End-to-end tests for complete training and inference workflows.

This module tests complete workflows including:
- Data loading → Training step → Loss computation
- Inference pipeline device and shape integrity
- Multi-epoch consistency
- Error recovery
"""

import pytest
import torch

from spectramr.data.batch_types import BatchAdapter, TrainingBatch


class TestEnd2EndTrainingWorkflow:
    """End-to-end tests for training workflow."""

    def test_complete_training_batch_preparation(self):
        """Test complete batch preparation for training."""
        # Step 1: Raw data from loader
        raw_batch_dict = {
            "input": torch.randn(4, 36, 256, 256),
            "target": torch.randn(4, 36, 256, 256),
            "mask": torch.ones(4, 1, 256, 256),
            "filename": "test.h5",
            "coils": 18,
        }

        # Step 2: Adapt to TrainingBatch
        batch = BatchAdapter.from_dict(raw_batch_dict)
        assert isinstance(batch, TrainingBatch)
        assert batch.input.shape == (4, 36, 256, 256)
        assert batch.target.shape == (4, 36, 256, 256)

        # Step 3: Move to device
        moved = batch.to("cpu")
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"

        # Step 4: Verify metadata preserved
        assert moved.metadata["coils"] == 18
        assert moved.metadata["filename"] == "test.h5"

    def test_complete_inference_workflow(self):
        """Test complete inference workflow."""
        # Step 1: Create inference batch (batch size 1)
        batch_dict = {
            "input": torch.randn(1, 36, 256, 256),
            "target": torch.randn(1, 36, 256, 256),
        }

        # Step 2: Adapt
        batch = BatchAdapter.from_dict(batch_dict)
        assert batch.input.shape[0] == 1

        # Step 3: Move to device
        moved = batch.to("cpu")
        assert moved.input.device.type == "cpu"

        # Step 4: Simulate model output
        model_output = torch.randn_like(moved.input)
        assert model_output.device == moved.input.device
        assert model_output.shape == moved.input.shape

    def test_multi_batch_consistency(self):
        """Test consistency across multiple batches in epoch."""
        num_batches = 5
        batch_size = 4

        for batch_idx in range(num_batches):
            batch_dict = {
                "input": torch.randn(batch_size, 36, 256, 256),
                "target": torch.randn(batch_size, 36, 256, 256),
                "batch_idx": batch_idx,
            }

            batch = BatchAdapter.from_dict(batch_dict)
            moved = batch.to("cpu")

            # All batches should have same structure
            assert moved.input.shape == (batch_size, 36, 256, 256)
            assert moved.target.shape == (batch_size, 36, 256, 256)
            assert moved.metadata["batch_idx"] == batch_idx

    def test_multi_epoch_batch_consistency(self):
        """Test batch consistency across multiple epochs."""
        num_epochs = 3
        batches_per_epoch = 5

        for epoch in range(num_epochs):
            for batch_idx in range(batches_per_epoch):
                batch_dict = {
                    "input": torch.randn(4, 2, 256, 256),
                    "target": torch.randn(4, 2, 256, 256),
                }

                batch = BatchAdapter.from_dict(batch_dict)
                moved = batch.to("cpu")

                # Verify consistency
                assert moved.input.shape[0] == 4
                assert moved.target.shape[0] == 4


class TestBatchConsistency:
    """Test consistency of batch objects through training workflow."""

    def test_batch_tensor_reference_integrity(self):
        """Test tensor reference integrity through workflow."""
        input_tensor = torch.randn(4, 2, 256, 256)
        target_tensor = torch.randn(4, 2, 256, 256)

        batch = TrainingBatch(input=input_tensor, target=target_tensor)

        # References should be preserved
        assert batch.input is input_tensor
        assert batch.target is target_tensor

        # After device move (both on CPU)
        moved = batch.to("cpu")
        # New tensors but same values
        assert torch.allclose(moved.input, input_tensor)
        assert torch.allclose(moved.target, target_tensor)

    def test_batch_metadata_immutability_through_workflow(self):
        """Test that metadata remains consistent through workflow."""
        metadata = {
            "filename": "test.h5",
            "slice_idx": 10,
            "acquisition_date": "2024-01-01",
        }

        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
            metadata=metadata.copy(),
        )

        # Move to device
        moved = batch.to("cpu")

        # Metadata should be identical
        assert moved.metadata == batch.metadata
        assert moved.metadata["filename"] == "test.h5"
        assert moved.metadata["slice_idx"] == 10

    def test_batch_shape_immutability_through_workflow(self):
        """Test that shapes remain consistent through workflow."""
        shape = (4, 36, 256, 256)

        batch = TrainingBatch(
            input=torch.randn(*shape),
            target=torch.randn(*shape),
        )

        # Initial shape
        assert batch.input.shape == shape

        # After device move
        moved = batch.to("cpu")
        assert moved.input.shape == shape

        # After multiple device moves
        moved2 = moved.to("cpu")
        assert moved2.input.shape == shape


class TestColdDiffusionCompleteWorkflow:
    """Test complete cold diffusion workflow."""

    def test_cold_diffusion_end2end(self):
        """Test complete cold diffusion training workflow."""
        # Step 1: Raw data with undersampling
        num_coils = 18
        num_channels = num_coils * 2

        # Create full k-space
        full_kspace = torch.randn(4, num_channels, 256, 256)

        # Create undersampled k-space
        undersampled = full_kspace.clone()
        # Zero out 75% for 4x acceleration
        mask = torch.rand(1, 1, 256, 256) > 0.75
        undersampled = undersampled * mask.float()

        # Step 2: Create batch
        batch_dict = {
            "input": undersampled,
            "target": full_kspace,
            "mask": mask.float(),
            "coils": num_coils,
            "acceleration": 4.0,
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Step 3: Verify batch routing
        assert torch.equal(batch.input, undersampled)
        assert torch.equal(batch.target, full_kspace)
        assert torch.equal(batch.mask, mask.float())

        # Step 4: Move to device
        moved = batch.to("cpu")
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"

        # Step 5: Verify metadata
        assert moved.metadata["coils"] == num_coils
        assert moved.metadata["acceleration"] == 4.0

    def test_cold_diffusion_mask_consistency(self):
        """Test mask consistency in cold diffusion workflow."""
        # Create mask and undersampled data
        mask = torch.ones(4, 1, 256, 256)
        # Create center frequency region (typical Cartesian mask)
        center_region = 64
        mask[:, :, :center_region, :] = 1.0
        mask[:, :, center_region:, :] = 0.25

        batch_dict = {
            "input": torch.randn(4, 36, 256, 256),
            "target": torch.randn(4, 36, 256, 256),
            "mask": mask,
        }

        batch = BatchAdapter.from_dict(batch_dict)
        moved = batch.to("cpu")

        # Mask should be accessible and have correct shape
        assert moved.mask.shape == (4, 1, 256, 256)
        # Mask values should be in valid range [0, 1]
        assert (moved.mask >= 0).all() and (moved.mask <= 1).all()


class TestSuperResolutionWorkflow:
    """Test super-resolution training workflow."""

    def test_super_resolution_batch_preparation(self):
        """Test super-resolution batch preparation."""
        # Low-res and high-res images
        batch_dict = {
            "input": torch.randn(4, 1, 64, 64),  # Low-res
            "target": torch.randn(4, 1, 256, 256),  # High-res
        }

        batch = BatchAdapter.from_dict(batch_dict)

        # Verify different spatial sizes are handled
        assert batch.input.shape[-2:] != batch.target.shape[-2:]
        assert batch.input.shape[0] == batch.target.shape[0]  # Same batch size

        # Move to device
        moved = batch.to("cpu")
        assert moved.input.device == moved.target.device

    def test_super_resolution_scaling_integrity(self):
        """Test scaling integrity in super-resolution."""
        upscale_factor = 4

        for low_res_size in [32, 64, 128]:
            high_res_size = low_res_size * upscale_factor

            batch_dict = {
                "input": torch.randn(4, 1, low_res_size, low_res_size),
                "target": torch.randn(4, 1, high_res_size, high_res_size),
            }

            batch = BatchAdapter.from_dict(batch_dict)

            # Verify upscaling
            assert batch.target.shape[-1] == upscale_factor * batch.input.shape[-1]
            assert batch.target.shape[-2] == upscale_factor * batch.input.shape[-2]


class TestBatchHandlingErrors:
    """Test error handling in batch workflows."""

    def test_invalid_batch_format_error(self):
        """Test error handling for invalid batch format."""
        batch_dict = {
            "unknown_field": torch.randn(4, 2, 256, 256),
        }

        with pytest.raises(ValueError):
            BatchAdapter.from_dict(batch_dict)

    def test_empty_batch_handling(self):
        """Test handling of edge case with minimal batch."""
        batch_dict = {
            "input": torch.randn(1, 1, 32, 32),  # Minimal batch
            "target": torch.randn(1, 1, 32, 32),
        }

        batch = BatchAdapter.from_dict(batch_dict)
        assert batch.input.shape[0] == 1

    def test_large_batch_handling(self):
        """Test handling of large batches."""
        batch_dict = {
            "input": torch.randn(256, 2, 256, 256),
            "target": torch.randn(256, 2, 256, 256),
        }

        batch = BatchAdapter.from_dict(batch_dict)
        assert batch.input.shape[0] == 256


class TestDataTypeConsistency:
    """Test data type consistency through workflows."""

    def test_dtype_preservation_workflow(self):
        """Test dtype preservation through complete workflow."""
        for dtype in [torch.float32, torch.float64]:
            batch_dict = {
                "input": torch.randn(4, 2, 256, 256, dtype=dtype),
                "target": torch.randn(4, 2, 256, 256, dtype=dtype),
            }

            batch = BatchAdapter.from_dict(batch_dict)
            moved = batch.to("cpu")

            assert moved.input.dtype == dtype
            assert moved.target.dtype == dtype

    def test_mixed_dtype_workflow(self):
        """Test workflow with mixed dtypes."""
        batch_dict = {
            "input": torch.randn(4, 2, 256, 256, dtype=torch.float32),
            "target": torch.randn(4, 2, 256, 256, dtype=torch.float32),
            "mask": torch.ones(4, 1, 256, 256, dtype=torch.bool),
        }

        batch = BatchAdapter.from_dict(batch_dict)
        moved = batch.to("cpu")

        assert moved.input.dtype == torch.float32
        assert moved.mask.dtype == torch.bool


class TestDeviceMovementPatterns:
    """Test common device movement patterns."""

    def test_cpu_to_cpu_movement(self):
        """Test moving batch from CPU to CPU (no-op)."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, device="cpu"),
            target=torch.randn(4, 2, 256, 256, device="cpu"),
        )

        moved = batch.to("cpu")

        # Should remain on CPU
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"

    def test_repeated_device_movement(self):
        """Test repeated device movement."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
        )

        # Multiple device moves should work
        for _ in range(5):
            batch = batch.to("cpu")
            assert batch.input.device.type == "cpu"

    def test_nonblocking_movement_consistency(self):
        """Test consistency of non-blocking movements."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
        )

        # Non-blocking move
        moved = batch.to("cpu", non_blocking=True)

        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"


class TestNumericalConsistency:
    """Test numerical consistency through workflows."""

    def test_batch_value_consistency(self):
        """Test that batch values remain consistent through device moves."""
        input_vals = torch.randn(4, 2, 256, 256)
        target_vals = torch.randn(4, 2, 256, 256)

        batch = TrainingBatch(input=input_vals, target=target_vals)
        moved = batch.to("cpu")

        # Values should be numerically identical
        assert torch.allclose(moved.input, input_vals)
        assert torch.allclose(moved.target, target_vals)

    def test_batch_statistics_consistency(self):
        """Test that batch statistics remain consistent."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
        )

        # Compute statistics before
        input_mean_before = batch.input.mean()
        target_mean_before = batch.target.mean()

        # Move to device
        moved = batch.to("cpu")

        # Compute statistics after
        input_mean_after = moved.input.mean()
        target_mean_after = moved.target.mean()

        # Should be numerically identical
        assert torch.allclose(input_mean_before, input_mean_after)
        assert torch.allclose(target_mean_before, target_mean_after)


class TestGradientFlowIntegrity:
    """Test gradient flow through batch workflow."""

    def test_gradient_flow_through_batch(self):
        """Test that gradients flow correctly through batch."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, requires_grad=True),
            target=torch.randn(4, 2, 256, 256),
        )

        # Compute loss
        loss = (batch.input * 2).sum()

        # Backward pass
        loss.backward()

        # Gradients should be computed
        assert batch.input.grad is not None
        assert batch.input.grad.shape == batch.input.shape

    def test_gradient_flow_after_device_move(self):
        """Test gradient flow after device movement."""
        input_with_grad = torch.randn(4, 2, 256, 256, requires_grad=True)

        batch = TrainingBatch(
            input=input_with_grad,
            target=torch.randn(4, 2, 256, 256),
        )

        # Verify requires_grad preserved
        assert batch.input.requires_grad is True

        # After device move (CPU to CPU)
        moved = batch.to("cpu")
        assert moved.input.requires_grad is True
