import pytest
import torch

from mriforge.data.batch_types import BatchAdapter, TrainingBatch

# ==============================================================================
# Helpers and Mocks
# ==============================================================================


class MockTorchIOImage:
    """Mocks a TorchIO image object which has a `.tensor` but no `.to()`."""

    def __init__(self, tensor):
        self.tensor = tensor


# ==============================================================================
# Regular Behavior (Happy Path)
# ==============================================================================


def test_training_batch_creation_and_access():
    """Test creating a TrainingBatch and accessing its attributes via dict/tuple."""
    input_t = torch.zeros(1, 1, 2, 2)
    target_t = torch.ones(1, 1, 2, 2)
    mask_t = torch.ones(1, 1, 2, 2)
    metadata = {"fname": "test.nii", "slice": 5}

    batch = TrainingBatch(
        input=input_t, target=target_t, mask=mask_t, metadata=metadata
    )

    # Direct access
    assert torch.equal(batch.input, input_t)
    assert torch.equal(batch.target, target_t)
    assert torch.equal(batch.mask, mask_t)

    # Tuple unpacking access
    unpacked_input, unpacked_target = batch[0], batch[1]
    assert torch.equal(unpacked_input, input_t)
    assert torch.equal(unpacked_target, target_t)

    # Dict-like access
    assert torch.equal(batch["input"], input_t)
    assert torch.equal(batch["target"], target_t)
    assert torch.equal(batch["mask"], mask_t)
    assert batch["fname"] == "test.nii"

    # .get() access
    assert torch.equal(batch.get("input"), input_t)
    assert batch.get("missing_key", "default") == "default"

    # `in` operator
    assert "input" in batch
    assert "fname" in batch
    assert "missing_key" not in batch


def test_training_batch_to_device():
    """Test the recursively moving to device behavior of TrainingBatch."""
    # We'll use CPU to CPU so we don't require CUDA for testing, but we verify it returns new tensors correctly.
    input_t = torch.zeros(1, 1, 2, 2)
    target_t = torch.ones(1, 1, 2, 2)

    batch = TrainingBatch(
        input=input_t,
        target=target_t,
        metadata={"list": [torch.ones(1)], "dict": {"t": torch.zeros(1)}, "scalar": 5},
    )

    # Move to 'cpu' explicitly
    device_batch = batch.to("cpu")

    assert torch.equal(device_batch.input, input_t)
    assert torch.equal(device_batch.target, target_t)

    # Check metadata recursion
    assert torch.equal(device_batch.metadata["list"][0], torch.ones(1))
    assert torch.equal(device_batch.metadata["dict"]["t"], torch.zeros(1))
    assert device_batch.metadata["scalar"] == 5


# ==============================================================================
# Edge Cases & Special Handling
# ==============================================================================


def test_training_batch_invalid_access():
    """Test indexing TrainingBatch with invalid keys and indices."""
    batch = TrainingBatch(input=torch.zeros(1), target=torch.ones(1))

    with pytest.raises(IndexError, match="only supports unpacking \\(input, target\\)"):
        _ = batch[2]

    with pytest.raises(KeyError, match="Key 'invalid' not found"):
        _ = batch["invalid"]

    with pytest.raises(TypeError, match="indices must be int or str"):
        _ = batch[1.5]  # type: ignore


def test_batch_adapter_missing_keys():
    """Test BatchAdapter with missing canonical keys."""
    # Missing input
    with pytest.raises(
        ValueError, match="requires canonical keys \\('input', 'target'\\)"
    ):
        BatchAdapter.from_dict({"target": torch.ones(1)})

    # Missing target
    with pytest.raises(
        ValueError, match="requires canonical keys \\('input', 'target'\\)"
    ):
        BatchAdapter.from_dict({"input": torch.ones(1)})


def test_training_batch_to_device_with_none():
    """Test `.to()` handling `None` values gracefully."""
    batch = TrainingBatch(input=torch.ones(1), target=torch.ones(1), mask=None)
    device_batch = batch.to("cpu")

    assert device_batch.mask is None


# ==============================================================================
# Integration & E2E
# ==============================================================================


def test_batch_adapter_with_torchio_mock():
    """Simulate unpacking a dictionary coming from tio.SubjectsLoader."""
    input_t = torch.zeros(1, 1, 10, 10)
    target_t = torch.ones(1, 1, 10, 10)

    # TorchIO wraps arrays in a nested dictionary
    batch_data = {
        "input": {"data": input_t, "affine": torch.eye(4)},
        "target": {"data": target_t, "affine": torch.eye(4)},
        "subject_id": ["subj_1"],
    }

    batch = BatchAdapter.from_dict(batch_data)

    assert isinstance(batch, TrainingBatch)
    assert torch.equal(batch.input, input_t)
    assert torch.equal(batch.target, target_t)
    assert batch.mask is None
    assert batch.metadata["subject_id"] == ["subj_1"]


def test_torchio_image_to_device():
    """Test that TorchIO mock objects without `.to` are converted by extracting `.tensor`."""
    input_t = torch.zeros(2, 2)
    mock_tio_img = MockTorchIOImage(input_t)

    # We cheat the type hinting slightly for test purposes
    batch = TrainingBatch(input=mock_tio_img, target=torch.ones(2, 2))  # type: ignore

    # The `to` function should extract the `.tensor` attribute from `MockTorchIOImage`
    device_batch = batch.to("cpu")

    # The extracted object should now be a pure tensor, not a MockTorchIOImage
    assert isinstance(device_batch.input, torch.Tensor)
    assert torch.equal(device_batch.input, input_t)


# ==============================================================================
# Gradient Flow Audit (CRITICAL)
# ==============================================================================


def test_gradient_flow_preservation():
    """
    Test the backpropagation graph.
    Ensure that `TrainingBatch.to()` does not accidentally detach tensors that require gradients.
    """
    # Initialize batch where inputs require gradients
    input_t = torch.tensor([1.0, 2.0], requires_grad=True)
    target_t = torch.tensor([3.0, 4.0], requires_grad=True)

    batch = TrainingBatch(input=input_t, target=target_t)

    # Move batch to device (even if it's the same device, it recreates the batch)
    device_batch = batch.to("cpu")

    # Assert requires_grad is strictly preserved
    assert device_batch.input.requires_grad is True
    assert device_batch.target.requires_grad is True

    # Run a dummy backward pass to ensure graph is intact
    loss = (device_batch.input * 2).sum() + (device_batch.target * 3).sum()
    loss.backward()

    # Assert gradients exist and are correct (input grad = 2, target grad = 3)
    assert input_t.grad is not None
    assert target_t.grad is not None
    assert torch.all(input_t.grad == 2.0)
    assert torch.all(target_t.grad == 3.0)

    # Ensure no NaNs or Infs
    assert not torch.isnan(input_t.grad).any()
    assert not torch.isinf(input_t.grad).any()


def test_detached_flow_preservation():
    """
    Ensure detached tensors remain detached after batch operations.
    """
    input_t = torch.tensor([1.0, 2.0], requires_grad=False)
    target_t = torch.tensor([3.0, 4.0], requires_grad=False)

    batch = TrainingBatch(input=input_t, target=target_t)
    device_batch = batch.to("cpu")

    assert device_batch.input.requires_grad is False
    assert device_batch.target.requires_grad is False
