import torch
import torchio as tio

from spectramr.data.builders.torchio_queue_builder import (
    TorchIOQueueBuilder,
    TorchIOQueueConfig,
)


def test_builder_selects_slab_strategy():
    # Config with slab mode enabled
    config = TorchIOQueueConfig(
        queue_length=10,
        samples_per_volume=2,
        patch_size=(10, 10, 3),  # 3D patch
        batch_size=2,
        enable_slab_mode=True,
    )

    # Dummy dataset
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=torch.randn(1, 20, 20, 5)),  # [C, W, H, D]
        target=tio.ScalarImage(tensor=torch.randn(1, 20, 20, 5)),
    )
    dataset = tio.SubjectsDataset([subject])

    # Build queue
    queue, loader = TorchIOQueueBuilder.build_train_queue(dataset, config)

    # Verify collate_fn is wrapped or is the strategy method
    # In build_train_queue: collate_fn=collate_strategy.collate
    # We can't easily inspect the instance bound method class, but we can test behavior.

    # Fetch a batch
    batch = next(iter(loader))

    # Check shapes
    # Input should be flattened: [B, C*D, H, W] -> [2, 1*3, 10, 10] = [2, 3, 10, 10]
    assert batch["input"].shape == (2, 3, 10, 10)

    # Target should be middle slice: [B, C, H, W] -> [2, 1, 10, 10]
    assert batch["target"].shape == (2, 1, 10, 10)

    print("Slab mode collation verified successfully!")


def test_builder_defaults_to_image_strategy():
    # Config with slab mode DISABLED
    config = TorchIOQueueConfig(
        queue_length=10,
        samples_per_volume=2,
        patch_size=(10, 10, 3),  # 3D patch
        batch_size=2,
        enable_slab_mode=False,
    )

    subject = tio.Subject(
        input=tio.ScalarImage(tensor=torch.randn(1, 20, 20, 5)),
        target=tio.ScalarImage(tensor=torch.randn(1, 20, 20, 5)),
    )
    dataset = tio.SubjectsDataset([subject])

    queue, loader = TorchIOQueueBuilder.build_train_queue(dataset, config)

    batch = next(iter(loader))

    # Standard 3D behavior (no squeeze depth because D=3)
    # [2, 1, 10, 10, 3]
    assert batch["input"].shape == (2, 1, 10, 10, 3)
    assert batch["target"].shape == (2, 1, 10, 10, 3)

    print("Standard image mode collation verified!")


if __name__ == "__main__":
    try:
        test_builder_selects_slab_strategy()
        test_builder_defaults_to_image_strategy()
        print("All tests passed.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
