import torch
import torchio as tio

from mriforge.data.transforms.slab_transforms import SelectMiddleSlice, SlabToChannel


def test_slab_to_channel():
    # Create a 3D subject: 1 channel, 10x10 spatial, Depth 3
    # TorchIO data convention: [C, W, H, D]
    tensor = torch.randn(1, 10, 10, 3)
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=tensor),
        target=tio.ScalarImage(tensor=tensor.clone()),
    )

    transform = SlabToChannel(include=("input",))
    transformed = transform(subject)

    # Input should be flattened: [1*3, 10, 10, 1]
    assert transformed["input"].data.shape == (3, 10, 10, 1)

    # Target should be unchanged: [1, 10, 10, 3]
    assert transformed["target"].data.shape == (1, 10, 10, 3)

    # Check values preserved
    # Original: [1, 10, 10, 3] -> Permute(0,3,1,2) -> [1, 3, 10, 10] -> Reshape -> [3, 10, 10]
    expected_slice0 = tensor[0, :, :, 0]
    result_slice0 = transformed["input"].data[0, :, :, 0]
    assert torch.allclose(expected_slice0, result_slice0)


def test_select_middle_slice():
    # Depth 3
    tensor = torch.randn(1, 10, 10, 3)
    # Mark middle slice distinctively
    tensor[0, :, :, 1] = 100.0

    subject = tio.Subject(
        input=tio.ScalarImage(tensor=tensor), target=tio.ScalarImage(tensor=tensor)
    )

    transform = SelectMiddleSlice(include=("target",))
    transformed = transform(subject)

    # Target should be [1, 10, 10, 1]
    assert transformed["target"].data.shape == (1, 10, 10, 1)

    # Value should be 100
    assert torch.allclose(transformed["target"].data.mean(), torch.tensor(100.0))

    # Input unchanged
    assert transformed["input"].data.shape == (1, 10, 10, 3)


if __name__ == "__main__":
    test_slab_to_channel()
    test_select_middle_slice()
    print("All slab transform tests passed!")
