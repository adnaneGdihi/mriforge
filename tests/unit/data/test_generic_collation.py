import torch

from spectramr.data.collation.strategies import ImageCollateStrategy


class MockTioImage:
    def __init__(self, data, affine=None):
        self.data = data
        self.affine = affine or torch.eye(4)


def test_collate_generic_tensors():
    """Test collating generic dictionary keys."""
    strategy = ImageCollateStrategy()

    batch = [
        {"kspace": torch.randn(2, 64, 64, 1), "target": torch.randn(2, 64, 64, 1)},
        {"kspace": torch.randn(2, 64, 64, 1), "target": torch.randn(2, 64, 64, 1)},
    ]

    result = strategy.collate(batch)

    assert "kspace" in result
    assert "target" in result
    assert result["kspace"].shape == (2, 2, 64, 64, 1)


def test_collate_stacks_scalar_field_strength_target_to_batch():
    """A 0-dim ``field_strength_target`` (stamped by the nifti_paired path for the
    ULF→HF task2 port) must collate to a ``[B]`` tensor, since ulf_dps /
    monotone_field / field_conditioned_inr / generative_refiner / ulf_redegrad_tta
    do ``batch['field_strength_target'].reshape(-1)``. Pins that the generic loop
    stacks it exactly like the proven ``contrast_idx`` key."""
    strategy = ImageCollateStrategy()

    batch = [
        {
            "input": torch.randn(1, 8, 8, 1),
            "target": torch.randn(1, 8, 8, 1),
            "field_strength_target": torch.tensor(3.0),
        },
        {
            "input": torch.randn(1, 8, 8, 1),
            "target": torch.randn(1, 8, 8, 1),
            "field_strength_target": torch.tensor(3.0),
        },
    ]

    result = strategy.collate(batch)

    assert "field_strength_target" in result
    fst = result["field_strength_target"]
    assert isinstance(fst, torch.Tensor)
    assert fst.shape == (2,)
    assert fst.reshape(-1).tolist() == [3.0, 3.0]


def test_collate_torchio_images():
    """Test collating TorchIO Image objects."""
    strategy = ImageCollateStrategy()

    batch = [
        {"kspace": MockTioImage(torch.randn(2, 64, 64, 1))},
        {"kspace": MockTioImage(torch.randn(2, 64, 64, 1))},
    ]

    result = strategy.collate(batch)
    assert "kspace" in result
    assert isinstance(result["kspace"], torch.Tensor)
    assert result["kspace"].shape == (2, 2, 64, 64, 1)


def test_collate_mixed_types():
    """Test collating tensors mixed with metadata."""
    strategy = ImageCollateStrategy()

    batch = [
        {"image": torch.randn(1, 10, 10), "id": "sub1"},
        {"image": torch.randn(1, 10, 10), "id": "sub2"},
    ]

    result = strategy.collate(batch)
    assert "image" in result
    assert result["image"].shape == (2, 1, 10, 10)
    assert result["id"] == ["sub1", "sub2"]


def test_squeeze_depth():
    """Test squeezing depth dimension."""
    strategy = ImageCollateStrategy(squeeze_depth_dim=True)

    # (C, H, W, D=1)
    batch = [{"vol": torch.randn(1, 32, 32, 1)}, {"vol": torch.randn(1, 32, 32, 1)}]

    result = strategy.collate(batch)
    # Result should be (B, C, H, W) -> (2, 1, 32, 32)
    assert result["vol"].shape == (2, 1, 32, 32)
