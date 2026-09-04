import torch

from spectramr.data.collation.strategies import SlabCollateStrategy


def test_slab_collation_logic():
    # Create dummy 2.5D patches: [C, H, W, D] (TorchIO usually gives 4D for ScalarImage data)
    # But wait, ImageCollateStrategy receives a list of Subjects or Dicts.
    # The elements in the dict are Tensors or Images.
    # CollateStrategy extracts tensors.

    # Simulating a batch of 2 samples
    # Shape: [C=1, D=3, H=10, W=10]? No, TorchIO convention: [C, W, H, D]
    # But let's check what our builder does:
    # "squeeze_depth_dim=False" means we keep D.

    # Assume we get tensors [C, H, W, D] or [C, D, H, W]?
    # TorchIO `ScalarImage.data` is [C, W, H, D] usually.
    # But let's assume standard PyTorch [C, D, H, W] for our logic check if that's what we expect.
    # Wait, SlabCollateStrategy logic:
    # inp_perm = inp.permute(0, 1, 4, 2, 3)
    # Logic implies inp is [B, C, H, W, D].
    # So we expect D to be the LAST dimension.

    # Let's create input tensors of shape [1, 10, 10, 3] (C,H,W,D)
    t1 = torch.randn(1, 10, 10, 3)
    t2 = torch.randn(1, 10, 10, 3)

    batch = [{"input": t1, "target": t1.clone()}, {"input": t2, "target": t2.clone()}]

    strategy = SlabCollateStrategy(
        padding_mode="zeros", flat_input=True, target_mode="middle"
    )

    collated = strategy.collate(batch)

    # Input Processing
    # [2, 1, 10, 10, 3] -> [2, 1, 3, 10, 10] (permute 0,1,4,2,3) -> [2, 3, 10, 10] (reshape)
    assert "input" in collated
    inp = collated["input"]
    print(f"Input shape: {inp.shape}")

    assert inp.shape == (2, 3, 10, 10)  # B, C*D, H, W

    # Verify values for first sample
    # Original t1: [1, 10, 10, 3]. D is last.
    # Flattened: C=0, D=0 -> Channel 0. D=1 -> Channel 1.
    # Let's check slice 0 (D=0)
    # t1[0, :, :, 0] should equal inp[0, 0, :, :]
    assert torch.allclose(t1[0, :, :, 0], inp[0, 0, :, :])
    assert torch.allclose(t1[0, :, :, 1], inp[0, 1, :, :])
    assert torch.allclose(t1[0, :, :, 2], inp[0, 2, :, :])

    # Target Processing (middle slice)
    # [2, 1, 10, 10, 3] -> pick index 1 at dim -1 -> [2, 1, 10, 10]
    tgt = collated["target"]
    print(f"Target shape: {tgt.shape}")
    assert tgt.shape == (2, 1, 10, 10)

    # Verify value: middle slice (index 1)
    assert torch.allclose(t1[0, :, :, 1], tgt[0, 0, :, :])

    print("Slab collation logic verified!")


if __name__ == "__main__":
    test_slab_collation_logic()
