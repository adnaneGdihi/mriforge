import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.append(str(Path(__file__).parents[2]))

from spectramr.data.transforms.graph_encoding import GraphBatchEncoder


def test_graph_training_logic():
    print(">>> Testing Graph Training Logic...")

    # Simulate TorchIO Batch (B, C, H, W)
    # C=2 for Complex
    B, C, H, W = 2, 2, 64, 64
    x = torch.randn(B, C, H, W)
    gt = torch.randn(B, C, H, W)

    print(f"    Input Shape: {x.shape}")

    encoder = GraphBatchEncoder()

    # 1. Input Path
    nodes_in = encoder(x)
    print(f"    Input Nodes: {nodes_in.shape}")  # Expect [2, 4096, 5]

    # 2. Target Path
    # We simulate what happens in train.py: Encode then slice
    nodes_gt_full = encoder(gt)
    target = nodes_gt_full[..., :2]  # We compare Signal only
    print(f"    Target Nodes: {target.shape}")  # Expect [2, 4096, 2]

    # 3. Simulate Model Prediction
    # Model takes [B, N, 5] and outputs [B, N, 2] (Signal Reconstruction)
    pred = torch.randn_like(target)

    # 4. Loss Calculation
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, target)
    print(f"    Loss calculated: {loss.item()}")

    # Assertions
    assert nodes_in.shape == (
        B,
        H * W,
        5,
    ), f"Input nodes shape incorrect: {nodes_in.shape}"
    assert target.shape == (
        B,
        H * W,
        2,
    ), f"Target nodes shape incorrect: {target.shape}"
    assert nodes_in.device == x.device, "Device mismatch"

    print(">>> SUCCESS: Graph training logic is valid.")


if __name__ == "__main__":
    test_graph_training_logic()
