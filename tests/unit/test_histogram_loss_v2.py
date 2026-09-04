import torch

from spectramr.models.losses.histogram_loss import HistogramConsistencyLoss


def test_histogram_loss_optimization_strong():
    # Setup with wider sigma for better gradients
    # Range [-2, 2]. Distance 1.0.
    loss_fn = HistogramConsistencyLoss(bins=50, min_val=-2.0, max_val=2.0, sigma=0.5)

    # Target: 0.5
    torch.manual_seed(42)
    target = torch.randn(10, 1, 32, 32) + 0.5

    # Input: -0.5
    input_param = torch.randn(10, 1, 32, 32) - 0.5
    input_param.requires_grad = True

    initial_loss = None
    # High LR to overcome mean reduction
    optimizer = torch.optim.SGD([input_param], lr=20.0)

    print(f"Initial Mean: {input_param.mean().item():.4f}")

    for i in range(50):
        optimizer.zero_grad()
        loss = loss_fn(input_param, target)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Step {i}, Loss: {loss.item():.4f}, Mean: {input_param.mean().item():.4f}"
            )

    final_mean = input_param.mean().item()
    print(f"Final Mean: {final_mean:.4f}")

    # Verify: loss decreases, confirming the loss is differentiable and functional
    assert (
        loss.item() < initial_loss
    ), f"Loss did not decrease: {initial_loss:.4f} -> {loss.item():.4f}"
    # Verify gradients flow (loss is non-zero and differentiable)
    assert loss.item() < 1.0, f"Loss {loss.item()} unexpectedly high"
    print("Optimization confirmed!")


if __name__ == "__main__":
    try:
        test_histogram_loss_optimization_strong()
    except Exception as e:
        print(f"Test Failed: {e}")
        exit(1)
