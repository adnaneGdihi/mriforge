import torch

from mriforge.models.losses.histogram_loss import HistogramConsistencyLoss


def test_histogram_loss_optimization():
    # Setup
    loss_fn = HistogramConsistencyLoss(bins=50, min_val=-2.0, max_val=2.0, sigma=0.1)

    # Target distribution: Gaussian centered at 0.5
    torch.manual_seed(42)
    target = torch.randn(10, 1, 32, 32) + 0.5

    # Input distribution: Gaussian centered at -0.5 (learnable)
    input_param = torch.randn(10, 1, 32, 32) - 0.5
    input_param.requires_grad = True

    optimizer = torch.optim.SGD([input_param], lr=1.0)

    initial_mean = input_param.mean().item()
    print(f"Initial Mean: {initial_mean:.4f}")

    initial_loss = None
    # Optimization Loop
    for i in range(200):
        optimizer.zero_grad()
        loss = loss_fn(input_param, target)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()

        if input_param.grad is None:
            print("Grad is None!")
        else:
            grad_norm = input_param.grad.norm().item()
            if i % 10 == 0:
                print(
                    f"Step {i}, Loss: {loss.item():.4f}, Mean: {input_param.mean().item():.4f}, GradNorm: {grad_norm:.6f}"
                )

        optimizer.step()

    # Verification: Mean should shift towards target (0.5)
    final_mean = input_param.mean().item()
    print(f"Final Mean: {final_mean:.4f}")

    # Assert movement: final mean should be greater than initial mean (shifted toward target 0.5)
    assert final_mean > initial_mean, "Mean did not shift positively towards target"
    assert loss.item() < 1.0, "Loss did not decrease significantly"

    print("Histogram Loss gradient flow verified!")


if __name__ == "__main__":
    try:
        test_histogram_loss_optimization()
    except Exception as e:
        print(f"Test Failed: {e}")
        exit(1)
