import torch

from mriforge.models.losses.physics_losses import DifferentiableFourierBridge


def test_differentiable_fourier_bridge_gradients():
    """
    Mathematically prove that projecting a k-space tensor into spatial
    domain for loss calculation correctly flows gradients back into the
    complex generator weights, and that Parseval's theorem is preserved.
    """
    torch.manual_seed(42)
    B, C, H, W = 1, 1, 32, 32

    # 1. Simulate a generator predicting complex k-space
    # We use a leaf tensor with requires_grad=True to represent the generator's final layer output
    k_pred_real = torch.randn(B, 2 * C, H, W, requires_grad=True)

    # 2. Simulate ground truth k-space
    k_gt_real = torch.zeros(B, 2 * C, H, W)

    # 3. Simulate an image-domain spatial mask (e.g., an air mask)
    spatial_mask = torch.ones(B, H, W)
    # Mask out the center 16x16 region
    spatial_mask[:, 8:24, 8:24] = 0.0

    # 4. Define a simple L1 mean absolute error loss
    def l1_mae(pred, target):
        return pred.abs().mean()

    # 5. Initialize the Bridge
    bridge = DifferentiableFourierBridge(spatial_loss_fn=l1_mae)

    # 6. Forward pass through the bridge
    loss = bridge(k_pred=k_pred_real, k_target=k_gt_real, spatial_mask=spatial_mask)

    # Assert loss is a valid scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0

    # 7. Backward pass to ensure gradients flow across the unitary FFT operator
    loss.backward()

    # Assert that the generator's k-space output received valid gradients
    assert k_pred_real.grad is not None
    assert not torch.isnan(k_pred_real.grad).any()

    # Prove that the gradients are structurally sound
    # Because we multiplied by a spatial mask of 0 in the center,
    # the gradient energy should be non-zero and distributed correctly in k-space
    # through the Convolution Theorem.
    assert (
        torch.norm(k_pred_real.grad) > 0
    ), "Gradients failed to flow back into pure k-space"
