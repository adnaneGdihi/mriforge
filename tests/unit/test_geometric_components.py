import torch

from mriforge.models.geometric.b0_estimator import B0Estimator
from mriforge.models.layers.spatial_transformer import SpatialTransformer
from mriforge.models.losses.registration import LocalCrossCorrelationLoss, SmoothnessLoss


def test_spatial_transformer_identity():
    """Test that zero flow results in identity transformation."""
    B, C, H, W = 1, 1, 32, 32
    stn = SpatialTransformer((H, W))

    # Create a grid image
    img = torch.zeros(B, C, H, W)
    img[:, :, 10:20, 10:20] = 1.0

    # Zero flow
    flow = torch.zeros(B, 2, H, W)

    warped = stn(img, flow)

    assert torch.allclose(img, warped, atol=1e-5)


def test_spatial_transformer_shift():
    """Test that constant flow shifts the image."""
    B, C, H, W = 1, 1, 32, 32
    stn = SpatialTransformer((H, W))

    img = torch.zeros(B, C, H, W)
    img[:, :, 16, 16] = 1.0  # Center pixel

    # Shift by 1 pixel in x and y (normalized coords are tricky, let's just check non-identity)
    # Flow of 2/(size-1) corresponds to 1 pixel shift in grid_sample [-1, 1] space
    shift_val = 2.0 / (32 - 1)
    flow = torch.ones(B, 2, H, W) * shift_val

    warped = stn(img, flow)

    # The peak should have moved
    assert warped[0, 0, 16, 16] < 0.9  # Should be interpolated/moved away
    # Should maintain approx energy (not strictly true for border, but inside ok)
    assert warped.sum() > 0.0


def test_b0_estimator_shapes():
    """Test B0Estimator input/output shapes."""
    B, C, H, W = 2, 1, 64, 64
    model = B0Estimator(input_shape=(H, W), input_channels=2)

    moving = torch.randn(B, 1, H, W)
    fixed = torch.randn(B, 1, H, W)

    corrected, flow = model(moving, fixed)

    assert corrected.shape == (B, 1, H, W)
    assert flow.shape == (B, 2, H, W)  # 2 channels for 2D flow


def test_lncc_loss():
    """Test Local Cross Correlation Loss."""
    loss_fn = LocalCrossCorrelationLoss(kernel_size=5)

    # Perfect correlation (identical images)
    I = torch.randn(1, 1, 30, 30)
    J = I.clone()

    loss = loss_fn(I, J)
    # LNCC is negative correlation. Perfect correlation = 1.0. Loss = -1.0
    # Note: padding might affect border pixels, so close to -1
    assert -1.0 <= loss.item() <= -0.8

    # Anticorrelation (I = -J)
    loss_anti = loss_fn(I, -J)
    # Squared correlation, so (-1)^2 = 1. Loss should still be ~ -1
    assert -1.0 <= loss_anti.item() <= -0.8

    # Uncorrelated
    K = torch.randn(1, 1, 30, 30)
    loss_rand = loss_fn(I, K)
    # Random images -> correlation near 0. Loss near 0.
    assert -0.2 <= loss_rand.item() <= 0.2


def test_smoothness_loss():
    """Test Smoothness Loss."""
    loss_fn = SmoothnessLoss()

    # Smooth flow (constant) -> 0 loss
    flow_smooth = torch.ones(1, 2, 32, 32)
    assert loss_fn(flow_smooth).item() == 0.0

    # Jagged flow
    flow_jagged = torch.rand(1, 2, 32, 32)
    assert loss_fn(flow_jagged).item() > 0.0
