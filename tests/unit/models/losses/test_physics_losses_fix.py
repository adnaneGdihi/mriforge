import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.physics_losses import (
    ParallelImagingKSpaceLoss,
    PhysicsConstraintLoss,
    PhysicsInformedLoss,
)


class TestPhysicsLossesFix:
    def test_kspace_loss_centered(self):
        """Verify that the loss generates CENTERED K-space (fft2c) matches."""
        loss_fn = ParallelImagingKSpaceLoss(fft_transformer=None)

        B, C, H, W = 1, 1, 32, 32
        img = torch.randn(B, C, H, W, dtype=torch.complex64)

        # Ground truth K-space: Computed via fft2c
        # If we computed it via fft2(uncentered), it would be shifted
        gt_kspace = fft2c(img)

        # If loss_fn uses fft2c internally, loss should be 0
        loss = loss_fn(img, gt_kspace)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

        # Verification: If we passed uncentered kspace, loss should be high
        wrong_kspace = torch.fft.fft2(img, norm="ortho")
        loss_wrong = loss_fn(img, wrong_kspace)
        # They are different by shift.
        assert loss_wrong > 0.1

    def test_parallel_imaging_loss_centered(self):
        """Verify parallel imaging path uses sense_forward (centered)."""
        loss_fn = ParallelImagingKSpaceLoss(fft_transformer=None)

        B, H, W = 1, 16, 16
        imgs = torch.randn(B, 1, H, W, dtype=torch.complex64)
        smaps = torch.randn(B, 2, H, W, dtype=torch.complex64)

        # Expected: fft2c(img * smap)
        coil_imgs = imgs * smaps
        gt_kspace = fft2c(coil_imgs)

        loss = loss_fn(imgs, gt_kspace, coil_sensitivities=smaps)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestPhysicsConstraintLossGradient:
    def test_total_loss_is_differentiable(self):
        """The accumulated total must carry gradient.

        Pre-fix ``total_loss += (...).item()`` produced a gradient-free constant,
        so ``backward()`` either errored or trained on nothing.
        """
        loss_fn = PhysicsConstraintLoss(
            smoothness_weight=0.1, energy_weight=0.1, kspace_weight=0.0
        )
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 1, 16, 16)

        total, metrics = loss_fn(pred, target)
        assert total.requires_grad and total.grad_fn is not None
        total.backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0
        assert "loss/physics_smoothness" in metrics


class TestPhysicsInformedLossRegularization:
    def test_multichannel_tv_and_box_blur_run(self):
        """The TV + depthwise box-blur regulariser must run for C > 1.

        Pre-fix the grouped conv used a ``(1, C, k, k)`` kernel, which raises for
        ``C > 1`` (out_channels must be divisible by groups). The fix uses
        ``(C, 1, k, k)``. Also exercises the TV along the H and W spatial axes.
        """
        loss_fn = PhysicsInformedLoss(lambda_l2=1.0, lambda_dc=0.0, lambda_reg=0.1)
        pred = torch.randn(2, 2, 16, 16, requires_grad=True)  # C = 2
        target = torch.randn(2, 2, 16, 16)

        losses = loss_fn(pred, target)
        assert "regularization" in losses
        total = losses["total"]
        assert total.requires_grad
        total.backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0
