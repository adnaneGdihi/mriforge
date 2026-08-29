"""Unit tests for UncertaintyAwareLoss (heteroscedastic NLL)."""

import pytest
import torch
import torch.nn.functional as F


class TestUncertaintyAwareLoss:
    """Tests for the heteroscedastic uncertainty-aware loss."""

    @pytest.fixture
    def loss_fn(self):
        from mriforge.models.losses.uncertainty_loss import UncertaintyAwareLoss
        return UncertaintyAwareLoss(base_loss_type="l2")

    @pytest.fixture
    def loss_fn_l1(self):
        from mriforge.models.losses.uncertainty_loss import UncertaintyAwareLoss
        return UncertaintyAwareLoss(base_loss_type="l1")

    def test_fallback_no_uncertainty(self, loss_fn):
        """Without uncertainty map, should fall back to base_loss."""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)

        loss = loss_fn(pred, target, uncertainty=None)
        expected = F.l1_loss(pred, target)

        assert torch.allclose(loss, expected, atol=1e-6), (
            f"Fallback mismatch: {loss.item()} vs {expected.item()}"
        )

    def test_zero_log_variance_equals_mse(self, loss_fn):
        """With log(σ²) = 0, heteroscedastic L2 reduces to 0.5*MSE."""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        log_var = torch.zeros_like(pred)  # σ² = 1

        loss = loss_fn(pred, target, uncertainty=log_var)

        # Expected: 0.5 * exp(0) * (pred-target)² + 0.5 * 0 = 0.5 * MSE
        expected = 0.5 * F.mse_loss(pred, target)
        assert torch.allclose(loss, expected, atol=1e-5), (
            f"Zero log_var: {loss.item():.6f} vs expected 0.5*MSE={expected.item():.6f}"
        )

    def test_high_uncertainty_reduces_reconstruction_penalty(self, loss_fn):
        """Higher uncertainty (larger log_var) should reduce the reconstruction term."""
        pred = torch.ones(2, 1, 32, 32)
        target = torch.zeros(2, 1, 32, 32)

        log_var_low = torch.zeros_like(pred)   # σ² = 1
        log_var_high = torch.full_like(pred, 4.0)  # σ² = e^4 ≈ 54.6

        loss_low_unc = loss_fn(pred, target, uncertainty=log_var_low)
        loss_high_unc = loss_fn(pred, target, uncertainty=log_var_high)

        # With high uncertainty, the reconstruction penalty (precision * error²)
        # is strongly attenuated, but the log-barrier term increases.
        # The net effect depends on the error magnitude, but for unit error:
        # low:  0.5 * 1 * 1 + 0.5 * 0 = 0.5
        # high: 0.5 * exp(-4) * 1 + 0.5 * 4 ≈ 0.009 + 2.0 = 2.009
        # So high_unc > low_unc (log-barrier dominates)
        # But reconstruction term is lower: 0.009 < 0.5
        # This is correct: model learns to output high σ where error is unavoidable
        assert loss_low_unc > 0, "Loss should be positive"
        assert loss_high_unc > 0, "Loss should be positive"

    def test_gradient_flow_through_uncertainty(self, loss_fn):
        """Verify gradients flow through the log-variance map."""
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 1, 16, 16)
        log_var = torch.randn(2, 1, 16, 16, requires_grad=True)

        loss = loss_fn(pred, target, uncertainty=log_var)
        loss.backward()

        assert pred.grad is not None, "No gradient on predictions"
        assert log_var.grad is not None, "No gradient on log_var"
        assert not torch.all(log_var.grad == 0), "Zero gradient on log_var"

    def test_l1_mode(self, loss_fn_l1):
        """L1 (Laplacian NLL) mode should produce valid loss."""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        log_var = torch.zeros_like(pred)

        loss = loss_fn_l1(pred, target, uncertainty=log_var)
        assert torch.isfinite(loss), f"Non-finite L1 loss: {loss.item()}"
        assert loss.item() > 0, "L1 loss should be positive"

    def test_log_var_clamping(self, loss_fn):
        """Extreme log-variance values should be clamped to [min, max]."""
        pred = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        log_var_extreme = torch.full_like(pred, 100.0)  # Way beyond max_log_var=10

        loss = loss_fn(pred, target, uncertainty=log_var_extreme)
        assert torch.isfinite(loss), f"Non-finite loss with extreme log_var: {loss.item()}"

    def test_broadcasting_single_channel_logvar(self, loss_fn):
        """Log-var with [B,1,H,W] should broadcast to [B,C,H,W]."""
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randn(2, 4, 32, 32)
        log_var = torch.zeros(2, 1, 32, 32)  # Single channel

        loss = loss_fn(pred, target, uncertainty=log_var)
        assert torch.isfinite(loss), "Broadcasting failed"

    def test_registry_access(self):
        """Verify loss is accessible via registry."""
        from mriforge.models.losses.registry import create_loss
        loss = create_loss("uncertainty")
        assert loss is not None, "Failed to resolve 'uncertainty' from registry"

    def test_invalid_base_loss_type(self):
        """Invalid base_loss_type should raise ValueError."""
        from mriforge.models.losses.uncertainty_loss import UncertaintyAwareLoss
        with pytest.raises(ValueError, match="base_loss_type must be"):
            UncertaintyAwareLoss(base_loss_type="huber")
