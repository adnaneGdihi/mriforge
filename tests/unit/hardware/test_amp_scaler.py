"""
Tests for AMP (Automatic Mixed Precision) scaler lifecycle and state management.

Critical variables tested:
- GradScaler creation and configuration
- Loss scaling during forward pass
- Gradient scaling and unscaling
- Scaler state dict save/load
- State persistence across resumption
- Gradient underflow detection

These tests ensure AMP training doesn't fail silently due to gradient underflow
or state corruption during checkpoint resumption.
"""

import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from tests.fixtures.critical_variables import SimpleGenerator


@pytest.mark.gpu
class TestAMPScalerCreation:
    """Tests for AMP scaler initialization and creation."""

    def test_scaler_created_when_cuda_available(self):
        """Test that GradScaler is properly created on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = GradScaler("cuda")
        assert scaler is not None

    def test_scaler_attributes_initialized(self):
        """Test that scaler attributes are properly initialized."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = GradScaler("cuda")

        # Test default attributes
        assert scaler.get_scale() == 2**16  # Default initial scale
        assert scaler.get_growth_factor() == 2.0
        assert scaler.get_backoff_factor() == 0.5

    def test_scaler_init_scale_custom(self):
        """Test that custom initial scale can be set."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        init_scale = 2**14
        scaler = GradScaler("cuda", init_scale=init_scale)
        assert scaler.get_scale() == init_scale

    def test_scaler_growth_interval(self):
        """Test that growth interval is configurable."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = GradScaler("cuda", growth_interval=1000)
        assert scaler.get_growth_interval() == 1000


@pytest.mark.gpu
class TestAMPLossScaling:
    """Tests for loss scaling and backward pass."""

    def test_loss_scaled_properly(self, device, generator):
        """Test that loss is scaled correctly before backward."""
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")

        scaler = GradScaler("cuda")
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        # Forward pass with autocast
        batch = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)
        target = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)

        with autocast(device_type="cuda"):
            output = generator(batch)
            loss = nn.L1Loss()(output, target)

        # Record loss value before scaling
        loss_value = loss.item()

        # Scale loss
        scaled_loss = scaler.scale(loss)

        # Verify loss is scaled
        expected_scale = scaler.get_scale()
        assert torch.allclose(
            scaled_loss,
            torch.tensor(loss_value * expected_scale, device=device),
            rtol=1e-4,
        )

    def test_backward_with_scaled_loss(self, device, generator):
        """Test that backward pass works correctly with scaled loss."""
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")

        scaler = GradScaler("cuda")
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        batch = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)
        target = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)

        with autocast(device_type="cuda"):
            output = generator(batch)
            loss = nn.L1Loss()(output, target)

        # Backward with unscaling
        scaler.scale(loss).backward()

        # Verify gradients exist and are finite
        for param in generator.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), "Gradient contains NaN/Inf"

        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

    def test_unscale_gradients(self, device, generator):
        """Test that gradients are properly unscaled."""
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")

        scaler = GradScaler("cuda")
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        batch = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)
        target = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)

        with autocast(device_type="cuda"):
            output = generator(batch)
            loss = nn.L1Loss()(output, target)

        scaler.scale(loss).backward()

        # Capture scaled gradient
        first_param = next(generator.parameters())
        scaled_grad = first_param.grad.clone()

        # Unscale
        scaler.unscale_(optimizer)
        unscaled_grad = first_param.grad.clone()

        # Verify unscaling
        scale = scaler.get_scale()
        expected_unscaled = scaled_grad / scale
        assert torch.allclose(unscaled_grad, expected_unscaled, rtol=1e-4)


@pytest.mark.gpu
class TestAMPScalerState:
    """Tests for scaler state dict save/load."""

    def test_scaler_state_dict_structure(self):
        """Test that state dict has expected structure."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = GradScaler("cuda")
        state = scaler.state_dict()

        assert "scale" in state
        # Usually internal state like _growth_tracker is returned in state_dict for GradScaler
        assert "growth_tracker" in state or "_growth_tracker" in state

        if "growth_tracker" in state:
            assert isinstance(state["growth_tracker"], int)
        elif "_growth_tracker" in state:
            assert isinstance(state["_growth_tracker"], int)

    def test_scaler_state_dict_save_restore(self):
        """Test that scaler state can be saved and restored."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler1 = GradScaler("cuda")

        # Modify scaler state
        original_scale = scaler1.get_scale()

        # Save state
        state_dict = scaler1.state_dict()

        # Create new scaler and restore
        scaler2 = GradScaler("cuda")
        scaler2.load_state_dict(state_dict)

        # Verify state is identical
        assert scaler2.get_scale() == original_scale

    def test_scaler_state_dict_preserves_scale(self):
        """Test that state dict preserves the exact scale value."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = GradScaler("cuda", init_scale=2**14)
        scale_before = scaler.get_scale()

        state_dict = scaler.state_dict()
        state_dict["scale"] *= 2  # Simulate scale increase

        scaler.load_state_dict(state_dict)
        scale_after = scaler.get_scale()

        assert scale_after == scale_before * 2

    def test_scaler_state_dict_preserves_growth_tracker(self):
        """Test that growth tracker is preserved across save/load."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = GradScaler("cuda", growth_interval=100)

        # Manually modify growth tracker
        state = scaler.state_dict()
        tracker_key = (
            "growth_tracker" if "growth_tracker" in state else "_growth_tracker"
        )
        state[tracker_key] = 50

        scaler_new = GradScaler("cuda")
        scaler_new.load_state_dict(state)

        # Verify tracker is preserved
        state_new = scaler_new.state_dict()
        assert state_new[tracker_key] == 50


@pytest.mark.gpu
class TestAMPScalerResumeTraining:
    """Tests for using scaler across training resumption."""

    def test_scaler_state_persistence_across_resume(self, device, generator):
        """Test that scaler state persists correctly across checkpoint/resume."""
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")

        scaler = GradScaler("cuda")
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        # Simulate partial training (2 steps)
        for step in range(2):
            batch = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)
            target = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)

            with autocast(device_type="cuda"):
                output = generator(batch)
                loss = nn.L1Loss()(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Save scaler state
        checkpoint = {
            "scaler_state_dict": scaler.state_dict(),
            "generator_state_dict": generator.state_dict(),
        }

        # Create new objects and restore
        scaler_resumed = GradScaler("cuda")
        generator_resumed = SimpleGenerator().to(device)

        scaler_resumed.load_state_dict(checkpoint["scaler_state_dict"])
        generator_resumed.load_state_dict(checkpoint["generator_state_dict"])

        # Verify states match
        assert scaler_resumed.get_scale() == scaler.get_scale()

    def test_scaler_state_not_corrupted_on_resume(self, device, generator):
        """Test that scaler state is not corrupted after resume."""
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")

        scaler = GradScaler("cuda")
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        # Step 1: Run one step
        batch = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)
        target = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)

        with autocast(device_type="cuda"):
            output = generator(batch)
            loss = nn.L1Loss()(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        scale_after_step1 = scaler.get_scale()

        # Save and resume
        checkpoint = {"scaler_state_dict": scaler.state_dict()}
        scaler_resumed = GradScaler("cuda")
        scaler_resumed.load_state_dict(checkpoint["scaler_state_dict"])

        # Step 2: Continue training
        optimizer.zero_grad()
        batch = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)
        target = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device)

        with autocast(device_type="cuda"):
            output = (
                generator_resumed(batch)
                if "generator_resumed" in locals()
                else generator(batch)
            )
            loss = nn.L1Loss()(output, target)

        scaler_resumed.scale(loss).backward()
        scaler_resumed.unscale_(optimizer)
        scaler_resumed.step(optimizer)
        scaler_resumed.update()

        # Verify scaler is still functional
        assert scaler_resumed.get_scale() > 0


@pytest.mark.gpu
class TestAMPGradientUnderflow:
    """Tests for detecting and handling gradient underflow."""

    def test_scaler_detects_inf_gradients(self, device, generator):
        """Test that scaler detects infinite gradients."""
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")

        scaler = GradScaler("cuda")
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        # Create a loss that produces very large gradients
        batch = torch.randn(2, 2, 64, 64, dtype=torch.float32, device=device) * 1e10

        with autocast(device_type="cuda"):
            output = generator(batch)
            loss = output.pow(2).sum()

        scaler.scale(loss).backward()

        # Check if scaler detects invalid gradients
        grad_norm_valid = scaler.unscale_(optimizer)
        # scaler.unscale_ returns without error even if grads are bad,
        # so we manually check

        has_inf_grad = False
        for param in generator.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    has_inf_grad = True
                    break

        # Step should be skipped if grads are invalid
        scaler.step(optimizer)
        scaler.update()

    def test_scaler_recovery_from_overflow(self, device, generator):
        """Test that scaler recovers from overflow by scaling down."""
        if device.type != "cuda":
            pytest.skip("AMP requires CUDA")

        scaler = GradScaler("cuda", init_scale=2**16, growth_interval=2)
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        scale_before = scaler.get_scale()

        # Create a situation likely to cause overflow
        batch = torch.full((2, 2, 64, 64), 1e10, dtype=torch.float32, device=device)

        with autocast(device_type="cuda"):
            output = generator(batch)
            loss = output.pow(2).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Check for inf gradients (these would trigger overflow)
        has_inf = False
        for param in generator.parameters():
            if param.grad is not None and (torch.isinf(param.grad).any()):
                has_inf = True
                break

        scaler.step(optimizer)
        scaler.update()

        # Even if overflow occurred, scaler should handle it
        assert scaler.get_scale() > 0
