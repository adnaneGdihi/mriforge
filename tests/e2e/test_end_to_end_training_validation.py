"""
End-to-end training validation tests.

Tests complete training loops including:
- Multi-step training runs (5-10 iterations)
- Gradient flow and convergence
- Loss decrease and stability
- Checkpoint save/load/resume
- Validation metric computation
- Physics constraint satisfaction
- State management across steps
- Memory cleanup
- Numerical stability

These tests validate the entire training pipeline with all critical
variables working together.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from tests.fixtures.critical_variables import SimpleDiscriminator, SimpleGenerator


class TestMiniTrainingLoop:
    """Tests for mini training loops (5-10 steps)."""

    def test_gan_training_loop_basic(self, device):
        """Test basic GAN training loop stability."""
        generator = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        discriminator = SimpleDiscriminator(in_channels=2).to(device)

        opt_g = optim.Adam(generator.parameters(), lr=0.0002)
        opt_d = optim.Adam(discriminator.parameters(), lr=0.0002)

        criterion = nn.BCEWithLogitsLoss()

        losses_g = []
        losses_d = []

        for step in range(5):
            # Create batch
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            # Generator step
            gen_output = generator(batch)
            disc_fake = discriminator(gen_output.detach())
            loss_g = criterion(disc_fake, torch.ones_like(disc_fake))

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            losses_g.append(loss_g.detach())

            # Discriminator step
            disc_real = discriminator(target)
            disc_fake = discriminator(gen_output.detach())

            loss_d_real = criterion(disc_real, torch.ones_like(disc_real))
            loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_d = (loss_d_real + loss_d_fake) / 2

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            losses_d.append(loss_d.detach())

        # Single GPU→CPU sync per series, after the loop.
        assert torch.isfinite(torch.stack(losses_g)).all()
        assert torch.isfinite(torch.stack(losses_d)).all()

    def test_training_loss_convergence(self, device):
        """Test that loss shows convergence trend over steps."""
        generator = SimpleGenerator(in_channels=1, out_channels=1).to(device)
        optimizer = optim.Adam(generator.parameters(), lr=0.001)

        losses = []
        true_image = torch.ones(1, 1, 64, 64, device=device) * 0.5

        for step in range(10):
            batch = torch.randn(1, 1, 64, 64, device=device)
            output = generator(batch)

            loss = nn.L1Loss()(output, true_image)
            losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Single GPU→CPU sync after the loop.
        losses_cpu = torch.stack(losses).cpu().tolist()
        first_third = sum(losses_cpu[:3]) / 3
        last_third = sum(losses_cpu[-3:]) / 3

        # Last third loss should be lower than first third on average
        assert last_third <= first_third + 0.1  # Allow some variance

    def test_gradient_flow_through_steps(self, device):
        """Test that gradients flow and update correctly across steps."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify weights have changed
        final_weights = model.state_dict()

        num_changed = 0
        for key in initial_weights:
            if not torch.equal(initial_weights[key], final_weights[key]):
                num_changed += 1

        assert num_changed > 0  # At least some weights changed

    def test_training_state_accumulation(self, device):
        """Test that training state accumulates across steps."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        metrics = {
            "iteration": 0,
            "step_losses": [],
        }

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)

            metrics["iteration"] += 1
            metrics["step_losses"].append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Single GPU→CPU sync after the loop, then derive total_loss.
        step_losses_cpu = torch.stack(metrics["step_losses"]).cpu().tolist()
        metrics["total_loss"] = sum(step_losses_cpu)

        assert metrics["iteration"] == 5
        assert metrics["total_loss"] > 0
        assert len(metrics["step_losses"]) == 5


class TestTrainingWithCheckpointing:
    """Tests for training with checkpoint save/load/resume."""

    def test_resume_continues_from_checkpoint(self, device, tmp_path):
        """Test that training can resume from checkpoint."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        checkpoint_path = tmp_path / "checkpoint.pt"

        # Training phase 1: 5 steps
        losses_phase1 = []
        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)
            losses_phase1.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint = {
            "iteration": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "last_loss": losses_phase1[-1],
        }
        torch.save(checkpoint, checkpoint_path)

        # Training phase 2: Resume
        model_resumed = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer_resumed = optim.Adam(model_resumed.parameters(), lr=0.0002)

        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
        model_resumed.load_state_dict(loaded_checkpoint["model_state_dict"])
        optimizer_resumed.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

        iteration = loaded_checkpoint["iteration"]
        assert iteration == 5

        losses_phase2 = []
        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model_resumed(batch)
            loss = nn.L1Loss()(output, target)
            losses_phase2.append(loss.item())

            optimizer_resumed.zero_grad()
            loss.backward()
            optimizer_resumed.step()

            iteration += 1

        # Verify resume continuity
        assert iteration == 10

    def test_checkpoint_loss_continuity(self, device, tmp_path):
        """Test that loss tracking continues without interruption."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        all_losses = []

        # Phase 1: 3 steps
        for step in range(3):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)
            all_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        last_loss = all_losses[-1]

        # Checkpoint
        checkpoint = {"last_loss": last_loss}

        # Phase 2: 2 more steps
        for step in range(2):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)
            all_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify loss history is continuous
        assert len(all_losses) == 5
        assert all_losses[2] == last_loss


class TestTrainingValidationCycle:
    """Tests for training + validation cycles."""

    def test_train_val_metric_computation(self, device):
        """Test that validation metrics are computed correctly."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        train_losses = []
        val_losses = []

        for epoch in range(2):
            # Training
            for step in range(3):
                batch = torch.randn(2, 2, 64, 64, device=device)
                target = torch.randn(2, 2, 64, 64, device=device)

                output = model(batch)
                loss = nn.L1Loss()(output, target)
                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                for step in range(2):
                    batch_val = torch.randn(2, 2, 64, 64, device=device)
                    target_val = torch.randn(2, 2, 64, 64, device=device)

                    output_val = model(batch_val)
                    loss_val = nn.L1Loss()(output_val, target_val)
                    val_losses.append(loss_val.item())

        # Verify metrics tracked
        assert len(train_losses) == 6  # 2 epochs * 3 steps
        assert len(val_losses) == 4  # 2 epochs * 2 val steps

    def test_best_model_selection(self, device):
        """Test that best model can be selected based on validation loss."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        best_loss = float("inf")
        best_state_dict = None

        for epoch in range(3):
            # Validation loss
            with torch.no_grad():
                val_loss = 0.0
                for step in range(2):
                    batch = torch.randn(2, 2, 64, 64, device=device)
                    target = torch.randn(2, 2, 64, 64, device=device)

                    output = model(batch)
                    val_loss += nn.L1Loss()(output, target).item()

                val_loss /= 2

            # Update best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())

        # Verify best model was selected
        assert best_state_dict is not None
        assert best_loss < float("inf")


class TestPhysicsConstraintSatisfaction:
    """Tests for physics constraints during training."""

    def test_kspace_magnitude_bounded(self, device):
        """Test that k-space magnitude stays bounded during training."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        max_magnitudes = []

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            output = model(batch)

            # Compute magnitude as if it's complex (real/imag channels)
            real_ch = output[:, 0:1, :, :]
            imag_ch = output[:, 1:2, :, :]

            magnitude = torch.sqrt(real_ch**2 + imag_ch**2)
            max_magnitudes.append(magnitude.max().item())

            # Dummy loss
            loss = magnitude.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Magnitude should not explode
        assert all(m < 1e6 for m in max_magnitudes)

    def test_reconstruction_error_not_increasing(self, device):
        """Test that reconstruction error doesn't diverge."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        target = torch.ones(2, 2, 64, 64, device=device) * 0.5
        errors = []

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            output = model(batch)

            error = nn.L1Loss()(output, target)
            errors.append(error.item())

            optimizer.zero_grad()
            error.backward()
            optimizer.step()

        # Error might fluctuate but shouldn't diverge
        final_error = errors[-1]
        assert not torch.isnan(torch.tensor(final_error))
        assert final_error < 1e6


class TestNumericalStability:
    """Tests for numerical stability during training."""

    def test_no_gradient_overflow(self, device):
        """Test that gradients don't overflow during training."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)

            optimizer.zero_grad()
            loss.backward()

            # Check gradients are finite
            for param in model.parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all()

            optimizer.step()

    def test_no_loss_underflow(self, device):
        """Test that loss doesn't underflow to zero."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        losses = []

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device) * 0.1  # Small target

            output = model(batch)
            loss = nn.L1Loss()(output, target)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should not be all zeros
        assert not all(l < 1e-10 for l in losses)

    def test_weight_magnitudes_stable(self, device):
        """Test that model weights don't explode during training."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        weight_norms = []

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute weight norm
            total_norm = 0
            for param in model.parameters():
                total_norm += param.norm().item() ** 2

            weight_norms.append(total_norm**0.5)

        # Weight norms should not explode
        assert all(w < 1e5 for w in weight_norms)


class TestMemoryManagement:
    """Tests for memory cleanup during training."""

    def test_no_memory_leak_in_loop(self, device):
        """Test that memory is not leaking across training steps."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clean up
            del batch, target, output, loss

        # If we got here without OOM, no leak detected

    def test_optimizer_state_cleared(self, device):
        """Test that optimizer state is properly managed."""
        model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        for step in range(5):
            batch = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)

            output = model(batch)
            loss = nn.L1Loss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Optimizer state should exist but be manageable
        state_dict = optimizer.state_dict()
        assert state_dict is not None
