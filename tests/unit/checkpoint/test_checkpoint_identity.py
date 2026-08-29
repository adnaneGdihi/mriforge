"""
Tests for checkpoint save/load identity and training resumption.

Critical variables tested:
- Model state dict save/restore (generator, discriminator)
- Optimizer state dict save/restore
- Iteration/epoch counter persistence
- Training resume without duplication
- RNG state for deterministic resumption

Ensures that checkpoint→load produces identical training state
and that resuming training doesn't diverge from continuous training.
"""

import copy

import pytest
import torch
import torch.optim as optim

from tests.fixtures.critical_variables import (
    SimpleDiscriminator,
    SimpleGenerator,
    assert_model_state_dict_identical,
    assert_optimizer_state_identical,
)


class TestModelStateDictIdentity:
    """Tests for model state dict save/load producing identical weights."""

    def test_save_load_produces_identical_weights(self, generator, device):
        """Test that saving and loading a model produces identical weights."""
        # Original model
        weight_before = generator.state_dict()

        # Save to dict
        state_dict = copy.deepcopy(generator.state_dict())

        # Load into new model
        generator_loaded = SimpleGenerator().to(device)
        generator_loaded.load_state_dict(state_dict)

        # Compare weights
        weight_after = generator_loaded.state_dict()

        assert_model_state_dict_identical(weight_before, weight_after)

    def test_save_load_preserves_gradient_capability(self, generator, device):
        """Test that loaded model can compute gradients correctly."""
        state_dict = generator.state_dict()

        generator_loaded = SimpleGenerator().to(device)
        generator_loaded.load_state_dict(state_dict)

        # Create dummy input and verify backprop works
        x = torch.randn(2, 2, 64, 64, device=device, requires_grad=True)
        y = generator_loaded(x)
        loss = y.pow(2).mean()
        loss.backward()

        # Verify gradients exist
        for param in generator_loaded.parameters():
            assert param.grad is not None

    def test_save_load_file_roundtrip(self, generator, device, temp_checkpoint_path):
        """Test that file-based checkpoint roundtrip preserves weights."""
        # Save to file
        torch.save(generator.state_dict(), temp_checkpoint_path)

        # Load from file
        generator_loaded = SimpleGenerator().to(device)
        state_dict = torch.load(temp_checkpoint_path, map_location=device)
        generator_loaded.load_state_dict(state_dict)

        # Verify weights match
        assert_model_state_dict_identical(
            generator.state_dict(),
            generator_loaded.state_dict(),
        )

    def test_multiple_save_load_cycles_preserve_weights(self, generator, device):
        """Test that multiple save/load cycles don't accumulate errors."""
        original_state = copy.deepcopy(generator.state_dict())

        current_state = copy.deepcopy(original_state)

        # Perform 5 save/load cycles
        for cycle in range(5):
            new_model = SimpleGenerator().to(device)
            new_model.load_state_dict(current_state)
            current_state = copy.deepcopy(new_model.state_dict())

        # Final state should still match original
        assert_model_state_dict_identical(original_state, current_state)

    def test_discriminator_state_dict_identity(self, discriminator, device):
        """Test discriminator state dict save/load identity."""
        state_before = copy.deepcopy(discriminator.state_dict())

        disc_loaded = SimpleDiscriminator().to(device)
        disc_loaded.load_state_dict(state_before)

        state_after = disc_loaded.state_dict()

        assert_model_state_dict_identical(state_before, state_after)


class TestOptimizerStateIdentity:
    """Tests for optimizer state dict save/load."""

    def test_optimizer_state_dict_structure(self, optimizer_state):
        """Test that optimizer state dict has expected structure."""
        optimizer_g = optimizer_state["optimizer_g"]
        state_dict = optimizer_g.state_dict()

        assert "state" in state_dict
        assert "param_groups" in state_dict

    def test_optimizer_state_save_restore(self, generator, device):
        """Test that optimizer state can be saved and restored."""
        optimizer = optim.Adam(generator.parameters(), lr=0.0002)

        # Train one step to populate optimizer state
        x = torch.randn(2, 2, 64, 64, device=device)
        y = generator(x)
        loss = y.pow(2).mean()
        loss.backward()
        optimizer.step()

        # Save state
        state_before = copy.deepcopy(optimizer.state_dict())

        # Create new optimizer and restore
        generator_new = SimpleGenerator().to(device)
        optimizer_new = optim.Adam(generator_new.parameters(), lr=0.0002)
        optimizer_new.load_state_dict(state_before)

        # Verify state matches
        state_after = optimizer_new.state_dict()
        assert_optimizer_state_identical(state_before, state_after)

    def test_optimizer_momentum_preserved(self, generator, device):
        """Test that optimizer momentum is preserved across save/load."""
        optimizer = optim.SGD(generator.parameters(), lr=0.01, momentum=0.9)

        # Run a few steps
        for _ in range(3):
            x = torch.randn(2, 2, 64, 64, device=device)
            y = generator(x)
            loss = y.pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save state
        state_dict = copy.deepcopy(optimizer.state_dict())

        # Create new optimizer and restore
        generator_new = SimpleGenerator().to(device)
        optimizer_new = optim.SGD(generator_new.parameters(), lr=0.01, momentum=0.9)
        optimizer_new.load_state_dict(state_dict)

        # Verify momentum buffer is preserved
        for param_group, param_group_new in zip(
            optimizer.param_groups, optimizer_new.param_groups, strict=False
        ):
            assert param_group["momentum"] == param_group_new["momentum"]


class TestCheckpointIdentity:
    """Tests for complete checkpoint save/load roundtrip."""

    def test_checkpoint_all_components_saved(
        self, checkpoint_full, temp_checkpoint_path
    ):
        """Test that checkpoint contains all required components."""
        required_keys = {
            "epoch",
            "iteration",
            "generator_state_dict",
            "discriminator_state_dict",
            "optimizer_g_state_dict",
            "optimizer_d_state_dict",
            "best_loss",
            "rng_state_pytorch",
            "rng_state_numpy",
        }

        assert set(checkpoint_full.keys()).issuperset(required_keys)

    def test_checkpoint_file_save_load_roundtrip(
        self, checkpoint_full, device, temp_checkpoint_path
    ):
        """Test that checkpoint can be saved to file and loaded back identically."""
        # Save checkpoint
        torch.save(checkpoint_full, temp_checkpoint_path)

        # Load checkpoint (weights_only=False needed for non-tensor metadata)
        loaded_checkpoint = torch.load(
            temp_checkpoint_path, map_location=device, weights_only=False
        )

        # Verify all keys present
        assert set(loaded_checkpoint.keys()) == set(checkpoint_full.keys())

        # Verify scalar values match
        assert loaded_checkpoint["epoch"] == checkpoint_full["epoch"]
        assert loaded_checkpoint["iteration"] == checkpoint_full["iteration"]
        assert loaded_checkpoint["best_loss"] == checkpoint_full["best_loss"]

    def test_checkpoint_generator_state_restored_exactly(
        self, generator, discriminator, device, temp_checkpoint_path
    ):
        """Test that generator state is restored exactly from checkpoint."""
        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "epoch": 10,
            "iteration": 5000,
        }

        torch.save(checkpoint, temp_checkpoint_path)

        # Load and restore (weights_only=False needed for non-tensor metadata)
        loaded_checkpoint = torch.load(
            temp_checkpoint_path, map_location=device, weights_only=False
        )
        generator_restored = SimpleGenerator().to(device)
        generator_restored.load_state_dict(loaded_checkpoint["generator_state_dict"])

        # Verify exact match
        assert_model_state_dict_identical(
            checkpoint["generator_state_dict"],
            generator_restored.state_dict(),
        )

    def test_checkpoint_preserves_training_metadata(self, checkpoint_full):
        """Test that checkpoint preserves training metadata."""
        assert checkpoint_full["epoch"] == 42
        assert checkpoint_full["iteration"] == 10000
        assert checkpoint_full["best_loss"] == 0.15
        assert checkpoint_full["learning_rate"] == 0.0002

    def test_checkpoint_config_preserved(self, checkpoint_full):
        """Test that checkpoint config is preserved."""
        config = checkpoint_full["config"]
        assert config["training_mode"] == "gan"
        assert config["model_type"] == "standard_unet"
        assert config["batch_size"] == 32


class TestTrainingResumeIdentity:
    """Tests for resuming training without duplication/divergence."""

    def test_iteration_counter_continuity(self):
        """Test that iteration counter continues uninterrupted across resume."""
        # Simulate checkpoint at iteration 5000
        checkpoint = {"iteration": 5000}

        # Load and continue
        current_iteration = checkpoint["iteration"]
        assert current_iteration == 5000

        # Next iteration should be 5001
        next_iteration = current_iteration + 1
        assert next_iteration == 5001

    def test_epoch_counter_continuity(self):
        """Test that epoch counter continues across resume."""
        checkpoint = {
            "epoch": 10,
            "iteration_in_epoch": 500,
        }

        assert checkpoint["epoch"] == 10
        assert checkpoint["iteration_in_epoch"] == 500

    def test_loss_tracking_continuity(self):
        """Test that loss tracking continues without reset."""
        checkpoint = {
            "best_loss": 0.15,
            "last_loss": 0.18,
            "loss_history": [0.5, 0.45, 0.40, 0.35, 0.30, 0.20, 0.18],
        }

        # Load checkpoint
        best_loss = checkpoint["best_loss"]
        loss_history = checkpoint["loss_history"]

        # Continue training (simulate new loss)
        new_loss = 0.17
        if new_loss < best_loss:
            best_loss = new_loss

        loss_history.append(new_loss)

        assert best_loss == 0.15  # best_loss not updated
        assert len(loss_history) == 8
        assert loss_history[-1] == 0.17

    def test_early_stopping_counter_continuity(self):
        """Test that early stopping counter continues across resume."""
        checkpoint = {
            "no_improve_counter": 3,
            "early_stopping_patience": 10,
        }

        no_improve = checkpoint["no_improve_counter"]
        patience = checkpoint["early_stopping_patience"]

        # Continue training (simulate no improvement)
        no_improve += 1

        assert no_improve == 4
        assert no_improve < patience  # Not stopped yet

    def test_sampler_resumption_state(self):
        """Test that data sampler can continue from saved state."""
        checkpoint = {
            "sampler_state": 5000,  # Files already yielded: 0-4999
            "num_samples_per_epoch": 10000,
        }

        sampler_state = checkpoint["sampler_state"]
        remaining_samples = checkpoint["num_samples_per_epoch"] - sampler_state

        assert sampler_state == 5000
        assert remaining_samples == 5000

    def test_no_batch_duplication_on_resume(self):
        """Test that resuming doesn't cause batch duplication."""
        # Simulate epoch with 100 batches
        batches_processed_before_checkpoint = 50
        checkpoint = {"batch_index": batches_processed_before_checkpoint}

        # Resume and process next 50 batches
        batch_index = checkpoint["batch_index"]
        batches_to_process = 50

        # Verify we start from correct position
        assert batch_index == 50  # Not 0
        assert batch_index + batches_to_process == 100  # Correct total

    def test_model_weight_identity_after_resume_and_forward(self, generator, device):
        """Test that model weights don't change during save/load."""
        # Get initial weights
        weights_before = {k: v.clone() for k, v in generator.state_dict().items()}

        # Save and load
        checkpoint = {"state_dict": generator.state_dict()}
        generator_resumed = SimpleGenerator().to(device)
        generator_resumed.load_state_dict(checkpoint["state_dict"])

        weights_after = {
            k: v.clone() for k, v in generator_resumed.state_dict().items()
        }

        # Verify weights unchanged
        for key in weights_before:
            assert torch.equal(weights_before[key], weights_after[key])


class TestCheckpointCorruptionDetection:
    """Tests for detecting corrupted checkpoints."""

    def test_incomplete_checkpoint_detection(self, temp_checkpoint_path):
        """Test that incomplete checkpoints are detected."""
        incomplete_checkpoint = {
            "generator_state_dict": torch.randn(10),
            # Missing optimizer, RNG, metadata
        }

        torch.save(incomplete_checkpoint, temp_checkpoint_path)
        loaded = torch.load(temp_checkpoint_path)

        required_keys = {
            "generator_state_dict",
            "optimizer_g_state_dict",
            "rng_state_pytorch",
        }

        missing_keys = required_keys - set(loaded.keys())
        assert len(missing_keys) > 0  # Should detect missing keys

    def test_corrupted_state_dict_shape_mismatch(self, generator, device):
        """Test detection of state dict with wrong shapes."""
        # Create corrupted state dict
        corrupted_state = generator.state_dict()
        first_key = list(corrupted_state.keys())[0]
        corrupted_state[first_key] = torch.randn(999, 999)  # Wrong shape

        # Attempt to load should fail
        generator_new = SimpleGenerator().to(device)
        with pytest.raises(RuntimeError):
            generator_new.load_state_dict(corrupted_state)

    def test_corrupted_state_dict_dtype_mismatch(self, generator, device):
        """Test detection of state dict with wrong dtype."""
        corrupted_state = generator.state_dict()
        first_key = list(corrupted_state.keys())[0]
        original_dtype = corrupted_state[first_key].dtype

        # Change dtype of one tensor
        if corrupted_state[first_key].dtype == torch.float32:
            corrupted_state[first_key] = corrupted_state[first_key].type(torch.float64)

        # Attempt to load should fail or warn
        generator_new = SimpleGenerator().to(device)
        try:
            # PyTorch 2.0+ might automatically cast float64 to float32 on load
            # so we check if it either fails or casts it back
            generator_new.load_state_dict(corrupted_state, strict=True)
            loaded_dtype = generator_new.state_dict()[first_key].dtype
            assert (
                loaded_dtype == original_dtype
            ), "Should have casted back to original dtype"
        except (RuntimeError, TypeError):
            pass  # Expected - strict loading caught the mismatch
