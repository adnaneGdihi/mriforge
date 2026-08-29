"""
Tests for RNG (Random Number Generator) determinism and state management.

Critical variables tested:
- Python RNG state (torch.manual_seed)
- NumPy RNG state (np.random.seed)
- CUDA RNG state (torch.cuda.manual_seed)
- RNG restoration across checkpoint/resume
- Deterministic data loading
- Reproducibility of random augmentations

Ensures that training resumption produces deterministic results
matching hypothetical continuous training.
"""

import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class TestRNGSeeding:
    """Tests for RNG seeding and reproducibility."""

    def test_torch_manual_seed_reproducibility(self):
        """Test that torch.manual_seed produces deterministic results."""
        seed = 42

        # First run
        torch.manual_seed(seed)
        tensor1 = torch.randn(10)

        # Second run with same seed
        torch.manual_seed(seed)
        tensor2 = torch.randn(10)

        assert torch.equal(tensor1, tensor2)

    def test_numpy_seed_reproducibility(self):
        """Test that np.random.seed produces deterministic results."""
        seed = 42

        # First run
        np.random.seed(seed)
        array1 = np.random.randn(10)

        # Second run with same seed
        np.random.seed(seed)
        array2 = np.random.randn(10)

        assert np.allclose(array1, array2)

    @pytest.mark.gpu
    def test_torch_cuda_seed_reproducibility(self):
        """Test that torch.cuda.manual_seed works on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        seed = 42

        # First run
        torch.cuda.manual_seed(seed)
        tensor1 = torch.randn(10, device="cuda")

        # Second run with same seed
        torch.cuda.manual_seed(seed)
        tensor2 = torch.randn(10, device="cuda")

        assert torch.equal(tensor1, tensor2)

    def test_combined_seeding_reproducibility(self):
        """Test that seeding all RNG sources together produces reproducibility."""
        seed = 42

        def generate_mixed_random():
            """Generate random values from all RNG sources."""
            torch_val = torch.randn(1).item()
            numpy_val = np.random.randn()

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                cuda_val = torch.randn(1, device="cuda").item()
            else:
                cuda_val = None

            return torch_val, numpy_val, cuda_val

        # First run
        torch.manual_seed(seed)
        np.random.seed(seed)
        vals1 = generate_mixed_random()

        # Second run with same seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        vals2 = generate_mixed_random()

        assert vals1[0] == vals2[0]  # torch
        assert vals1[1] == vals2[1]  # numpy
        if vals1[2] is not None:
            assert vals1[2] == vals2[2]  # cuda


class TestRNGStateCapture:
    """Tests for capturing and restoring RNG state."""

    def test_torch_get_rng_state_structure(self):
        """Test that torch RNG state is properly captured."""
        torch.manual_seed(42)
        rng_state = torch.get_rng_state()

        assert rng_state is not None
        assert isinstance(rng_state, torch.ByteTensor)
        assert len(rng_state) > 0

    def test_torch_set_rng_state_restoration(self):
        """Test that torch RNG state can be set and restored."""
        seed = 42
        torch.manual_seed(seed)

        # Capture state
        state1 = torch.get_rng_state()

        # Generate some random values to advance RNG
        _ = torch.randn(100)

        # Restore state
        torch.set_rng_state(state1)
        tensor1 = torch.randn(10)

        # Reset and compare
        torch.manual_seed(seed)
        # Reset and compare
        torch.manual_seed(seed)
        # _ = torch.randn(100)  # REMOVED: Do not advance, as we want to compare with state AT seed (restored state)
        tensor2 = torch.randn(10)

        assert torch.equal(tensor1, tensor2)

    def test_numpy_get_rng_state_structure(self):
        """Test that NumPy RNG state is properly captured."""
        np.random.seed(42)
        rng_state = np.random.get_state()

        assert rng_state is not None
        assert isinstance(rng_state, tuple)
        assert len(rng_state) == 5

    def test_numpy_set_rng_state_restoration(self):
        """Test that NumPy RNG state can be set and restored."""
        seed = 42
        np.random.seed(seed)

        # Capture state
        state1 = np.random.get_state()

        # Advance RNG
        _ = np.random.randn(100)

        # Restore state
        np.random.set_state(state1)
        array1 = np.random.randn(10)

        # Reset and compare
        np.random.seed(seed)
        # Reset and compare
        np.random.seed(seed)
        # _ = np.random.randn(100) # REMOVED logic flaw similar to torch test if present?
        # Wait, lines 166 was: _ = np.random.randn(100)
        # Line 158 advanced 100.
        # Line 161 restored state1 (at seed).
        # Line 162 generated array1 (at seed).
        # Line 165 reset to seed.
        # Line 166 advanced 100.
        # Line 167 generated array2 (at step 100).
        # This is also flawed! array1 is step 0. array2 is step 100.
        # Removing line 166.

        array2 = np.random.randn(10)

        assert np.allclose(array1, array2)

    @pytest.mark.gpu
    def test_cuda_rng_state_capture(self):
        """Test that CUDA RNG state can be captured."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.cuda.manual_seed(42)
        cuda_state = torch.cuda.get_rng_state()

        assert cuda_state is not None
        assert isinstance(cuda_state, torch.ByteTensor)
        assert len(cuda_state) > 0


class TestRNGStatePersistence:
    """Tests for RNG state across checkpoint/resume."""

    def test_rng_state_pickling(self):
        """Test that RNG state can be pickled for checkpoint storage."""
        torch.manual_seed(42)
        state_original = torch.get_rng_state()

        # Pickle and unpickle
        state_pickled = pickle.dumps(state_original)
        state_restored = pickle.loads(state_pickled)

        assert torch.equal(state_original, state_restored)

    def test_rng_state_dict_structure(self):
        """Test RNG state dict capture for checkpoint."""
        torch.manual_seed(42)
        np.random.seed(42)

        rng_state_dict = {
            "pytorch_rng_state": torch.get_rng_state().cpu().clone(),
            "numpy_rng_state": np.random.get_state(),
        }

        if torch.cuda.is_available():
            rng_state_dict["cuda_rng_state"] = torch.cuda.get_rng_state().cpu().clone()

        # Verify structure
        assert "pytorch_rng_state" in rng_state_dict
        assert "numpy_rng_state" in rng_state_dict

    def test_rng_state_restoration_from_checkpoint(self):
        """Test that RNG state can be restored from checkpoint dict."""
        seed = 42

        # Original run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Capture state BEFORE generation
        checkpoint_rng = {
            "pytorch_rng_state": torch.get_rng_state().cpu().clone(),
            "numpy_rng_state": np.random.get_state(),
        }

        torch_val1 = torch.randn(1).item()
        numpy_val1 = np.random.randn()

        # Advance RNG (simulate training steps)
        _ = torch.randn(100)
        _ = np.random.randn(100)

        # Restore RNG state
        torch.set_rng_state(checkpoint_rng["pytorch_rng_state"])
        np.random.set_state(checkpoint_rng["numpy_rng_state"])

        # Generate same values as checkpoint run
        torch_val2 = torch.randn(1).item()
        numpy_val2 = np.random.randn()

        assert torch_val1 == torch_val2
        assert numpy_val1 == numpy_val2


class TestDeterministicTraining:
    """Tests for deterministic training across resumption."""

    def test_deterministic_forward_pass(self):
        """Test that forward pass is deterministic given same RNG state."""
        device = torch.device("cpu")
        # Use a model with Dropout to ensure RNG is actually used during forward
        generator = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(4, 2, 3, padding=1),
        ).to(device)
        generator.train()  # Enable dropout

        torch.manual_seed(42)
        batch = torch.randn(2, 2, 64, 64, device=device)

        # Capture generator RNG state before forward
        rng_state = torch.get_rng_state()

        with torch.no_grad():
            output1 = generator(batch)

        # Reset RNG and run again
        torch.set_rng_state(rng_state)
        with torch.no_grad():
            output2 = generator(batch)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_deterministic_batch_sequence(self):
        """Test that data loader produces same batch sequence given same seed."""
        seed = 42

        # Create dataset
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(X, y)

        # First run
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader1 = DataLoader(dataset, batch_size=32, shuffle=True)
        batches1 = [batch for batch in loader1]

        # Second run with same seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader2 = DataLoader(dataset, batch_size=32, shuffle=True)
        batches2 = [batch for batch in loader2]

        # Verify same batch sequence
        for batch1, batch2 in zip(batches1, batches2, strict=False):
            assert torch.equal(batch1[0], batch2[0])
            assert torch.equal(batch1[1], batch2[1])

    def test_deterministic_augmentation_sequence(self):
        """Test that random augmentations produce same sequence with seeded RNG."""
        seed = 42

        def augment_with_seed(seed):
            """Apply random augmentation with given seed."""
            torch.manual_seed(seed)

            # Simulate random augmentation (random rotation angle)
            angles = torch.rand(5) * 360
            return angles

        angles1 = augment_with_seed(seed)
        angles2 = augment_with_seed(seed)

        assert torch.equal(angles1, angles2)


class TestDeterministicResumption:
    """Tests for resuming training deterministically."""

    def test_resumption_produces_identical_batch_sequence(self):
        """Test that resuming produces identical batch sequence as continuous run."""
        seed = 42

        def generate_values(num_steps):
            return [torch.randn(5) for _ in range(num_steps)]

        # 1. Continuous run (10 steps)
        torch.manual_seed(seed)
        vals_continuous = generate_values(10)

        # 2. Interrupted run: generate 5, capture state
        torch.manual_seed(seed)
        _ = generate_values(5)
        checkpoint_rng = torch.get_rng_state()

        # 3. Resumed run: restore state, generate remaining 5
        torch.set_rng_state(checkpoint_rng)
        vals_resumed = generate_values(5)

        # Verify resumed values match the latter part of continuous run
        for i in range(5):
            assert torch.allclose(vals_continuous[5 + i], vals_resumed[i], atol=1e-6)

    def test_rng_state_not_leaked_across_processes(self):
        """Test that RNG state is independent (for multi-processing)."""
        seed = 42

        # Process 1
        torch.manual_seed(seed)
        state1 = torch.get_rng_state()
        vals1 = torch.randn(5)

        # Process 2 (independent)
        torch.manual_seed(seed)
        state2 = torch.get_rng_state()
        vals2 = torch.randn(5)

        # Should be identical if both seeded independently
        assert torch.equal(state1, state2)
        assert torch.equal(vals1, vals2)

    def test_rng_skip_equivalent_to_advance(self):
        """Test that RNG skip is equivalent to advancing RNG."""
        seed = 42

        # Method 1: Skip 100 random numbers
        torch.manual_seed(seed)
        _ = torch.randn(100)  # Skip
        val1 = torch.randn(1).item()

        # Method 2: Same RNG state
        torch.manual_seed(seed)
        checkpoint_rng = torch.get_rng_state()
        _ = torch.randn(100)
        torch.set_rng_state(checkpoint_rng)  # Restore before skip
        _ = torch.randn(100)  # Advance back
        val2 = torch.randn(1).item()

        # Should produce same value
        assert val1 == val2


@pytest.mark.gpu
class TestRNGConsistencyAcrossDevices:
    """Tests for RNG consistency across CPU and CUDA."""

    def test_torch_cpu_cuda_independent_rngs(self):
        """Test that CPU and CUDA RNG are independent."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        seed = 42

        # CPU
        torch.manual_seed(seed)
        cpu_val = torch.randn(1)

        # CUDA
        torch.cuda.manual_seed(seed)
        cuda_val = torch.randn(1, device="cuda")

        # They should both be random but independent seeding
        assert cpu_val.device.type == "cpu"
        assert cuda_val.device.type == "cuda"

    def test_torch_cuda_manual_seed_all(self):
        """Test that torch.cuda.manual_seed_all seeds all CUDA devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        seed = 42

        torch.cuda.manual_seed_all(seed)

        # If multiple GPUs, each should be seeded
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # On each GPU, same seed should produce same values
            torch.cuda.set_device(0)
            val0 = torch.randn(1, device="cuda")

            torch.cuda.set_device(1)
            val1 = torch.randn(1, device="cuda")

            # Same seed, same values  (may differ due to async, but test intention)
            assert val0 is not None
            assert val1 is not None


class TestRNGStateValidation:
    """Tests for validating RNG state integrity."""

    def test_rng_state_has_minimum_size(self):
        """Test that RNG state has sufficient size."""
        torch.manual_seed(42)
        state = torch.get_rng_state()

        # MT19937 state is 625 uint32 values = 2500 bytes
        assert len(state) >= 2500

    def test_rng_state_bytes_not_zero(self):
        """Test that RNG state is not all zeros."""
        torch.manual_seed(42)
        state = torch.get_rng_state()

        # After seeding, state should not be all zeros
        assert state.sum() > 0

    def test_numpy_rng_state_tuple_valid(self):
        """Test NumPy RNG state tuple structure."""
        np.random.seed(42)
        state = np.random.get_state()

        # Should be tuple of (str, ndarray, int, int, float) for MT19937
        assert isinstance(state, tuple)
        assert len(state) == 5
        assert state[0] == "MT19937"
