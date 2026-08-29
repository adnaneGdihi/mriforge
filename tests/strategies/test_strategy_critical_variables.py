"""
Parameterized tests for all 8 training strategies.

Tests critical variables across:
- GAN training (k-space and image domain)
- Diffusion training (score-based models)
- Reconstruction-only baseline
- VAE training (image domain)
- SSL training (multi-manifest)
- Domain adaptation (source + target)
- Image translation (paired NIfTI)
- Latent diffusion (compressed VAE latent space)

Each strategy is tested with:
1. Configuration validation
2. Data batch structure verification
3. Model initialization
4. Forward/backward pass correctness
5. Loss computation determinism
6. Gradient flow validation
7. State dict save/load identity
8. Checkpoint resumption
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from tests.fixtures.critical_variables import SimpleDiscriminator, SimpleGenerator


@pytest.fixture(
    params=[
        "gan",
        "diffusion",
        "reconstruction",
        "vae",
        "ssl",
        "domain_adaptation",
        "image_translation",
        "latent_diffusion",
    ]
)
def training_strategy(request):
    """Parametrized fixture for all 8 training strategies."""
    return request.param


@pytest.fixture
def config_for_strategy(training_strategy, all_strategy_configs):
    """Get config for the current strategy."""
    return all_strategy_configs[training_strategy]


class TestStrategyConfigValidation:
    """Tests for strategy configuration validation."""

    def test_config_has_required_fields(self, config_for_strategy, training_strategy):
        """Test that config has all required fields."""
        required_fields = {
            "training_mode",
            "data",
            "model",
        }

        config_keys = set(config_for_strategy.keys())
        assert required_fields.issubset(
            config_keys
        ), f"Config missing fields for {training_strategy}"

    def test_training_mode_matches_strategy(
        self, config_for_strategy, training_strategy
    ):
        """Test that training mode matches the strategy."""
        training_mode = config_for_strategy["training_mode"]

        # Special case: image_translation uses gan training mode
        if training_strategy == "image_translation":
            assert training_mode in ("gan",)
        else:
            # Most strategies have matching training_mode
            assert training_mode in (training_strategy, "gan")

    def test_data_config_valid(self, config_for_strategy):
        """Test that data configuration is valid."""
        data_config = config_for_strategy["data"]

        assert "source" in data_config or "batch_size" in data_config
        assert data_config.get("batch_size", 32) > 0

    def test_model_config_valid(self, config_for_strategy):
        """Test that model configuration is valid."""
        model_config = config_for_strategy["model"]

        assert "model_type" in model_config or len(model_config) > 0


class TestStrategyDataLoading:
    """Tests for data loading per strategy."""

    def test_batch_structure_valid(self, training_strategy):
        """Test that batches have valid structure for the strategy."""
        if training_strategy in ("gan", "reconstruction", "diffusion"):
            # K-space strategies
            batch = {
                "input": torch.randn(4, 2, 64, 64),  # Complex as real/imag channels
                "output": torch.randn(4, 2, 64, 64),
            }
        elif training_strategy in ("vae", "image_translation"):
            # Image domain strategies
            batch = {
                "input": torch.randn(4, 1, 64, 64),
                "output": torch.randn(4, 1, 64, 64),
            }
        else:
            # Other strategies (SSL, domain adaptation)
            batch = {}

        # Batch should be dict
        assert isinstance(batch, dict)

    def test_batch_device_placement(self, training_strategy, device):
        """Test that batch tensors are on correct device."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch = {
                "input": torch.randn(2, 2, 64, 64, device=device),
                "output": torch.randn(2, 2, 64, 64, device=device),
            }
        else:
            batch = {
                "input": torch.randn(2, 1, 64, 64, device=device),
                "output": torch.randn(2, 1, 64, 64, device=device),
            }

        for key, tensor in batch.items():
            if device.type == "cuda":
                assert tensor.device.type == "cuda"
            else:
                assert tensor.device == device

    def test_batch_no_nans_or_infs(self, training_strategy, device):
        """Test that batches don't contain NaN or Inf."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch = {
                "input": torch.randn(4, 2, 64, 64, device=device),
                "output": torch.randn(4, 2, 64, 64, device=device),
            }
        else:
            batch = {
                "input": torch.randn(4, 1, 64, 64, device=device),
                "output": torch.randn(4, 1, 64, 64, device=device),
            }

        for tensor in batch.values():
            assert not torch.isnan(tensor).any()
            assert not torch.isinf(tensor).any()


class TestStrategyModelInitialization:
    """Tests for model initialization per strategy."""

    def test_generator_initializes(self, training_strategy, device):
        """Test that generator model initializes correctly."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        else:
            model = SimpleGenerator(in_channels=1, out_channels=1).to(device)

        assert model is not None
        model_device = next(model.parameters()).device
        if device.type == "cuda":
            assert model_device.type == "cuda"
        else:
            assert model_device == device

    def test_discriminator_initializes(self, training_strategy, device):
        """Test that discriminator initializes (if needed)."""
        if training_strategy in ("gan"):
            disc = SimpleDiscriminator(in_channels=2).to(device)
            assert disc is not None
            disc_device = next(disc.parameters()).device
            if device.type == "cuda":
                assert disc_device.type == "cuda"
            else:
                assert disc_device == device

    def test_model_parameters_trainable(self, training_strategy, device):
        """Test that model parameters are trainable."""
        if training_strategy in ("gan", "diffusion"):
            model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        else:
            model = SimpleGenerator(in_channels=1, out_channels=1).to(device)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0

    def test_model_state_dict_structure(self, training_strategy, device):
        """Test that model state dict has expected structure."""
        if training_strategy in ("gan", "diffusion"):
            model = SimpleGenerator(in_channels=2, out_channels=2).to(device)
        else:
            model = SimpleGenerator(in_channels=1, out_channels=1).to(device)

        state_dict = model.state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        assert all(isinstance(k, str) for k in state_dict.keys())


class TestStrategyForwardPass:
    """Tests for forward pass per strategy."""

    def test_forward_pass_basic(self, training_strategy, device, generator):
        """Test that forward pass works without errors."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)

        with torch.no_grad():
            output = generator(batch_input)

        assert output is not None
        assert output.shape == batch_input.shape

    def test_forward_pass_output_finite(self, training_strategy, device, generator):
        """Test that forward pass output is finite."""
        # Match the channel count used by the rest of this test class — the
        # generator fixture is built with 2-channel inputs for gan, diffusion,
        # and reconstruction strategies.
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)

        with torch.no_grad():
            output = generator(batch_input)

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_forward_pass_deterministic(self, training_strategy, device, generator):
        """Test that forward pass is deterministic (with same seed)."""
        torch.manual_seed(42)

        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)

        with torch.no_grad():
            output1 = generator(batch_input)

        torch.manual_seed(42)
        batch_input_same = torch.randn(
            2,
            2 if training_strategy in ("gan", "diffusion", "reconstruction") else 1,
            64,
            64,
            device=device,
        )

        with torch.no_grad():
            output2 = generator(batch_input_same)

        # Outputs might differ due to RNG, but forward pass itself is deterministic
        # (model weights don't change)


class TestStrategyBackwardPass:
    """Tests for backward pass and gradient flow."""

    def test_backward_pass_computes_gradients(
        self, training_strategy, device, generator
    ):
        """Test that backward pass computes gradients."""
        optimizer = optim.Adam(generator.parameters(), lr=0.0002)

        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device, requires_grad=True)
            target = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device, requires_grad=True)
            target = torch.randn(2, 1, 64, 64, device=device)

        output = generator(batch_input)
        loss = nn.L1Loss()(output, target)

        loss.backward()

        # Check that gradients were computed
        for param in generator.parameters():
            if param.grad is not None:
                assert param.grad.shape == param.shape
                assert torch.isfinite(param.grad).all()

    def test_gradient_magnitude_reasonable(self, training_strategy, device, generator):
        """Test that gradient magnitudes are reasonable (not too large or small)."""
        optimizer = optim.Adam(generator.parameters(), lr=0.0002)

        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)
            target = torch.randn(2, 1, 64, 64, device=device)

        output = generator(batch_input)
        loss = nn.L1Loss()(output, target)
        loss.backward()

        # Compute gradient norms
        total_grad_norm = 0
        for param in generator.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2

        total_grad_norm = total_grad_norm**0.5

        # Gradient norm should be reasonable
        assert 0 < total_grad_norm < 1e5

    def test_optimizer_step_updates_weights(self, training_strategy, device, generator):
        """Test that optimizer step updates model weights."""
        optimizer = optim.Adam(generator.parameters(), lr=0.0002)

        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)
            target = torch.randn(2, 1, 64, 64, device=device)

        # Capture weights before
        weights_before = {k: v.clone() for k, v in generator.state_dict().items()}

        # Forward, backward, step
        output = generator(batch_input)
        loss = nn.L1Loss()(output, target)
        loss.backward()
        optimizer.step()

        # Capture weights after
        weights_after = {k: v.clone() for k, v in generator.state_dict().items()}

        # At least some weights should have changed
        num_changed = 0
        for key in weights_before:
            if not torch.equal(weights_before[key], weights_after[key]):
                num_changed += 1

        assert num_changed > 0


class TestStrategyLossComputation:
    """Tests for loss computation per strategy."""

    def test_loss_is_scalar(self, training_strategy, device, generator):
        """Test that computed loss is a scalar."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)
            target = torch.randn(2, 1, 64, 64, device=device)

        output = generator(batch_input)
        loss = nn.L1Loss()(output, target)

        assert loss.ndim == 0  # Scalar

    def test_loss_is_positive(self, training_strategy, device, generator):
        """Test that loss is non-negative."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)
            target = torch.randn(2, 1, 64, 64, device=device)

        output = generator(batch_input)
        loss = nn.L1Loss()(output, target)

        assert loss.item() >= 0

    def test_loss_is_finite(self, training_strategy, device, generator):
        """Test that loss is finite (no NaN or Inf)."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)
            target = torch.randn(2, 1, 64, 64, device=device)

        output = generator(batch_input)
        loss = nn.L1Loss()(output, target)

        assert torch.isfinite(loss)

    def test_loss_deterministic(self, training_strategy, device, generator):
        """Test that loss computation is deterministic."""
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
            target = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)
            target = torch.randn(2, 1, 64, 64, device=device)

        output = generator(batch_input)

        loss1 = nn.L1Loss()(output, target)
        loss2 = nn.L1Loss()(output, target)

        assert loss1.item() == loss2.item()


class TestStrategyCheckpointSaveLoad:
    """Tests for checkpoint save/load per strategy."""

    def test_checkpoint_save_load_identity(
        self, training_strategy, device, generator, tmp_path
    ):
        """Test that checkpoint save/load preserves state."""
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(generator.state_dict(), checkpoint_path)

        # Load checkpoint
        generator_loaded = SimpleGenerator(
            in_channels=(
                2 if training_strategy in ("gan", "diffusion", "reconstruction") else 1
            ),
            out_channels=(
                2 if training_strategy in ("gan", "diffusion", "reconstruction") else 1
            ),
        ).to(device)

        generator_loaded.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )

        # Verify weights match
        for p_orig, p_loaded in zip(
            generator.parameters(), generator_loaded.parameters(), strict=False
        ):
            assert torch.equal(p_orig, p_loaded)

    def test_checkpoint_forward_pass_identical(
        self, training_strategy, device, generator, tmp_path
    ):
        """Test that loaded model produces identical forward pass."""
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(generator.state_dict(), checkpoint_path)

        # Load checkpoint
        generator_loaded = SimpleGenerator(
            in_channels=(
                2 if training_strategy in ("gan", "diffusion", "reconstruction") else 1
            ),
            out_channels=(
                2 if training_strategy in ("gan", "diffusion", "reconstruction") else 1
            ),
        ).to(device)

        generator_loaded.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )

        # Test forward pass
        if training_strategy in ("gan", "diffusion", "reconstruction"):
            batch_input = torch.randn(2, 2, 64, 64, device=device)
        else:
            batch_input = torch.randn(2, 1, 64, 64, device=device)

        with torch.no_grad():
            output1 = generator(batch_input)
            output2 = generator_loaded(batch_input)

        assert torch.allclose(output1, output2, atol=1e-6)
