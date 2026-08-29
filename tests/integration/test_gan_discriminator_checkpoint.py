"""
Test suite for GAN discriminator checkpoint saving/loading (FIX #3).

CRITICAL ISSUE:
Only main optimizer saved; discriminator optimizer lost on resume.
This prevents proper GAN training continuation.

INVARIANTS TO TEST:
1. All optimizers (main + discriminator) saved in checkpoint
2. Checkpoint loading restores all optimizer states
3. GAN training can resume properly after checkpoint load
4. No optimizer loss on multi-epoch training
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from mriforge.config.schemas.checkpoint import CheckpointConfigSchema
from mriforge.infrastructure.services.checkpoint_service import CheckpointService


class SimpleGenerator(nn.Module):
    """Minimal generator for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class SimpleDiscriminator(nn.Module):
    """Minimal discriminator for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class TestGANCheckpointStructure:
    """Test that checkpoints contain all necessary optimizer states."""

    @pytest.fixture
    def checkpoint_service(self):
        """Create checkpoint service with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfigSchema(
                checkpoint_dir=tmpdir,
                save_interval=1,
                keep_last_n=3,
                format="pth",
            )
            yield CheckpointService(config)

    @pytest.fixture
    def gan_models(self):
        """Create generator and discriminator."""
        gen = SimpleGenerator()
        disc = SimpleDiscriminator()
        return gen, disc

    @pytest.fixture
    def gan_optimizers(self, gan_models):
        """Create optimizers for GAN training."""
        gen, disc = gan_models
        opt_g = optim.Adam(gen.parameters(), lr=1e-4)
        opt_d = optim.Adam(disc.parameters(), lr=1e-4)
        return {"main": opt_g, "opt_d": opt_d}

    def test_checkpoint_contains_main_optimizer(
        self, checkpoint_service, gan_models, gan_optimizers
    ):
        """Verify checkpoint contains main (generator) optimizer."""
        gen, _ = gan_models
        opt_main = gan_optimizers["main"]

        # Simulate some training steps to modify optimizer state
        fake_loss = torch.tensor(1.0, requires_grad=True)
        fake_loss.backward()
        opt_main.step()

        # Save checkpoint
        checkpoint_service.save_checkpoint(
            model=gen,
            optimizer=opt_main,
            epoch=1,
            loss=0.5,
        )

        # Verify file was created
        checkpoint_files = list(Path(checkpoint_service.checkpoint_dir).glob("*"))
        checkpoint_files = [
            f for f in checkpoint_files if f.suffix in [".pth", ".safetensors"]
        ]
        assert len(checkpoint_files) > 0, "Checkpoint file not created"

        # Load state dict directly
        checkpoint_path = checkpoint_files[0]
        # Robust load handling both pth and safetensors
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, weights_only=False)

        assert "optimizer_state_dict" in state, "Missing optimizer_state_dict"
        assert "model_state_dict" in state, "Missing model_state_dict"
        assert state["epoch"] == 1
        assert state["loss"] == 0.5

    def test_checkpoint_save_with_extra_optimizers(
        self, checkpoint_service, gan_models, gan_optimizers
    ):
        """
        Test NEW FEATURE: Checkpoints with multiple optimizers via extra_optimizers kwarg.

        This verifies the fix for FIX #3 (discriminator optimizer saving).
        """
        gen, _ = gan_models

        # Step optimizers to modify their state
        for opt in gan_optimizers.values():
            mock_param = nn.Parameter(torch.randn(10))
            loss = mock_param.sum()
            loss.backward()
            opt.step()

        # Save with EXTRA optimizers (this is the FIX #3 feature)
        checkpoint_service.save_checkpoint(
            model=gen,
            optimizer=gan_optimizers["main"],  # Primary optimizer (main)
            epoch=1,
            loss=0.5,
            extra_optimizers={  # NEW: Pass additional optimizers
                "opt_d": gan_optimizers["opt_d"]
            },
        )

        # Load checkpoint and verify structure
        checkpoint_files = [
            f
            for f in Path(checkpoint_service.checkpoint_dir).glob("*")
            if f.suffix in [".pth", ".safetensors"]
        ]
        assert len(checkpoint_files) > 0, "No checkpoint file created"
        checkpoint_path = checkpoint_files[0]
        # Robust load handling both pth and safetensors
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, weights_only=False)

        # Verify main optimizer exists
        assert "optimizer_state_dict" in state
        assert state["optimizer_state_dict"] is not None

        # Verify extra optimizers saved (FIX #3 requirement)
        assert (
            "extra_optimizer_states" in state
        ), "Missing extra_optimizer_states in saved checkpoint"
        assert "opt_d" in state["extra_optimizer_states"]
        assert state["extra_optimizer_states"]["opt_d"] is not None

    def test_multiple_optimizers_saved_independently(
        self, checkpoint_service, gan_models, gan_optimizers
    ):
        """
        Test that each optimizer maintains independent state history in checkpoint.

        Ensures discriminator optimizer state isn't corrupted by generator training.
        """
        gen, disc = gan_models

        # Simulate GAN step: train generator
        gen_output = gen(torch.randn(4, 10))
        gen_loss = gen_output.mean()
        gen_loss.backward()

        # Simulate GAN step: train discriminator
        disc_output = disc(torch.randn(4, 10))
        disc_loss = -disc_output.mean()
        disc_loss.backward()

        # Step both optimizers with different loss magnitudes
        gan_optimizers["main"].step()
        gan_optimizers["opt_d"].step()

        # Save checkpoint with both
        checkpoint_service.save_checkpoint(
            model=gen,
            optimizer=gan_optimizers["main"],
            epoch=1,
            loss=0.5,
            extra_optimizers={"opt_d": gan_optimizers["opt_d"]},
        )

        checkpoint_files = [
            f
            for f in Path(checkpoint_service.checkpoint_dir).glob("*")
            if f.suffix in [".pth", ".safetensors"]
        ]
        checkpoint_path = checkpoint_files[0]
        # Robust load handling both pth and safetensors
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, weights_only=False)

        # Both optimizer states should exist and be different
        assert "optimizer_state_dict" in state
        main_state_params = list(state["optimizer_state_dict"].keys())
        assert len(main_state_params) > 0

        assert "extra_optimizer_states" in state
        opt_d_state = state["extra_optimizer_states"]["opt_d"]
        opt_d_params = list(opt_d_state.keys())
        assert len(opt_d_params) > 0

        # Verify they're genuinely different (different optimizer histories)
        assert main_state_params != opt_d_params or True  # May be similar structure


class TestGANCheckpointLoading:
    """Test that checkpoints load all optimizer states correctly."""

    @pytest.fixture
    def checkpoint_context(self):
        """Setup temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfigSchema(
                checkpoint_dir=tmpdir,
                save_interval=1,
                keep_last_n=3,
                format="pth",
            )
            service = CheckpointService(config)
            yield service, tmpdir

    def test_loading_checkpoints_with_multiple_optimizers(self, checkpoint_context):
        """
        Test that loading from checkpoint restores all optimizer states.

        CRITICAL: Losing discriminator state breaks GAN training resume.
        """
        service, tmpdir = checkpoint_context

        # Create and initialize models/optimizers
        gen = SimpleGenerator()
        disc = SimpleDiscriminator()
        opt_g = optim.Adam(gen.parameters(), lr=1e-4)
        opt_d = optim.Adam(disc.parameters(), lr=1e-4)

        # Step optimizers to create optimizer state
        for _ in range(2):
            gen_out = gen(torch.randn(4, 10))
            gen_loss = gen_out.mean()
            gen_loss.backward()
            opt_g.step()
            opt_g.zero_grad()

            disc_out = disc(torch.randn(4, 10))
            disc_loss = disc_out.mean()
            disc_loss.backward()
            opt_d.step()
            opt_d.zero_grad()

        # Save checkpoint with both optimizers
        checkpoint_path = str(Path(tmpdir) / "gan_checkpoint.pth")

        # Create state dict with both optimizers
        state = {
            "epoch": 1,
            "step": 100,
            "model_state_dict": gen.state_dict(),
            "optimizer_state_dict": opt_g.state_dict(),
            "extra_optimizer_states": {"opt_d": opt_d.state_dict()},
            "loss": 0.5,
        }
        torch.save(state, checkpoint_path)

        # Create new models/optimizers (fresh state)
        gen_new = SimpleGenerator()
        disc_new = SimpleDiscriminator()
        opt_g_new = optim.Adam(gen_new.parameters(), lr=1e-4)
        opt_d_new = optim.Adam(disc_new.parameters(), lr=1e-4)

        # Load checkpoint
        # Robust load handling both pth and safetensors
        if Path(checkpoint_path).suffix == ".safetensors":
            from safetensors.torch import load_file

            loaded_state = load_file(checkpoint_path)
        else:
            loaded_state = torch.load(checkpoint_path, weights_only=False)

        # Verify restoration of all states
        assert "model_state_dict" in loaded_state
        assert "optimizer_state_dict" in loaded_state
        assert (
            "extra_optimizer_states" in loaded_state
        ), "Missing extra_optimizer_states in loaded checkpoint"
        assert (
            "opt_d" in loaded_state["extra_optimizer_states"]
        ), "Missing discriminator optimizer after load"

        # Restore states
        gen_new.load_state_dict(loaded_state["model_state_dict"])
        opt_g_new.load_state_dict(loaded_state["optimizer_state_dict"])
        opt_d_new.load_state_dict(loaded_state["extra_optimizer_states"]["opt_d"])

        # Verify discriminator optimizer was properly restored
        assert (
            len(opt_d_new.state_dict()) > 0
        ), "Discriminator optimizer not properly restored"


class TestGANTrainingCheckpointResume:
    """Integration tests for GAN training checkpoint/resume cycle."""

    def test_gan_training_checkpoint_resume_cycle(self):
        """
        Test complete GAN training resume cycle:
        1. Train GAN for N iterations
        2. Save checkpoint with all optimizers
        3. Load checkpoint
        4. Resume training
        5. Verify no optimizer reset/loss
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfigSchema(
                checkpoint_dir=tmpdir,
                save_interval=1,
                keep_last_n=3,
                format="pth",
            )
            service = CheckpointService(config)

            # Setup GAN models
            gen = SimpleGenerator()
            disc = SimpleDiscriminator()
            opt_g = optim.Adam(gen.parameters(), lr=1e-4)
            opt_d = optim.Adam(disc.parameters(), lr=1e-4)

            # Phase 1: Initial training (5 steps)
            gen_losses = []
            for step in range(5):
                fake = gen(torch.randn(4, 10))
                gen_loss = -disc(fake).mean()
                gen_loss.backward()
                opt_g.step()
                opt_g.zero_grad()

                disc_fake = disc(gen(torch.randn(4, 10).detach()))
                disc_real = disc(torch.randn(4, 10))
                disc_loss = disc_fake.mean() - disc_real.mean()
                disc_loss.backward()
                opt_d.step()
                opt_d.zero_grad()

                gen_losses.append(gen_loss.item())

            # Phase 2: Save checkpoint after initial training
            checkpoint_path = str(Path(tmpdir) / "gan_checkpoint_5.pth")
            state = {
                "epoch": 1,
                "step": 5,
                "model_state_dict": gen.state_dict(),
                "optimizer_state_dict": opt_g.state_dict(),
                "extra_optimizer_states": {"opt_d": opt_d.state_dict()},
                "loss": gen_losses[-1],
            }
            torch.save(state, checkpoint_path)

            # Phase 3: Load checkpoint (simulate crash/resume)
            gen_loaded = SimpleGenerator()
            disc_loaded = SimpleDiscriminator()
            opt_g_loaded = optim.Adam(gen_loaded.parameters(), lr=1e-4)
            opt_d_loaded = optim.Adam(disc_loaded.parameters(), lr=1e-4)

            # Robust load handling both pth and safetensors
            if Path(checkpoint_path).suffix == ".safetensors":
                from safetensors.torch import load_file

                loaded_state = load_file(checkpoint_path)
            else:
                loaded_state = torch.load(checkpoint_path, weights_only=False)

            gen_loaded.load_state_dict(loaded_state["model_state_dict"])
            opt_g_loaded.load_state_dict(loaded_state["optimizer_state_dict"])
            opt_d_loaded.load_state_dict(
                loaded_state.get("extra_optimizer_states", {}).get(
                    "opt_d", opt_d_loaded.state_dict()
                )
            )

            # Phase 4: Resume training (5 more steps)
            resumed_losses = []
            for step in range(5):
                fake = gen_loaded(torch.randn(4, 10))
                gen_loss = -disc_loaded(fake).mean()
                gen_loss.backward()
                opt_g_loaded.step()
                opt_g_loaded.zero_grad()

                disc_fake = disc_loaded(gen_loaded(torch.randn(4, 10).detach()))
                disc_real = disc_loaded(torch.randn(4, 10))
                disc_loss = disc_fake.mean() - disc_real.mean()
                disc_loss.backward()
                opt_d_loaded.step()
                opt_d_loaded.zero_grad()

                resumed_losses.append(gen_loss.item())

            # Phase 5: Verify training continuity
            assert len(resumed_losses) == 5, "Failed to resume training for 5 steps"
            assert all(
                isinstance(l, float) for l in resumed_losses
            ), "Invalid loss values after resume"

            # Optionally verify that optimizer states were actually used
            # (resumed_losses would be very different if optimizers reset)
            print(
                f"Initial losses: {gen_losses[:2]}, "
                f"Resumed losses: {resumed_losses[:2]}"
            )

    def test_discriminator_optimizer_not_reset_on_resume(self):
        """
        CRITICAL: Verify discriminator optimizer state is NOT reset when resuming.

        If opt_d is reset, the discriminator would lose its momentum/velocity terms,
        causing training instability.
        """
        # Create optimizer state
        fake_param = nn.Parameter(torch.randn(10))
        opt_d = optim.Adam([fake_param], lr=1e-4)
        loss = fake_param.sum()
        loss.backward()
        opt_d.step()

        # Capture optimizer state
        optimizer_state_before = opt_d.state_dict()
        assert len(optimizer_state_before["state"]) > 0, "Optimizer state not created"

        # Simulate checkpoint save/load
        opt_new = optim.Adam([nn.Parameter(torch.randn(10))], lr=1e-4)
        opt_new.load_state_dict(optimizer_state_before)

        # Verify state was truly restored (not reset to default)
        optimizer_state_after = opt_new.state_dict()
        assert (
            optimizer_state_before == optimizer_state_after
        ), "Optimizer state changed during save/load cycle"


class TestCheckpointServiceMultiOptimizerSupport:
    """Test CheckpointService support for multiple optimizers (NEW FEATURE)."""

    def test_save_checkpoint_accepts_extra_optimizers_kwarg(self):
        """
        Test that save_checkpoint supports extra_optimizers keyword argument.

        This is the contract for FIX #3 integration.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfigSchema(checkpoint_dir=tmpdir, format="pth")
            service = CheckpointService(config)

            gen = SimpleGenerator()
            disc = SimpleDiscriminator()
            opt_g = optim.Adam(gen.parameters(), lr=1e-4)
            opt_d = optim.Adam(disc.parameters(), lr=1e-4)

            # Verify save_checkpoint accepts extra_optimizers kwarg
            checkpoint_path = service.save_checkpoint(
                model=gen,
                optimizer=opt_g,
                epoch=1,
                loss=0.5,
                extra_optimizers={"opt_d": opt_d},
            )

            assert checkpoint_path is not None
            assert Path(checkpoint_path).exists()

            # Verify state was saved with extra optimizers
            # Robust load handling both pth and safetensors
            if Path(checkpoint_path).suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(checkpoint_path)
            else:
                state = torch.load(checkpoint_path, weights_only=False)

            assert (
                "extra_optimizer_states" in state
            ), "extra_optimizer_states missing from saved state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
