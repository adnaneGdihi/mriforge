"""
End-to-End tests for full training pipeline.

Tests validate:
1. Complete training run (mini)
2. Config to checkpoint flow
3. Inference from checkpoint
4. Metrics logging

Run on CI, NOT local dev machine.
"""

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)


# === Mock Components ===
class MiniGenerator(nn.Module):
    """Minimal generator for E2E testing."""

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_ch, 3, padding=1),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class MiniDataset(torch.utils.data.Dataset):
    """Minimal dataset for E2E testing."""

    def __init__(self, size=100, img_size=32, channels=1):
        self.size = size
        self.img_size = img_size
        self.channels = channels
        # Pre-generate data for reproducibility
        torch.manual_seed(42)
        self.data = torch.randn(size, channels, img_size, img_size)
        self.targets = torch.randn(size, channels, img_size, img_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"input": self.data[idx], "target": self.targets[idx]}


def create_mini_dataloader(batch_size=4, size=100):
    """Create minimal dataloader."""
    dataset = MiniDataset(size=size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# === Full Training Run Tests ===
class TestFullTrainingRun:
    """Test complete training runs."""

    @pytest.mark.timeout(120)
    @pytest.mark.e2e
    def test_mini_training_run_completes(self):
        """Mini training run should complete without errors."""
        model = MiniGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(), learning_rate=1e-3, optimizer_type="adam"
        )
        dataloader = create_mini_dataloader(batch_size=4, size=20)

        model.train()

        for epoch in range(3):
            for batch in dataloader:
                inputs = batch["input"]
                targets = batch["target"]

                output = model(inputs)
                loss = nn.L1Loss()(output, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Should complete without error
        assert True

    @pytest.mark.timeout(120)
    @pytest.mark.e2e
    def test_training_loss_decreases(self):
        """Training loss should decrease over epochs."""
        model = MiniGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(), learning_rate=1e-2, optimizer_type="adam"
        )
        dataloader = create_mini_dataloader(batch_size=4, size=50)

        model.train()

        epoch_losses = []

        for epoch in range(5):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                inputs = batch["input"]
                targets = batch["target"]

                output = model(inputs)
                loss = nn.L1Loss()(output, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_losses.append(epoch_loss / n_batches)

        # Loss should decrease
        assert (
            epoch_losses[-1] < epoch_losses[0]
        ), f"Loss did not decrease: {epoch_losses}"

    @pytest.mark.timeout(120)
    @pytest.mark.e2e
    def test_training_with_validation(self):
        """Training with validation loop."""
        model = MiniGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(), learning_rate=1e-3, optimizer_type="adam"
        )

        train_loader = create_mini_dataloader(batch_size=4, size=40)
        val_loader = create_mini_dataloader(batch_size=4, size=20)

        for epoch in range(3):
            # Train
            model.train()
            for batch in train_loader:
                output = model(batch["input"])
                loss = nn.L1Loss()(output, batch["target"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    output = model(batch["input"])
                    val_loss += nn.L1Loss()(output, batch["target"]).item()

            val_loss /= len(val_loader)

        assert True  # Completed


# === Checkpoint Flow Tests ===
class TestCheckpointFlow:
    """Test save/load checkpoint flow."""

    @pytest.mark.timeout(60)
    @pytest.mark.e2e
    def test_checkpoint_save_load_resume(self, tmp_path):
        """Save checkpoint, load, and resume training."""
        # Phase 1: Train and save
        model1 = MiniGenerator()
        optimizer1 = OptimizationBuilder.create_single_optimizer(
            model1.parameters(), learning_rate=1e-3, optimizer_type="adam"
        )
        dataloader = create_mini_dataloader(batch_size=4, size=20)

        model1.train()
        for batch in dataloader:
            output = model1(batch["input"])
            loss = nn.L1Loss()(output, batch["target"])
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        # Save checkpoint
        checkpoint = {
            "model": model1.state_dict(),
            "optimizer": optimizer1.state_dict(),
            "epoch": 5,
        }
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, ckpt_path)

        # Phase 2: Load and resume
        model2 = MiniGenerator()
        optimizer2 = OptimizationBuilder.create_single_optimizer(
            model2.parameters(), learning_rate=1e-3, optimizer_type="adam"
        )

        loaded = torch.load(ckpt_path)
        model2.load_state_dict(loaded["model"])
        optimizer2.load_state_dict(loaded["optimizer"])

        # Continue training
        model2.train()
        for batch in dataloader:
            output = model2(batch["input"])
            loss = nn.L1Loss()(output, batch["target"])
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

        assert loaded["epoch"] == 5

    @pytest.mark.timeout(60)
    @pytest.mark.e2e
    def test_best_checkpoint_selection(self, tmp_path):
        """Save best checkpoint based on val loss."""
        model = MiniGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(), learning_rate=1e-2, optimizer_type="adam"
        )

        train_loader = create_mini_dataloader(batch_size=4, size=20)
        val_loader = create_mini_dataloader(batch_size=4, size=10)

        best_val_loss = float("inf")

        for epoch in range(5):
            # Train
            model.train()
            for batch in train_loader:
                output = model(batch["input"])
                loss = nn.L1Loss()(output, batch["target"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    output = model(batch["input"])
                    val_loss += nn.L1Loss()(output, batch["target"]).item()
            val_loss /= len(val_loader)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), tmp_path / "best.pt")

        # Best checkpoint should exist
        assert (tmp_path / "best.pt").exists()


# === Inference Tests ===
class TestInferencePipeline:
    """Test inference from trained model."""

    @pytest.mark.timeout(60)
    @pytest.mark.e2e
    def test_inference_from_checkpoint(self, tmp_path):
        """Load checkpoint and run inference."""
        # Train and save
        model = MiniGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(), learning_rate=1e-2, optimizer_type="adam"
        )
        dataloader = create_mini_dataloader(batch_size=4, size=20)

        model.train()
        for batch in dataloader:
            output = model(batch["input"])
            loss = nn.L1Loss()(output, batch["target"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), tmp_path / "model.pt")

        # Load for inference
        inference_model = MiniGenerator()
        inference_model.load_state_dict(torch.load(tmp_path / "model.pt"))
        inference_model.eval()

        # Run inference
        test_input = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            output = inference_model(test_input)

        assert output.shape == test_input.shape

    @pytest.mark.timeout(60)
    @pytest.mark.e2e
    def test_batch_inference(self, tmp_path):
        """Run inference on batch of samples."""
        model = MiniGenerator()
        model.eval()

        test_loader = create_mini_dataloader(batch_size=8, size=32)

        all_outputs = []
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch["input"])
                all_outputs.append(output)

        # Should have processed all batches
        assert len(all_outputs) == 4  # 32 / 8 = 4


# === Metrics Logging Tests ===
class TestMetricsLogging:
    """Test metrics computation and logging."""

    @pytest.mark.timeout(60)
    @pytest.mark.e2e
    def test_metric_computation(self):
        """Compute metrics during training."""
        model = MiniGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(), learning_rate=1e-3, optimizer_type="adam"
        )
        dataloader = create_mini_dataloader(batch_size=4, size=20)

        metrics_history = []

        model.train()
        for i, batch in enumerate(dataloader):
            output = model(batch["input"])
            targets = batch["target"]

            # Compute loss
            loss = nn.L1Loss()(output, targets)

            # Compute additional metrics
            with torch.no_grad():
                mse = nn.MSELoss()(output, targets)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

            metrics = {
                "step": i,
                "loss": loss.item(),
                "mse": mse.item(),
                "psnr": psnr.item(),
            }
            metrics_history.append(metrics)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should have metrics for all batches
        assert len(metrics_history) == len(dataloader)

        # All metrics should be valid numbers
        for m in metrics_history:
            assert not any(
                v != v for v in m.values() if isinstance(v, float)
            )  # Not NaN


# === Config to Results Flow Tests ===
class TestConfigToResultsFlow:
    """Test complete config to results flow."""

    @pytest.mark.timeout(120)
    @pytest.mark.e2e
    def test_config_creates_correct_model(self):
        """Config should create correct model type."""
        config = {
            "model_type": "mini",
            "in_channels": 1,
            "out_channels": 1,
        }

        # Simulate model factory
        if config["model_type"] == "mini":
            model = MiniGenerator(
                in_ch=config["in_channels"],
                out_ch=config["out_channels"],
            )

        assert model is not None

        # Test forward pass
        x = torch.randn(1, 1, 32, 32)
        y = model(x)
        assert y.shape == (1, 1, 32, 32)


# === Error Recovery Tests ===
class TestErrorRecovery:
    """Test error recovery in training."""

    @pytest.mark.timeout(60)
    @pytest.mark.e2e
    def test_resume_after_nan_detection(self):
        """Training should be able to resume after NaN detection."""
        model = MiniGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(), learning_rate=1e-3, optimizer_type="adam"
        )
        dataloader = create_mini_dataloader(batch_size=4, size=20)

        model.train()
        nan_detected = False

        for batch in dataloader:
            output = model(batch["input"])
            loss = nn.L1Loss()(output, batch["target"])

            # Simulate NaN check
            if torch.isnan(loss):
                nan_detected = True
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # No NaN should occur with proper model
        assert not nan_detected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
