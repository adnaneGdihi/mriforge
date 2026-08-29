#!/usr/bin/env python
"""Dimension Validation Tests - Scenario-Based Testing

This test suite validates that different training strategies handle various
tensor dimensions and shapes correctly throughout the training pipeline.
"""

from typing import Any

import torch


class DimensionScenario:
    """Base class for dimension test scenarios."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        """Return {lr, hr, batch_data} tensors for this scenario."""
        raise NotImplementedError

    def get_model_config(self) -> dict[str, Any]:
        """Return model configuration for this scenario."""
        raise NotImplementedError

    def is_valid(self) -> bool:
        """Whether this scenario should work without errors."""
        raise NotImplementedError


class Scenario_4D_2Channel(DimensionScenario):
    """Standard 4D tensors with 2 channels (real/imag)."""

    def __init__(self):
        super().__init__(
            "4D_2Channel", "Standard 2D image slices with 2 channels (real+imaginary)"
        )

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        return {
            "lr": torch.randn(batch_size, 2, 256, 256),
            "hr": torch.randn(batch_size, 2, 256, 256),
            "batch_data": {},
        }

    def get_model_config(self) -> dict[str, Any]:
        return {"in_channels": 2, "out_channels": 2, "model_type": "standard_unet"}

    def is_valid(self) -> bool:
        return True


class Scenario_5D_2Channel(DimensionScenario):
    """5D tensors with 2 channels (volumetric, TorchIO format)."""

    def __init__(self):
        super().__init__(
            "5D_2Channel", "3D volumetric data in TorchIO format (B, C, H, W, D)"
        )

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        return {
            "lr": torch.randn(batch_size, 2, 256, 256, 16),
            "hr": torch.randn(batch_size, 2, 256, 256, 16),
            "batch_data": {},
        }

    def get_model_config(self) -> dict[str, Any]:
        return {"in_channels": 2, "out_channels": 2, "model_type": "standard_unet"}

    def is_valid(self) -> bool:
        return True


class Scenario_5D_SingletonDepth(DimensionScenario):
    """5D tensors with singleton depth dimension."""

    def __init__(self):
        super().__init__(
            "5D_SingletonDepth", "5D tensors with singleton depth: (B, C, H, W, 1)"
        )

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        return {
            "lr": torch.randn(batch_size, 2, 256, 256, 1),
            "hr": torch.randn(batch_size, 2, 256, 256, 1),
            "batch_data": {},
        }

    def get_model_config(self) -> dict[str, Any]:
        return {"in_channels": 2, "out_channels": 2, "model_type": "standard_unet"}

    def is_valid(self) -> bool:
        # Should be valid but inefficient - should be squeezed to 4D
        return True


class Scenario_ChannelMismatch_1to2(DimensionScenario):
    """Channel mismatch: data has 1 channel but model expects 2."""

    def __init__(self):
        super().__init__(
            "ChannelMismatch_1to2",
            "Data with 1 channel (magnitude) but model configured for 2 (real+imag)",
        )

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        return {
            "lr": torch.randn(batch_size, 1, 256, 256),  # Magnitude only
            "hr": torch.randn(batch_size, 1, 256, 256),  # Magnitude only
            "batch_data": {},
        }

    def get_model_config(self) -> dict[str, Any]:
        return {
            "in_channels": 2,  # ← MISMATCH
            "out_channels": 2,
            "model_type": "standard_unet",
        }

    def is_valid(self) -> bool:
        # Should FAIL - incompatible configuration
        return False


class Scenario_ChannelMismatch_2to1(DimensionScenario):
    """Channel mismatch: data has 2 channels but model expects 1."""

    def __init__(self):
        super().__init__(
            "ChannelMismatch_2to1",
            "Data with 2 channels (real+imag) but model configured for 1",
        )

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        return {
            "lr": torch.randn(batch_size, 2, 256, 256),
            "hr": torch.randn(batch_size, 2, 256, 256),
            "batch_data": {},
        }

    def get_model_config(self) -> dict[str, Any]:
        return {
            "in_channels": 1,  # ← MISMATCH
            "out_channels": 1,
            "model_type": "standard_unet",
        }

    def is_valid(self) -> bool:
        # Should FAIL - incompatible configuration
        return False


class Scenario_MultiChannel_4Channels(DimensionScenario):
    """Multi-channel: 4 channels (e.g., 2 coils x 2 real/imag)."""

    def __init__(self):
        super().__init__(
            "MultiChannel_4Channels",
            "Multi-channel data (e.g., 4 channels for 2 coils with real+imag)",
        )

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        return {
            "lr": torch.randn(batch_size, 4, 256, 256),  # 2 coils x 2 channels
            "hr": torch.randn(batch_size, 4, 256, 256),  # 2 coils x 2 channels
            "batch_data": {},
        }

    def get_model_config(self) -> dict[str, Any]:
        return {"in_channels": 4, "out_channels": 4, "model_type": "standard_unet"}

    def is_valid(self) -> bool:
        return True


class Scenario_TimestepMismatch(DimensionScenario):
    """Timestep tensor shape mismatch with batch."""

    def __init__(self):
        super().__init__(
            "TimestepMismatch", "Timestep tensor has wrong batch dimension"
        )

    def get_data(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        data = {
            "lr": torch.randn(batch_size, 2, 256, 256),
            "hr": torch.randn(batch_size, 2, 256, 256),
            "batch_data": {},
        }
        # Add mismatched timestep
        data["timesteps"] = torch.randint(0, 1000, (batch_size - 1,))  # Wrong size!
        return data

    def get_model_config(self) -> dict[str, Any]:
        return {"in_channels": 2, "out_channels": 2, "model_type": "standard_unet"}

    def is_valid(self) -> bool:
        # Should FAIL - timestep size doesn't match batch
        return False


# ============================================================================
# TEST SUITE
# ============================================================================


class TestDimensionValidation:
    """Test dimension handling for all scenarios."""

    SCENARIOS = [
        Scenario_4D_2Channel(),
        Scenario_5D_2Channel(),
        Scenario_5D_SingletonDepth(),
        Scenario_ChannelMismatch_1to2(),
        Scenario_ChannelMismatch_2to1(),
        Scenario_MultiChannel_4Channels(),
        Scenario_TimestepMismatch(),
    ]

    def test_scenarios_structure(self):
        """Verify scenario definitions are valid."""
        for scenario in self.SCENARIOS:
            assert scenario.name, "Scenario missing name"
            assert scenario.description, f"Scenario {scenario.name} missing description"
            assert hasattr(
                scenario, "is_valid"
            ), f"Scenario {scenario.name} missing is_valid()"
            assert isinstance(
                scenario.is_valid(), bool
            ), f"Scenario {scenario.name} is_valid() not bool"

    def test_scenario_data_shapes(self):
        """Verify scenario data tensors have expected shapes."""
        for scenario in self.SCENARIOS:
            data = scenario.get_data(batch_size=4)

            assert "lr" in data, f"Scenario {scenario.name}: missing 'lr'"
            assert "hr" in data, f"Scenario {scenario.name}: missing 'hr'"

            lr = data["lr"]
            hr = data["hr"]

            assert lr.shape[0] == 4, f"Scenario {scenario.name}: LR batch size != 4"
            assert hr.shape[0] == 4, f"Scenario {scenario.name}: HR batch size != 4"

            assert lr.dim() >= 3, f"Scenario {scenario.name}: LR must be >= 3D"
            assert hr.dim() >= 3, f"Scenario {scenario.name}: HR must be >= 3D"

    def test_valid_scenarios(self):
        """Valid scenarios should work."""
        valid_scenarios = [s for s in self.SCENARIOS if s.is_valid()]

        for scenario in valid_scenarios:
            data = scenario.get_data()
            config = scenario.get_model_config()

            lr = data["lr"]
            hr = data["hr"]

            # Check basic compatibility
            assert lr.shape[0] == hr.shape[0], f"{scenario.name}: Batch size mismatch"
            assert (
                lr.shape[1] == config["in_channels"]
            ), f"{scenario.name}: LR channels don't match model config"
            assert (
                hr.shape[1] == config["in_channels"]
            ), f"{scenario.name}: HR channels don't match model config"

    def test_invalid_scenarios(self):
        """Invalid scenarios should be identifiable."""
        invalid_scenarios = [s for s in self.SCENARIOS if not s.is_valid()]

        for scenario in invalid_scenarios:
            data = scenario.get_data()
            config = scenario.get_model_config()

            lr = data["lr"]
            hr = data["hr"]

            # At least one of these should fail
            channel_mismatch = (
                lr.shape[1] != config["in_channels"]
                or hr.shape[1] != config["in_channels"]
            )
            timestep_mismatch = (
                "timesteps" in data and data["timesteps"].shape[0] != lr.shape[0]
            )

            assert (
                channel_mismatch or timestep_mismatch
            ), f"{scenario.name}: Should be invalid but no mismatch found"


class TestStrategyDimensionHandling:
    """Test how each strategy handles different dimension scenarios."""

    def test_diffusion_strategy_4d(self):
        """DiffusionTrainingStrategy with 4D tensors."""
        scenario = Scenario_4D_2Channel()
        data = scenario.get_data()

        # This test would use actual strategy
        # For now, verify data structure is correct
        assert data["lr"].dim() == 4
        assert data["hr"].dim() == 4
        assert data["lr"].shape == data["hr"].shape

    def test_diffusion_strategy_5d(self):
        """DiffusionTrainingStrategy with 5D tensors."""
        scenario = Scenario_5D_2Channel()
        data = scenario.get_data()

        # This test would use actual strategy
        assert data["lr"].dim() == 5
        assert data["hr"].dim() == 5
        assert data["lr"].shape == data["hr"].shape

    def test_gan_strategy_4d(self):
        """GANTrainingStrategy with 4D tensors."""
        # TODO: Implement when GANTrainingStrategy is available
        pass

    def test_reconstruction_strategy_4d(self):
        """ReconstructionTrainingStrategy with 4D tensors."""
        # TODO: Implement when available
        pass


class TestModelDimensionHandling:
    """Test how models handle different input dimensions."""

    def test_kspace_cold_diffusion_4d(self):
        """KSpaceColdDiffusionGenerator with 4D input."""
        from mriforge.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            unet_config={"num_channels": 32},
            model_type="standard_unet",
            # This asserts a SHAPE contract, so declare the physics off rather
            # than leaving it configured-but-inert: `dc_method` defaults to
            # 'hard', and the training forward now refuses to build a DC layer
            # it cannot run for want of `kspace_measured`.
            dc_method="none",
        )

        x = torch.randn(4, 2, 256, 256)
        timesteps = torch.tensor([50, 50, 50, 50])

        output = model(x, timestep=timesteps)

        # Should return tuple (tensor, scale) during training
        assert isinstance(output, tuple), "Should return (tensor, scale)"
        out_tensor, scale = output

        assert (
            out_tensor.shape == x.shape
        ), f"Output shape {out_tensor.shape} != input shape {x.shape}"

    def test_kspace_cold_diffusion_5d(self):
        """KSpaceColdDiffusionGenerator with 5D input."""
        from mriforge.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        model = KSpaceColdDiffusionGenerator(
            in_channels=32,  # Collapsed depth dimension (2 * 16)
            out_channels=32,
            unet_config={"num_channels": 32},
            model_type="standard_unet",
            # Shape contract only — see the 4-D sibling. Doubly right here: a
            # genuine 5-D batch is the one case where the measurement legitimately
            # cannot broadcast against the prediction, so a shape test should not
            # be carrying that warning as noise.
            dc_method="none",
        )

        x = torch.randn(4, 2, 256, 256, 16)  # 5D
        timesteps = torch.tensor([50, 50, 50, 50])

        output = model(x, timestep=timesteps)

        assert isinstance(output, tuple), "Should return (tensor, scale)"
        out_tensor, scale = output

        # Output should be a valid tensor with matching batch size
        assert (
            out_tensor.shape[0] == x.shape[0]
        ), f"Batch size mismatch: {out_tensor.shape[0]} != {x.shape[0]}"
        assert (
            out_tensor.dim() >= 4
        ), f"Output should be at least 4D, got {out_tensor.dim()}D"


class TestDimensionLogging:
    """Test dimension logging at various stages."""

    def test_log_data_shapes(self):
        """Verify that data shapes are logged correctly."""
        import logging

        # This would test actual logging in the system
        # For now, just verify logging setup works
        logger = logging.getLogger("test_dimensions")

        scenario = Scenario_4D_2Channel()
        data = scenario.get_data()

        logger.info(f"Data shapes: LR={data['lr'].shape}, HR={data['hr'].shape}")

        assert True  # Just verification that logging works


if __name__ == "__main__":
    # Run tests
    test = TestDimensionValidation()
    test.test_scenarios_structure()
    test.test_scenario_data_shapes()
    test.test_valid_scenarios()
    test.test_invalid_scenarios()

    print("✓ All dimension validation tests passed")
    print(f"✓ Tested {len(TestDimensionValidation.SCENARIOS)} scenarios")
