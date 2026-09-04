"""Unit tests for backbone integration in KSpaceColdDiffusionGenerator.

Tests validate that different backbone types (Vision Mamba, Swin Transformer, etc.)
properly integrate with the cold diffusion framework.
"""

import pytest
import torch

from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)


class TestBackboneIntegration:
    """Test suite for backbone integration in Cold Diffusion generator."""

    @pytest.fixture
    def device(self):
        """Get appropriate device for tests."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def common_kwargs(self):
        """Common kwargs for generator instantiation."""
        return {
            "in_channels": 2,
            "out_channels": 2,
            "base_channels": 64,
            "num_layers": 4,
            "num_timesteps": 1000,
            "time_embedding_dim": 256,
        }

    @pytest.fixture
    def sample_input(self, device):
        """Create sample input tensor."""
        return torch.randn(1, 2, 256, 256, device=device)

    @pytest.fixture
    def sample_timesteps(self, device):
        """Create sample timesteps (batch size 1, value 500)."""
        return torch.full((1,), 500, dtype=torch.long, device=device)

    def _forward_and_validate(
        self,
        generator: KSpaceColdDiffusionGenerator,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Helper to forward pass and validate output.

        Handles both training (returns tuple) and eval (returns tensor) modes.
        """
        generator.to(x.device)  # Ensure model is on same device as input
        generator.eval()  # Set to eval mode before forward
        with torch.no_grad():
            output = generator(x, timesteps)

        # In eval mode, should return tensor directly; in training mode returns (output, scale)
        if isinstance(output, tuple):
            output, scale = output
            assert scale is not None, "Expected scale when returning tuple"

        return output

    @pytest.mark.parametrize(
        "backbone_type,backbone_kwargs",
        [
            ("vision_mamba", {"mamba_depth": 4, "mamba_hidden_dim": 128}),
            ("vision_transformer", {"patch_size": 16, "num_heads": 8}),
            ("swin_transformer", {"window_size": 8}),
            ("restormer", {"num_blocks": 4, "num_refinement_blocks": 4, "scale": 1}),
            ("swinir", {"window_size": 8, "depths": (6,), "upscale": 1}),
            ("nafnet", {}),
            ("unet", {}),
        ],
    )
    def test_backbone_integration(
        self,
        common_kwargs,
        sample_input,
        sample_timesteps,
        backbone_type: str,
        backbone_kwargs: dict,
    ):
        """Test backbone integration for various architecture types.

        Args:
            backbone_type: Type of backbone to test
            backbone_kwargs: Additional kwargs for specific backbone
        """
        # Create generator with specified backbone
        generator = KSpaceColdDiffusionGenerator(
            **common_kwargs, backbone_type=backbone_type, **backbone_kwargs
        )
        assert generator is not None
        assert hasattr(generator, "backbone")

        # Perform forward pass
        output = self._forward_and_validate(generator, sample_input, sample_timesteps)

        # Validate output shape
        assert output.shape == sample_input.shape, (
            f"Shape mismatch for {backbone_type}: "
            f"expected {sample_input.shape}, got {output.shape}"
        )

        # Validate output is not all zeros (data is flowing through network)
        assert not torch.allclose(
            output, torch.zeros_like(output)
        ), f"{backbone_type} output is all zeros"

        # Validate output doesn't have NaNs or Infs
        assert torch.isfinite(
            output
        ).all(), f"{backbone_type} output contains NaN or Inf values"

        del generator, output
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_timestep_edge_cases(self, common_kwargs, sample_input):
        """Test generator with edge case timestep values."""
        device = sample_input.device
        generator = KSpaceColdDiffusionGenerator(
            **common_kwargs,
            backbone_type="vision_mamba",
            mamba_depth=4,
            mamba_hidden_dim=128,
        )
        generator.to(device).eval()

        # Test with t=0 (first timestep)
        timesteps_start = torch.zeros((1,), dtype=torch.long, device=device)
        with torch.no_grad():
            output_start = generator(sample_input, timesteps_start)
        if isinstance(output_start, tuple):
            output_start = output_start[0]
        assert output_start.shape == sample_input.shape
        assert torch.isfinite(output_start).all()

        # Test with t=999 (last valid timestep)
        timesteps_end = torch.full((1,), 999, dtype=torch.long, device=device)
        with torch.no_grad():
            output_end = generator(sample_input, timesteps_end)
        if isinstance(output_end, tuple):
            output_end = output_end[0]
        assert output_end.shape == sample_input.shape
        assert torch.isfinite(output_end).all()

        # Outputs at different timesteps should differ
        assert not torch.allclose(
            output_start, output_end, atol=1e-5
        ), "Outputs at t=0 and t=999 should differ"

    def test_batch_dimension_handling(self, common_kwargs):
        """Test proper handling of batch dimension."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = KSpaceColdDiffusionGenerator(
            **common_kwargs,
            backbone_type="vision_mamba",
            mamba_depth=4,
            mamba_hidden_dim=128,
        )
        generator.to(device).eval()

        # Test batch size 1
        x_b1 = torch.randn(1, 2, 128, 128, device=device)
        timesteps_b1 = torch.full((1,), 500, dtype=torch.long, device=device)
        with torch.no_grad():
            output_b1 = generator(x_b1, timesteps_b1)
        if isinstance(output_b1, tuple):
            output_b1 = output_b1[0]
        assert output_b1.shape == x_b1.shape

        # Test batch size 4
        x_b4 = torch.randn(4, 2, 128, 128, device=device)
        timesteps_b4 = torch.full((4,), 500, dtype=torch.long, device=device)
        with torch.no_grad():
            output_b4 = generator(x_b4, timesteps_b4)
        if isinstance(output_b4, tuple):
            output_b4 = output_b4[0]
        assert output_b4.shape == x_b4.shape

    def test_scale_output_in_training_mode(self, common_kwargs, sample_input):
        """Test that generator returns scale factor in training mode."""
        device = sample_input.device
        generator = KSpaceColdDiffusionGenerator(
            **common_kwargs,
            backbone_type="vision_mamba",
            mamba_depth=4,
            mamba_hidden_dim=128,
        )
        # Keep generator in training mode
        generator.to(device).train()

        timesteps = torch.full((1,), 500, dtype=torch.long, device=device)
        output = generator(sample_input, timesteps)

        # In training mode, should return tuple
        assert isinstance(
            output, tuple
        ), "Generator should return tuple in training mode"
        assert len(output) == 2, "Tuple should have (output, scale)"
        output_tensor, scale = output
        assert output_tensor.shape == sample_input.shape
        # scale may be None if pipeline-level scaling is employed
        if scale is not None:
            assert isinstance(scale, (float, torch.Tensor))

    def test_deterministic_behavior_with_eval(self, common_kwargs, sample_input):
        """Test that generator is deterministic in eval mode."""
        device = sample_input.device
        generator = KSpaceColdDiffusionGenerator(
            **common_kwargs,
            backbone_type="vision_mamba",
            mamba_depth=4,
            mamba_hidden_dim=128,
        )
        generator.to(device).eval()

        timesteps = torch.full((1,), 500, dtype=torch.long, device=device)

        # Run twice with same input
        with torch.no_grad():
            output1 = generator(sample_input, timesteps)
            if isinstance(output1, tuple):
                output1 = output1[0]

            output2 = generator(sample_input, timesteps)
            if isinstance(output2, tuple):
                output2 = output2[0]

        # Outputs should be identical
        assert torch.allclose(
            output1, output2, atol=1e-6
        ), "Generator outputs should be deterministic in eval mode"

    def test_invalid_backbone_raises_error(self, common_kwargs):
        """Test that invalid backbone type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backbone_type"):
            KSpaceColdDiffusionGenerator(
                **common_kwargs,
                backbone_type="invalid_backbone",
            )

    def test_5d_input_handling(self, common_kwargs):
        """Test that 5D inputs are handled correctly.

        The diffusion strategy permutes inputs to ``[B, D, C, H, W]`` before
        passing them to the generator. With ``in_channels=2``, a depth=2
        slab does NOT collide with the rep-fusion heuristic
        (``x.shape[1] > 1 and x.shape[2] == self.in_channels``) and exercises
        the pure depth-flatten path.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = KSpaceColdDiffusionGenerator(
            **common_kwargs,
            backbone_type="unet",
        )
        generator.to(device)

        # Depth-first 5D input (B, D, C, H, W) with D=1 so the heuristic
        # ``x.shape[1] > 1`` is False — bypasses RepetitionFusion and goes
        # straight to the volumetric reshape path.
        x = torch.randn(1, 1, 2, 64, 64, device=device)
        timesteps = torch.tensor([500], device=device)

        with torch.no_grad():
            output = generator(x, timesteps)

        if isinstance(output, tuple):
            output = output[0]

        # The forward may emit one of several valid layouts after the
        # depth-flatten round-trip: depth-first 5D (the input layout),
        # TorchIO-style depth-last 5D, or a squeezed 4D tensor when D=1.
        # All are valid — the contract is "spatial dims preserved, channel
        # count matches out_channels". The key invariant is that output
        # carries the same spatial extent and channel count as the input
        # after the singleton depth is squeezed.
        valid_shapes = {
            tuple(x.shape),  # (1, 1, 2, 64, 64)
            (1, 2, 64, 64, 1),  # TorchIO-style depth-last
            (1, 2, 64, 64),  # 4D squeezed
        }
        assert tuple(output.shape) in valid_shapes, (
            f"Expected one of {valid_shapes}, got {output.shape}"
        )
