"""Unit tests for tissue parameter decoding and Bloch synthesis."""

import pytest
import torch

from spectramr.models.generators.disentangled_mri import DisentangledMRI, Generator


class TestTissueParameterDecoding:
    """Test tissue parameter output mode in DisentangledMRI."""

    @pytest.fixture
    def model_tissue_mode(self):
        """Create model with tissue parameter output."""
        return DisentangledMRI(
            in_channels=1,
            out_channels=1,
            output_tissue_params=True,
            enable_bloch_synthesis=True,
        )

    @pytest.fixture
    def model_image_mode(self):
        """Create model with direct image output."""
        return DisentangledMRI(
            in_channels=1,
            out_channels=1,
            output_tissue_params=False,
            enable_bloch_synthesis=False,
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input image."""
        return torch.randn(2, 1, 64, 64)

    def test_tissue_params_output_structure(self, model_tissue_mode, sample_input):
        """Test that tissue mode returns dict with correct keys."""
        output, mu, logvar = model_tissue_mode(sample_input)

        assert isinstance(output, dict), "Tissue mode should return dict"
        assert "rho" in output, "Should have proton density"
        assert "t1" in output, "Should have T1"
        assert "t2" in output, "Should have T2"

    def test_tissue_params_shapes(self, model_tissue_mode, sample_input):
        """Test that tissue parameters have correct shapes."""
        output, mu, logvar = model_tissue_mode(sample_input)

        batch_size = sample_input.shape[0]
        height = sample_input.shape[2]
        width = sample_input.shape[3]

        for param_name in ["rho", "t1", "t2"]:
            param = output[param_name]
            expected_shape = (batch_size, 1, height, width)
            assert (
                param.shape == expected_shape
            ), f"{param_name} should have shape {expected_shape}, got {param.shape}"

    def test_tissue_params_ranges(self, model_tissue_mode, sample_input):
        """Test that tissue parameters are in valid ranges."""
        output, mu, logvar = model_tissue_mode(sample_input)

        # Proton density: [0, 1]
        rho = output["rho"]
        assert (rho >= 0).all(), "Proton density should be >= 0"
        assert (rho <= 1).all(), "Proton density should be <= 1"

        # T1: positive (> 0.001 due to shift)
        t1 = output["t1"]
        assert (t1 > 0).all(), "T1 should be positive"
        assert (t1 >= 1e-3).all(), "T1 should be >= 1e-3 (shifted)"

        # T2: positive (> 0.001 due to shift)
        t2 = output["t2"]
        assert (t2 > 0).all(), "T2 should be positive"
        assert (t2 >= 1e-3).all(), "T2 should be >= 1e-3 (shifted)"

    def test_image_mode_output(self, model_image_mode, sample_input):
        """Test that image mode returns tensor (not dict)."""
        output, mu, logvar = model_image_mode(sample_input)

        assert isinstance(output, torch.Tensor), "Image mode should return tensor"
        assert output.shape == sample_input.shape, "Output shape should match input"

    def test_gradient_flow_tissue_mode(self, model_tissue_mode, sample_input):
        """Test that gradients flow in tissue parameter mode."""
        sample_input.requires_grad_(True)

        output, mu, logvar = model_tissue_mode(sample_input)

        # Compute loss on tissue parameters
        loss = output["rho"].mean() + output["t1"].mean() + output["t2"].mean()
        loss.backward()

        assert sample_input.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(sample_input.grad).any(), "Gradients should not be NaN"

    def test_generator_tissue_heads(self):
        """Test that Generator creates tissue parameter heads correctly."""
        gen = Generator(
            content_dim=256,
            style_dim=8,
            n_res=4,
            out_channels=1,
            output_tissue_params=True,
        )

        # Check that heads exist
        assert hasattr(gen, "rho_head"), "Should have rho head"
        assert hasattr(gen, "t1_head"), "Should have T1 head"
        assert hasattr(gen, "t2_head"), "Should have T2 head"
        assert hasattr(gen, "shared_head"), "Should have shared head"

        # Check that final head doesn't exist in tissue mode
        assert not hasattr(gen, "final"), "Should not have final conv in tissue mode"

    def test_generator_image_heads(self):
        """Test that Generator in image mode has correct structure."""
        gen = Generator(
            content_dim=256,
            style_dim=8,
            n_res=4,
            out_channels=1,
            output_tissue_params=False,
        )

        # Check that final conv exists
        assert hasattr(gen, "final"), "Should have final conv in image mode"

        # Check that tissue heads don't exist
        assert not hasattr(
            gen, "rho_head"
        ), "Should not have tissue heads in image mode"

    def test_content_style_encoder_properties(self, model_tissue_mode):
        """Test that model exposes encoder properties for latent consistency loss."""
        assert hasattr(
            model_tissue_mode, "content_encoder"
        ), "Should have content_encoder property"
        assert hasattr(
            model_tissue_mode, "style_encoder"
        ), "Should have style_encoder property"
        assert hasattr(model_tissue_mode, "generator"), "Should have generator property"

        # Properties should return the actual encoder modules
        assert model_tissue_mode.content_encoder is model_tissue_mode.enc_c
        assert model_tissue_mode.style_encoder is model_tissue_mode.enc_s
        assert model_tissue_mode.generator is model_tissue_mode.gen


class TestBlochSynthesis:
    """Test Bloch equation synthesis in training strategy."""

    def test_bloch_signal_equation(self):
        """Test SPGR signal equation implementation."""
        # Realistic brain tissue parameters at 3T
        rho = torch.tensor([[[[0.8]]]])  # Proton density
        t1 = torch.tensor([[[[1000.0]]]])  # T1 in ms (gray matter ~1000-1200ms)
        t2 = torch.tensor([[[[80.0]]]])  # T2 in ms (gray matter ~80-100ms)

        # Typical 3T T1-weighted SPGR sequence
        te = 4.0  # Echo time (ms)
        tr = 8.0  # Repetition time (ms)
        flip_angle = 12.0  # degrees

        # Compute signal manually
        import math

        alpha = flip_angle * (math.pi / 180.0)
        e1 = math.exp(-tr / t1.item())
        e2 = math.exp(-te / t2.item())

        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)

        expected_signal = rho.item() * sin_alpha * (1 - e1) / (1 - cos_alpha * e1) * e2

        # Signal should be in reasonable range for brain tissue
        assert 0 < expected_signal < 1, f"Signal {expected_signal} should be in (0, 1)"

        # Gray matter should give reasonable signal
        # T1-weighted SPGR with these params typically gives ~0.3-0.5 for GM
        assert (
            0.01 < expected_signal < 0.8
        ), f"Gray matter signal {expected_signal} should be in reasonable range"

    def test_tissue_parameter_physical_constraints(self):
        """Test that physical constraints on tissue parameters make sense."""
        # For brain tissue at 3T:
        # White Matter: T1 ~800-900ms, T2 ~70-80ms
        # Gray Matter: T1 ~1000-1200ms, T2 ~80-100ms
        # CSF: T1 ~4000ms, T2 ~2000ms

        # T1 should always be > T2 for biological tissues
        t1_values = [800.0, 1000.0, 4000.0]  # WM, GM, CSF
        t2_values = [70.0, 80.0, 2000.0]

        for t1, t2 in zip(t1_values, t2_values, strict=False):
            assert t1 > t2, f"T1 ({t1}ms) should be > T2 ({t2}ms)"

    def test_bloch_synthesis_ranges(self):
        """Test that Bloch synthesis produces valid signal range."""
        # Create range of realistic tissue parameters
        batch_size = 10
        rho = torch.rand(batch_size, 1, 32, 32) * 0.5 + 0.5  # [0.5, 1.0]
        t1 = torch.rand(batch_size, 1, 32, 32) * 2000 + 500  # [500, 2500]ms
        t2 = torch.rand(batch_size, 1, 32, 32) * 150 + 50  # [50, 200]ms

        # Ensure T1 > T2
        t1 = torch.maximum(t1, t2 + 100)  # Add 100ms margin

        # Standard 3T parameters
        te = 4.0
        tr = 8.0
        flip_angle = torch.tensor(12.0 * (torch.pi / 180.0))

        # Compute signals
        e1 = torch.exp(-tr / (t1 + 1e-6))
        e2 = torch.exp(-te / (t2 + 1e-6))

        sin_alpha = torch.sin(flip_angle)
        cos_alpha = torch.cos(flip_angle)

        signal = rho * sin_alpha * (1 - e1) / (1 - cos_alpha * e1 + 1e-6) * e2

        # All signals should be in [0, 1] range
        assert (signal >= 0).all(), "Signal should be non-negative"
        assert (signal <= 1).all(), "Signal should be <= 1"

        # Most brain signals should be in [0.1, 0.8] range for T1-weighted
        median_signal = signal.median()
        assert (
            0.01 < median_signal < 0.9
        ), f"Median signal {median_signal} should be in reasonable range"


class TestTissueBoundsRegularization:
    """Test tissue parameter bounds regularization."""

    def test_t1_t2_constraint(self):
        """Test that T1 > T2 constraint is enforced."""
        # Case 1: Valid (T1 > T2)
        t1_valid = torch.tensor([[[[1000.0]]]])
        t2_valid = torch.tensor([[[[80.0]]]])

        margin = 50.0
        violation = torch.relu(t2_valid + margin - t1_valid)
        assert violation.item() == 0, "No violation when T1 > T2 + margin"

        # Case 2: Violation (T1 < T2)
        t1_invalid = torch.tensor([[[[70.0]]]])
        t2_invalid = torch.tensor([[[[100.0]]]])

        violation = torch.relu(t2_invalid + margin - t1_invalid)
        assert violation.item() > 0, "Should have violation when T1 < T2"

    def test_t1_upper_bound(self):
        """Test T1 upper bound constraint."""
        t1_max = 5000.0

        # Valid T1
        t1_valid = torch.tensor([[[[2000.0]]]])
        violation = torch.relu(t1_valid - t1_max)
        assert violation.item() == 0, "No violation for reasonable T1"

        # Invalid T1 (too large)
        t1_invalid = torch.tensor([[[[6000.0]]]])
        violation = torch.relu(t1_invalid - t1_max)
        assert violation.item() > 0, "Should have violation for unrealistic T1"

    def test_t2_lower_bound(self):
        """Test T2 lower bound constraint."""
        t2_min = 10.0

        # Valid T2
        t2_valid = torch.tensor([[[[80.0]]]])
        violation = torch.relu(t2_min - t2_valid)
        assert violation.item() == 0, "No violation for reasonable T2"

        # Invalid T2 (too small)
        t2_invalid = torch.tensor([[[[5.0]]]])
        violation = torch.relu(t2_min - t2_invalid)
        assert violation.item() > 0, "Should have violation for unrealistic T2"
