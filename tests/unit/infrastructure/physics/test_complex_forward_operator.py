"""Unit tests for ComplexForwardOperator.

Tests the full MRI signal equation forward degradation operator:
- FFT/IFFT roundtrip with coil sensitivity
- B0 phase application
- Complex Gaussian noise properties
- Composite operator correctness
"""

import torch

from spectramr.infrastructure.physics.complex_forward_operator import (
    ComplexForwardOperator,
    add_complex_noise,
    apply_b0_phase,
    apply_coil_sensitivity,
    apply_sampling_mask,
)
from spectramr.infrastructure.physics.fft_ops import fft2c


class TestApplyCoilSensitivity:
    """Tests for coil sensitivity modulation."""

    def test_none_maps_returns_unchanged(self):
        """No sensitivity maps → identity."""
        img = torch.randn(2, 1, 32, 32) + 0j
        out = apply_coil_sensitivity(img, None)
        assert torch.allclose(out, img)

    def test_single_channel_broadcast(self):
        """Single-channel image broadcasts to N coils."""
        img = torch.complex(torch.randn(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))
        smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
        out = apply_coil_sensitivity(img, smaps)
        assert out.shape == (1, 4, 16, 16)

    def test_parseval_energy(self):
        """Coil-modulated image should have energy proportional to coil norms."""
        img = torch.complex(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
        smaps = torch.complex(torch.ones(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))
        out = apply_coil_sensitivity(img, smaps)
        assert torch.allclose(out.abs(), torch.ones(1, 2, 8, 8))


class TestApplyB0Phase:
    """Tests for off-resonance phase accrual."""

    def test_zero_echo_time_no_change(self):
        """TE=0 → no phase shift."""
        img = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        b0 = torch.randn(1, 1, 16, 16)
        out = apply_b0_phase(img, b0, echo_time=0.0)
        assert torch.allclose(out, img)

    def test_none_b0_returns_unchanged(self):
        """No B0 map → identity."""
        img = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        out = apply_b0_phase(img, None, echo_time=0.01)
        assert torch.allclose(out, img)

    def test_magnitude_preserved(self):
        """Phase modulation should preserve magnitude."""
        img = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        b0 = torch.randn(1, 1, 16, 16) * 100.0
        out = apply_b0_phase(img, b0, echo_time=0.01)
        assert torch.allclose(out.abs(), img.abs(), atol=1e-5)

    def test_phase_shift_direction(self):
        """Positive B0 with positive TE should add negative phase."""
        img = torch.complex(torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
        b0 = torch.ones(1, 1, 4, 4) * 100.0  # 100 rad/s
        te = 0.01  # 10ms
        out = apply_b0_phase(img, b0, echo_time=te)
        expected_phase = -100.0 * 0.01  # -1.0 rad
        assert torch.allclose(out.angle(), torch.full_like(out.real, expected_phase), atol=1e-5)


class TestApplySamplingMask:
    """Tests for k-space undersampling."""

    def test_no_mask_full_density(self):
        """No mask + density=1 → identity."""
        k = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        out = apply_sampling_mask(k, None, density=1.0)
        assert torch.allclose(out, k)

    def test_binary_mask_application(self):
        """Binary mask zeros out unsampled frequencies."""
        k = torch.complex(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4, :] = 1.0
        out = apply_sampling_mask(k, mask, density=1.0)
        assert out[..., 4:, :].abs().sum() == 0.0
        assert out[..., :4, :].abs().sum() > 0.0


class TestAddComplexNoise:
    """Tests for complex Gaussian noise injection."""

    def test_zero_sigma_no_noise(self):
        """σ=0 → no noise added."""
        k = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        out = add_complex_noise(k, sigma=0.0)
        assert torch.allclose(out, k)

    def test_noise_zero_mean(self):
        """Large-sample noise should be approximately zero-mean."""
        k = torch.zeros(1, 1, 256, 256, dtype=torch.complex64)
        out = add_complex_noise(k, sigma=1.0)
        # Check mean is near zero (large sample → CLT)
        assert out.real.mean().abs() < 0.1
        assert out.imag.mean().abs() < 0.1

    def test_noise_variance(self):
        """Noise variance should match σ²."""
        k = torch.zeros(1, 1, 512, 512, dtype=torch.complex64)
        sigma = 0.5
        out = add_complex_noise(k, sigma=sigma)
        var_real = out.real.var().item()
        var_imag = out.imag.var().item()
        assert abs(var_real - sigma**2) < 0.05
        assert abs(var_imag - sigma**2) < 0.05


class TestComplexForwardOperator:
    """Tests for the composite forward operator."""

    def test_identity_at_t0(self):
        """At t=0 (clean), operator should produce mostly-intact k-space."""
        op = ComplexForwardOperator(num_timesteps=100, max_sigma=0.0)
        img = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
        k_out = op(img, timestep=0)
        # At t=0: density=1.0, sigma=0 → should match FFT of input
        k_expected = fft2c(img)
        assert torch.allclose(k_out, k_expected, atol=1e-5)

    def test_degradation_increases_with_t(self):
        """Higher timestep should produce more degraded output."""
        op = ComplexForwardOperator(num_timesteps=100, max_sigma=0.1, min_density=0.1)
        img = torch.complex(torch.randn(1, 1, 32, 32), torch.zeros(1, 1, 32, 32))
        mask = torch.ones(1, 1, 32, 32)

        torch.manual_seed(0)
        k_low = op(img, timestep=10, mask=mask)
        torch.manual_seed(0)
        k_high = op(img, timestep=90, mask=mask)

        # Higher t → more energy removed (lower norm expected)
        # At t=90, density is very low, so energy should be much lower
        assert k_high.abs().sum() < k_low.abs().sum()

    def test_output_shape_preserved(self):
        """Output shape should match input k-space dimensions."""
        op = ComplexForwardOperator()
        img = torch.complex(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64))
        k_out = op(img, timestep=50)
        assert k_out.shape == (2, 1, 64, 64)

    def test_with_coils_and_b0(self):
        """Full signal equation with all physical operators."""
        op = ComplexForwardOperator(num_timesteps=50, max_sigma=0.01)
        img = torch.complex(torch.randn(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))
        smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
        b0 = torch.randn(1, 1, 16, 16) * 50.0
        mask = torch.ones(1, 1, 16, 16)

        k_out = op(
            img, timestep=25,
            mask=mask, sensitivity_maps=smaps, b0_map=b0, echo_time=0.005,
        )
        assert k_out.shape == (1, 4, 16, 16)  # Expanded to N_coils
        assert torch.is_complex(k_out)

    def test_cosine_schedule(self):
        """Cosine density schedule produces slower initial drop-off."""
        op_lin = ComplexForwardOperator(density_schedule="linear")
        op_cos = ComplexForwardOperator(density_schedule="cosine")

        # At t = T/4, cosine should retain more density than linear
        d_lin = op_lin._density_at(25)
        d_cos = op_cos._density_at(25)
        assert d_cos > d_lin

    def test_2channel_real_input(self):
        """Accept 2-channel real/imag input format."""
        op = ComplexForwardOperator(num_timesteps=10, max_sigma=0.0)
        img_ri = torch.randn(1, 2, 32, 32)  # 2-channel real/imag
        k_out = op(img_ri, timestep=0)
        assert torch.is_complex(k_out)
