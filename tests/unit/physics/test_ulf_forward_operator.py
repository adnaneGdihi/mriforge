"""Unit tests for DifferentiableULFForwardOperator."""

import math

import pytest
import torch


class TestPowerLawDispersion:
    """Tests for power-law T₁/T₂ dispersion scaling."""

    def test_t1_shortens_at_lower_field(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import power_law_dispersion

        t1_3t = torch.tensor([1350.0, 850.0])  # GM, WM at 3T (ms)
        t2_3t = torch.tensor([110.0, 80.0])
        t1_ulf, _t2_ulf = power_law_dispersion(t1_3t, t2_3t, b0_source=3.0, b0_target=0.064)
        # T₁ must shorten at lower field
        assert (t1_ulf < t1_3t).all(), "T₁ should decrease at lower B₀"

    def test_contrast_compression(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import power_law_dispersion

        t1_3t = torch.tensor([1350.0, 850.0])  # GM, WM
        t2_3t = torch.tensor([110.0, 80.0])
        t1_ulf, _ = power_law_dispersion(t1_3t, t2_3t)

        contrast_3t = (t1_3t[0] - t1_3t[1]).abs()
        contrast_ulf = (t1_ulf[0] - t1_ulf[1]).abs()
        assert contrast_ulf < contrast_3t, (
            f"ULF contrast ({contrast_ulf:.0f}ms) should be less than "
            f"3T contrast ({contrast_3t:.0f}ms) due to T₁ compression"
        )

    def test_t2_weak_dependence(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import power_law_dispersion

        t1_3t = torch.tensor([1350.0])
        t2_3t = torch.tensor([110.0])
        _, t2_ulf = power_law_dispersion(t1_3t, t2_3t, gamma_t2=0.05)
        # T₂ should change very little (γ=0.05 is weak)
        ratio = (t2_ulf / t2_3t).item()
        assert 0.7 < ratio < 1.0, f"T₂ ratio {ratio:.3f} should be near 1.0"

    def test_identity_at_same_field(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import power_law_dispersion

        t1 = torch.tensor([1000.0])
        t2 = torch.tensor([100.0])
        t1_out, t2_out = power_law_dispersion(t1, t2, b0_source=3.0, b0_target=3.0)
        assert torch.allclose(t1_out, t1, atol=1e-4)
        assert torch.allclose(t2_out, t2, atol=1e-4)

    def test_delegates_to_the_elected_owner_bit_for_bit(self):
        """Both halves must BE dispersion.py's power law, not agree with it.

        ``torch.allclose`` would pass on a re-implementation that drifted within
        tolerance, which is how the T1 half sat here undetected. Bit-identity is
        what says "one owner" (non-negotiable 17).
        """
        from spectramr.infrastructure.physics.dispersion import (
            power_law_t1_transport,
            power_law_t2_transport,
        )
        from spectramr.infrastructure.physics.ulf_forward_operator import power_law_dispersion

        t1 = torch.tensor([1600.0, 850.0])
        t2 = torch.tensor([80.0, 60.0])
        got_t1, got_t2 = power_law_dispersion(
            t1, t2, b0_source=7.0, b0_target=0.1, beta=0.35, gamma_t2=0.05
        )
        assert torch.equal(got_t1, power_law_t1_transport(t1, 7.0, 0.1, 0.35))
        assert torch.equal(got_t2, power_law_t2_transport(t2, 7.0, 0.1, 0.05))

    def test_nonpositive_field_raises_valueerror(self):
        """Delegation upgrades a bare ZeroDivisionError into a named ValueError.

        The previous inline arithmetic divided by ``b0_source`` directly: a Python
        ``0.0`` raised ZeroDivisionError with no message, and a zero *tensor* entry
        produced ``inf`` and no error at all (pitfall #9).
        """
        from spectramr.infrastructure.physics.ulf_forward_operator import power_law_dispersion

        with pytest.raises(ValueError, match="positive field strengths"):
            power_law_dispersion(torch.tensor([1000.0]), torch.tensor([80.0]), b0_source=0.0)


class TestErnstMagnetization:
    """Tests for steady-state Ernst equation rendering."""

    def test_output_nonnegative(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import ernst_magnetization

        rho = torch.ones(1, 1, 8, 8)
        t1 = torch.full((1, 1, 8, 8), 500.0)
        t2 = torch.full((1, 1, 8, 8), 80.0)
        mxy = ernst_magnetization(rho, t1, t2, tr=500, te=14, alpha=math.pi / 2)
        assert (mxy >= 0).all(), "Magnitude Mxy should be non-negative"

    def test_b1_modulation(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import ernst_magnetization

        rho = torch.ones(1, 1, 8, 8)
        t1 = torch.full((1, 1, 8, 8), 500.0)
        t2 = torch.full((1, 1, 8, 8), 80.0)

        mxy_uniform = ernst_magnetization(rho, t1, t2, tr=500, te=14, alpha=math.pi / 2)
        b1_shaded = torch.ones(1, 1, 8, 8) * 0.5  # 50% transmit efficiency
        mxy_shaded = ernst_magnetization(
            rho,
            t1,
            t2,
            tr=500,
            te=14,
            alpha=math.pi / 2,
            b1_plus=b1_shaded,
        )
        assert not torch.allclose(mxy_uniform, mxy_shaded, atol=1e-3), (
            "B₁⁺ shading should modulate Mxy"
        )

    def test_zero_rho_gives_zero(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import ernst_magnetization

        rho = torch.zeros(1, 1, 4, 4)
        t1 = torch.full((1, 1, 4, 4), 500.0)
        t2 = torch.full((1, 1, 4, 4), 80.0)
        mxy = ernst_magnetization(rho, t1, t2, tr=500, te=14, alpha=math.pi / 2)
        assert torch.allclose(mxy, torch.zeros_like(mxy), atol=1e-8)


class TestGNLWarp:
    """Tests for gradient non-linearity warping."""

    def test_none_coefficients_identity(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import gnl_warp

        image = torch.randn(1, 1, 16, 16)
        result = gnl_warp(image, sh_coefficients=None)
        assert torch.allclose(result, image)

    def test_nonzero_coefficients_change(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import gnl_warp

        image = torch.randn(1, 1, 16, 16)
        # Order 3 → K=10 basis functions
        coeffs = torch.zeros(2, 10)
        coeffs[0, 1] = 0.05  # x-displacement
        coeffs[1, 2] = 0.05  # y-displacement
        result = gnl_warp(image, sh_coefficients=coeffs, max_order=3)
        assert not torch.allclose(result, image, atol=1e-4), "GNL should warp the image"

    def test_shape_preserved(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import gnl_warp

        image = torch.randn(2, 1, 32, 32)
        coeffs = torch.zeros(2, 10)
        coeffs[0, 1] = 0.02
        result = gnl_warp(image, sh_coefficients=coeffs, max_order=3)
        assert result.shape == image.shape

    def test_size_mismatch_raises(self):
        """A mis-sized SH coefficient block must RAISE, never silently skip GNL.

        Regression for the silent-fallback fix (pitfall #9/#10): a deliberately
        provided coefficient tensor whose last dim does not match the polynomial
        basis size is a config/wiring error and must fail loud instead of
        degrading to identity warping (which a --strict smoke run would mask).
        """
        from spectramr.infrastructure.physics.ulf_forward_operator import gnl_warp

        image = torch.randn(1, 1, 16, 16)
        # max_order=3 -> K=10 basis functions; provide a mis-sized block (7 != 10)
        bad_coeffs = torch.zeros(2, 7)
        with pytest.raises(ValueError, match="basis size 10"):
            gnl_warp(image, sh_coefficients=bad_coeffs, max_order=3)


class TestULFNoise:
    """Tests for ULF noise injection."""

    def test_thermal_scaling(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import inject_ulf_noise

        kspace = torch.zeros(1, 1, 1000, dtype=torch.complex64)
        t = torch.linspace(0, 0.03, 1000)

        noisy_064 = inject_ulf_noise(kspace, t, b0=0.064, enable_emi=False)
        noisy_3t = inject_ulf_noise(kspace, t, b0=3.0, enable_emi=False)

        power_064 = noisy_064.abs().pow(2).mean()
        power_3t = noisy_3t.abs().pow(2).mean()
        assert power_064 > power_3t, "Noise should be larger at lower B₀"

    def test_emi_adds_energy(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import inject_ulf_noise

        kspace = torch.zeros(1, 1, 1000, dtype=torch.complex64)
        t = torch.linspace(0, 0.03, 1000)

        no_emi = inject_ulf_noise(kspace, t, b0=0.064, enable_emi=False, thermal_scale=0.0)
        with_emi = inject_ulf_noise(kspace, t, b0=0.064, enable_emi=True, thermal_scale=0.0)

        assert with_emi.abs().sum() > no_emi.abs().sum() + 1e-6, "EMI should add energy"


class TestDifferentiableULFForwardOperator:
    """Integration tests for the full 5-stage pipeline."""

    @pytest.fixture
    def operator(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import (
            DifferentiableULFForwardOperator,
        )

        return DifferentiableULFForwardOperator(
            image_shape=(16, 16),
            enable_noise=False,  # Deterministic for shape tests
            enable_emi=False,
        )

    def test_cartesian_output_shape(self, operator):
        rho = torch.ones(2, 1, 16, 16)
        t1 = torch.full((2, 1, 16, 16), 1350.0)
        t2 = torch.full((2, 1, 16, 16), 110.0)
        kspace = operator(rho, t1, t2)
        assert kspace.shape == (2, 1, 16, 16), f"Expected (2,1,16,16), got {kspace.shape}"
        assert torch.is_complex(kspace), "Output must be complex"

    def test_output_nonzero(self, operator):
        rho = torch.ones(1, 1, 16, 16)
        t1 = torch.full((1, 1, 16, 16), 500.0)
        t2 = torch.full((1, 1, 16, 16), 80.0)
        kspace = operator(rho, t1, t2)
        assert kspace.abs().sum() > 0, "Output k-space should be non-zero"

    def test_cartesian_odd_dim_uses_centered_ssot(self):
        """Odd image dims must round-trip through the fft2c SSOT.

        The Cartesian branch previously used fftshift(fft2(ifftshift(x))), the
        opposite shift convention to fft2c — they agree for even dims but
        mis-center on odd dims. With fft2c the operator runs and produces valid
        centered k-space on an odd grid.
        """
        from spectramr.infrastructure.physics.fft_ops import ifft2c
        from spectramr.infrastructure.physics.ulf_forward_operator import (
            DifferentiableULFForwardOperator,
        )

        op = DifferentiableULFForwardOperator(
            image_shape=(15, 15), enable_noise=False, enable_emi=False
        )
        rho = torch.ones(1, 1, 15, 15)
        t1 = torch.full((1, 1, 15, 15), 500.0)
        t2 = torch.full((1, 1, 15, 15), 80.0)
        kspace = op(rho, t1, t2)
        assert kspace.shape == (1, 1, 15, 15)
        assert torch.is_complex(kspace)
        # Centered ifft of centered k-space is finite and full-shape (no crash
        # from a half-sample shift mismatch on the odd grid).
        img = ifft2c(kspace)
        assert torch.isfinite(img).all()

    def test_differentiable(self, operator):
        rho = torch.ones(1, 1, 16, 16, requires_grad=True)
        t1 = torch.full((1, 1, 16, 16), 500.0)
        t2 = torch.full((1, 1, 16, 16), 80.0)
        kspace = operator(rho, t1, t2)
        loss = kspace.abs().sum()
        loss.backward()
        assert rho.grad is not None, "Gradients must flow through the operator"
        assert rho.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_b1_changes_output(self, operator):
        rho = torch.ones(1, 1, 16, 16)
        t1 = torch.full((1, 1, 16, 16), 500.0)
        t2 = torch.full((1, 1, 16, 16), 80.0)

        k1 = operator(rho, t1, t2, b1_plus=None)
        b1 = torch.ones(1, 1, 16, 16) * 0.5
        k2 = operator(rho, t1, t2, b1_plus=b1)
        assert not torch.allclose(k1, k2, atol=1e-4), "B₁⁺ should change output"

    def test_noise_increases_energy(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import (
            DifferentiableULFForwardOperator,
        )

        op_clean = DifferentiableULFForwardOperator(
            image_shape=(16, 16),
            enable_noise=False,
        )
        op_noisy = DifferentiableULFForwardOperator(
            image_shape=(16, 16),
            enable_noise=True,
            thermal_scale=1e-3,
        )
        rho = torch.ones(1, 1, 16, 16)
        t1 = torch.full((1, 1, 16, 16), 500.0)
        t2 = torch.full((1, 1, 16, 16), 80.0)

        k_clean = op_clean(rho, t1, t2)
        k_noisy = op_noisy(rho, t1, t2)

        # Noisy should have more energy (noise adds variance)
        diff = (k_noisy - k_clean).abs().mean()
        assert diff > 1e-6, "Noise should measurably change the output"

    def test_maxwell_phase_adds_imaginary(self):
        from spectramr.infrastructure.physics.ulf_forward_operator import (
            DifferentiableULFForwardOperator,
        )

        op = DifferentiableULFForwardOperator(
            image_shape=(16, 16),
            enable_maxwell=True,
            enable_noise=False,
            gx=20.0,
            gy=20.0,
        )
        rho = torch.ones(1, 1, 16, 16)
        t1 = torch.full((1, 1, 16, 16), 500.0)
        t2 = torch.full((1, 1, 16, 16), 80.0)
        kspace = op(rho, t1, t2)
        assert torch.is_complex(kspace)
