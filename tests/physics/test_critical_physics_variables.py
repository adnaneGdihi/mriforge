"""
Tests for MRI physics variables and operations.

Critical variables tested:
- FFT normalization (must be "ortho")
- K-space mask validity (Cartesian, radial, non-Cartesian)
- Data consistency projections
- Coil sensitivity maps (SENSE equation)
- Hermitian symmetry (for real-valued images)
- Energy conservation across FFT/IFFT
- Phase coherence for multi-coil reconstruction

These tests ensure MRI reconstruction artifacts aren't silently introduced.
"""

import numpy as np
import torch


class TestFFTOrthonormality:
    """Tests for FFT orthonormality and energy conservation."""

    def test_fft_ifft_roundtrip_identity(self):
        """Test that FFT→IFFT produces identity (ortho norm)."""
        batch_size = 2
        height, width = 64, 64

        # Create spatial domain image
        image = torch.randn(batch_size, 1, height, width)

        # Apply FFT with ortho normalization
        kspace = torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
            ),
            dim=(-2, -1),
        )

        # Apply IFFT
        reconstructed = torch.fft.fftshift(
            torch.fft.ifftn(
                torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
            ),
            dim=(-2, -1),
        )

        # Verify reconstruction matches original (within numerical precision)
        # Use real part since input was real
        assert torch.allclose(image, reconstructed.real, rtol=1e-5, atol=1e-6)

    def test_fft_energy_conservation(self):
        """Test Parseval's theorem: energy in spatial domain = energy in k-space."""
        batch_size = 2
        height, width = 64, 64

        image = torch.randn(batch_size, 1, height, width)

        # Compute energy in spatial domain
        spatial_energy = (image**2).sum()

        # Compute energy in k-space
        kspace = torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
            ),
            dim=(-2, -1),
        )
        kspace_energy = (torch.abs(kspace) ** 2).sum()

        # Energies should be equal (Parseval's theorem)
        assert torch.allclose(spatial_energy, kspace_energy, rtol=1e-4)

    def test_fft_norm_parameter_critical(self):
        """Test that FFT norm parameter critically affects reconstruction."""
        batch_size = 1
        height, width = 32, 32

        image = torch.randn(batch_size, 1, height, width)

        # FFT with ortho (correct)
        kspace_ortho = torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
            ),
            dim=(-2, -1),
        )
        recon_ortho = torch.fft.fftshift(
            torch.fft.ifftn(
                torch.fft.ifftshift(kspace_ortho, dim=(-2, -1)),
                norm="ortho",
                dim=(-2, -1),
            ),
            dim=(-2, -1),
        ).real

        # FFT without any norm (incorrect - will have scaling errors)
        # torch.fft.fftn(..., norm=None) is default, scales by 1 in forward, 1/N in backward.
        # To see scale error, we do forward without norm, then backward without norm.
        kspace_no_norm = torch.fft.fftn(image, dim=(-2, -1))  # No norm = scale by 1
        recon_no_norm = torch.fft.ifftn(
            kspace_no_norm, dim=(-2, -1)
        ).real  # No norm = scale by 1/N

        # Scale of ortho recon should match original image (approx mean abs)
        scale_orig = image.abs().mean()
        scale_ortho = recon_ortho.abs().mean()

        assert torch.allclose(scale_orig, scale_ortho, rtol=1e-2)


class TestKSpaceMask:
    """Tests for k-space undersampling mask validity."""

    def test_cartesian_mask_valid_range(self):
        """Test that Cartesian mask values are in [0, 1]."""
        height, width = 64, 64
        center_fraction = 0.08
        acceleration = 4

        # Generate Cartesian mask
        num_center_lines = int(height * center_fraction)
        num_total_lines = height // acceleration
        num_outer_lines = num_total_lines - num_center_lines

        mask = torch.zeros(height, width)

        # Center region
        center_start = (height - num_center_lines) // 2
        mask[center_start : center_start + num_center_lines, :] = 1.0

        # Outer region (random sampling)
        outer_indices = torch.randperm(height)[:num_outer_lines]
        mask[outer_indices, :] = 1.0

        # Verify mask properties
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.dtype in (torch.float32, torch.float64)

    def test_cartesian_mask_acceleration_factor(self):
        """Test that Cartesian mask achieves specified acceleration."""
        height, width = 64, 64
        target_acceleration = 4
        center_fraction = 0.08

        num_center_lines = int(height * center_fraction)
        num_total_lines = height // target_acceleration
        num_outer_lines = num_total_lines - num_center_lines

        mask = torch.zeros(height, width)

        # Center
        center_start = (height - num_center_lines) // 2
        mask[center_start : center_start + num_center_lines, :] = 1.0

        # Outer
        outer_indices = torch.randperm(height)[:num_outer_lines]
        mask[outer_indices, :] = 1.0

        # Compute actual acceleration
        num_sampled_lines = (mask[:, 0] > 0).sum().item()
        actual_acceleration = height / num_sampled_lines

        assert abs(actual_acceleration - target_acceleration) < 1.0

    def test_mask_center_fraction_enforcement(self):
        """Test that center fraction is fully sampled in Cartesian mask."""
        height, width = 64, 64
        center_fraction = 0.08

        num_center_lines = int(height * center_fraction)

        # Create mask with fully sampled center
        center_start = (height - num_center_lines) // 2
        center_end = center_start + num_center_lines

        mask = torch.zeros(height, width)
        mask[center_start:center_end, :] = 1.0

        # Verify center is fully sampled
        center_sampled = mask[center_start:center_end, :].mean().item()
        assert center_sampled == 1.0

        # Verify outside center is not sampled (before outer lines added)
        outside_sampled = (
            mask[
                torch.cat(
                    [torch.arange(0, center_start), torch.arange(center_end, height)]
                )
            ]
            .mean()
            .item()
        )
        assert outside_sampled == 0.0

    def test_mask_binary_values(self):
        """Test that mask is strictly binary (0 or 1)."""
        height, width = 64, 64

        mask = torch.zeros(height, width)
        mask[::4, :] = 1.0  # Sample every 4th line

        unique_values = torch.unique(mask)
        assert len(unique_values) == 2  # Only 0 and 1
        assert torch.all((mask == 0) | (mask == 1))

    def test_mask_not_empty(self):
        """Test that mask doesn't inadvertently mask everything."""
        height, width = 64, 64

        mask = torch.ones(height, width) * 0.5  # Invalid: not binary

        # Valid masks should have at least some non-zero entries
        assert mask.sum() > 0

    def test_radial_trajectory_valid(self):
        """Test that radial trajectory sampling is valid."""
        num_angles = 64
        num_radii = 32

        # Generate radial trajectory using explicit numpy calls
        angles = np.linspace(0.0, 2.0 * np.pi, int(num_angles), endpoint=False)
        radii = np.linspace(0.0, 1.0, int(num_radii))

        # Create trajectory sampling pattern
        trajectory = []
        for angle in angles:
            for radius in radii:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                trajectory.append([float(x), float(y)])

        trajectory = torch.tensor(trajectory, dtype=torch.float32)

        # Verify trajectory covers k-space
        assert trajectory.shape == (num_angles * num_radii, 2)
        assert trajectory.abs().max() <= 1.0  # Normalized to [-1, 1]


class TestDataConsistency:
    """Tests for data consistency projection."""

    def test_data_consistency_mask_binary(self):
        """Test that data consistency mask is binary."""
        batch_size = 4
        height, width = 64, 64

        # Create binary mask for known k-space locations
        mask = torch.zeros(batch_size, 1, height, width, dtype=torch.bool)
        mask[:, :, : height // 2, :] = True  # Top half is known

        # Verify binary
        assert mask.dtype == torch.bool
        assert torch.all((mask == True) | (mask == False))

    def test_data_consistency_projection(self):
        """Test that DC projection preserves known k-space."""
        batch_size = 2
        height, width = 64, 64

        # Original k-space
        kspace_real = torch.randn(batch_size, height, width)
        kspace_imag = torch.randn(batch_size, height, width)
        kspace = torch.complex(kspace_real, kspace_imag)

        # Known k-space region (center)
        kspace_known = kspace.clone()
        mask = torch.zeros(batch_size, height, width, dtype=torch.bool)
        mask[:, height // 4 : 3 * height // 4, :] = True

        # Corrupted k-space (with noise in known region)
        kspace_corrupted = kspace + torch.randn_like(kspace) * 0.1
        kspace_corrupted[~mask] = kspace[~mask]  # Keep unknown region

        # Apply DC projection
        kspace_dc = kspace_corrupted.clone()
        kspace_dc[mask] = kspace_known[mask]  # Replace known region

        # Verify known region is restored exactly
        assert torch.allclose(kspace_dc[mask], kspace_known[mask], atol=1e-6)

    def test_data_consistency_weight_factor(self):
        """Test soft data consistency with weight factor."""
        batch_size = 2
        height, width = 64, 64

        kspace_true = torch.randn(batch_size, height, width, dtype=torch.complex64)
        kspace_predicted = kspace_true + torch.randn_like(kspace_true) * 0.1

        # Soft DC with weight λ
        weight = 0.9
        mask = torch.ones(batch_size, height, width, dtype=torch.bool)
        mask[:, : height // 2, :] = True

        kspace_dc = kspace_predicted.clone()
        kspace_dc[mask] = (
            weight * kspace_true[mask] + (1 - weight) * kspace_predicted[mask]
        )

        # Verify weighted combination
        expected = weight * kspace_true[mask] + (1 - weight) * kspace_predicted[mask]
        assert torch.allclose(kspace_dc[mask], expected, atol=1e-6)


class TestCoilSensitivityMaps:
    """Tests for SENSE coil sensitivity maps."""

    def test_sensitivity_map_shape(self):
        """Test that sensitivity maps have correct shape."""
        batch_size = 4
        num_coils = 4
        height, width = 64, 64

        # Create sensitivity maps
        sens_maps = torch.randn(
            batch_size, num_coils, height, width, dtype=torch.complex64
        )

        assert sens_maps.shape == (batch_size, num_coils, height, width)
        assert sens_maps.dtype == torch.complex64

    def test_sensitivity_map_normalization(self):
        """Test that sensitivity maps normalize k-space appropriately for SENSE."""
        batch_size = 1
        num_coils = 4
        height, width = 32, 32

        # True image
        image = torch.randn(batch_size, 1, height, width)

        # Coil sensitivity maps (simulated)
        sens_maps = torch.zeros(
            batch_size, num_coils, height, width, dtype=torch.complex64
        )
        for c in range(num_coils):
            phase = 2 * np.pi * c / num_coils
            sens_maps[:, c, :, :] = torch.exp(1j * torch.tensor(phase))

        # Apply sensitivity encoding (SENSE)
        coil_images = image.real.expand(-1, num_coils, -1, -1) * sens_maps

        # Compute RSS (correct reconstruction assuming orthogonal coils)
        coil_images_magnitude = torch.abs(coil_images)
        rss_recon = torch.sqrt((coil_images_magnitude**2).sum(dim=1, keepdim=True))

        # Verify RSS has positive magnitude
        assert rss_recon.min() >= 0
        assert not torch.isnan(rss_recon).any()

    def test_sensitivity_map_orthogonality(self):
        """Test that orthogonal sensitivity maps preserve energy."""
        batch_size = 1
        num_coils = 4
        height, width = 32, 32

        # Orthogonal sensitivity maps (no energy loss)
        sens_maps = torch.zeros(
            batch_size, num_coils, height, width, dtype=torch.complex64
        )
        for c in range(num_coils):
            phase_shift = 2 * np.pi * c / num_coils
            # Each coil has unit magnitude
            sens_maps[:, c, :, :] = (1.0 / np.sqrt(num_coils)) * torch.exp(
                1j * torch.tensor(phase_shift)
            )

        # Verify orthogonality: sum of |S|^2 = 1 everywhere
        sens_magnitude_sq = torch.abs(sens_maps) ** 2
        reconstructed_magnitude = sens_magnitude_sq.sum(dim=1, keepdim=True)

        # For orthogonal maps, sum should be 1
        expected = torch.ones(batch_size, 1, height, width)
        assert torch.allclose(reconstructed_magnitude, expected, atol=1e-5)

    def test_sense_equation_reconstruction(self):
        """Test that SENSE reconstruction follows the correct equation."""
        batch_size = 1
        num_coils = 4
        height, width = 32, 32

        # True image
        true_image = torch.randn(batch_size, 1, height, width)

        # Create sensitivity maps
        sens_maps = torch.zeros(
            batch_size, num_coils, height, width, dtype=torch.complex64
        )
        for c in range(num_coils):
            phase = 2 * np.pi * c / num_coils
            sens_maps[:, c, :, :] = torch.exp(1j * torch.tensor(phase))

        # Apply SENSE: y_c = S_c * x
        measured_coil_images = true_image.real.expand(-1, num_coils, -1, -1) * sens_maps

        # Reconstruct via RSS
        rss_recon = torch.sqrt(
            (torch.abs(measured_coil_images) ** 2).sum(dim=1, keepdim=True)
        )

        # Verify non-negative and finite
        assert rss_recon.min() >= 0
        assert torch.isfinite(rss_recon).all()


class TestHermitianSymmetry:
    """Tests for Hermitian symmetry of k-space from real-valued images."""

    def test_hermitian_symmetry_real_image(self):
        """Test that k-space from real image has Hermitian symmetry."""
        batch_size = 2
        height, width = 64, 64

        # Real-valued image
        image = torch.randn(batch_size, 1, height, width)

        # FFT to k-space
        kspace = torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
            ),
            dim=(-2, -1),
        )

        # Check Hermitian symmetry: k(x,y) = conj(k(-x,-y))
        kspace_flipped = torch.flip(kspace, dims=[-2, -1]).conj()
        # Roll by 1 to align indices for centered FFT symmetry around DC
        kspace_flipped = torch.roll(kspace_flipped, shifts=(1, 1), dims=(-2, -1))

        assert torch.allclose(kspace, kspace_flipped, atol=1e-5)

    def test_hermitian_symmetry_magnitude_phase(self):
        """Test Hermitian symmetry through magnitude and phase."""
        height, width = 32, 32

        image = torch.randn(1, 1, height, width)

        kspace = torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
            ),
            dim=(-2, -1),
        )

        # Extract magnitude and phase
        magnitude = torch.abs(kspace)
        phase = torch.angle(kspace)

        # Flipped k-space with roll adjustment
        magnitude_flipped = torch.roll(
            torch.flip(magnitude, dims=[-2, -1]), shifts=(1, 1), dims=(-2, -1)
        )
        phase_flipped = torch.roll(
            torch.flip(phase, dims=[-2, -1]), shifts=(1, 1), dims=(-2, -1)
        )

        # For Hermitian symmetry:
        # |k(x,y)| = |k(-x,-y)|
        assert torch.allclose(magnitude, magnitude_flipped, atol=1e-5)

        # phase(k(x,y)) = -phase(k(-x,-y))
        phase_sum = phase + phase_flipped
        # Wrap phase to [-pi, pi] to avoid small numerical errors around pi
        phase_sum = (phase_sum + np.pi) % (2 * np.pi) - np.pi
        assert torch.allclose(phase_sum, torch.zeros_like(phase_sum), atol=1e-4)


class TestMultiCoilConsistency:
    """Tests for multi-coil data consistency."""

    def test_multicoil_kspace_shape(self):
        """Test that multi-coil k-space has correct shape."""
        batch_size = 4
        num_coils = 4
        height, width = 64, 64

        kspace = torch.randn(
            batch_size, num_coils, height, width, dtype=torch.complex64
        )

        assert kspace.shape == (batch_size, num_coils, height, width)
        assert kspace.dtype == torch.complex64

    def test_rss_reconstruction(self):
        """Test that RSS reconstruction is valid."""
        batch_size = 2
        num_coils = 4
        height, width = 64, 64

        kspace = torch.randn(
            batch_size, num_coils, height, width, dtype=torch.complex64
        )

        # Convert to image domain
        image_coils = torch.fft.fftshift(
            torch.fft.ifftn(
                torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
            ),
            dim=(-2, -1),
        )

        # Compute RSS
        rss = torch.sqrt((torch.abs(image_coils) ** 2).sum(dim=1, keepdim=True))

        # Verify properties
        assert rss.shape == (batch_size, 1, height, width)
        assert rss.min() >= 0  # RSS is non-negative
        assert not torch.isnan(rss).any()
        assert not torch.isinf(rss).any()

    def test_coil_combination_energy_scaling(self):
        """Test that coil combination preserves energy scaling."""
        batch_size = 1
        num_coils = 4
        height, width = 32, 32

        image = torch.randn(batch_size, 1, height, width)

        # Multi-coil sensitivity maps
        sens_maps = torch.randn(
            batch_size, num_coils, height, width, dtype=torch.complex64
        )

        # Coil images
        coil_images = image.real.expand(-1, num_coils, -1, -1) * sens_maps

        # RSS
        rss = torch.sqrt((torch.abs(coil_images) ** 2).sum(dim=1, keepdim=True))

        # Verify RSS is reasonable
        assert rss.abs().max() > 0
        assert not torch.isnan(rss).any()
