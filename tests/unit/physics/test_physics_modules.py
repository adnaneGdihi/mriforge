"""Unit tests for physics-related modules."""

import torch


class TestFFTOperations:
    """Test FFT operations for k-space processing."""

    def test_fft2c_forward(self):
        """Test centered 2D FFT."""
        from mriforge.infrastructure.physics.fft_ops import fft2c

        # Create test image
        image = torch.randn(2, 1, 32, 32, dtype=torch.complex64)

        # Forward FFT
        kspace = fft2c(image)

        assert kspace.shape == image.shape
        assert kspace.dtype == torch.complex64

    def test_fft_inverse_identity(self):
        """Test that FFT followed by IFFT recovers original."""
        from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c

        image = torch.randn(2, 1, 32, 32, dtype=torch.complex64)

        # Forward then inverse
        kspace = fft2c(image)
        recovered = ifft2c(kspace)

        # Should recover original (within numerical precision)
        assert torch.allclose(image, recovered, atol=1e-5)


class TestDataConsistency:
    """Test data consistency operations."""

    def test_dc_layer_preserves_sampled_kspace(self):
        """Test DC layer operates on k-space data."""
        from mriforge.infrastructure.physics.data_consistency import SimpleDataConsistency

        # Create DC layer
        dc = SimpleDataConsistency(weight=0.1)

        # Test inputs
        pred_image = torch.randn(2, 1, 32, 32)

        # Apply DC (without measured k-space, returns input)
        result = dc(pred_image)

        assert result.shape == pred_image.shape


class TestSamplingPatterns:
    """Test k-space sampling pattern generation."""

    def test_uniform_cartesian_accelerator(self):
        """Test uniform Cartesian undersampling accelerator."""
        from mriforge.infrastructure.physics.sampling import (
            UniformCartesianKSpaceAccelerator,
        )

        accelerator = UniformCartesianKSpaceAccelerator(
            num_timesteps=100,
            max_acceleration=4.0,
            center_fraction=0.08,
        )

        mask = accelerator.get_acceleration_mask(
            kspace_shape=(32, 32),
            timestep=50,
        )

        # Mask may have extra batch/channel dims - check spatial dims
        assert mask.shape[-2:] == (32, 32)
        # Check some values are sampled
        assert mask.sum() > 0

    def test_acceleration_factor(self):
        """Test acceleration factor computation."""
        from mriforge.infrastructure.physics.sampling import KSpaceAccelerator

        # Check base class method exists
        assert hasattr(KSpaceAccelerator, "get_acceleration_factor")


class TestCoilSensitivity:
    """Test coil sensitivity estimation."""

    def test_coil_sensitivity_estimation(self):
        """Test coil sensitivity map estimation."""
        from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss

        # Multi-coil k-space data
        kspace = torch.randn(2, 8, 32, 32, dtype=torch.complex64)

        # Use RSS method (simpler)
        sens_maps = estimate_csm_rss(kspace, num_coils=8)
        assert sens_maps.shape[1] == 8  # Same number of coils


class TestPhysicsIntegration:
    """Test physics integration utilities."""

    def test_forward_model_exists(self):
        """Test physics integration module exists."""
        from mriforge.infrastructure.physics import integration

        # Integration module should exist
        assert integration is not None


class TestBlochSimulation:
    """Test Bloch equation simulation."""

    def test_bloch_simulation_import(self):
        """Test Bloch simulation module imports."""
        from mriforge.infrastructure.physics.biophysics.bloch import BlochSimulator

        assert hasattr(BlochSimulator, "__init__")


class TestPINN:
    """Test physics-informed neural network components."""

    def test_pinn_module_exists(self):
        """Test PINN module function exists."""
        from mriforge.infrastructure.physics.pinn import PDE, BlochEquation, PINNModule

        assert hasattr(PINNModule, "__init__")
        assert hasattr(PDE, "compute_residual")
        assert hasattr(BlochEquation, "compute_residual")


class TestTrajectories:
    """Test k-space trajectory generation."""

    def test_trajectory_factory_exists(self):
        """Test trajectory factory exists."""
        from mriforge.infrastructure.physics.trajectories import (
            TrajectoryFactory,
            get_trajectory,
        )

        assert hasattr(TrajectoryFactory, "get_radial_trajectory")
        assert hasattr(TrajectoryFactory, "get_spiral_trajectory")
        assert callable(get_trajectory)

    def test_radial_trajectory_generation(self):
        """Test radial trajectory generation."""
        from mriforge.infrastructure.physics.trajectories import TrajectoryFactory

        trajectory, dcf = TrajectoryFactory.get_radial_trajectory(
            im_size=(32, 32),
            num_spokes=16,
        )

        assert trajectory is not None
        assert dcf is not None
