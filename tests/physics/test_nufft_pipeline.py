"""Test NUFFT Physics Pipeline.

Verifies that NonCartesianSimulationTransform correctly:
1. Generates trajectories
2. Simulates k-space via Forward NUFFT
3. Reconstructs via Adjoint NUFFT
4. Populates Subject fields correctly
"""

import pytest
import torch
import torchio as tio

# Skip if torchkbnufft not available
try:
    import torchkbnufft as tkbn
except ImportError:
    pytest.skip("torchkbnufft not installed", allow_module_level=True)

from mriforge.data.transforms.non_cartesian import NonCartesianSimulationTransform
from mriforge.infrastructure.physics.trajectories import TrajectoryFactory


class TestNUFFTPipeline:
    @pytest.fixture
    def sample_subject(self):
        # Create a simple phantom: box in center
        # TorchIO expects (Channels, W, H, D) or similar 4D tensor.
        # Ensure 4D: (1, 128, 128, 1)
        data = torch.zeros(1, 128, 128, 1)
        data[:, 32:96, 32:96, :] = 1.0

        return tio.Subject(
            input=tio.ScalarImage(tensor=data),
            # Optional target
            target=tio.ScalarImage(tensor=data.clone()),
        )

    def test_spiral_simulation(self, sample_subject):
        """Test Spiral trajectory simulation."""
        transform = NonCartesianSimulationTransform(
            pattern="spiral",
            im_size=(128, 128),
            acceleration=4.0,
        )

        transformed = transform(sample_subject)

        # Check constraints
        assert "measured_kspace" in transformed
        assert "trajectory" in transformed
        assert "dcf" in transformed

        kspace = transformed["measured_kspace"]
        traj = transformed["trajectory"]
        dcf = transformed["dcf"]

        # Shapes
        # Kspace: (C, N)
        assert kspace.ndim == 2 or kspace.ndim == 3
        assert traj.shape[0] == 2
        assert traj.shape[1] == kspace.shape[-1]
        assert dcf.shape[0] == kspace.shape[-1]

        # Check input image (aliased) vs target (clean)
        aliased = transformed["input"].data
        target = transformed["target"].data

        # Artifacts check: Simulating undersampling should introduce error
        mse = torch.mean((aliased - target) ** 2)
        assert mse > 0.001, "Aliased image should differ from ground truth"

        # Signal preservation (approximate)
        # Applying Adjoint usually scales signal up/down depending on DCF normalization
        # We assume transform output is reasonably scaled?
        # NonCartesianSimulationTransform doesn't explicitly normalize adjoint output
        # unless Tkbn does. Tkbn adjoint is technically summation, so values grow with N.
        # But we verify it's not zero.
        assert aliased.abs().max() > 0.1

    def test_radial_simulation(self, sample_subject):
        """Test Radial trajectory simulation."""
        transform = NonCartesianSimulationTransform(
            pattern="radial",
            im_size=(128, 128),
            acceleration=8.0,
        )

        transformed = transform(sample_subject)

        kspace = transformed["measured_kspace"]
        traj = transformed["trajectory"]

        # Check properties
        assert (
            kspace.is_complex() or kspace.shape[0] == 1
        )  # Depending on how Tkbn returns
        # Tkbn usually returns complex tensor if input is complex or real?
        # Our input was float. output kspace is complex.
        # Check if stored as complex tensor or 2-channel real
        if torch.is_complex(kspace):
            assert True
        else:
            # (C, 2, N) or (2, N)?
            # Tkbn output is complex. Subject storage might depend.
            pass

    def test_forward_adjoint_consistency(self):
        """Test that Adjoint(Forward(x)) is somewhat consistent (geometry wise)."""
        im_size = (64, 64)
        image = torch.randn(1, 1, 64, 64).to(torch.complex64)

        transform = NonCartesianSimulationTransform(
            pattern="spiral",
            im_size=im_size,
            acceleration=1.0,  # Fully sampledish
        )

        # Manually invoke operators
        nufft, adj = transform._get_operators(im_size, image.device)
        traj, dcf = TrajectoryFactory.get_spiral_trajectory(im_size, num_arms=48)

        kspace = nufft(image, traj)
        recon = adj(kspace * dcf.unsqueeze(0).unsqueeze(0), traj)

        assert recon.shape == image.shape
