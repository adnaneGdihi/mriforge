import torch

from mriforge.infrastructure.physics.coordinate_utils import CoordinateSampler
from mriforge.models.generators.hybrid_pinn import HybridQMapGenerator


class TestHybridArchitecture:
    """
    Verification tests for the Hybrid Grid-INR architecture.
    Checks gradient flow, coordinate sampling consistency, and super-resolution properties.
    """

    def test_gradient_flow_to_input(self):
        """
        Verify that gradients flow back through the decoder MLP and grid_sample
        to the CNN encoder and input images.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridQMapGenerator(input_dim=1, feature_dim=32).to(device)

        img = torch.randn(1, 1, 32, 32, device=device, requires_grad=True)

        # Forward pass
        t1, t2, pd, b0, b1 = model(img)

        # Loss on predicted map
        loss = t1.mean() + t2.mean() + pd.mean() + b0.mean() + b1.mean()
        loss.backward()

        # Check gradients
        assert img.grad is not None
        assert img.grad.abs().sum() > 0

        # Check encoder parameters
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_coordinate_sampling_grid_match(self):
        """
        Verify that sampling on a regular grid matches the full-grid inference output.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridQMapGenerator(input_dim=1).to(device)
        img = torch.randn(1, 1, 16, 16, device=device)

        # 1. Standard grid inference
        t1_grid, _, _, _, _ = model(img)  # (1, 1, 16, 16)

        # 2. Manual coordinate inference
        coords = CoordinateSampler.get_full_grid(1, (16, 16), device)
        t1_sampled, _, _, _, _ = model(img, query_coords=coords)  # (1, 1, 256)

        # Reshape sampled to match grid
        t1_sampled_reshaped = t1_sampled.view(1, 1, 16, 16)

        assert torch.allclose(t1_grid, t1_sampled_reshaped, atol=1e-5)

    def test_super_resolution_query(self):
        """
        Ensure the model can be queried at higher resolution than the input grid.
        Input: 16x16, Query: 32x32.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridQMapGenerator(input_dim=1).to(device)
        img = torch.randn(1, 1, 16, 16, device=device)

        # High-res coordinates
        hr_coords = CoordinateSampler.get_full_grid(1, (32, 32), device)

        # Query
        t1, t2, pd, b0, b1 = model(img, query_coords=hr_coords)

        assert t1.shape == (1, 1, 1024)  # 32*32
        assert torch.isfinite(t1).all()

    def test_roi_sampling(self):
        """
        Verify localized ROI sampling returns expected shapes and finite values.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridQMapGenerator(input_dim=1).to(device)
        img = torch.randn(1, 1, 32, 32, device=device)

        # Sample center ROI
        roi_coords = CoordinateSampler.sample_roi(
            1, 100, center=(0, 0), scale=0.1, device=device
        )

        t1, t2, pd, b0, b1 = model(img, query_coords=roi_coords)

        assert t1.shape == (1, 1, 100)
        assert torch.isfinite(t1).all()
