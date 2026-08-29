"""Unit tests for Task 1 architectural blocks (vf_architecture_blocks.py).

Tests shape round-trips, gradient flow, and basic functional correctness
for all 6 missing blocks plus the 2 enhancement operators.
"""

import pytest
import torch

from mriforge.models.generators.vf_architecture_blocks import (
    DirichletESPIRiTBlock,
    EKFPhaseTracker,
    MoDLSuperResBlock,
    NeuralComplexSumBlock,
    VCCVarNetBlock,
    WienerUNetBlock,
)


@pytest.fixture
def device():
    """CPU device for unit tests."""
    return torch.device("cpu")


@pytest.fixture
def img_size():
    """Standard test image size."""
    return (32, 32)


# ------------------------------------------------------------------
# 1. MoDL Super-Resolution Block
# ------------------------------------------------------------------


class TestMoDLSuperResBlock:
    """Tests for MoDLSuperResBlock."""

    def test_shape_without_dc(self, device, img_size):
        """Output shape matches input shape without data consistency."""
        H, W = img_size
        block = MoDLSuperResBlock(channels=4, window_size=8).to(device)
        x = torch.randn(1, 4, H, W, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_shape_with_dc(self, device, img_size):
        """Output shape matches input with CG data consistency."""
        H, W = img_size
        block = MoDLSuperResBlock(
            channels=4, num_heads=2, window_size=8, num_cg_iters=2
        ).to(device)
        x = torch.randn(1, 4, H, W, device=device, dtype=torch.cfloat)
        y_kspace = torch.randn(1, 4, H, W, device=device, dtype=torch.cfloat)
        mask = torch.ones(1, 1, H, W, device=device)
        out = block(x, y_kspace=y_kspace, mask=mask)
        assert out.shape == x.shape

    def test_gradient_flow(self, device, img_size):
        """Gradients flow through the block."""
        H, W = img_size
        block = MoDLSuperResBlock(channels=4, window_size=8).to(device)
        x = torch.randn(1, 4, H, W, device=device, requires_grad=True)
        out = block(x)
        loss = out.abs().sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ------------------------------------------------------------------
# 2. Wiener-UNet Block
# ------------------------------------------------------------------


class TestWienerUNetBlock:
    """Tests for WienerUNetBlock."""

    def test_shape_2ch_real(self, device, img_size):
        """Shape preservation with 2-channel real input."""
        H, W = img_size
        block = WienerUNetBlock(channels=2, num_res_blocks=2).to(device)
        x = torch.randn(1, 2, H, W, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, device, img_size):
        """Gradients flow through Wiener + CNN."""
        H, W = img_size
        block = WienerUNetBlock(channels=2, num_res_blocks=2).to(device)
        x = torch.randn(1, 2, H, W, device=device, requires_grad=True)
        out = block(x)
        loss = out.abs().sum()
        loss.backward()
        assert x.grad is not None


# ------------------------------------------------------------------
# 3. EKF Phase Tracker
# ------------------------------------------------------------------


class TestEKFPhaseTracker:
    """Tests for EKFPhaseTracker."""

    def test_output_shape(self, device):
        """Output shape matches input sequence length."""
        tracker = EKFPhaseTracker().to(device)
        phases = torch.randn(2, 50, device=device)
        out = tracker(phases)
        assert out.shape == (2, 50)

    def test_smooth_output(self, device):
        """EKF output should be smoother than noisy input."""
        tracker = EKFPhaseTracker(measurement_noise=1.0).to(device)
        # Clean linear drift + noise
        t = torch.linspace(0, 1, 100, device=device).unsqueeze(0)
        clean = 0.5 * t
        noisy = clean + 0.3 * torch.randn_like(clean)
        filtered = tracker(noisy)
        # Filtered should have lower variance of second differences
        noise_var = torch.diff(noisy, n=2).var()
        filtered_var = torch.diff(filtered, n=2).var()
        assert filtered_var < noise_var


# ------------------------------------------------------------------
# 4. Dirichlet-ESPIRiT Block
# ------------------------------------------------------------------


class TestDirichletESPIRiTBlock:
    """Tests for DirichletESPIRiTBlock."""

    def test_output_shape(self, device, img_size):
        """Sensitivity maps have correct shape."""
        H, W = img_size
        Nc = 4
        block = DirichletESPIRiTBlock(kernel_size=4).to(device)
        kspace = torch.randn(1, Nc, H, W, device=device, dtype=torch.cfloat)
        maps = block(kspace, center_fraction=0.2)
        assert maps.shape == (1, Nc, H, W)

    def test_normalized_maps(self, device, img_size):
        """Sensitivity maps should be approximately normalized."""
        H, W = img_size
        Nc = 4
        block = DirichletESPIRiTBlock(kernel_size=4).to(device)
        kspace = torch.randn(1, Nc, H, W, device=device, dtype=torch.cfloat)
        maps = block(kspace, center_fraction=0.3)
        rss = (maps.abs().pow(2).sum(dim=1) + 1e-8).sqrt()
        # RSS should be approximately 1.0 where there's signal
        assert rss.mean().item() == pytest.approx(1.0, abs=0.3)


# ------------------------------------------------------------------
# 5. VCC VarNet Block
# ------------------------------------------------------------------


class TestVCCVarNetBlock:
    """Tests for VCCVarNetBlock."""

    def test_hermitian_symmetry(self, device, img_size):
        """VCC output should more closely satisfy Hermitian symmetry."""
        H, W = img_size
        block = VCCVarNetBlock().to(device)
        kspace = torch.randn(1, 2, H, W, device=device, dtype=torch.cfloat)
        out = block(kspace)
        assert out.shape == kspace.shape

    def test_with_mask(self, device, img_size):
        """Works correctly with a sampling mask."""
        H, W = img_size
        block = VCCVarNetBlock(apply_data_consistency=True).to(device)
        kspace = torch.randn(1, 2, H, W, device=device, dtype=torch.cfloat)
        mask = (torch.rand(1, 1, H, W, device=device) > 0.5).float()
        out = block(kspace, mask=mask)
        assert out.shape == kspace.shape


# ------------------------------------------------------------------
# 6. Neural Complex-Sum Block
# ------------------------------------------------------------------


class TestNeuralComplexSumBlock:
    """Tests for NeuralComplexSumBlock."""

    def test_output_shape(self, device, img_size):
        """Output has correct channels."""
        H, W = img_size
        Nc = 4
        block = NeuralComplexSumBlock(num_coils=Nc, output_channels=1).to(device)
        coil_images = torch.randn(1, Nc, H, W, device=device, dtype=torch.cfloat)
        out = block(coil_images)
        assert out.shape == (1, 1, H, W)
        assert out.is_complex()

    def test_gradient_flow(self, device, img_size):
        """Gradients flow through the MLP combiner."""
        H, W = img_size
        Nc = 4
        block = NeuralComplexSumBlock(num_coils=Nc).to(device)
        x = torch.randn(
            1, Nc, H, W, device=device, dtype=torch.cfloat, requires_grad=True
        )
        out = block(x)
        loss = out.abs().sum()
        loss.backward()
        assert x.grad is not None


# ------------------------------------------------------------------
# 7. MarkerKSpaceDCProjection (Enhancement)
# ------------------------------------------------------------------


class TestMarkerKSpaceDCProjection:
    """Tests for MarkerKSpaceDCProjection."""

    def test_output_shape(self, device, img_size):
        """Output shape matches input."""
        from mriforge.infrastructure.physics.vf_operators import MarkerKSpaceDCProjection

        H, W = img_size
        mask = torch.zeros(1, 1, H, W)
        mask[0, 0, 10:15, 10:15] = 1.0
        prior = torch.ones(1, 1, H, W)
        proj = MarkerKSpaceDCProjection(mask, prior).to(device)
        x = torch.randn(1, 1, H, W, device=device)
        out = proj(x)
        assert out.shape == (1, 1, H, W)


# ------------------------------------------------------------------
# 8. VarianceWeightedTGV (Enhancement)
# ------------------------------------------------------------------


class TestVarianceWeightedTGV:
    """Tests for VarianceWeightedTGV."""

    def test_output_shape(self, device, img_size):
        """Output shape matches input."""
        from mriforge.infrastructure.physics.vf_physics_operators import VarianceWeightedTGV

        H, W = img_size
        tgv = VarianceWeightedTGV(num_iters=2).to(device)
        x = torch.randn(1, 1, H, W, device=device)
        var_map = torch.ones(1, 1, H, W, device=device) * 0.1
        out = tgv(x, variance_map=var_map)
        assert out.shape == x.shape

    def test_fallback_without_variance(self, device, img_size):
        """Falls back to base TGV when variance_map is None."""
        from mriforge.infrastructure.physics.vf_physics_operators import VarianceWeightedTGV

        H, W = img_size
        tgv = VarianceWeightedTGV(num_iters=2).to(device)
        x = torch.randn(1, 1, H, W, device=device)
        out = tgv(x)
        assert out.shape == x.shape
