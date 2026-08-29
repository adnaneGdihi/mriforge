"""Unit tests for all VF Method modules (M2–M10).

Tests cover: Phase Steganography (M7), DC Navigator (M3),
Hyperelastic Jacobian Loss (M9), Null-Space Erasure (M2),
BioHarmonicTransition (M4), and Koopman Operator (M8).
"""

import math

import pytest
import torch

# ──────────────────────────────────────────────────────────────────────
# M7: Phase Steganography
# ──────────────────────────────────────────────────────────────────────


class TestPhaseSteganography:
    """Tests for PhaseSteganographicMarker."""

    @pytest.fixture
    def marker(self):
        from mriforge.infrastructure.physics.phase_steganography import (
            PhaseSteganographicMarker,
        )

        return PhaseSteganographicMarker(
            im_size=(64, 64), spatial_freq_x=4.0, spatial_freq_y=4.0, amplitude=0.5
        )

    def test_inject_complex(self, marker):
        """Phase injection preserves magnitude."""
        img = torch.complex(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64))
        marked = marker.inject(img)
        assert marked.shape == img.shape
        assert marked.is_complex()
        # Magnitude must be identical
        torch.testing.assert_close(marked.abs(), img.abs(), atol=1e-5, rtol=1e-5)

    def test_inject_stacked(self, marker):
        """Phase injection works with 2-channel stacked tensors."""
        img = torch.randn(2, 2, 64, 64)  # 2ch stacked
        marked = marker.inject(img)
        assert marked.shape == img.shape
        assert not marked.is_complex()

    def test_erase_is_magnitude(self, marker):
        """Erasure via magnitude perfectly removes the marker."""
        img = torch.complex(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64))
        marked = marker.inject(img)
        erased = marker.erase(marked)
        original_mag = img.abs()
        torch.testing.assert_close(erased, original_mag, atol=1e-5, rtol=1e-5)

    def test_extract_strain_shape(self, marker):
        """Strain extraction returns correct shape."""
        img = torch.complex(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64))
        marked = marker.inject(img)
        strain = marker.extract_strain(marked)
        assert strain.shape == (2, 2, 64, 64)

    def test_forward_is_inject(self, marker):
        """Forward pass is an alias for inject."""
        img = torch.complex(torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64))
        torch.testing.assert_close(marker(img), marker.inject(img))


# ──────────────────────────────────────────────────────────────────────
# M3: DC Navigator
# ──────────────────────────────────────────────────────────────────────


class TestDCNavigator:
    """Tests for DCNavigatorExtractor."""

    @pytest.fixture
    def nav(self):
        from mriforge.infrastructure.physics.dc_navigator import DCNavigatorExtractor

        return DCNavigatorExtractor(
            sample_rate=100.0, resp_band=(0.1, 0.5), card_band=(0.8, 2.0)
        )

    def test_extract_dc_shape(self, nav):
        """DC extraction returns [T] vector."""
        kspace = torch.randn(10, 2, 1, 32, 32) + 0j  # [T, B, C, H, W]
        dc = nav.extract_dc(kspace)
        assert dc.shape == (10,)

    def test_extract_dc_single_frame(self, nav):
        """Single frame DC extraction works."""
        kspace = torch.randn(2, 1, 32, 32) + 0j
        dc = nav.extract_dc(kspace)
        assert dc.shape == (1,)

    def test_decompose_shapes(self, nav):
        """Decomposition returns respiratory + cardiac + residual."""
        dc = torch.randn(100) + 0j
        components = nav.decompose(dc)
        assert "respiratory" in components
        assert "cardiac" in components
        assert "residual" in components
        for v in components.values():
            assert v.shape == (100,)

    def test_forward_pipeline(self, nav):
        """Full forward pipeline returns all keys."""
        kspace = torch.randn(50, 1, 1, 16, 16) + 0j
        result = nav(kspace)
        assert "dc" in result
        assert "respiratory" in result
        assert "cardiac" in result

    def test_energy_conservation(self, nav):
        """resp + card + residual ≈ original dc signal."""
        dc = torch.randn(200) + 0j
        comps = nav.decompose(dc)
        reconstructed = comps["respiratory"] + comps["cardiac"] + comps["residual"]
        torch.testing.assert_close(reconstructed, dc, atol=1e-5, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# M9: Hyperelastic Jacobian Loss
# ──────────────────────────────────────────────────────────────────────


class TestHyperelasticJacobianLoss:
    """Tests for HyperelasticJacobianLoss."""

    @pytest.fixture
    def loss_fn(self):
        from mriforge.models.losses.hyperelastic_jacobian_loss import (
            HyperelasticJacobianLoss,
        )

        return HyperelasticJacobianLoss(fold_penalty_weight=10.0)

    def test_identity_field_zero_loss(self, loss_fn):
        """Identity displacement field (zeros) → det(J)=1 → loss≈0."""
        field = torch.zeros(2, 2, 32, 32)  # zero displacement
        loss = loss_fn(field)
        assert loss.item() < 1e-10

    def test_nonzero_displacement_positive_loss(self, loss_fn):
        """Non-trivial displacement → positive loss."""
        field = torch.randn(2, 2, 32, 32) * 0.1
        loss = loss_fn(field)
        assert loss.item() > 0

    def test_3d_field(self, loss_fn):
        """3-D deformation field is supported."""
        field = torch.zeros(1, 3, 8, 8, 8)
        loss = loss_fn(field)
        assert loss.item() < 1e-10

    def test_folding_penalty(self):
        """Negative det(J) triggers extra folding penalty."""
        from mriforge.models.losses.hyperelastic_jacobian_loss import (
            HyperelasticJacobianLoss,
        )

        loss_with = HyperelasticJacobianLoss(fold_penalty_weight=100.0)
        loss_without = HyperelasticJacobianLoss(fold_penalty_weight=0.0)

        # Create a field with spatially varying displacement that causes
        # strong compression (∂ux/∂x << -1 → det(J) < 0)
        field = torch.zeros(1, 2, 16, 16)
        # Exponential ramp: u_x = -exp(x) → ∂ux/∂x grows strongly negative
        x = torch.linspace(0, 3.0, 16)
        field[0, 0, :, :] = -torch.exp(x).unsqueeze(0).expand(16, -1)

        l_with = loss_with(field)
        l_without = loss_without(field)
        assert l_with.item() > l_without.item()

    def test_registered_in_loss_registry(self):
        """Loss is accessible via create_loss."""
        from mriforge.models.losses.registry import create_loss

        loss = create_loss("hyperelastic_jacobian")
        assert loss is not None


# ──────────────────────────────────────────────────────────────────────
# M2: Null-Space Erasure
# ──────────────────────────────────────────────────────────────────────


class TestNullSpaceErasure:
    """Tests for NullSpaceMarkerEraser."""

    def test_orthogonal_projection_removes_marker(self):
        """Projection onto null-space zeroes marker energy."""
        from mriforge.infrastructure.physics.null_space_erasure import NullSpaceMarkerEraser

        # Simple marker: single Gaussian peak
        marker = torch.zeros(1, 1, 16, 16)
        marker[0, 0, 8, 8] = 1.0

        eraser = NullSpaceMarkerEraser(marker_basis=marker, epsilon=1e-6)

        # Corrupted = anatomy + marker
        anatomy = torch.randn(2, 1, 16, 16)
        corrupted = anatomy + marker

        cleaned = eraser(corrupted)

        # Marker inner product should be near zero after erasure
        marker_flat = marker.reshape(1, -1)
        cleaned_flat = cleaned.reshape(2, -1)
        inner = (cleaned_flat * marker_flat).sum(dim=-1)
        assert torch.all(inner.abs() < 1e-3)

    def test_preserves_orthogonal_signal(self):
        """Signal orthogonal to marker is preserved exactly."""
        from mriforge.infrastructure.physics.null_space_erasure import NullSpaceMarkerEraser

        # Marker in one pixel
        marker = torch.zeros(1, 1, 4, 4)
        marker[0, 0, 0, 0] = 1.0

        eraser = NullSpaceMarkerEraser(marker_basis=marker)

        # Signal in a different pixel (orthogonal)
        signal = torch.zeros(1, 1, 4, 4)
        signal[0, 0, 3, 3] = 5.0

        cleaned = eraser(signal)
        torch.testing.assert_close(
            cleaned[0, 0, 3, 3], torch.tensor(5.0), atol=1e-4, rtol=1e-4
        )


# ──────────────────────────────────────────────────────────────────────
# M4: BioHarmonicTransition
# ──────────────────────────────────────────────────────────────────────


class TestBioHarmonicTransition:
    """Tests for BioHarmonicTransition."""

    @pytest.fixture
    def transition(self):
        from mriforge.models.mamba.bloch_mamba import BioHarmonicTransition

        return BioHarmonicTransition(d_model=16, f_resp=0.3, f_card=1.2)

    def test_output_shape(self, transition):
        """Output shape matches input."""
        h = torch.randn(4, 16)
        h_new = transition(h)
        assert h_new.shape == h.shape

    def test_eigenvalues_at_correct_frequencies(self, transition):
        """Eigenvalue imaginary parts match respiratory/cardiac frequencies."""
        w_resp = 2 * math.pi * 0.3
        w_card = 2 * math.pi * 1.2

        imag = transition.A_imag.detach()
        # First 4 eigenvalues should be ±w_resp, ±w_card
        assert abs(imag[0].item() - w_resp) < 1e-4
        assert abs(imag[1].item() + w_resp) < 1e-4
        assert abs(imag[2].item() - w_card) < 1e-4
        assert abs(imag[3].item() + w_card) < 1e-4

    def test_decay_causes_amplitude_reduction(self, transition):
        """Negative decay reduces state amplitude over time."""
        h = torch.ones(1, 16)
        for _ in range(100):
            h = transition(h)
        # After 100 steps, amplitude should be significantly reduced
        assert h.abs().max().item() < 1.0

    def test_batched_input(self, transition):
        """Works with batched 3-D input."""
        h = torch.randn(4, 10, 16)  # [B, L, D]
        h_new = transition(h)
        assert h_new.shape == h.shape


# ──────────────────────────────────────────────────────────────────────
# M8: Koopman Operator
# ──────────────────────────────────────────────────────────────────────


class TestKoopmanOperator:
    """Tests for KoopmanMotionFilter."""

    @pytest.fixture
    def koopman(self):
        from mriforge.models.temporal.koopman_operator import KoopmanMotionFilter

        return KoopmanMotionFilter(
            input_dim=2, latent_dim=16, hidden_dim=32, num_layers=2
        )

    def test_encode_decode_shape(self, koopman):
        """Encoder/decoder produce correct shapes."""
        x = torch.randn(2, 50, 2)
        z = koopman.encode(x)
        assert z.shape == (2, 50, 16)
        x_hat = koopman.decode(z)
        assert x_hat.shape == (2, 50, 2)

    def test_propagate_shape(self, koopman):
        """Propagation produces correct multi-step shape."""
        z = torch.randn(2, 16)
        z_future = koopman.propagate(z, steps=5)
        assert z_future.shape == (2, 5, 16)

    def test_filter_signal_keys(self, koopman):
        """filter_signal returns all expected keys."""
        signal = torch.randn(2, 50, 2)
        result = koopman.filter_signal(signal)
        assert set(result.keys()) == {
            "filtered",
            "aperiodic",
            "reconstruction",
            "prediction",
        }

    def test_koopman_loss_keys(self, koopman):
        """koopman_loss returns all expected loss keys."""
        signal = torch.randn(2, 50, 2)
        losses = koopman.koopman_loss(signal)
        assert set(losses.keys()) == {
            "reconstruction",
            "prediction",
            "linearity",
            "total",
        }
        assert losses["total"].item() > 0

    def test_forward_2d_input(self, koopman):
        """Forward handles 2-D input [T, D] by adding batch dim."""
        signal = torch.randn(50, 2)
        result = koopman(signal)
        assert result["filtered"].shape[0] == 1


# ──────────────────────────────────────────────────────────────────────
# M6: ContinuousBlochODE
# ──────────────────────────────────────────────────────────────────────


class TestContinuousBlochODE:
    """Tests for ContinuousBlochODE."""

    @pytest.fixture
    def ode(self):
        from mriforge.infrastructure.physics.bloch_simulation import ContinuousBlochODE

        return ContinuousBlochODE(dt=5.0, slice_thickness=5.0)

    def test_output_shape(self, ode):
        """Output signal series has correct shape."""
        tissue = torch.rand(2, 3, 16, 16) * 1000 + 1  # PD, T1, T2
        signals = ode(tissue, n_steps=10, tr=50.0)
        assert signals.shape == (10, 2, 1, 16, 16)

    def test_unread_te_argument_removed(self, ode):
        """The unread ``te`` knob was removed (pitfall #15); passing it raises.

        The output is the full per-step trajectory, never a TE-sampled echo, so
        ``te`` had no effect and was an advertised-but-unread knob.
        """
        import pytest

        tissue = torch.rand(1, 3, 8, 8) * 1000 + 1
        with pytest.raises(TypeError):
            ode(tissue, n_steps=5, tr=50.0, te=20.0)

    def test_signal_decays(self, ode):
        """Mxy decays over time after excitation."""
        tissue = torch.tensor([[[[1.0]]]] * 3).permute(1, 0, 2, 3)
        tissue[:, 1:2] = 1000.0  # T1=1000ms
        tissue[:, 2:3] = 50.0  # T2=50ms
        signals = ode(tissue, n_steps=20, tr=10.0)
        # After a few steps post-excitation, signal should decay
        assert signals[-1].abs().max() < signals[2].abs().max() + 0.1

    def test_through_plane_reset(self, ode):
        """Through-plane motion resets spin state to equilibrium."""
        tissue = torch.ones(1, 3, 4, 4) * 500.0
        tissue[:, 0:1] = 1.0  # PD = 1
        # Deformation that moves everything out of slice
        deformation = torch.ones(5, 1, 1, 4, 4) * 100.0  # way beyond threshold
        signals = ode(tissue, n_steps=5, tr=50.0, deformation_z_sequence=deformation)
        assert signals.shape == (5, 1, 1, 4, 4)

    def test_equilibrium_start(self, ode):
        """Initial state starts at equilibrium (Mz=M0, Mxy=0)."""
        tissue = torch.ones(1, 3, 4, 4)
        tissue[:, 0:1] = 1.0
        tissue[:, 1:2] = 1000.0
        tissue[:, 2:3] = 100.0
        # First step before RF: Mxy should be near 0
        signals = ode(tissue, n_steps=1, tr=1000.0)  # TR very long
        # Signal (Mxy) should be very small at first step (no RF yet since TR=1000)
        # Actually TR=1000 > dt=5, so no RF fires yet → Mxy stays 0
        assert signals[0].abs().max() < 0.01


# ──────────────────────────────────────────────────────────────────────
# M10: DynamicMRNeRF
# ──────────────────────────────────────────────────────────────────────


class TestDynamicMRNeRF:
    """Tests for DynamicMRNeRF."""

    @pytest.fixture
    def nerf(self):
        from mriforge.models.generators.dynamic_mr_nerf import DynamicMRNeRF

        return DynamicMRNeRF(
            in_channels=2,
            out_channels=2,
            num_frequencies=4,
            hidden_dim=32,
            num_layers=3,
            im_size=(16, 16),
        )

    def test_render_image_shape(self, nerf):
        """Rendered image has correct shape."""
        marker = torch.zeros(2)
        img = nerf.render_image(marker)
        assert img.shape == (2, 2, 16, 16)

    def test_forward_shape(self, nerf):
        """Forward pass returns correct image shape."""
        x = torch.randn(2, 2, 16, 16)
        y = nerf(x)
        assert y.shape == (2, 2, 16, 16)

    def test_query_arbitrary_coords(self, nerf):
        """Query works with arbitrary coordinate sets."""
        coords = torch.rand(2, 100, 2) * 2 - 1  # random coords in [-1, 1]
        marker = torch.zeros(2)
        intensity = nerf.query(coords, marker)
        assert intensity.shape == (2, 100, 2)

    def test_marker_conditioning(self, nerf):
        """Different marker signals produce different outputs."""
        x = torch.randn(1, 2, 16, 16)
        y1 = nerf(x, marker_signal=torch.tensor([0.0]))
        y2 = nerf(x, marker_signal=torch.tensor([1.0]))
        # With different conditioning, outputs should differ
        assert not torch.allclose(y1, y2, atol=1e-3)
