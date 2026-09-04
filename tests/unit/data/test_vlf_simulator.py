import torch
import torch.nn.functional as F

from spectramr.data.synthesis.vlf_simulator import VLFSimulator


class TestVLFSimulator:
    def test_initialization(self):
        """Test default and custom initialization."""
        sim = VLFSimulator()
        assert sim.target_snr == 10.0
        assert sim.resolution_scale == 0.6
        assert sim.b0_warp_strength == 0.08
        assert sim.bias_field_strength == 0.4

        sim_custom = VLFSimulator(target_snr=20.0, bias_field_strength=0.5)
        assert sim_custom.target_snr == 20.0
        assert sim_custom.bias_field_strength == 0.5

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        sim = VLFSimulator()
        B, C, H, W = 2, 1, 128, 128
        image = torch.ones(B, C, H, W)
        output = sim(image)
        assert output.shape == (B, C, H, W)

    def test_bias_field_effect(self):
        """Test that bias field changes intensity statistics."""
        sim = VLFSimulator(
            b0_warp_strength=0.0,
            resolution_scale=1.0,
            target_snr=100.0,
            bias_field_strength=0.5,
        )
        B, C, H, W = 4, 1, 64, 64
        image = torch.ones(B, C, H, W)

        # Manually call bias field simulation
        biased = sim.simulate_bias_field(image)

        # Check that it's not uniform anymore
        assert biased.std() > 0.01, "Bias field should introduce variance"

        # Check range (approximate, since it's random)
        # Intensity should vary around 1.0
        assert biased.mean() > 0.5 and biased.mean() < 1.5

    def test_warping_effect(self):
        """Test that geometric warping is applied."""
        sim = VLFSimulator(b0_warp_strength=0.2, bias_field_strength=0.0)
        # Create a grid image
        image = torch.zeros(1, 1, 64, 64)
        image[:, :, ::8, :] = 1.0
        image[:, :, :, ::8] = 1.0

        warped = sim.simulate_b0_warping(image)

        # Simple check: pixel match should be low
        # (Though with interpolation it might be tricky, but exact match implies NO warp)
        assert not torch.allclose(image, warped, atol=1e-4)

    def test_gibbs_ringing_effect(self):
        """Test that k-space truncation produces Gibbs ringing (unlike bilinear)."""
        sim = VLFSimulator(
            b0_warp_strength=0.0,
            bias_field_strength=0.0,
            resolution_scale=0.25,
            target_snr=100.0,
        )

        # Create a sharp box (ideal for ringing)
        image = torch.zeros(1, 1, 128, 128)
        image[:, :, 32:96, 32:96] = 1.0  # White square in middle

        # 1. Apply Simulator (FFT Truncation)
        output_gibbs = sim.simulate_gibbs_ringing(image)

        # 2. Apply Bilinear Interpolation (Old way)
        output_bilinear = F.interpolate(
            F.interpolate(image, scale_factor=0.25, mode="bilinear"),
            size=image.shape[2:],
            mode="bilinear",
        )

        # Gibbs ringing should cause:
        # 1. Undershoot (negative values or dips below 0 near edges, though we take abs() so > 0 but distinct)
        # 2. Overshoot (values > 1.0 near edges)

        # Check max value (overshoot)
        # Bilinear is bounded by input [0, 1] (usually slightly less due to averaging)
        assert output_bilinear.max() <= 1.0 + 1e-5

        # Gibbs should overshoot > 1.0 near edges
        assert (
            output_gibbs.max() > 1.02
        ), f"Gibbs ringing should overshoot, got max {output_gibbs.max()}"

        # Check energy conservation (Parseval's theorem implies scale changes, but relative structure matches)
        # Just ensure it's not empty
        assert output_gibbs.mean() > 0.1

    def test_generate_from_segmentation(self):
        """Test generative mode from segmentation map."""
        sim = VLFSimulator()

        # Create dummy segmentation: (B, 1, H, W)
        # 0=BG, 1=CSF, 2=GM, 3=WM
        seg = torch.zeros(1, 1, 64, 64)
        seg[:, :, 10:20, 10:20] = 1  # CSF
        seg[:, :, 20:30, 20:30] = 2  # GM
        seg[:, :, 30:40, 30:40] = 3  # WM

        # TR=500, TE=15 (T1-weighted approximate)
        vlf_image = sim.generate_from_segmentation(seg, TR=500.0, TE=15.0)

        assert vlf_image.shape == seg.shape
        assert vlf_image.min() >= 0
        assert not torch.isnan(vlf_image).any()

        # Check that we have signal in tissue areas
        # Background should be noise (since Rician noise adds to 0 signal)
        # But tissues should be brighter

        # Extract signal in WM region
        wm_signal = vlf_image[:, :, 35, 35].mean()
        # Extract signal in BG region (far corner)
        bg_signal = vlf_image[:, :, 60, 60].mean()

        # WM should be significantly brighter than noise floor (roughly)
        # Note: Rician noise bias means BG is not 0, it's sigma * sqrt(pi/2)
        # But signal regions should be > BG
        assert wm_signal > bg_signal
