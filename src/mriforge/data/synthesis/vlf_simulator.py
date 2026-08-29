"""Synthetic Very Low Field (VLF) MRI Data Generator.

Simulates 64mT (ULF) data from 3T (HF) clean data using paired dataset parameters.
Based on van den Broek et al. paired 64mT-3T brain MRI dataset (2025).

Physics-based degradation pipeline:
1. Contrast Remapping (Physics-informed via Bloch equations).
2. B0 Inhomogeneity (Geometric warping from low field homogeneity).
3. B1 Bias Field (Low-frequency intensity modulation).
4. Resolution Loss (K-space truncation -> Gibbs ringing).
5. Rician Noise (Low SNR characteristic of 64mT).

Rician noise model:
    M = sqrt((I + n_r)^2 + (Q + n_i)^2)
    P(M) = (M/sigma^2) * exp(-(M^2 + A^2)/(2*sigma^2)) * I_0(M*A/sigma^2)
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from mriforge.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator
from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.infrastructure.physics.tissue_properties import TissuePropertyRegistry


class VLFSimulator:
    """Simulates Very Low Field acquisition artifacts."""

    def __init__(
        self,
        target_snr: float = 10.0,  # 64mT typical SNR
        resolution_scale: float = 0.6,  # 64mT: ~1.6mm vs 3T: ~1.0mm in-plane
        b0_warp_strength: float = 0.08,  # Stronger at low field
        bias_field_strength: float = 0.4,  # More B1 inhomogeneity at 64mT
        sequence_type: str = "SE",  # Spin Echo for T2w
        use_paired_dataset_params: bool = True,  # Use actual 64mT/3T params
    ):
        """__init__.

        Args:
            target_snr (float): Description.
            resolution_scale (float): Description.
            b0_warp_strength (float): Description.
            bias_field_strength (float): Description.
            sequence_type (str): Description.
            use_paired_dataset_params (bool): Description.
        """
        self.target_snr = target_snr
        self.resolution_scale = resolution_scale
        self.b0_warp_strength = b0_warp_strength
        self.bias_field_strength = bias_field_strength
        self.use_paired_dataset_params = use_paired_dataset_params

        # Physics components
        self.bloch_sim = DifferentiableBlochSimulator(sequence_type=sequence_type)

        # Paired dataset sequence parameters (from Tables 1 & 2)
        # 64mT (ULF) parameters
        self.ulf_params = {
            "T1w": {"TR": 880.0, "TE": 5.03, "TI": 340.0, "flip": 90.0},
            "T2w": {"TR": 2000.0, "TE": 194.8, "flip": 90.0},
            "FLAIR": {"TR": 3500.0, "TE": 175.1, "TI": 1290.0, "flip": 90.0},
            "DWI": {
                "TR": 1000.0,
                "TE": 76.22,
                "TI": 100.0,
                "b_value": 900.0,
                "flip": 90.0,
            },
        }

        # 3T (HF) low-res parameters (resolution-matched)
        self.hf_params = {
            "T1w": {"TR": 10.0, "TE": 4.58, "flip": 12.0},  # 3D SPGR
            "T2w": {"TR": 4645.0, "TE": 80.0, "flip": 90.0},
            "FLAIR": {"TR": 11000.0, "TE": 125.0, "TI": 2800.0, "flip": 90.0},
            "DWI": {"TR": 2827.0, "TE": 73.0, "b_value": 1000.0, "flip": 90.0},
        }

    def generate_from_segmentation(
        self,
        segmentation: Tensor,
        contrast: str = "T1w",
        TR: float | None = None,
        TE: float | None = None,
    ) -> Tensor:
        """Generate a synthetic VLF image from a tissue segmentation map.

        Args:
            segmentation: [B, 1, H, W] integer tensor (0=BG, 1=CSF, 2=GM, 3=WM)
            contrast: Sequence type ('T1w', 'T2w', 'FLAIR', 'DWI')
            TR: Repetition Time (ms), uses dataset default if None
            TE: Echo Time (ms), uses dataset default if None

        Returns:
            vlf_image: [B, 1, H, W] Simulated VLF image
        """
        # Use paired dataset parameters if not specified
        if self.use_paired_dataset_params and contrast in self.ulf_params:
            if TR is None:
                TR = self.ulf_params[contrast]["TR"]
            if TE is None:
                TE = self.ulf_params[contrast]["TE"]
        else:
            # Fallback defaults
            TR = TR if TR is not None else 500.0
            TE = TE if TE is not None else 20.0
        # 1. Map segmentation -> (PD, T1, T2) at 64mT
        # Returns [B, 3, H, W]
        tissue_maps = TissuePropertyRegistry.get_map_tensor(segmentation, field_strength="64mT")

        # 2. Run Bloch Simulation -> Clean "Physics" Image
        # Bloch returns signal intensity
        clean_image = self.bloch_sim(tissue_maps, TR=TR, TE=TE)

        # Bloch sim might result in values > 1.0 depending on PD scaling,
        # but typically PD is 0-1 so signal is 0-1.

        # 3. Apply VLF Degradation Pipeline
        # We reuse the __call__ logic but on this synthetic image
        return self.__call__(clean_image)

    def apply_rician_noise(self, magnitude: Tensor) -> Tensor:
        """Add Rician noise.

        Rician noise arises from Gaussian noise in real/imag channels of k-space.
        M = sqrt( (I + n_r)^2 + (Q + n_i)^2 )
        """
        # Estimate signal level
        # If signal is 0, we assume some base noise level or fix sigma
        signal_power = torch.mean(magnitude**2)
        if signal_power == 0:
            # Fallback: assume latent noise floor?
            # For test case with 0 signal, we want pure noise.
            # If target_snr is relative to signal, and signal is 0, calc is weird.
            # Let's assume a small epsilon signal for SNR calc OR
            signal_power = torch.tensor(1.0, device=magnitude.device)

        noise_power = signal_power / (10 ** (self.target_snr / 10.0))
        sigma = torch.sqrt(noise_power / 2)

        noise_r = torch.randn_like(magnitude) * sigma
        noise_i = torch.randn_like(magnitude) * sigma

        # Assuming input is magnitude, we treat it as Real component + 0 Imag?
        # Or split it.
        # Simple approximation:
        noisy_magnitude = torch.sqrt((magnitude + noise_r) ** 2 + noise_i**2)
        return noisy_magnitude

    def simulate_b0_warping(self, image: Tensor) -> Tensor:
        """Simulate geometric distortion due to low field homogeneity."""
        B, C, H, W = image.shape
        device = image.device

        # Create coordinate grid
        # [-1, 1]
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Create a smooth field map (e.g. low freq cosine)
        # Delta = strength * cos(pi * x) * cos(pi * y)
        field_map_y = torch.cos(math.pi * grid_y) * torch.cos(math.pi * grid_x * 0.5)
        field_map_x = torch.sin(math.pi * grid_y * 0.5) * torch.cos(math.pi * grid_x)

        # Distort coordinates
        # VLF typically has warping along readout (say x)
        warp_y = grid_y + self.b0_warp_strength * field_map_y
        warp_x = grid_x + self.b0_warp_strength * field_map_x

        # Stack for grid_sample: (B, H, W, 2) -> (x, y) order!
        grid = torch.stack([warp_x, warp_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Values outside [-1, 1] might be handled by padding (zeros)
        distorted = F.grid_sample(image, grid, align_corners=True, padding_mode="zeros")
        return distorted

    def simulate_bias_field(self, image: Tensor) -> Tensor:
        """Simulate B1 transmit inhomogeneity (Bias Field).

        Creates a smooth, low-frequency intensity modulation using
        polynomial basis functions.
        """
        B, C, H, W = image.shape
        device = image.device

        # Create coordinate grid [-1, 1]
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Basis functions for smooth variation
        # We use a mix of 2nd order polynomials to create a 'dome' or 'saddle'
        # B1 maps are typically multiplicative

        # 1. Center brightening/darkening (Quadratic)
        # r^2 = x^2 + y^2
        r2 = grid_x**2 + grid_y**2
        basis_1 = 1.0 - 0.5 * r2  # Center is 1, edges are 0.5

        # 2. Linear gradients (tilting)
        basis_2 = 0.5 * grid_x
        basis_3 = 0.5 * grid_y

        # Randomize coefficients per batch element
        # coeffs: [B, 3] -> bias map: c1*b1 + c2*b2 + c3*b3
        # We want the field to mean ~1.0 but vary spatially

        bias_field = torch.zeros(B, 1, H, W, device=device)

        for i in range(B):
            # Random mix
            c1 = 1.0 + torch.randn(1, device=device).item() * 0.2  # Main dome
            c2 = torch.randn(1, device=device).item() * 0.2  # Tilt X
            c3 = torch.randn(1, device=device).item() * 0.2  # Tilt Y

            field = c1 * basis_1 + c2 * basis_2 + c3 * basis_3

            # Normalize field to be centered around 1.0 + strength
            # Actually we want multiplicative factor: I' = I * Field
            # Field should range approx [1-strength, 1+strength]

            # Apply strength scaling
            field = (field - 1.0) * self.bias_field_strength + 1.0
            bias_field[i, 0] = field

        # Apply bias field
        corrupted = image * bias_field
        return corrupted

    def simulate_gibbs_ringing(self, image: Tensor) -> Tensor:
        """Simulate resolution loss via K-Space Truncation (Gibbs Ringing).

        1. FFT -> K-space
        2. Crop center (low frequencies)
        3. Zero-pad back to full size
        4. IFFT -> Image with ringing
        """
        # image is magnitude (real). We treat it as real part of complex signal.
        kspace = fft2c(image.to(torch.complex64))  # (B, C, H, W) complex

        B, C, H, W = kspace.shape

        # Calculate crop size
        h_crop = int(H * self.resolution_scale)
        w_crop = int(W * self.resolution_scale)

        # Ensure crop is even to verify centering easily, though python slicing handles it
        if h_crop % 2 != 0:
            h_crop -= 1
        if w_crop % 2 != 0:
            w_crop -= 1

        # Calculate start/end indices for center crop
        h_start = (H - h_crop) // 2
        w_start = (W - w_crop) // 2
        h_end = h_start + h_crop
        w_end = w_start + w_crop

        # Create mask (or just slice and pad)
        # We want to ZERO OUT high frequencies
        mask = torch.zeros_like(kspace, dtype=torch.bool)  # False everywhere
        mask[:, :, h_start:h_end, w_start:w_end] = True  # True in center

        # Apply truncation
        kspace_truncated = kspace * mask

        # In theory, if we want to simulate *downsampling*, we would crop.
        # But here we simulate *upsampled low-res* (blurred on HR grid),
        # so zero-padding (which we just did by masking) effectively interpolates
        # the low-freq info onto the high-res grid with Sinc interpolation kernel -> Ringing!

        # IFFT
        image_ringing = ifft2c(kspace_truncated)

        return torch.abs(image_ringing)  # Magnitude

    def estimate_segmentation_heuristic(self, image: Tensor) -> Tensor:
        """Estimate tissue segmentation from T1-weighted image using intensity."""
        # Simple k-means like heuristic or fixed thresholds
        # Assumption: image is T1w normalized [0,1]
        # CSF (dark) < GM (gray) < WM (bright)

        # Thresholds (approximate for normalized T1)
        # 0.0 - 0.15: Background
        # 0.15 - 0.4: CSF
        # 0.4 - 0.75: GM
        # 0.75 - 1.0: WM

        seg = torch.zeros_like(image, dtype=torch.long)

        # BG is already 0

        # CSF
        mask_csf = (image >= 0.1) & (image < 0.3)
        seg[mask_csf] = 1

        # GM
        mask_gm = (image >= 0.3) & (image < 0.65)
        seg[mask_gm] = 2

        # WM
        mask_wm = image >= 0.65
        seg[mask_wm] = 3

        return seg

    def simulate_contrast_remapping(
        self,
        image: Tensor,
        segmentation: Tensor | None = None,
        contrast: str = "T1w",
        tr_hf: float | None = None,
        te_hf: float | None = None,
        tr_vlf: float | None = None,
        te_vlf: float | None = None,
    ) -> Tensor:
        """Remap contrast from High Field (3T) to Low Field (64mT).

        Uses "Physics-Informed Style Transfer" with paired dataset parameters:
        1. Estimate tissue maps (PD, T1, T2) at 3T and 64mT.
        2. Simulate expected signal S_3T and S_64mT using Bloch equations.
        3. Compute contrast ratio map R = S_64mT / S_3T.
        4. Apply R to real 3T image: I_64mT = I_3T * R.

        This preserves anatomical texture while remapping contrast to ULF.

        Args:
            image: [B, C, H, W] High field image
            segmentation: [B, 1, H, W] Tissue segmentation (optional)
            contrast: Sequence type for parameter lookup
            tr_hf/te_hf: 3T sequence parameters (uses dataset defaults if None)
            tr_vlf/te_vlf: 64mT sequence parameters (uses dataset defaults if None)
        """
        # Use paired dataset parameters if enabled
        if self.use_paired_dataset_params and contrast in self.hf_params:
            if tr_hf is None:
                tr_hf = self.hf_params[contrast]["TR"]
            if te_hf is None:
                te_hf = self.hf_params[contrast]["TE"]
            if tr_vlf is None:
                tr_vlf = self.ulf_params[contrast]["TR"]
            if te_vlf is None:
                te_vlf = self.ulf_params[contrast]["TE"]
        else:
            # Fallback defaults
            tr_hf = tr_hf if tr_hf is not None else 1900.0
            te_hf = te_hf if te_hf is not None else 2.5
            tr_vlf = tr_vlf if tr_vlf is not None else 500.0
            te_vlf = te_vlf if te_vlf is not None else 20.0
        if segmentation is None:
            segmentation = self.estimate_segmentation_heuristic(image)

        # 1. Get Tissue Properties for HF and LF
        maps_hf = TissuePropertyRegistry.get_map_tensor(segmentation, field_strength="3.0T")
        maps_lf = TissuePropertyRegistry.get_map_tensor(segmentation, field_strength="64mT")

        # 2. Simulate expected signals
        # Scale PD? Assuming maps_hf/lf relative PD is same, but T1/T2 differ.
        with torch.no_grad():
            s_hf = self.bloch_sim(maps_hf, TR=tr_hf, TE=te_hf)
            s_lf = self.bloch_sim(maps_lf, TR=tr_vlf, TE=te_vlf)

        # 3. Compute Ratio (avoid div by zero)
        # Soft division
        epsilon = 1e-6
        contrast_ratio = (s_lf + epsilon) / (s_hf + epsilon)

        # 4. Apply to image
        # Clamp ratio to avoid exploding gradients or artifacts?
        # contrast_ratio = torch.clamp(contrast_ratio, 0.1, 10.0)

        remapped = image * contrast_ratio

        return remapped

    def __call__(
        self,
        hf_image: Float[Tensor, "B C H W"],
        segmentation: Float[Tensor, "B 1 H W"] = None,
        contrast: str = "T1w",
        enable_contrast_remapping: bool = True,
    ) -> Float[Tensor, "B C H W"]:
        """Simulate 64mT acquisition from 3T image.

        Args:
            hf_image: 3T clean image (Magnitude).
            segmentation: Optional segmentation map. If None and remapping enabled, uses heuristic.
            contrast: Sequence type for parameter selection ('T1w', 'T2w', 'FLAIR', 'DWI')
            enable_contrast_remapping: Whether to apply physics-based contrast translation.

        Returns:
            vlf_image: Simulated 64mT image with realistic degradations.
        """
        output = hf_image

        # 0. Contrast Remapping (Physics-informed 3T->64mT)
        if enable_contrast_remapping:
            output = self.simulate_contrast_remapping(
                output, segmentation=segmentation, contrast=contrast
            )

        # 1. Geometric Warping (B0 Inhomogeneity)
        warped = self.simulate_b0_warping(output)

        # 2. Bias Field (B1 Inhomogeneity)
        biased = self.simulate_bias_field(warped)

        # 3. K-Space Truncation (Gibbs Ringing) - REPLACES Bilinear Blur
        # This is the physics-correct way to lose resolution
        low_res = self.simulate_gibbs_ringing(biased)

        # 4. Rician Noise (applied on magnitude)
        noisy = self.apply_rician_noise(low_res)

        return noisy
