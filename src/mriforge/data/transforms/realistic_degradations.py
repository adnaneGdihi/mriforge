#!/usr/bin/env python3
"""
Realistic Degradations for Data Pipeline
========================================

Advanced degradation models for medical imaging data augmentation.
Includes Rician noise, motion blur, k-space undersampling, and
other realistic artifacts commonly found in MRI data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio

from mriforge.infrastructure.physics.fft_ops import (
    fft_volume_full3d_uncentered,
    ifft_volume_full3d_uncentered,
)
from mriforge.infrastructure.physics.field_simulation import B0MapSimulator


class B0GeometricDistortion(nn.Module):
    """
    Simulates geometric distortion caused by B0 field inhomogeneity.

    This is common in EPI sequences where off-resonance leads to pixel shifts
    along the phase-encoding direction.

    Formula: delta_y = gamma * Delta_B0(x,y,z) * T_readout
    """

    def __init__(self, strength: float = 1.0, readout_dim: int = 2):
        """__init__.

        Args:
            strength (float): Description.
            readout_dim (int): Description.
        """
        super().__init__()
        self.strength = strength
        self.readout_dim = readout_dim  # 2 for height (y), 3 for width (x) usually [B, C, D, H, W]

    def forward(self, x: torch.Tensor, b0_map: torch.Tensor) -> torch.Tensor:
        """
        Apply distortion.

        Args:
            x: [B, C, D, H, W]
            b0_map: [B, 1, H, W] or [B, 1, D, H, W] (Hz)

        forward method for B0GeometricDistortion.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b0_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.strength == 0:
            return x

        B, C, D, H, W = x.shape
        device = x.device

        # Ensure b0_map matches 3D shape if provided as 2D
        if b0_map.dim() == 4:  # [B, 1, H, W]
            b0_map = b0_map.unsqueeze(2).expand(-1, -1, D, -1, -1)

        # Create identity grid
        vectors = [torch.linspace(-1, 1, steps=s, device=device) for s in (D, H, W)]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij"), dim=-1)  # [D, H, W, 3]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, D, H, W, 3]

        # Calculate displacement
        # Normalize strength relative to FOV (simple approx: strength is pixel shift)
        # Grid range is [-1, 1], so 2 units = full FOV.
        # Shift in normalized units = (pixels / size) * 2

        size_dim = [D, H, W][self.readout_dim - 2]  # readout_dim 2->D? No.
        # input is [B, C, D, H, W] -> dims 2, 3, 4
        # grid is [D, H, W, 3] -> 0, 1, 2 (corresponding to Z, Y, X)

        # Map readout_dim (2,3,4) to grid index (0,1,2)
        grid_dim = self.readout_dim - 2

        # B0 map is in Hz. We treat 'strength' as scaling factor to pixels.
        # displacement (pixels) = b0 * strength
        # displacement (grid) = (b0 * strength / size) * 2

        disp_pixels = b0_map * self.strength
        disp_grid = (disp_pixels / size_dim) * 2.0

        # Add to grid
        # grid[..., grid_dim] += disp_grid.squeeze(1)
        # Note: grid is [..., 3], disp_grid is [B, 1, D, H, W]

        # We need to broadcast carefully.
        # grid: [B, D, H, W, 3]
        # disp: [B, D, H, W] (after squeeze)

        disp_grid = (
            disp_grid.squeeze(1).permute(0, 2, 3, 4, 1)
            if disp_grid.shape[1] > 1
            else disp_grid.squeeze(1)
        )
        # Actually disp_grid is [B, 1, D, H, W] -> squeeze -> [B, D, H, W]

        # Create full flow field [B, D, H, W, 3] initialized to 0
        flow = torch.zeros_like(grid)
        flow[..., grid_dim] = disp_grid

        # Apply
        new_grid = torch.clamp(grid + flow, -1, 1)

        return F.grid_sample(
            x, new_grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )


class RandomB0Distortion(tio.Transform):
    """
    TorchIO wrapper for B0 Geometric Distortion.
    """

    def __init__(self, max_displacement: float = 2.0, p: float = 0.5, field_strength: float = 3.0):
        """__init__.

        Args:
            max_displacement (float): Description.
            p (float): Description.
            field_strength (float): Description.
        """
        super().__init__(p=p)
        self.max_displacement = max_displacement
        self.simulator = B0MapSimulator(field_strength=field_strength)
        self.distortion = B0GeometricDistortion(
            strength=1.0
        )  # Strength scaling handled dynamically

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Generate random B0 map
        # Assume first image shape
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        shape = subject.shape[1:]  # [H, W, D] usually for TorchIO?
        # TorchIO stores as [C, W, H, D] or [C, H, W, D]?
        # TIO is (C, W, H, D) usually but let's check input tensor

        # We will work on tensor data directly from 'input'
        for image_name in subject.get_images_names():
            image = subject[image_name]
            data = image.data  # [C, W, H, D]

            # Map generation expects [H, W] (2D) or we need 3D?
            # B0MapSimulator is 2D. We can extrude or generate 2D.
            # Let's generate 2D map matching H, W (dims 2, 1 in TIO W, H, D)

            spatial_shape = data.shape[1:]  # W, H, D
            h, w = spatial_shape[1], spatial_shape[0]

            # Generate 2D map
            device = data.device
            b0_map = self.simulator.generate_smooth_b0((h, w), device=device)  # [1, 1, H, W]

            # Random strength for this sample
            strength = (torch.rand(1, device=device).item() * 2 - 1) * self.max_displacement

            # Apply distortion
            # We need to bridge [C, W, H, D] to [B, C, D, H, W] expected by B0GeometricDistortion
            # TIO: W, H, D -> D, H, W for module

            # Permute [C, W, H, D] -> [1, C, D, H, W]
            tensor_in = data.permute(3, 2, 0, 1).unsqueeze(0)  # [D, H, W, C] -> wait
            # data: [C, W, H, D]
            # target: [1, C, D, H, W]
            tensor_in = data.permute(0, 3, 2, 1).unsqueeze(0)  # [1, C, D, H, W]

            # Adjust map to [1, 1, D, H, W] (extrude D)
            # b0_map is [1, 1, H, W].

            # Multiply map by strength before passing (per-sample dynamic strength).
            b0_weighted = b0_map * strength

            distorted = self.distortion(tensor_in.contiguous(), b0_weighted)

            # Permute back: [1, C, D, H, W] -> [C, W, H, D]
            tensor_out = distorted.squeeze(0).permute(0, 3, 2, 1)

            image.set_data(tensor_out)

        return subject


class RicianNoise(nn.Module):
    """
    Rician noise model for MRI data.

    MRI magnitude images follow Rician distribution due to the
    presence of complex Gaussian noise in k-space.
    """

    def __init__(self, noise_level: float = 0.1):
        """__init__.

        Args:
            noise_level (float): Description.
        """
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Rician noise to input tensor.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W]

        Returns:
            Noisy tensor with same shape

        forward method for RicianNoise.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.noise_level == 0:
            return x

        # Generate complex Gaussian noise
        noise_real = torch.randn_like(x) * self.noise_level
        noise_imag = torch.randn_like(x) * self.noise_level

        # Compute magnitude with Rician noise
        noisy_magnitude = torch.sqrt((x + noise_real) ** 2 + noise_imag**2)

        return noisy_magnitude


class MotionBlur3D(nn.Module):
    """
    3D Motion blur for volumetric data.

    Simulates patient motion during acquisition of 3D volumes.
    """

    def __init__(
        self,
        blur_kernel_size: int = 5,
        motion_intensity: float = 0.5,
        motion_direction: tuple[int, int, int] | None = None,
    ):
        """__init__.

        Args:
            blur_kernel_size (int): Description.
            motion_intensity (float): Description.
            motion_direction (Optional[tuple[int, int, int]]): Description.
        """
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        self.motion_intensity = motion_intensity

        if motion_direction is None:
            # Random motion direction
            self.motion_direction = torch.randn(3, dtype=torch.float32)
        else:
            self.motion_direction = torch.tensor(motion_direction, dtype=torch.float32)

        self.motion_direction = F.normalize(self.motion_direction, dim=0)

    def _create_motion_kernel(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Create motion blur kernel."""
        D, H, W = shape[-3:]

        # Create coordinate grids
        d_coords = torch.linspace(-1, 1, D, device=self.motion_direction.device)
        h_coords = torch.linspace(-1, 1, H, device=self.motion_direction.device)
        w_coords = torch.linspace(-1, 1, W, device=self.motion_direction.device)

        D_grid, H_grid, W_grid = torch.meshgrid(d_coords, h_coords, w_coords, indexing="ij")

        # Compute motion blur weights
        motion_vector = torch.stack([D_grid, H_grid, W_grid], dim=-1)

        # Project onto motion direction
        projection = torch.sum(motion_vector * self.motion_direction.view(1, 1, 1, 3), dim=-1)

        # Create blur kernel
        blur_weights = torch.exp(-self.motion_intensity * projection**2)
        blur_weights = blur_weights / blur_weights.sum()

        return blur_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply motion blur to 3D volume.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Blurred tensor with same shape

        forward method for MotionBlur3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.motion_intensity == 0:
            return x

        B, C, D, H, W = x.shape
        k = self.blur_kernel_size

        # 1. Generate ONE kernel for the whole batch (much faster)
        # Randomize direction per batch for variety, or keep consistent?
        # User suggested: "simply use a consistent kernel across the batch (randomized per batch...)"
        # We re-randomize direction if needed, but here we just use the current state or re-init?
        # The original code had fixed direction in __init__ unless random.
        # But if we want random per batch, we should generate it here.
        # However, _create_motion_kernel uses self.motion_direction.
        # Let's assume we use the current self.motion_direction or re-randomize it?
        # The user's snippet just calls _create_motion_kernel.

        # Optimization: Generate kernel once per batch
        kernel = self._create_motion_kernel((k, k, k)).to(x.device)

        # 2. Reshape kernel for depth-wise convolution
        # We want to apply the same spatial kernel to every channel of every image independently
        # OR replicate it to match input channels.
        # Grouped conv with groups=B*C means we need B*C kernels.
        # We repeat the single kernel B*C times.
        kernel = kernel.view(1, 1, k, k, k).repeat(B * C, 1, 1, 1, 1)

        # 3. Apply batched convolution
        # padding='same' logic manually calculated
        pad = k // 2

        # Reshape input for grouped convolution: [1, B*C, D, H, W]
        x_reshaped = x.view(1, B * C, D, H, W)

        output = F.conv3d(
            x_reshaped,
            kernel,
            padding=pad,
            groups=B * C,
        )

        return output.view(B, C, D, H, W)


class KSpaceUndersampling(nn.Module):
    """
    K-space undersampling for MRI acceleration.

    Simulates parallel imaging and compressed sensing
    undersampling patterns.
    """

    def __init__(
        self,
        acceleration_factor: int = 2,
        pattern: str = "uniform",
        center_fraction: float = 0.1,
    ):
        """__init__.

        Args:
            acceleration_factor (int): Description.
            pattern (str): Description.
            center_fraction (float): Description.
        """
        super().__init__()
        self.acceleration_factor = acceleration_factor
        self.pattern = pattern
        self.center_fraction = center_fraction

    def _create_mask(self, shape: tuple[int, int, int]) -> torch.Tensor:
        """Create undersampling mask."""
        D, H, W = shape

        if self.pattern == "uniform":
            # Uniform random undersampling
            mask = torch.rand(D, H, W) < (1.0 / self.acceleration_factor)

        elif self.pattern == "equispaced":
            # Equispaced undersampling
            mask = torch.zeros(D, H, W, dtype=torch.bool)
            step = self.acceleration_factor
            mask[:, ::step, :] = True

        elif self.pattern == "gaussian":
            # Gaussian random undersampling
            coords = torch.meshgrid(
                torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
            )
            center_d, center_h, center_w = D // 2, H // 2, W // 2

            # Compute distances from center
            distances = torch.sqrt(
                (coords[0] - center_d) ** 2
                + (coords[1] - center_h) ** 2
                + (coords[2] - center_w) ** 2
            )

            # Gaussian probability
            prob = torch.exp(-(distances**2) / (2 * (min(D, H, W) / 4) ** 2))
            mask = torch.rand(D, H, W) < prob

        else:
            raise ValueError(f"Unknown undersampling pattern: {self.pattern}")

        # Always keep center region (low frequencies)
        center_size_d = max(1, int(D * self.center_fraction))
        center_size_h = max(1, int(H * self.center_fraction))
        center_size_w = max(1, int(W * self.center_fraction))

        center_start_d = (D - center_size_d) // 2
        center_start_h = (H - center_size_h) // 2
        center_start_w = (W - center_size_w) // 2

        mask[
            center_start_d : center_start_d + center_size_d,
            center_start_h : center_start_h + center_size_h,
            center_start_w : center_start_w + center_size_w,
        ] = True

        return mask.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply k-space undersampling.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Undersampled tensor with same shape

        forward method for KSpaceUndersampling.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.acceleration_factor == 1:
            return x

        B, C, D, H, W = x.shape

        # Generate masks
        if self.pattern == "equispaced":
            # Deterministic mask, same for all
            mask = self._create_mask((D, H, W)).to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        else:
            # Random mask, different for each sample/channel
            # We can generate one large random tensor and threshold it for uniform
            if self.pattern == "uniform":
                mask = torch.rand(B, C, D, H, W, device=x.device) < (1.0 / self.acceleration_factor)
                # Always keep center
                center_size_d = max(1, int(D * self.center_fraction))
                center_size_h = max(1, int(H * self.center_fraction))
                center_size_w = max(1, int(W * self.center_fraction))

                center_start_d = (D - center_size_d) // 2
                center_start_h = (H - center_size_h) // 2
                center_start_w = (W - center_size_w) // 2

                mask[
                    :,
                    :,
                    center_start_d : center_start_d + center_size_d,
                    center_start_h : center_start_h + center_size_h,
                    center_start_w : center_start_w + center_size_w,
                ] = True
                mask = mask.float()
            else:
                # Fallback to loop for complex mask generation (gaussian)
                masks = []
                for _ in range(B * C):
                    masks.append(self._create_mask((D, H, W)))
                mask = torch.stack(masks).view(B, C, D, H, W).to(x.device)

        # FFT to k-space (batched). fft_volume_full3d_uncentered returns an
        # *unshifted* spectrum (DC at the corner). The mask in
        # `_create_mask` places its "centre region" at the geometric
        # middle of the array, so we must fftshift the k-space before
        # masking and ifftshift after — otherwise the mask never
        # protects the DC bin and an all-ones input collapses to zero.
        k_space = fft_volume_full3d_uncentered(x)
        k_space = torch.fft.fftshift(k_space, dim=(-3, -2, -1))

        k_space_masked = k_space * mask

        k_space_masked = torch.fft.ifftshift(k_space_masked, dim=(-3, -2, -1))
        reconstructed = ifft_volume_full3d_uncentered(k_space_masked)

        return reconstructed


class ElasticDeformation3D(nn.Module):
    """Applies elastic deformation to a 3D tensor volume.

    This transform simulates physical deformations (like subject motion or
    tissue compression) by generating a smoothed random displacement field
    and resampling the input volume accordingly using `F.grid_sample`.

    Attributes:
        alpha (float): Scaling factor for the displacement field. Controls the
            magnitude of the deformation. Defaults to 1.0.
        sigma (float): Standard deviation for the Gaussian filter used to
            smooth the displacement field. Larger values create smoother
            deformations. Defaults to 10.0.
    """

    def __init__(self, alpha: float = 1.0, sigma: float = 10.0):
        """Initializes the ElasticDeformation3D transform.

        Args:
            alpha (float, optional): Scaling factor for displacement. Defaults to 1.0.
            sigma (float, optional): Standard deviation for smoothing. Defaults to 10.0.
        """
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated elastic deformation using flow fields.
        x: [B, C, D, H, W]

        forward method for ElasticDeformation3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.alpha == 0:
            return x

        B, C, D, H, W = x.shape
        device = x.device

        # 1. Create identity grid [-1, 1]
        vectors = [torch.linspace(-1, 1, steps=s, device=device) for s in (D, H, W)]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij"), dim=-1)  # [D, H, W, 3]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, D, H, W, 3]

        # 2. Generate Gaussian flow field on low-res then upsample (Simulates sigma)
        # Generating noise at full res and smoothing is slow.
        # Generating at low res and interpolating is equivalent to smoothing.
        scale_factor = 8  # Approximation of sigma smoothing
        small_shape = (
            max(1, D // scale_factor),
            max(1, H // scale_factor),
            max(1, W // scale_factor),
        )

        noise = torch.randn(B, 3, *small_shape, device=device) * self.alpha
        # Upsample noise to valid flow field
        flow = F.interpolate(noise, size=(D, H, W), mode="trilinear", align_corners=False)
        flow = flow.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        # 3. Apply deformation
        # grid_sample expects [B, D, H, W, 3] coordinates in range [-1, 1]
        deformed_grid = torch.clamp(grid + flow, -1, 1)

        # Reshape x for grid_sample (it processes all channels together)
        return F.grid_sample(
            x,
            deformed_grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )


class MultiContrastProcessor(nn.Module):
    """
    Multi-contrast MRI processing for T1, T2, FLAIR fusion.

    Handles different MRI contrast types and provides fusion methods
    for multi-modal image processing.
    """

    def __init__(self, contrasts: list[str] = None):
        """__init__.

        Args:
            contrasts (list[str]): Description.
        """
        if contrasts is None:
            contrasts = ["t1", "t2", "flair"]
        super().__init__()
        self.contrasts = contrasts
        self.n_contrasts = len(contrasts)

        # Fusion layers
        self.fusion_conv = nn.Conv3d(self.n_contrasts, self.n_contrasts, 3, padding=1)
        self.fusion_norm = nn.InstanceNorm3d(self.n_contrasts)
        self.fusion_act = nn.ReLU(inplace=True)

        # Attention mechanism for contrast weighting
        self.attention = nn.Sequential(
            nn.Conv3d(self.n_contrasts, self.n_contrasts, 1), nn.Sigmoid()
        )

    def forward(self, contrast_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process multi-contrast images.

        Args:
            contrast_dict: Dictionary with contrast types as keys
                          {'t1': tensor, 't2': tensor, 'flair': tensor}

        Returns:
            Fused multi-contrast representation [B, C, D, H, W]

        forward method for MultiContrastProcessor.

        Executes PyTorch tensor operations.

        Args:
            contrast_dict (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Stack contrasts along channel dimension
        contrast_list = []
        for contrast in self.contrasts:
            if contrast in contrast_dict:
                contrast_list.append(contrast_dict[contrast])
            else:
                # Pad with zeros if contrast is missing
                shape = list(contrast_dict[self.contrasts[0]].shape)
                shape[1] = 1  # Single channel
                device = self.fusion_conv.weight.device
                contrast_list.append(torch.zeros(shape, device=device))

        # Stack: [B, C, D, H, W] -> [B, n_contrasts, D, H, W]
        stacked = torch.cat(contrast_list, dim=1)

        # Apply fusion
        fused = self.fusion_conv(stacked)
        fused = self.fusion_norm(fused)
        fused = self.fusion_act(fused)

        # Apply attention weighting
        attention_weights = self.attention(stacked)
        fused = fused * attention_weights

        return fused


class RealisticDegradationPipeline(nn.Module):
    """
    Complete realistic degradation pipeline.

    Combines multiple degradation types for comprehensive
    data augmentation.
    """

    def __init__(
        self,
        rician_noise_level: float = 0.05,
        motion_blur_intensity: float = 0.3,
        undersampling_factor: int = 2,
        elastic_alpha: float = 0.5,
        elastic_sigma: float = 8.0,
    ):
        """__init__.

        Args:
            rician_noise_level (float): Description.
            motion_blur_intensity (float): Description.
            undersampling_factor (int): Description.
            elastic_alpha (float): Description.
            elastic_sigma (float): Description.
        """
        super().__init__()

        self.degradations = nn.ModuleList(
            [
                RicianNoise(noise_level=rician_noise_level),
                MotionBlur3D(motion_intensity=motion_blur_intensity),
                KSpaceUndersampling(acceleration_factor=undersampling_factor),
                ElasticDeformation3D(alpha=elastic_alpha, sigma=elastic_sigma),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply full degradation pipeline.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Degraded tensor with same shape

        forward method for RealisticDegradationPipeline.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        degraded = x
        for degradation in self.degradations:
            degraded = degradation(degraded)
        return degraded


# Convenience functions
def add_rician_noise(x: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Add Rician noise to tensor."""
    noise_model = RicianNoise(noise_level)
    return noise_model(x)


def apply_motion_blur_3d(x: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
    """Apply 3D motion blur."""
    blur_model = MotionBlur3D(motion_intensity=intensity)
    return blur_model(x)


def undersample_kspace(
    x: torch.Tensor, acceleration_factor: int = 2, pattern: str = "uniform"
) -> torch.Tensor:
    """Apply k-space undersampling."""
    undersampling = KSpaceUndersampling(acceleration_factor=acceleration_factor, pattern=pattern)
    return undersampling(x)


def apply_elastic_deformation_3d(
    x: torch.Tensor, alpha: float = 1.0, sigma: float = 10.0
) -> torch.Tensor:
    """Apply elastic deformation."""
    deformation = ElasticDeformation3D(alpha=alpha, sigma=sigma)
    return deformation(x)


def create_realistic_degradation_pipeline(
    rician_noise_level: float = 0.05,
    motion_blur_intensity: float = 0.3,
    undersampling_factor: int = 2,
    elastic_alpha: float = 0.5,
    elastic_sigma: float = 8.0,
) -> RealisticDegradationPipeline:
    """Create complete degradation pipeline."""
    return RealisticDegradationPipeline(
        rician_noise_level=rician_noise_level,
        motion_blur_intensity=motion_blur_intensity,
        undersampling_factor=undersampling_factor,
        elastic_alpha=elastic_alpha,
        elastic_sigma=elastic_sigma,
    )


__all__ = [
    "B0GeometricDistortion",
    "ElasticDeformation3D",
    "KSpaceUndersampling",
    "MotionBlur3D",
    "MultiContrastProcessor",
    "RandomB0Distortion",
    "RealisticDegradationPipeline",
    "RicianNoise",
    "add_rician_noise",
    "apply_elastic_deformation_3d",
    "apply_motion_blur_3d",
    "create_realistic_degradation_pipeline",
    "undersample_kspace",
]
