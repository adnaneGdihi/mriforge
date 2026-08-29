import warnings
from typing import Any

import torch
import torch.nn as nn


class PhysicsTransform(nn.Module):
    """Base class for physics-based transforms."""

    pass


class SimulateKSpace(PhysicsTransform):
    """
    Simulate k-space acquisition from ground truth images.
    Expects input shape matching the physics operator requirements.
    """

    def __init__(self, physics_operator: Any):
        """__init__.

        Args:
            physics_operator (Any): Description.
        """
        super().__init__()
        self.physics = physics_operator

    def forward(self, gt_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gt_image: (B, C, H, W) or (C, H, W)
        Returns:
            kspace: K-space data
        """
        return self.physics.forward(gt_image)


class ApplyMask(PhysicsTransform):
    """
    Apply undersampling mask to full k-space.
    """

    def __init__(self, mask: torch.Tensor):
        """__init__.

        Args:
            mask (torch.Tensor): Description.
        """
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, kspace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            kspace (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Description.
        """
        masked_kspace = kspace * self.mask
        return masked_kspace, self.mask


class AddGaussianNoise(PhysicsTransform):
    """
    Add Gaussian noise to k-space or image domain.
    """

    def __init__(self, std: float = 0.01):
        """__init__.

        Args:
            std (float): Description.
        """
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if self.std <= 0:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise


class LowFieldNoise(PhysicsTransform):
    """
    Simulates low-field MRI noise (Rician).
    """

    def __init__(self, target_snr: float = 15.0):
        """__init__.

        Args:
            target_snr (float): Description.
        """
        super().__init__()
        self.target_snr = target_snr

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Normalized image [0,1]
        """
        # Ensure float
        image = image.float()

        # Calculate signal power per batch item or global?
        # Typically per image, but let's assume batched
        # If batched, we need to handle dims.
        # For simplicity, we assume robust broadcasting or per-item loop if needed.
        # Here we do simple global estimation for batch or per-channel

        signal_power = torch.mean(image**2)
        snr_linear = 10 ** (self.target_snr / 10.0)
        noise_variance = signal_power / snr_linear

        noise_std = torch.sqrt(noise_variance / 2.0)

        # Rician noise = magnitude(real + n1, imag + n2)
        # Verify if input is complex or magnitude. Assuming magnitude input (real)
        noise_real = torch.randn_like(image) * noise_std
        noise_imag = torch.randn_like(image) * noise_std

        noisy_image = torch.sqrt((image + noise_real) ** 2 + noise_imag**2)
        return torch.clamp(noisy_image, 0, 1)


class B0Inhomogeneity(PhysicsTransform):
    """
    Simulates B0 field inhomogeneity artifacts via phase variation.
    """

    def __init__(self, strength: float = 0.05):
        """__init__.

        Args:
            strength (float): Description.
        """
        super().__init__()
        self.strength = strength

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            image (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        h, w = image.shape[-2:]
        device = image.device

        # Cache coordinate grid if possible, or generate on fly
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Low frequency phase map
        # sin(pi*y) * cos(pi*x)
        import math

        from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c

        phase_map = torch.sin(grid_y * math.pi) * torch.cos(grid_x * math.pi) * self.strength

        # FFT using canonical centered transform
        kspace = fft2c(image)
        # Apply phase shift
        kspace_shifted = kspace * torch.exp(1j * phase_map)
        # IFFT using canonical centered transform
        image_distorted = ifft2c(kspace_shifted).real

        return image_distorted


import torchio as tio


class LogMagnitudeScaling(tio.transforms.Transform):
    """
    Apply Log-Magnitude scaling to k-space data (Subject-based).
    Scales 'kspace', 'kspace_sub', and 'kspace_raw' if present.
    [Updated] Handles subject properly.
    """

    def __init__(self, quantile: float = 0.95, log_scale: bool = True, **kwargs):
        """__init__.

        Args:
            quantile (float): Description.
            log_scale (bool): Description.
        """
        super().__init__(**kwargs)
        self.quantile = quantile
        self.log_scale = log_scale

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # [UPDATED] Include canonical keys 'input' and 'target'
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        keys_to_transform = [
            k for k in ["kspace", "kspace_sub", "kspace_raw", "input", "target"] if k in subject
        ]

        # We should probably share the scaling factor if we want consistency?
        # Typically, reference scale comes from the fully sampled data (kspace).
        # But if we only have kspace_sub (test time), we assume robust stats on it.
        # For training, let's calculate scale from 'kspace' if available, else self.

        # BUT for Cold Diffusion, input is noise/degraded.
        # Let's scale each independently to normalize distribution,
        # OR use specific reference.
        # The user's snippet was simple function.
        # Let's apply independent robust normalization to start, or strict scale consistency?
        # If we use independent stats, input and target might deviate in absolute scale if masking removes peak energy (DC).
        # DC peak is crucial. Masking retains DC usually.
        # So independent stats might be safe.

        for key in keys_to_transform:
            image = subject[key]
            tensor = image.data

            # Helper to process tensor (C, W, H, D)
            # Typically C=2 (Complex).

            # 1. Compute Magnitude
            if tensor.shape[0] % 2 == 0:
                # Treat as complex pairs [Re, Im, Re, Im...]
                real = tensor[0::2]
                imag = tensor[1::2]
                mag = torch.sqrt(real**2 + imag**2)
            else:
                mag = torch.abs(tensor)

            # 2. Compute Scale (Robust Max)
            # Global per-image quantile
            q_val = torch.quantile(mag.float(), self.quantile)
            scale = q_val + 1e-8

            # 3. Apply Scaling
            scaled_tensor = tensor / scale

            # 4. Log Scaling (Optional)
            if self.log_scale:
                # Log(1 + |x|) * x/|x|
                # scaling factor = log(1 + mag) / (mag + eps)
                # Apply this scalar to the normalized tensor

                # Compute magnitude of normalized tensor
                if tensor.shape[0] % 2 == 0:
                    real_n = scaled_tensor[0::2]
                    imag_n = scaled_tensor[1::2]
                    mag_norm = torch.sqrt(real_n**2 + imag_n**2)
                else:
                    mag_norm = torch.abs(scaled_tensor)

                scale_log = torch.log1p(mag_norm) / (mag_norm + 1e-9)

                if tensor.shape[0] % 2 == 0:
                    # Expand scale_log [C/2, ...] to [C, ...]
                    # We repeat each element 2 times along dim 0
                    scale_log_expanded = torch.repeat_interleave(scale_log, 2, dim=0)
                    scaled_tensor = scaled_tensor * scale_log_expanded
                else:
                    scaled_tensor = scaled_tensor * scale_log

            # Update Subject
            # We create a new image to avoid in-place issues if shared
            new_image = tio.ScalarImage(tensor=scaled_tensor, affine=image.affine)
            subject[key] = new_image

            # [OPTIONAL] Store scale factor metadata?
            # subject[f"{key}_scale"] = scale

        return subject


class VirtualCoilCompression(tio.transforms.Transform):
    """
    Apply Virtual Coil Compression (VCC) using PCA/SVD.
    Standardizes variable multicoil data to a fixed number of virtual coils.

    Args:
        target_coils: Number of virtual coils to retain (e.g., 8).
    """

    def __init__(self, target_coils: int = 8, **kwargs):
        """__init__.

        Args:
            target_coils (int): Description.
        """
        warnings.warn(
            "VirtualCoilCompression is deprecated; use the canonical "
            "mriforge.data.transforms.coil_compression.SVDCoilCompressionTransform "
            "(or set physics.coil_processing.compression in config). It is kept "
            "importable only for backward compatibility and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
        self.target_coils = target_coils

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # [UPDATED] Include canonical keys 'input' and 'target'
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        keys_to_transform = [
            k for k in ["kspace", "kspace_sub", "kspace_raw", "input", "target"] if k in subject
        ]

        for key in keys_to_transform:
            image = subject[key]
            tensor = image.data  # (C_total, spatial...)

            # Handle Real/Imag stacked channels
            is_stacked = False
            if tensor.shape[0] % 2 == 0:
                # Assume stacked complex: [Re1, Im1, Re2, Im2...]
                # Convention in this codebase seems to be often alternating or split.
                # Let's assume alternating pairwise [R, I, R, I] for now based on LogMagnitudeScaling
                # (real = tensor[0::2], imag = tensor[1::2])
                real = tensor[0::2]
                imag = tensor[1::2]
                complex_data = torch.complex(real, imag)  # (Coils, Spatial...)
                is_stacked = True
            else:
                if torch.is_complex(tensor):
                    complex_data = tensor
                else:
                    continue

            # Flatten spatial dimensions
            num_coils = complex_data.shape[0]
            if num_coils <= self.target_coils:
                # Pad if too few
                if num_coils < self.target_coils:
                    pad = self.target_coils - num_coils
                    zeros = torch.zeros(
                        (pad, *complex_data.shape[1:]),
                        dtype=complex_data.dtype,
                        device=complex_data.device,
                    )
                    compressed = torch.cat([complex_data, zeros], dim=0)
                else:
                    compressed = complex_data
            else:
                # SVD Compression (PCA on coils)
                flattened = complex_data.reshape(num_coils, -1)  # (Coils, Voxels)
                gram = torch.matmul(flattened, flattened.conj().t())  # (C, C)

                # Eigen decomposition (Hermitian)
                L, V = torch.linalg.eigh(gram)

                # Top k vectors corresponding to largest eigenvalues
                compression_matrix = V[:, -self.target_coils :]  # (C_in, C_out)

                # Project: Y = P^H * X
                compressed_flat = torch.matmul(compression_matrix.conj().t(), flattened)
                compressed = compressed_flat.reshape(self.target_coils, *complex_data.shape[1:])

            # Restore original format
            if is_stacked:
                # Stack real/imag back
                out_real = compressed.real
                out_imag = compressed.imag
                # Interleave
                out_tensor = torch.stack([out_real, out_imag], dim=1).flatten(0, 1)
            else:
                out_tensor = compressed

            # Update image
            new_image = tio.ScalarImage(tensor=out_tensor, affine=image.affine)
            subject[key] = new_image

        return subject


class NormalizeTransform(PhysicsTransform):
    """Normalize tensor with mean and std.

    Used for ensuring consistent normalization statistics (e.g. using training stats for validation).
    """

    def __init__(self, mean: float, std: float):
        """__init__.

        Args:
            mean (float): Description.
            std (float): Description.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
        Returns:
            Normalized tensor: (x - mean) / std
        """
        return (x - self.mean) / self.std
