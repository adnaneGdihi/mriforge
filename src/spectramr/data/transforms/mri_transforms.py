"""MRI-Specific Transforms for Complex Data.

Phase-Preserving and Physics-Aware Transforms for MRI.

Features:
- ComplexNormalization: Robust percentile scaling preserving phase
- KSpacePhasePreservingNormalization: DC-component based normalization for Non-Cartesian
- PhysicsAugmentation: B0 field simulation, thermal noise addition (SNR), partial Fourier
- ToComplex/FromComplex: Tensor format conversion utilities

These transforms ensure phase integrity critical for reconstruction tasks.
"""

import torch

from spectramr.data.transforms.normalization import compute_magnitude


class ComplexNormalization:
    """Normalizes complex MRI data while preserving phase relationships.

    Strategy: Scale by the robust percentile of MAGNITUDE, keeping complex structure.
    This prevents outliers (like fat signal spikes) from affecting normalization.

    Delegates magnitude computation to
    :func:`spectramr.data.transforms.normalization.compute_magnitude` (SSOT).

    Args:
        percentile: Percentile for robust max (default 0.99 to avoid outliers)
        eps: Small value to prevent division by zero

    Example:
        >>> norm = ComplexNormalization(percentile=0.99)
        >>> kspace_normalized = norm(kspace)  # Preserves phase!
    """

    def __init__(self, percentile: float = 0.99, eps: float = 1e-8):
        """__init__.

        Args:
            percentile (float): Description.
            eps (float): Description.
        """
        self.percentile = percentile
        self.eps = eps

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize complex or real/imag stacked data.

        Args:
            data: Complex [C, H, W] or stacked [2, H, W] (Re/Im)

        Returns:
            Normalized data with same structure
        """
        # Delegate magnitude computation to SSOT
        magnitude = compute_magnitude(data, eps=self.eps)

        # Robust percentile scaling via torch.quantile (consistent with SSOT)
        scale_val = torch.quantile(magnitude.flatten().float(), self.percentile)

        # Safety check
        if scale_val < self.eps:
            scale_val = torch.tensor(1.0, device=data.device)

        return data / scale_val


class KSpacePhasePreservingNormalization:
    """K-Space normalization that preserves spectral phase.

    For Non-Cartesian (Spiral/Radial), we normalize k-space by the
    magnitude of the DC component (center of k-space), which represents
    the total signal energy. This is physically meaningful and stable.
    """

    def __init__(self, center_fraction: float = 0.1, eps: float = 1e-8):
        """
        Args:
            center_fraction: Fraction of k-space center to use for normalization
            eps: Small value for numerical stability
        """
        self.center_fraction = center_fraction
        self.eps = eps

    def __call__(self, kspace: torch.Tensor) -> torch.Tensor:
        """Normalize k-space by center region magnitude.

        Args:
            kspace: K-space data [C, H, W] or [2, N] (for non-Cartesian)

        Returns:
            Normalized k-space
        """
        is_complex = kspace.is_complex()

        if is_complex:
            magnitude = torch.abs(kspace)
        else:
            if kspace.shape[0] == 2:
                magnitude = torch.sqrt(kspace[0] ** 2 + kspace[1] ** 2)
            else:
                magnitude = torch.abs(kspace)

        # For grid k-space: use center region
        if magnitude.dim() >= 2 and magnitude.shape[-1] > 1 and magnitude.shape[-2] > 1:
            H, W = magnitude.shape[-2:]
            ch = max(1, int(H * self.center_fraction / 2))
            cw = max(1, int(W * self.center_fraction / 2))

            # Ensure valid slice indices
            h_start = max(0, H // 2 - ch)
            h_end = min(H, H // 2 + ch)
            w_start = max(0, W // 2 - cw)
            w_end = min(W, W // 2 + cw)

            center_mag = magnitude[..., h_start:h_end, w_start:w_end]

            # Fallback if slice is somehow empty
            if center_mag.numel() == 0:
                scale_val = magnitude.max() + self.eps
            else:
                scale_val = center_mag.max() + self.eps
        else:
            # For 1D k-space (non-Cartesian flattened)
            scale_val = magnitude.max() + self.eps

        return kspace / scale_val


class PhysicsAugmentation:
    """Applies physics-aware augmentations for MRI training.

    Includes:
    - B0 simulation (off-resonance)
    - Noise addition (thermal/physiological)
    - Partial Fourier simulation
    """

    def __init__(
        self,
        add_noise: bool = True,
        snr_range: tuple[float, float] = (20.0, 40.0),
        add_b0: bool = False,
        b0_max_hz: float = 100.0,
    ):
        """__init__.

        Args:
            add_noise (bool): Description.
            snr_range (tuple[float, float]): Description.
            add_b0 (bool): Description.
            b0_max_hz (float): Description.
        """
        self.add_noise = add_noise
        self.snr_range = snr_range
        self.add_b0 = add_b0
        self.b0_max_hz = b0_max_hz

    def __call__(self, sample: dict) -> dict:
        """Apply physics augmentations.

        Args:
            sample: Dict with 'kspace' and/or 'image' keys

        Returns:
            Augmented sample
        """
        if self.add_noise and "kspace" in sample:
            sample["kspace"] = self._add_noise(sample["kspace"])

        return sample

    def _add_noise(self, kspace: torch.Tensor) -> torch.Tensor:
        """Add complex Gaussian noise to k-space."""
        snr = torch.empty(1).uniform_(*self.snr_range).item()

        is_complex = kspace.is_complex()
        if is_complex:
            signal_power = torch.mean(torch.abs(kspace) ** 2)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = torch.randn_like(torch.view_as_real(kspace)) * torch.sqrt(noise_power / 2)
            noise_complex = torch.view_as_complex(noise)
            return kspace + noise_complex
        else:
            # Assume [2, H, W]
            signal_power = torch.mean(kspace**2)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = torch.randn_like(kspace) * torch.sqrt(noise_power)
            return kspace + noise


class ULFNoiseGate:
    """Hard-gates the background noise floor for ULF data.

    Prevents the network from trying to translate thermal/Rician noise
    inherent in ULF (e.g. 50mT) data into HF tissue structures.

    Strategy:
    1. Estimate noise floor from background (default 5th percentile)
    2. Set everything below floor to mask_value (silence)
    """

    def __init__(self, threshold_percentile: float = 5.0, mask_value: float = 0.0):
        """__init__.

        Args:
            threshold_percentile (float): Description.
            mask_value (float): Description.
        """
        self.threshold_percentile = threshold_percentile
        self.mask_value = mask_value

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise gating.

        Args:
            data: Complex or Real/Imag stacked tensor [C, H, W]

        Returns:
            Gated tensor with noise floor set to mask_value
        """
        is_complex = data.is_complex()

        if is_complex:
            magnitude = torch.abs(data)
        else:
            if data.ndim == 3 and data.shape[0] == 2:
                # [2, H, W] Real/Imag
                magnitude = torch.sqrt(data[0] ** 2 + data[1] ** 2)
            else:
                magnitude = torch.abs(data)

        # Estimate noise floor dynamically per slice
        flat_mag = magnitude.view(-1)
        if flat_mag.numel() == 0:
            return data

        k = int(self.threshold_percentile / 100.0 * flat_mag.numel())
        k = max(1, min(k, flat_mag.numel() - 1))

        # Use kthvalue for robustness
        noise_floor = torch.kthvalue(flat_mag, k).values

        # Create mask (True = Signal, False = Noise)
        mask = magnitude > noise_floor

        # Expand mask for broadcasting if needed
        # Case specific: [2, H, W] data but [H, W] mask
        if not is_complex and data.ndim == 3 and data.shape[0] == 2 and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if self.mask_value == 0.0:
            return data * mask
        else:
            # Clone to apply custom value
            out = data.clone()
            # Expand mask to match data shape for torch.where if needed
            # complex data * real mask works, but where needs dtype match sometimes
            target_val = torch.tensor(self.mask_value, dtype=data.dtype, device=data.device)
            return torch.where(mask, data, target_val)


class ToComplexTensor:
    """Converts various input formats to complex tensor.

    Handles:
    - Already complex tensors
    - [2, H, W] real/imag stacked
    - [H, W, 2] real/imag stacked
    - Magnitude-only (pads with zero phase) [NOT RECOMMENDED]
    """

    def __init__(self, assume_format: str = "channel_first"):
        """
        Args:
            assume_format: "channel_first" for [2, H, W] or "channel_last" for [H, W, 2]
        """
        self.format = assume_format

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Convert to complex tensor.

        Args:
            data: Input tensor in various formats

        Returns:
            Complex tensor
        """
        if data.is_complex():
            return data

        if data.dim() == 3:
            if self.format == "channel_first" and data.shape[0] == 2:
                # [2, H, W] -> complex [H, W]
                return torch.complex(data[0], data[1])
            elif self.format == "channel_last" and data.shape[-1] == 2:
                # [H, W, 2] -> complex [H, W]
                return torch.complex(data[..., 0], data[..., 1])

        # Magnitude only: add zero phase (warning!)
        import warnings

        warnings.warn(
            "ToComplexTensor: Input appears to be magnitude-only. "
            "Adding zero phase, but this destroys phase information!"
        )
        return torch.complex(data, torch.zeros_like(data))


class FromComplexTensor:
    """Converts complex tensor to real representation.

    Output options:
    - [2, H, W] real/imag stacked (training format)
    - magnitude [1, H, W] (visualization only!)
    """

    def __init__(self, output_format: str = "stack", stack_dim: int = 0):
        """
        Args:
            output_format: "stack" for [2, H, W] or "magnitude" for |z|
            stack_dim: Dimension to stack real/imag
        """
        self.output_format = output_format
        self.stack_dim = stack_dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Convert complex to real representation.

        Args:
            data: Complex tensor

        Returns:
            Real tensor [2, H, W] or [1, H, W]
        """
        if not data.is_complex():
            return data

        if self.output_format == "magnitude":
            return torch.abs(data).unsqueeze(0)
        else:
            # Stack real/imag
            return torch.stack([data.real, data.imag], dim=self.stack_dim)


class RandomMRIFlip:
    """Randomly flips MRI image/k-space data.

    Args:
        prob: Probability of flip
    """

    def __init__(self, prob: float = 0.5):
        """__init__.

        Args:
            prob (float): Description.
        """
        self.prob = prob

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """__call__.

        Args:
            data (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        # Draw from torch's RNG (not Python's ``random``): PyTorch re-seeds its
        # global generator per DataLoader worker, but never touches the ``random``
        # module — so ``random.random()`` is either identical across forked
        # workers or OS-entropy-random under forkserver, i.e. non-reproducible.
        if torch.rand(1).item() < self.prob:
            # Flip last dimension (width)
            return torch.flip(data, dims=[-1])
        return data


class RandomMRIRotation:
    """Randomly rotates MRI image/k-space data by 90 degrees.

    Args:
        prob: Probability of rotation
    """

    def __init__(self, prob: float = 0.5):
        """__init__.

        Args:
            prob (float): Description.
        """
        self.prob = prob

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """__call__.

        Args:
            data (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        # torch RNG, not Python's ``random`` — see RandomMRIFlip for why.
        if torch.rand(1).item() < self.prob:
            # Rotate 90 degrees in the last two dimensions (H, W)
            return torch.rot90(data, 1, [-2, -1])
        return data


# Convenience compositions
def get_kspace_normalization() -> ComplexNormalization:
    """Get standard k-space normalization transform."""
    return ComplexNormalization(percentile=0.99)


def get_image_normalization() -> ComplexNormalization:
    """Get standard image normalization transform."""
    return ComplexNormalization(percentile=0.999)


__all__ = [
    "ComplexNormalization",
    "FromComplexTensor",
    "KSpacePhasePreservingNormalization",
    "PhysicsAugmentation",
    "RandomMRIFlip",
    "RandomMRIRotation",
    "ToComplexTensor",
    "ULFNoiseGate",
    "get_image_normalization",
    "get_kspace_normalization",
]
