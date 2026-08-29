import numpy as np
import torch


def simulate_low_field_noise(image, target_snr=15.0) -> torch.Tensor:
    """Simulates low-field MRI noise by adding Rician noise.

    Args:
        image (torch.Tensor): The input clean image (normalized to [0, 1]).
        target_snr (float): The target Signal-to-Noise Ratio in dB.

    Returns:
        torch.Tensor: The noisy image.

    """
    if image.is_cuda:
        device = image.get_device()
    else:
        device = "cpu"

    # Ensure image is in float
    image = image.float()

    # Calculate signal power
    signal_power = torch.mean(image**2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (target_snr / 10.0)

    # Calculate noise variance
    noise_variance = signal_power / snr_linear

    # Generate Gaussian noise for real and imaginary channels
    noise_real = torch.randn_like(image) * torch.sqrt(noise_variance / 2.0)
    noise_imag = torch.randn_like(image) * torch.sqrt(noise_variance / 2.0)

    # Add noise to the signal (assuming signal is real)
    noisy_real = image + noise_real
    noisy_imag = noise_imag

    # Magnitude image with Rician noise
    noisy_image = torch.sqrt(noisy_real**2 + noisy_imag**2)

    # Clamp to original range
    noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image.to(device)


def simulate_b0_inhomogeneity(image, strength=0.05, readout_direction="y") -> torch.Tensor:
    """Simulates B0 inhomogeneity artifacts via true geometric distortion.
    Generates a spatial B0 field map and applies pixel displacement
    along the assumed phase-encoding direction, mimicking EPI distortion.

    Args:
        image (torch.Tensor): The input image [..., H, W].
        strength (float): The strength of the inhomogeneity (distortion magnitude).
        readout_direction (str): 'x' or 'y', defining the phase encoding direction constraint.

    Returns:
        torch.Tensor: Image with geometric B0 artifacts.
    """
    import torch.nn.functional as F

    if image.is_cuda:
        device = image.get_device()
    else:
        device = "cpu"

    h, w = image.shape[-2:]
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )

    # 1. Simulate a realistic B0 field map (low frequency structured off-resonance + noise)
    b0_field = torch.sin(y_coords * 1.5 * np.pi) * torch.cos(x_coords * 1.5 * np.pi)
    b0_field += torch.randn_like(b0_field) * 0.05
    # Smooth the field
    b0_field = F.avg_pool2d(b0_field.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()

    # Scale to displacement magnitude
    displacement = b0_field * strength

    # 2. Build the sampling grid for grid_sample
    # Base grid is just the identity mapping over [-1, 1]
    grid_y, grid_x = y_coords.clone(), x_coords.clone()

    # Apply displacement along the phase encode direction (typically y for EPI)
    if readout_direction == "y":  # Phase encoding along y
        grid_y = grid_y + displacement
    else:
        grid_x = grid_x + displacement

    # Stack into [1, H, W, 2] for grid_sample
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    # 3. Resample the image
    original_shape = image.shape
    if image.dim() == 2:
        img_4d = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        img_4d = image.unsqueeze(0)
    else:
        img_4d = image

    # Apply distortion
    distorted = F.grid_sample(
        img_4d, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )

    distorted = distorted.view(original_shape)
    return torch.clamp(distorted, 0, 1)


class DegradationPipeline:
    """A pipeline to apply a series of degradations to simulate low-field MRI."""

    def __init__(self, degradations):
        """__init__.

        Args:
            degradations (Any): Description.
        """
        self.degradations = degradations

    def __call__(self, image):
        """__call__.

        Args:
            image (Any): Description.
        Returns:
            Any: Description.
        """
        for degradation, params in self.degradations:
            image = degradation(image, **params)
        return image
