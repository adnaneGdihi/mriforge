r"""TSE Signal Model with B₀ Encoding for Low-Field MRI.

Implements the forward signal model from Koolstra et al. (2021):

.. math::

    y(\mathbf{k}(t)) = \int m(\mathbf{x})
        e^{-2\pi \Delta B_0(\mathbf{x}) i (t + t_{\text{shift}})}
        e^{-\mathbf{k}(t) \cdot \mathbf{x} i} \, d\mathbf{x}

This module provides:
1. A 2D Shepp-Logan phantom generator (oversampled 4× for intra-voxel simulation)
2. Synthetic Halbach-type B₀ field maps at 0.05 T
3. The TSE signal model that generates paired (unshifted, shifted) k-space data

Reference:
    Koolstra K et al., "Image distortion correction for MRI in low field
    permanent magnet systems with strong B₀ inhomogeneity and gradient
    field nonlinearities", MAGMA (2021) 34:631–642.
"""

from __future__ import annotations

import math

import torch

# ─────────────────────────────────────────────────────────────────────
# 1. Shepp-Logan Phantom
# ─────────────────────────────────────────────────────────────────────

# Standard Shepp-Logan parameters: (intensity, a, b, x0, y0, phi_deg)
_SHEPP_LOGAN_ELLIPSES = [
    (1.00, 0.6900, 0.9200, 0.0000, 0.0000, 0),
    (-0.8, 0.6624, 0.8740, 0.0000, -0.0184, 0),
    (-0.2, 0.1100, 0.3100, 0.2200, 0.0000, -18),
    (-0.2, 0.1600, 0.4100, -0.2200, 0.0000, 18),
    (0.10, 0.2100, 0.2500, 0.0000, 0.3500, 0),
    (0.10, 0.0460, 0.0460, 0.0000, 0.1000, 0),
    (0.10, 0.0460, 0.0460, 0.0000, -0.1000, 0),
    (0.10, 0.0460, 0.0230, -0.0800, -0.6050, 0),
    (0.10, 0.0230, 0.0230, 0.0000, -0.6060, 0),
    (0.10, 0.0230, 0.0460, 0.0600, -0.6050, 0),
]


def shepp_logan_2d(
    size: int = 128,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate a 2D Shepp-Logan phantom.

    Returns a real-valued spin-density image. The intensities follow
    the modified Shepp-Logan values to provide better visual contrast.

    Args:
        size: Output resolution N (creates N×N image).
        device: Target device.

    Returns:
        Real tensor [1, 1, size, size] with values in [0, 1].
    """
    image = torch.zeros(size, size, device=device)

    y_coords = torch.linspace(-1, 1, size, device=device)
    x_coords = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    for intensity, a, b, x0, y0, phi_deg in _SHEPP_LOGAN_ELLIPSES:
        phi = math.radians(phi_deg)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        # Rotated ellipse test
        xr = cos_phi * (xx - x0) + sin_phi * (yy - y0)
        yr = -sin_phi * (xx - x0) + cos_phi * (yy - y0)

        inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        image[inside] += intensity

    # Normalise to [0, 1]
    image = image - image.min()
    image = image / (image.max() + 1e-12)

    return image.unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────
# 2. Halbach-type B₀ Map Simulation
# ─────────────────────────────────────────────────────────────────────


def simulate_halbach_b0(
    shape: tuple[int, int],
    center_hz: float = 600.0,
    off_center_hz: float = 1500.0,
    off_center: bool = False,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    r"""Generate a synthetic B₀ deviation map mimicking a Halbach-array magnet.

    A k=1 cylindrical Halbach magnet has a dominant quadratic (2nd-order)
    inhomogeneity pattern. This function generates a realistic Halbach-like
    field map parameterised by peak deviations observed at the centre and
    off-centre slices of the paper's 0.05 T system.

    The field pattern is modelled as a **2nd-order polynomial** (dominant
    Halbach mode) plus random **higher-order perturbations** (manufacturing
    imperfections):

    .. math::

        \Delta B_0(x, y) = c_0 + c_1 x + c_2 y
            + c_3 x^2 + c_4 y^2 + c_5 x y + \epsilon(x, y)

    Args:
        shape: (H, W) output dimensions.
        center_hz: Peak ΔB₀ for a centre slice (~600 Hz in the paper).
        off_center_hz: Peak ΔB₀ for a 7.5 cm off-centre slice (~1500 Hz).
        off_center: If True, use off_center_hz; else use center_hz.
        seed: Random seed for reproducibility.
        device: Target device.

    Returns:
        B₀ deviation map [1, 1, H, W] in Hz.
    """
    if seed is not None:
        torch.manual_seed(seed)

    H, W = shape
    max_hz = off_center_hz if off_center else center_hz

    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Dominant 2nd-order Halbach mode (quadratic bowl)
    # Random coefficients scaled to produce the desired peak deviation
    c = torch.randn(6, device=device)
    field_2nd = (
        c[0] * 0.05
        + c[1] * xx * 0.1
        + c[2] * yy * 0.1
        + c[3] * xx**2
        + c[4] * yy**2
        + c[5] * xx * yy * 0.3
    )

    # Normalise so peak matches max_hz
    field_2nd = field_2nd - field_2nd.mean()
    field_2nd = field_2nd / (field_2nd.abs().max() + 1e-12) * max_hz * 0.8

    # Higher-order perturbations (manufacturing imperfections)
    noise_low = torch.randn(1, 1, max(2, H // 16), max(2, W // 16), device=device)
    noise_upsampled = (
        torch.nn.functional.interpolate(noise_low, size=(H, W), mode="bicubic", align_corners=False)
        .squeeze(0)
        .squeeze(0)
    )
    noise_upsampled = noise_upsampled / (noise_upsampled.abs().max() + 1e-12) * max_hz * 0.2

    b0_map = field_2nd + noise_upsampled
    return b0_map.unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────
# 3. TSE Signal Model
# ─────────────────────────────────────────────────────────────────────


class TSESignalModel:
    r"""Forward signal model for TSE with B₀ encoding via readout time-shift.

    Implements Eq. 1 of Koolstra et al.:

    .. math::

        y(\mathbf{k}(t)) = \sum_{\mathbf{x}} m(\mathbf{x})
            e^{-2\pi i \Delta B_0(\mathbf{x}) (t + t_{\text{shift}})}
            e^{-2\pi i \mathbf{k}(t) \cdot \mathbf{x}}

    The summation over x is computed on a 4× oversampled grid (as per the
    paper) to simulate multiple frequencies within each voxel.

    Args:
        n_readout: Number of frequency-encoding points per PE line.
        n_phase: Number of phase-encoding lines.
        fov_m: Field of view in metres.
        bw_hz: Readout bandwidth in Hz.
        oversample_factor: Grid oversampling for intra-voxel simulation.
    """

    def __init__(
        self,
        n_readout: int = 128,
        n_phase: int = 128,
        fov_m: float = 0.225,
        bw_hz: float = 20_000.0,
        oversample_factor: int = 4,
    ) -> None:
        self.n_ro = n_readout
        self.n_pe = n_phase
        self.fov = fov_m
        self.bw = bw_hz
        self.oversample = oversample_factor

        # Derived timing
        self.dwell_time = 1.0 / bw_hz  # seconds per sample
        self.pixel_bw = bw_hz / n_readout  # Hz/pixel

    def _build_time_vector(self, device: torch.device) -> torch.Tensor:
        """Build the readout time vector centred at the echo-top.

        Returns:
            Time vector [n_readout] in seconds, centred at 0.
        """
        total_readout = self.n_ro * self.dwell_time
        t = torch.linspace(
            -total_readout / 2,
            total_readout / 2,
            self.n_ro,
            device=device,
        )
        return t

    def _build_oversampled_image(
        self,
        image: torch.Tensor,
        b0_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upsample image and B₀ map to the oversampled grid.

        Args:
            image: Spin density [1, 1, H, W].
            b0_map: Field deviation [1, 1, H, W] in Hz.

        Returns:
            (image_os, b0_os) on the 4× oversampled grid.
        """
        os = self.oversample
        H, W = image.shape[-2:]
        H_os, W_os = H * os, W * os

        image_os = torch.nn.functional.interpolate(
            image, size=(H_os, W_os), mode="bilinear", align_corners=False
        )
        b0_os = torch.nn.functional.interpolate(
            b0_map, size=(H_os, W_os), mode="bilinear", align_corners=False
        )
        return image_os, b0_os

    def simulate_kspace(
        self,
        image: torch.Tensor,
        b0_map: torch.Tensor,
        t_shift: float = 0.0,
        snr: float | None = 20.0,
    ) -> torch.Tensor:
        r"""Generate 2D k-space data using the full signal model.

        For each phase-encoding step j and each frequency-encoding sample i,
        computes:

        .. math::

            y[j, i] = \sum_{\mathbf{x}} m(\mathbf{x})
                e^{-2\pi i \Delta B_0(\mathbf{x}) (t_i + t_{\text{shift}})}
                e^{-2\pi i (k_{x,i} x + k_{y,j} y)}

        Args:
            image: Spin density [1, 1, H, W], real-valued.
            b0_map: B₀ deviation [1, 1, H, W] in Hz.
            t_shift: Readout gradient time-shift in seconds.
            snr: Target SNR (adds white Gaussian noise). None disables noise.

        Returns:
            Complex k-space [1, 1, N_pe, N_ro].
        """
        device = image.device
        H, W = image.shape[-2:]

        # Oversample for intra-voxel simulation
        image_os, b0_os = self._build_oversampled_image(image, b0_map)
        H_os, W_os = image_os.shape[-2:]

        # Spatial coordinates on the oversampled grid (metres)
        x_os = torch.linspace(-self.fov / 2, self.fov / 2, W_os, device=device)
        y_os = torch.linspace(-self.fov / 2, self.fov / 2, H_os, device=device)

        # Readout time vector [N_ro]
        t_ro = self._build_time_vector(device)  # seconds, centred at 0

        # K-space coordinates
        dk_x = 1.0 / self.fov  # k-space step (1/m)
        dk_y = 1.0 / self.fov

        kx = (torch.arange(self.n_ro, device=device) - self.n_ro // 2) * dk_x  # [N_ro]
        ky = (torch.arange(self.n_pe, device=device) - self.n_pe // 2) * dk_y  # [N_pe]

        # Flatten the oversampled image for efficient matmul
        m_flat = image_os.squeeze().reshape(-1)  # [H_os * W_os]
        b0_flat = b0_os.squeeze().reshape(-1)  # [H_os * W_os]

        # Spatial coordinate grids (flattened)
        yy_os, xx_os = torch.meshgrid(y_os, x_os, indexing="ij")
        xx_flat = xx_os.reshape(-1)  # [M]
        yy_flat = yy_os.reshape(-1)  # [M]

        M = xx_flat.shape[0]

        # Build k-space line by line (memory-efficient)
        kspace = torch.zeros(self.n_pe, self.n_ro, dtype=torch.complex64, device=device)

        for j in range(self.n_pe):
            # B₀ phase: exp(-i * 2π * ΔB₀(x) * (t + t_shift))
            # t varies per readout sample, giving [N_ro, M]
            b0_phase = (
                -2.0 * math.pi * b0_flat.unsqueeze(0) * (t_ro.unsqueeze(1) + t_shift)
            )  # [N_ro, M]

            # Spatial encoding phase: exp(-i * 2π * (kx·x + ky·y))
            # kx varies per readout sample
            spatial_phase = (
                -2.0
                * math.pi
                * (kx.unsqueeze(1) * xx_flat.unsqueeze(0) + ky[j] * yy_flat.unsqueeze(0))
            )  # [N_ro, M]

            # Total phase
            total_phase = b0_phase + spatial_phase  # [N_ro, M]

            # Signal: sum over all spatial positions
            # y[j, i] = sum_x m(x) * exp(i * total_phase[i, x])
            signal_line = torch.sum(
                m_flat.unsqueeze(0) * torch.exp(1j * total_phase),
                dim=1,
            )  # [N_ro]

            kspace[j] = signal_line

        # Normalise by oversampling area
        kspace = kspace / (self.oversample**2)

        # Add noise
        if snr is not None:
            kspace = self._add_noise(kspace, snr)

        return kspace.unsqueeze(0).unsqueeze(0)

    def _add_noise(self, kspace: torch.Tensor, snr: float) -> torch.Tensor:
        """Add white Gaussian noise to achieve target SNR.

        Args:
            kspace: Complex k-space [N_pe, N_ro].
            snr: Target SNR (linear, not dB).

        Returns:
            Noisy complex k-space.
        """
        signal_power = kspace.abs().mean()
        sigma = signal_power / snr

        noise = (torch.randn_like(kspace.real) + 1j * torch.randn_like(kspace.real)) * (
            sigma / math.sqrt(2)
        )

        return kspace + noise

    def generate_paired_data(
        self,
        image: torch.Tensor,
        b0_map: torch.Tensor,
        t_shift: float = 100e-6,
        snr: float | None = 20.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate paired (unshifted, shifted) k-space acquisitions.

        This is the core data generation for the iterative B₀ mapping
        framework: one acquisition without time-shift and one with.

        Args:
            image: Spin density [1, 1, H, W].
            b0_map: B₀ deviation [1, 1, H, W] in Hz.
            t_shift: Readout gradient time-shift in seconds (e.g., 100e-6).
            snr: Target SNR. None disables noise.

        Returns:
            (kspace_unshifted, kspace_shifted), each [1, 1, N_pe, N_ro].
        """
        kspace_unshifted = self.simulate_kspace(image, b0_map, t_shift=0.0, snr=snr)
        kspace_shifted = self.simulate_kspace(image, b0_map, t_shift=t_shift, snr=snr)
        return kspace_unshifted, kspace_shifted


__all__ = [
    "TSESignalModel",
    "shepp_logan_2d",
    "simulate_halbach_b0",
]
