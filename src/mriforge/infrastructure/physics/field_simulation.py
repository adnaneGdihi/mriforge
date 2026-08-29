"""B0 Field Map Simulation for Multi-Field-Strength MRI Training.

Generates physically plausible B0 inhomogeneity maps for training
deep learning models to be robust to off-resonance effects.

### B0 Generation Logic
```mermaid
flowchart TD
    Start[Generate B0 Map] --> CheckField{Field Strength?}

    CheckField -->|High Field >= 1.5T| HighField[Susceptibility Dominant]
    CheckField -->|Low Field < 0.1T| LowField[Maxwell Terms Important]

    HighField --> Style{Style?}
    LowField --> Style

    Style -->|Smooth| GenSmooth[Low Res Noise -> Upsample]
    Style -->|Anatomical| GenAnat[Smooth Base + Gaussian Hotspots]

    GenSmooth --> Combine
    GenAnat --> Combine

    LowField -->|Calculate| Maxwell[Concomitant Gradients]
    Maxwell --> Combine{Combine Terms}

    Combine --> Final[Final Hz Map]
```

Field Strength Scaling:
- 3.0T: Off-resonance ≈ ±200-400 Hz (susceptibility dominant)
- 1.5T: Off-resonance ≈ ±100-200 Hz
- 0.55T: Off-resonance ≈ ±40-70 Hz (Low-Field)
- 0.064T: Off-resonance ≈ ±5-15 Hz (Ultra-Low-Field, Maxwell terms important)

B0 inhomogeneities in vivo are caused by:
- Air-tissue interfaces (sinuses, ears)
- Susceptibility differences (blood, fat, bone)
- Scanner magnet imperfections
- Concomitant gradients (Maxwell terms) at Ultra-Low-Field

References:
- Jezzard P, Balaban RS., "Correction for geometric distortion" MRM 1995
- Bernstein MA et al., "Concomitant gradient terms in phase contrast MR" MRM 1998
"""

from enum import Enum

import torch
import torch.nn.functional as F


class FieldStrength(Enum):
    """Standard clinical and research field strengths."""

    T3_0 = 3.0  # High Field (Clinical Standard)
    T1_5 = 1.5  # Standard Field (Clinical)
    T0_55 = 0.55  # Low Field (Lung/Heart MRI, Siena)
    T0_35 = 0.35  # Low Field (Open MRI)
    T0_064 = 0.064  # Ultra-Low Field (Point-of-Care, Hyperfine Swoop)


# Empirical off-resonance ranges (Hz) by field strength
FIELD_STRENGTH_HZ_MAP = {
    3.0: 250.0,  # ±250 Hz typical
    1.5: 120.0,  # ±120 Hz typical
    0.55: 60.0,  # ±60 Hz typical
    0.35: 40.0,  # ±40 Hz typical
    0.064: 15.0,  # ±15 Hz typical
}


def get_max_hz_for_field(field_strength: float) -> float:
    """Get approximate max off-resonance (Hz) for a given field strength.

    Uses linear interpolation based on empirical data.
    Off-resonance scales roughly linearly with B0.

    Args:
        field_strength: Main magnetic field in Tesla

    Returns:
        Maximum off-resonance in Hz
    """
    # Use lookup if available, otherwise linear approximation
    if field_strength in FIELD_STRENGTH_HZ_MAP:
        return FIELD_STRENGTH_HZ_MAP[field_strength]
    # Linear approximation: ~83.3 Hz per Tesla
    return 83.3 * field_strength


class B0MapSimulator:
    r"""Generates synthetic B0 inhomogeneity maps for MRI simulation.

    Supports multiple field strengths with appropriate scaling:
    - High field (3T): Susceptibility artifacts dominate
    - Low field (0.55T): Reduced artifacts, longer readouts
    - Ultra-low field (0.064T): Maxwell terms (concomitant gradients) become significant

    B0 maps represent the deviation from the main magnetic field (in Hz)
    at each spatial location. These deviations cause phase accumulation
    that leads to blurring and distortion in long-readout sequences.
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{B0}(x) = x \odot e^{-i \gamma B_0 TE}"""

    def __init__(self, field_strength: float = 3.0, concomitant_gradient_mt_m: float = 15.0):
        """Initialize simulator for a specific field strength.

        Args:
            field_strength: Main magnetic field in Tesla (default: 3.0T)
            concomitant_gradient_mt_m: Representative slice-select gradient
                amplitude (mT/m) used for the concomitant (Maxwell) field
                estimate. At a central axial slice the in-plane concomitant
                field is ``Gz²(x²+y²)/(8·B0)``, so a NON-ZERO slice gradient is
                required for ``include_maxwell`` to do anything — passing 0 (the
                previous behavior) made the advertised ULF Maxwell term a silent
                no-op (CLAUDE.md pitfall #16). 15 mT/m is a typical value.
        """
        self.field_strength = field_strength
        self.concomitant_gradient_mt_m = concomitant_gradient_mt_m
        self._max_hz = get_max_hz_for_field(field_strength)

    @property
    def max_hz(self) -> float:
        """Get maximum off-resonance in Hz for current field strength."""
        return self._max_hz

    @staticmethod
    def generate_smooth_b0(
        shape: tuple[int, int],
        max_hz: float = 100.0,
        smoothness: float = 8.0,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Generate a smooth, low-frequency B0 map.

        Creates a physically plausible field map by generating low-resolution
        noise and upsampling with bicubic interpolation.

        Args:
            shape: Output shape (H, W)
            max_hz: Maximum field deviation in Hz (±max_hz)
            smoothness: Downsampling factor (higher = smoother)
            seed: Random seed for reproducibility
            device: Target device

        Returns:
            B0 map [1, 1, H, W] in Hz, range [-max_hz, +max_hz]
        """
        if seed is not None:
            torch.manual_seed(seed)

        H, W = shape

        # Generate low-resolution noise
        small_H = max(2, int(H / smoothness))
        small_W = max(2, int(W / smoothness))
        low_res_noise = torch.randn(1, 1, small_H, small_W, device=device)

        # Upsample to full resolution (creates smooth field)
        b0_map = F.interpolate(
            low_res_noise,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )

        # Normalize to zero mean, unit std
        b0_map = (b0_map - b0_map.mean()) / (b0_map.std() + 1e-8)

        # Scale to Hz
        b0_map = b0_map * max_hz

        return b0_map

    @staticmethod
    def generate_anatomical_b0(
        shape: tuple[int, int],
        max_hz: float = 100.0,
        hotspot_count: int = 3,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Generate B0 map with anatomical-like hotspots.

        Simulates stronger field deviations at tissue boundaries
        (e.g., air-tissue, susceptibility interfaces).

        Args:
            shape: Output shape (H, W)
            max_hz: Maximum field deviation in Hz
            hotspot_count: Number of B0 hotspots (air cavities, etc.)
            seed: Random seed
            device: Target device

        Returns:
            B0 map [1, 1, H, W] in Hz
        """
        if seed is not None:
            torch.manual_seed(seed)

        H, W = shape

        # Start with smooth background
        b0_map = B0MapSimulator.generate_smooth_b0(
            shape, max_hz=max_hz * 0.3, smoothness=8.0, device=device
        )

        # Add Gaussian hotspots (simulating air cavities, susceptibility)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        for _ in range(hotspot_count):
            # Random center and size
            cx = (torch.rand(1, device=device) * 2 - 1) * 0.6
            cy = (torch.rand(1, device=device) * 2 - 1) * 0.6
            sigma = 0.1 + torch.rand(1, device=device) * 0.15
            amplitude = (torch.rand(1, device=device) - 0.5) * 2 * max_hz * 0.7

            # Gaussian blob
            hotspot = amplitude * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
            b0_map = b0_map + hotspot.unsqueeze(0).unsqueeze(0)

        # Clip to reasonable range
        b0_map = torch.clamp(b0_map, -max_hz * 1.5, max_hz * 1.5)

        return b0_map

    @staticmethod
    def generate_concomitant_field(
        shape: tuple[int, int],
        field_strength: float = 0.064,
        gx: float = 0.0,  # mT/m
        gy: float = 0.0,  # mT/m
        gz: float = 0.0,  # mT/m
        z_offset: float = 0.0,  # meters from isocenter
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Generate rigorous Maxwell term (concomitant gradient) field.

        Calculates 2nd-order corrections based on Bernstein et al. (1998):
        Bc = (1 / 2B0) * [
             (Gx^2 + Gy^2) * z^2
           + Gz^2 * (x^2 + y^2)/4
           - Gx*Gz*x*z - Gy*Gz*y*z
        ]

        Note: We simplify by assuming z_offset is constant across the slice (2D approximation).
        The cross-terms (Gx*Gz, Gy*Gz) depend on x*z and y*z.

        Args:
            shape: Output shape (H, W)
            field_strength: Main field in Tesla
            gx, gy, gz: Gradient amplitudes in mT/m
            z_offset: Slice position from isocenter (meters)
            device: Target device

        Returns:
            Concomitant field [1, 1, H, W] in Hz
        """
        # Ensure gradients are float
        if torch.is_tensor(gx):
            gx = gx.item()
        if torch.is_tensor(gy):
            gy = gy.item()
        if torch.is_tensor(gz):
            gz = gz.item()

        H, W = shape
        fov = 0.25  # 250mm

        y = torch.linspace(-fov / 2, fov / 2, H, device=device)
        x = torch.linspace(-fov / 2, fov / 2, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Convert mT/m to T/m
        Gx = gx * 1e-3
        Gy = gy * 1e-3
        Gz = gz * 1e-3
        B0 = field_strength
        gamma_hz = 42.576e6

        # 1. Transverse Gradients contribution (Readout/Phase)
        # Term: (Gx^2 + Gy^2) * z^2 / (2 * B0)
        # Valid if z is constant (2D slice)
        bc_transverse = ((Gx**2 + Gy**2) * z_offset**2) / (2 * B0)

        # 2. Longitudinal Gradient contribution (Slice Select)
        # Term: Gz^2 * (x^2 + y^2) / (8 * B0)
        # Matches the docstring header (1/(2*B0)) * [Gz^2 * (x^2+y^2)/4]
        # since 1/(2*B0) * 1/4 = 1/(8*B0); no numeric difference.
        bc_longitudinal = (Gz**2 * (xx**2 + yy**2)) / (8 * B0)

        # 3. Cross Terms (if z_offset != 0 and Gz != 0)
        # Bernstein 1998 concomitant field, cross term: -(Gx*x + Gy*y)*Gz*z / (2*B0).
        # The 1/(2*B0) prefactor is shared with the other two terms (see docstring);
        # dividing by B0 alone doubled this contribution.
        bc_cross = -(Gx * xx + Gy * yy) * Gz * z_offset / (2 * B0)

        delta_B = bc_transverse + bc_longitudinal + bc_cross

        return (delta_B * gamma_hz).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def generate_linear_gradient(
        shape: tuple[int, int],
        gradient_hz_per_pixel: float = 0.5,
        direction: str = "y",
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Generate a linear B0 gradient.

        Useful for testing gradient distortion correction.

        Args:
            shape: Output shape (H, W)
            gradient_hz_per_pixel: Field change per pixel
            direction: "x" or "y"
            device: Target device

        Returns:
            B0 map [1, 1, H, W] in Hz
        """
        H, W = shape

        # Centered *unit-step* ramp so the field actually changes by exactly
        # `gradient_hz_per_pixel` between adjacent pixels. linspace(-N/2, N/2, N)
        # has a per-sample step of N/(N-1) ≈ 1.0, not 1.0, which made the
        # parameter off by N/(N-1) (it only behaved as "per pixel" by accident
        # of that near-unit step).
        if direction == "y":
            ramp = torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2.0
            gradient = ramp.unsqueeze(1).expand(H, W)
        elif direction == "x":
            ramp = torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2.0
            gradient = ramp.unsqueeze(0).expand(H, W)
        else:
            raise ValueError(
                f"direction must be 'x' or 'y', got {direction!r} "
                "(no silent fallback — see CLAUDE.md pitfall #9)."
            )

        b0_map = gradient * gradient_hz_per_pixel

        return b0_map.unsqueeze(0).unsqueeze(0)

    def generate_for_field(
        self,
        shape: tuple[int, int],
        style: str = "smooth",
        include_maxwell: bool = True,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Generate B0 map scaled for the simulator's field strength.

        Automatically includes Maxwell terms for Ultra-Low-Field (< 0.1T).

        Args:
            shape: Output shape (H, W)
            style: "smooth" or "anatomical"
            include_maxwell: Include concomitant gradients for ULF
            device: Target device

        Returns:
            B0 map [1, 1, H, W] in Hz
        """
        # Generate base B0 map
        if style == "anatomical":
            b0_map = self.generate_anatomical_b0(shape, max_hz=self._max_hz, device=device)
        else:
            b0_map = self.generate_smooth_b0(shape, max_hz=self._max_hz, device=device)

        # Add Maxwell terms for Ultra-Low-Field. Pass the representative
        # slice-select gradient so the central-slice term Gz²(x²+y²)/(8·B0) is
        # actually non-zero (gz=0 made include_maxwell a no-op).
        if include_maxwell and self.field_strength < 0.1:
            maxwell = self.generate_concomitant_field(
                shape,
                field_strength=self.field_strength,
                gz=self.concomitant_gradient_mt_m,
                device=device,
            )
            b0_map = b0_map + maxwell

        return b0_map

    def generate_batch(
        self,
        batch_size: int,
        shape: tuple[int, int],
        max_hz: float | None = None,
        style: str = "smooth",
        smoothness: float = 8.0,
        include_maxwell: bool = True,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Generate a batch of B0 maps for training.

        Args:
            batch_size: Number of maps to generate
            shape: Output shape (H, W)
            max_hz: Override max Hz (uses field strength if None)
            style: "smooth" or "anatomical"
            smoothness: Smoothness factor for smooth style
            include_maxwell: Include Maxwell terms for ULF
            device: Target device

        Returns:
            B0 maps [B, 1, H, W] in Hz
        """
        hz = max_hz if max_hz is not None else self._max_hz

        maps = []
        for _ in range(batch_size):
            if style == "anatomical":
                b0 = self.generate_anatomical_b0(shape, max_hz=hz, device=device)
            else:
                b0 = self.generate_smooth_b0(shape, max_hz=hz, smoothness=smoothness, device=device)

            # Add Maxwell terms for ULF (non-zero slice gradient → real field).
            if include_maxwell and self.field_strength < 0.1:
                maxwell = self.generate_concomitant_field(
                    shape,
                    field_strength=self.field_strength,
                    gz=self.concomitant_gradient_mt_m,
                    device=device,
                )
                b0 = b0 + maxwell

            maps.append(b0)

        return torch.cat(maps, dim=0)


class B0AugmentationTransform:
    """Data augmentation transform that adds B0 effects to training data.

    Use this in your data pipeline to make models robust to field inhomogeneity.
    """

    def __init__(
        self,
        field_strength: float = 3.0,
        max_hz: float | None = None,
        smoothness_range: tuple[float, float] = (4.0, 12.0),
        prob: float = 0.5,
        style: str = "smooth",
        include_maxwell: bool = True,
    ):
        """
        Args:
            field_strength: Scanner field strength in Tesla
            max_hz: Override max Hz (uses field-appropriate default if None)
            smoothness_range: Range of smoothness values (random per sample)
            prob: Probability of applying B0 augmentation
            style: "smooth" or "anatomical"
            include_maxwell: Include Maxwell terms for ULF
        """
        self.simulator = B0MapSimulator(field_strength=field_strength)
        self.max_hz = max_hz if max_hz is not None else self.simulator.max_hz
        self.smoothness_range = smoothness_range
        self.prob = prob
        self.style = style
        self.include_maxwell = include_maxwell

    def __call__(self, sample: dict) -> dict:
        """Apply B0 augmentation to sample.

        Args:
            sample: Dict with 'image' or 'kspace' key

        Returns:
            Sample with added 'b0_map' key
        """
        if torch.rand(1).item() > self.prob:
            return sample

        # Get shape from image or kspace
        if "image" in sample:
            shape = sample["image"].shape[-2:]
            device = sample["image"].device
        elif "kspace" in sample:
            shape = sample["kspace"].shape[-2:]
            device = sample["kspace"].device
        else:
            return sample

        # Random smoothness
        smoothness = self.smoothness_range[0] + torch.rand(1).item() * (
            self.smoothness_range[1] - self.smoothness_range[0]
        )

        # Generate B0 map using field-aware method
        if self.style == "anatomical":
            b0_map = self.simulator.generate_anatomical_b0(shape, max_hz=self.max_hz, device=device)
        else:
            b0_map = self.simulator.generate_smooth_b0(
                shape, max_hz=self.max_hz, smoothness=smoothness, device=device
            )

        # Add Maxwell terms for ULF. Pass the slice-select gradient so the
        # concomitant term is actually non-zero — omitting gz defaulted it to 0,
        # making include_maxwell an advertised no-op (pitfall #16), unlike the
        # sibling generate_for_field / generate_batch paths.
        if self.include_maxwell and self.simulator.field_strength < 0.1:
            maxwell = self.simulator.generate_concomitant_field(
                shape,
                field_strength=self.simulator.field_strength,
                gz=self.simulator.concomitant_gradient_mt_m,
                device=device,
            )
            b0_map = b0_map + maxwell

        sample["b0_map"] = b0_map

        return sample
