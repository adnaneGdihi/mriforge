"""Synthetic Marker-Based B0/B1 Field Extraction.

All markers in this module are **Synthetic Digital Anchors** — mathematically
perfect arrays digitally injected into clean datasets *before* forward-physics
simulation.  No physical scanner is required.  The ``DigitalTwinSimulator``
handles injection and corruption; these extractors reverse-engineer the
simulated fields using only the known marker coordinates.

Methods
-------
- :class:`MarkerB0FromDisplacement`    — EPI pixel-shift → B0 (Hz)
- :class:`MarkerB0FromDualEcho`        — ΔΦ / ΔTE → B0 (Hz)
- :class:`MarkerB1TransmitDoubleAngle` — α/2α signal ratio → B1+ scale
- :class:`MarkerB1TransmitAFI`         — AFI with known T1 → B1+ scale
- :class:`MarkerB1ReceiveInversion`    — Bloch signal / measured → B1-
- :class:`LarmorDriftTracker`          — spectral peak shift → temporal B0 drift
- :class:`SusceptibilityB0Solver`      — χ interface → Maxwell inverse → B0
- :class:`BlochSiegertB1Estimator`     — off-resonance phase → B1+ (B0-free)
- :class:`SSFPBandingB0Estimator`      — dark band spacing → ∇B0
- :class:`STEAMSpinEchoB1Estimator`    — STE/SE ratio → B1+ scale
- :class:`K0PhaseDriftTracker`         — k-space center phase → temporal B0 drift
- :class:`PhaseUnwrappingSeed`         — marker-seeded spatial phase unwrapping

Reference
---------
    Systemic Physical Invariance: synthetic markers as absolute field probes.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# 1. B0 from EPI Pixel Shift
# ======================================================================


class MarkerB0FromDisplacement(nn.Module):
    r"""Extract B0 field map from EPI pixel displacement of the marker.

    In EPI, B0 offsets cause pixel shift along the phase-encode (PE)
    direction:

    .. math::

        \Delta y_{\text{pix}} = \frac{\Delta\omega}{2\pi \cdot BW_{\text{PE}}}

    By measuring the marker's displacement from its known true coordinates,
    we directly obtain the local B0 offset in Hz.

    Args:
        true_positions: Known marker centre positions ``[N_markers, 2]`` (y, x).
        pe_bandwidth: Phase-encode bandwidth in Hz/pixel.
    """

    def __init__(
        self,
        true_positions: torch.Tensor,
        pe_bandwidth: float = 30.0,
    ) -> None:
        super().__init__()
        self.register_buffer("true_positions", true_positions.float())
        self.pe_bandwidth = pe_bandwidth

    def forward(
        self,
        marker_image: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate B0 map from marker displacement.

        Args:
            marker_image: Acquired marker region ``[B, 1, H, W]`` magnitude.
            marker_mask: Binary mask for each marker ``[N_markers, 1, H, W]``.

        Returns:
            B0 field map ``[B, 1, H, W]`` in Hz.

        forward method for MarkerB0FromDisplacement.

        Executes PyTorch tensor operations.

        Args:
            marker_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, _, H, W = marker_image.shape
        b0_map = torch.zeros(B, 1, H, W, device=marker_image.device)

        for i in range(marker_mask.shape[0]):
            mask_i = marker_mask[i : i + 1]  # [1, 1, H, W]
            # Centre of mass of the acquired marker
            weighted = marker_image * mask_i
            total = weighted.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)

            y_coords = torch.arange(H, device=marker_image.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=marker_image.device, dtype=torch.float32)
            y_grid = y_coords.view(1, 1, H, 1).expand_as(weighted)
            x_grid = x_coords.view(1, 1, 1, W).expand_as(weighted)

            com_y = (weighted * y_grid).sum(dim=(-2, -1), keepdim=True) / total
            com_x = (weighted * x_grid).sum(dim=(-2, -1), keepdim=True) / total

            # Displacement from true position
            true_y, true_x = self.true_positions[i]
            dy = com_y.squeeze() - true_y  # pixels
            # B0 = dy_pix * BW_PE
            local_b0 = dy * self.pe_bandwidth  # Hz

            # Paint this B0 value into the marker region
            b0_map = b0_map + mask_i * local_b0.view(B, 1, 1, 1)

        return b0_map


# ======================================================================
# 2. B0 from Dual-Echo Phase
# ======================================================================


class MarkerB0FromDualEcho(nn.Module):
    r"""Extract B0 from dual-echo phase difference within the marker.

    .. math::

        \Delta B_0(\mathbf{r}) = \frac{
            \angle(S_2(\mathbf{r}) \cdot S_1^*(\mathbf{r}))
        }{2\pi \cdot \Delta TE}

    .. note::

        ``torch.angle`` wraps the phase difference to :math:`(-\pi, \pi]`, so
        this single-difference estimate (no temporal unwrapping) only recovers
        :math:`|\Delta B_0| < 1/(2\,\Delta TE)`. For the default
        ``delta_te = 2.46 ms`` that aliasing limit is ``±203 Hz``; larger fields
        wrap. Use a third echo or a spatial unwrap (``PhaseUnwrappingSeed``) to
        extend the range.

    Args:
        delta_te: Echo time difference in seconds.
    """

    def __init__(self, delta_te: float = 0.00246) -> None:
        super().__init__()
        self.delta_te = delta_te

    def forward(
        self,
        echo1: torch.Tensor,
        echo2: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute B0 map from two echo acquisitions.

        Args:
            echo1: Complex image at TE1 ``[B, 1, H, W]``.
            echo2: Complex image at TE2 ``[B, 1, H, W]``.
            marker_mask: Optional mask to restrict to marker region.

        Returns:
            B0 map ``[B, 1, H, W]`` in Hz.

        forward method for MarkerB0FromDualEcho.

        Executes PyTorch tensor operations.

        Args:
            echo1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            echo2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Phase difference: angle(S2 · S1*)
        # Contract: echo1/echo2 are COMPLEX (docstring). Real inputs would make
        # .conj() a no-op and angle() collapse to 0/π, silently yielding a bogus
        # all-zero B0 map (indistinguishable from a true no-field result). Raise
        # instead of degrading silently (pitfall #9).
        from spectramr.infrastructure.physics.fft_ops import _to_complex

        echo1 = _to_complex(echo1, strict=True)
        echo2 = _to_complex(echo2, strict=True)
        phase_diff = torch.angle(echo2 * echo1.conj())

        # B0 = ΔΦ / (2π · ΔTE)
        b0_map = phase_diff / (2.0 * math.pi * self.delta_te)

        if marker_mask is not None:
            b0_map = b0_map * marker_mask

        return b0_map


# ======================================================================
# 3. B1+ Transmit from Double Angle Method
# ======================================================================


class MarkerB1TransmitDoubleAngle(nn.Module):
    r"""B1+ transmit field estimation via the T1-corrected double-angle method.

    Two spoiled-GRE acquisitions at nominal flip angles α and 2α share the
    steady-state signal
    :math:`S(\alpha)=M_0\sin\alpha\,(1-E_1)/(1-E_1\cos\alpha)` with
    :math:`E_1=e^{-TR/T_1}`. Their ratio inverts in closed form for the actual
    flip angle (using :math:`\cos 2\alpha = 2\cos^2\alpha-1`):

    .. math::

        R=\frac{S_{2\alpha}}{S_\alpha}
         =\frac{2\cos\alpha\,(1-E_1\cos\alpha)}{1-E_1\cos 2\alpha}
        \;\Longrightarrow\;
        \cos\alpha=\frac{1-\sqrt{1-a\,d}}{a},

    with :math:`a=2E_1(1-R)` and :math:`d=R(1+E_1)`. As :math:`E_1\to0`
    (long-TR / full recovery) this reduces to the uncorrected long-TR estimate
    :math:`\cos\alpha=R/2`; the known marker :math:`T_1` removes the T1 bias of
    the plain double-angle method. B1+ scale =
    :math:`\alpha_{\text{actual}}/\alpha_{\text{nominal}}`.

    Args:
        t1_marker: Known T1 of marker material (ms).
        tr: Repetition time (ms).
        nominal_alpha: Nominal flip angle (degrees).
    """

    def __init__(
        self,
        t1_marker: float = 300.0,
        tr: float = 50.0,
        nominal_alpha: float = 30.0,
    ) -> None:
        super().__init__()
        self.e1 = math.exp(-tr / t1_marker)
        self.nominal_alpha_rad = math.radians(nominal_alpha)

    def forward(
        self,
        signal_alpha: torch.Tensor,
        signal_2alpha: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute B1+ scale map.

        Args:
            signal_alpha: Magnitude at flip angle α ``[B, 1, H, W]``.
            signal_2alpha: Magnitude at flip angle 2α ``[B, 1, H, W]``.
            marker_mask: Optional spatial mask.

        Returns:
            B1+ scale map ``[B, 1, H, W]`` (1.0 = perfect calibration).

        forward method for MarkerB1TransmitDoubleAngle.

        Executes PyTorch tensor operations.

        Args:
            signal_alpha (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            signal_2alpha (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        eps = 1e-8
        e1 = self.e1
        # Measured double-angle ratio R = S_2α / S_α (spoiled-GRE steady state).
        ratio = signal_2alpha / (signal_alpha + eps)

        # Exact T1-corrected inversion: cos α solves the quadratic
        #   a·cos²α − 2·cos α + d = 0,  a = 2 E1 (1−R),  d = R (1+E1).
        # Physical root (→ R/2 as E1→0):  cos α = (1 − √(1 − a·d)) / a.
        a = 2.0 * e1 * (1.0 - ratio)
        d = ratio * (1.0 + e1)
        disc = (1.0 - a * d).clamp_min(0.0)
        # Guard the a→0 branch (E1→0 or R→1): fall back to the uncorrected R/2
        # so no division by zero enters the value or its gradient.
        a_safe = torch.where(a.abs() < eps, torch.ones_like(a), a)
        cos_alpha = torch.where(a.abs() < eps, 0.5 * ratio, (1.0 - torch.sqrt(disc)) / a_safe)

        # Clamp for numerical stability, then invert.
        cos_alpha = cos_alpha.clamp(-0.999, 0.999)
        actual_alpha = torch.acos(cos_alpha)
        b1_scale = actual_alpha / self.nominal_alpha_rad

        if marker_mask is not None:
            b1_scale = b1_scale * marker_mask

        return b1_scale


# ======================================================================
# 4. B1+ from AFI with Known T1
# ======================================================================


class MarkerB1TransmitAFI(nn.Module):
    r"""B1+ estimation from Actual Flip-Angle Imaging (AFI) with known T1.

    AFI uses two interleaved TRs:

    .. math::

        r = \frac{S_2}{S_1}, \quad
        \cos(\alpha) = \frac{r \cdot n - 1}{n - r}

    where :math:`n = TR_2 / TR_1`.  Normally T1-biased, but with known
    marker T1 we can apply an exact correction:

    .. math::

        n_{\text{eff}} = n \cdot \frac{1 - E_1^{(1)}}{1 - E_1^{(2)}}

    Args:
        t1_marker: Known T1 of marker (ms).
        tr1: First TR (ms).
        tr2: Second TR (ms).
    """

    def __init__(
        self,
        t1_marker: float = 300.0,
        tr1: float = 20.0,
        tr2: float = 100.0,
    ) -> None:
        super().__init__()
        e1_1 = math.exp(-tr1 / t1_marker)
        e1_2 = math.exp(-tr2 / t1_marker)
        # Corrected ratio using known T1
        self.n_eff = (tr2 / tr1) * (1.0 - e1_1) / (1.0 - e1_2)

    def forward(
        self,
        signal_1: torch.Tensor,
        signal_2: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute B1+ scale from AFI signals.

        Args:
            signal_1: Signal from TR1 ``[B, 1, H, W]``.
            signal_2: Signal from TR2 ``[B, 1, H, W]``.
            marker_mask: Optional spatial mask.

        Returns:
            B1+ scale ``[B, 1, H, W]``.

        forward method for MarkerB1TransmitAFI.

        Executes PyTorch tensor operations.

        Args:
            signal_1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            signal_2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        r = signal_2 / (signal_1 + 1e-8)
        n = self.n_eff

        # cos(α) = (r·n - 1) / (n - r)
        cos_alpha = (r * n - 1.0) / (n - r + 1e-8)
        cos_alpha = cos_alpha.clamp(-0.999, 0.999)
        alpha_actual = torch.acos(cos_alpha)

        # B1+ scale = actual / nominal (nominal from sequence design)
        # Here we return raw actual flip angle; user divides by nominal
        b1_scale = alpha_actual

        if marker_mask is not None:
            b1_scale = b1_scale * marker_mask

        return b1_scale


# ======================================================================
# 5. B1- Receive from Analytical Signal Inversion
# ======================================================================


class MarkerB1ReceiveInversion(nn.Module):
    r"""Extract receive coil sensitivity (B1-) via Bloch equation inversion.

    Because the marker's T1, T2, and proton density are known, the Bloch
    equation predicts an exact signal magnitude.  Dividing the raw measured
    signal by this theoretical value isolates B1-:

    .. math::

        B_1^-(\mathbf{r}) = \frac{S_{\text{measured}}(\mathbf{r})}
                                  {S_{\text{Bloch}}(T_1, T_2, \rho, \alpha, TR, TE)}

    Args:
        t1: Marker T1 (ms).
        t2: Marker T2 (ms).
        rho: Marker proton density (a.u.).
        tr: Repetition time (ms).
        te: Echo time (ms).
        flip_angle: Flip angle (degrees).
    """

    def __init__(
        self,
        t1: float = 300.0,
        t2: float = 50.0,
        rho: float = 1.0,
        tr: float = 50.0,
        te: float = 10.0,
        flip_angle: float = 15.0,
    ) -> None:
        super().__init__()
        alpha = math.radians(flip_angle)
        e1 = math.exp(-tr / t1)
        e2_star = math.exp(-te / t2)

        # GRE steady-state signal: S = ρ·sin(α)·(1-E1)/(1-cos(α)·E1)·E2*
        self.s_bloch = rho * math.sin(alpha) * (1 - e1) / (1 - math.cos(alpha) * e1) * e2_star

    def forward(
        self,
        measured_signal: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract B1- (receive sensitivity) map.

        Args:
            measured_signal: Raw magnitude image ``[B, C, H, W]``.
            marker_mask: Spatial mask for marker region.

        Returns:
            B1- sensitivity map ``[B, C, H, W]``.

        forward method for MarkerB1ReceiveInversion.

        Executes PyTorch tensor operations.

        Args:
            measured_signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b1_receive = measured_signal / (self.s_bloch + 1e-8)

        if marker_mask is not None:
            b1_receive = b1_receive * marker_mask

        return b1_receive


# ======================================================================
# 6. Larmor Drift Tracker
# ======================================================================


class LarmorDriftTracker(nn.Module):
    r"""Track temporal B0 drift via spectral peak shift of marker compound.

    If the marker contains a uniform chemical compound (e.g. silicone),
    its spectral peak at known frequency :math:`f_0` provides an absolute
    anchor.  Temporal shift of this peak tracks B0 drift:

    .. math::

        \Delta B_0(t) = f_{\text{measured}}(t) - f_0

    Args:
        reference_freq_hz: Known Larmor frequency of marker compound (Hz).
        spectral_bandwidth: Bandwidth of spectral analysis window (Hz).
    """

    def __init__(
        self,
        reference_freq_hz: float = 0.0,
        spectral_bandwidth: float = 500.0,
    ) -> None:
        super().__init__()
        self.reference_freq_hz = reference_freq_hz
        self.spectral_bandwidth = spectral_bandwidth

    def forward(
        self,
        time_series_signal: torch.Tensor,
        dt: float = 0.001,
    ) -> torch.Tensor:
        """Estimate B0 drift from temporal signal of marker voxel.

        Args:
            time_series_signal: Complex signal ``[N_timepoints]`` or ``[B, N]``.
            dt: Temporal sampling interval (seconds).

        Returns:
            B0 drift ``[N_timepoints]`` in Hz.

        forward method for LarmorDriftTracker.

        Executes PyTorch tensor operations.

        Args:
            time_series_signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            dt (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if time_series_signal.ndim == 1:
            time_series_signal = time_series_signal.unsqueeze(0)

        # FFT along time dimension
        spectrum = torch.fft.fft(time_series_signal, dim=-1)
        N = spectrum.shape[-1]
        freqs = torch.fft.fftfreq(N, d=dt, device=spectrum.device)

        # Find peak frequency
        magnitudes = spectrum.abs()
        peak_idx = magnitudes.argmax(dim=-1)
        peak_freq = freqs[peak_idx]

        # Drift = measured - reference
        drift = peak_freq - self.reference_freq_hz

        return drift


# ======================================================================
# 7. Susceptibility-Based B0 Solver
# ======================================================================


class SusceptibilityB0Solver(nn.Module):
    r"""Estimate B0 from known susceptibility interface at marker boundary.

    The marker-air boundary creates a known magnetic susceptibility
    difference (:math:`\Delta\chi`).  The resulting B0 perturbation
    satisfies the dipole kernel convolution:

    .. math::

        \Delta B_0(\mathbf{r}) = B_0 \cdot d(\mathbf{r}) * \chi(\mathbf{r})

    where :math:`d(\mathbf{k}) = 1/3 - k_z^2/|\mathbf{k}|^2` is the dipole
    kernel.  With known :math:`\chi` of the marker we can reverse-solve for
    the background B0.

    Args:
        b0_tesla: Main field strength (Tesla).
    """

    def __init__(
        self,
        b0_tesla: float = 0.064,
    ) -> None:
        super().__init__()
        self.b0 = b0_tesla

    def _build_dipole_kernel_2d(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Build the 2D in-plane dipole-kernel surrogate in k-space.

        .. warning::

            This ``d(k) = 1/2 - k_y^2/|k|^2`` is a **non-physical 2-D surrogate**,
            NOT a projection of the true 3-D dipole kernel
            ``d(k) = 1/3 - k_z^2/|k|^2`` (the in-plane projection of the dipole
            field has no closed 2-D form). The recovered B0 is a heuristic; for a
            quantitatively correct inversion operate on a 3-D volume with the
            real kernel.

        Args:
            H: Height.
            W: Width.
            device: Device.

        Returns:
            Dipole kernel ``[1, 1, H, W]``.
        """
        ky = torch.fft.fftfreq(H, device=device).unsqueeze(1)
        kx = torch.fft.fftfreq(W, device=device).unsqueeze(0)
        k_sq = kx**2 + ky**2 + 1e-10
        # 2-D surrogate (see the docstring warning): d(k) = 1/2 - ky^2/|k|^2.
        dipole = 0.5 - ky**2 / k_sq
        return dipole.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        chi_map: torch.Tensor,
    ) -> torch.Tensor:
        """Compute B0 perturbation from susceptibility distribution.

        Args:
            chi_map: Susceptibility map ``[B, 1, H, W]`` in ppm.

        Returns:
            B0 perturbation ``[B, 1, H, W]`` in Hz.

        forward method for SusceptibilityB0Solver.

        Executes PyTorch tensor operations.

        Args:
            chi_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = chi_map.shape
        dipole = self._build_dipole_kernel_2d(H, W, chi_map.device)

        # Forward model: ΔB0 = B0 · IFFT(D · FFT(χ))
        chi_k = torch.fft.fft2(chi_map * 1e-6)  # ppm → dimensionless
        b0_perturbation_k = dipole * chi_k
        b0_perturbation = torch.fft.ifft2(b0_perturbation_k).real

        # Convert from field (T) to frequency (Hz): γ·ΔB / 2π
        gamma_bar = 42.576e6  # Hz/T
        b0_hz = b0_perturbation * self.b0 * gamma_bar

        return b0_hz


# ======================================================================
# 8. Bloch-Siegert Phase Shift (B1+)
# ======================================================================


class BlochSiegertB1Estimator(nn.Module):
    r"""B1+ estimation via Bloch-Siegert shift (B0-independent).

    An off-resonance RF pulse of frequency :math:`\omega_{\text{off}}`
    induces a phase shift proportional to :math:`|B_1^+|^2`:

    .. math::

        \phi_{\text{BS}} = \frac{|\omega_1|^2}{2 \omega_{\text{off}}} \cdot \tau

    where :math:`\omega_1 = \gamma B_1^+` and :math:`\tau` is the pulse
    duration.  Two acquisitions with :math:`\pm\omega_{\text{off}}` cancel
    B0 contributions, isolating B1+.

    Args:
        omega_off_hz: Off-resonance frequency of the BS pulse (Hz).
        pulse_duration_s: Duration of the off-resonance pulse (seconds).
    """

    def __init__(
        self,
        omega_off_hz: float = 4000.0,
        pulse_duration_s: float = 0.008,
    ) -> None:
        super().__init__()
        self.omega_off = omega_off_hz * 2.0 * math.pi  # rad/s
        self.tau = pulse_duration_s

    def forward(
        self,
        signal_pos: torch.Tensor,
        signal_neg: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute B1+ from Bloch-Siegert phase difference.

        Args:
            signal_pos: Complex image with :math:`+\\omega_{\text{off}}` ``[B, 1, H, W]``.
            signal_neg: Complex image with :math:`-\\omega_{\text{off}}` ``[B, 1, H, W]``.
            marker_mask: Optional spatial mask.

        Returns:
            B1+ scale map ``[B, 1, H, W]`` (relative units).

        forward method for BlochSiegertB1Estimator.

        Executes PyTorch tensor operations.

        Args:
            signal_pos (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            signal_neg (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Phase difference cancels B0: Δφ = φ_+ - φ_- = 2·φ_BS
        phase_diff = torch.angle(signal_pos * signal_neg.conj())
        phi_bs = phase_diff / 2.0  # Single BS shift

        # |ω1|² = 2·ω_off·φ_BS / τ
        omega1_sq = (2.0 * abs(self.omega_off) * phi_bs.abs()) / self.tau

        # B1+ scale: ω1 = γ·B1+ → relative scale ∝ √(|ω1|²)
        b1_scale = torch.sqrt(omega1_sq + 1e-10)

        # Normalise so marker centre ≈ 1.0
        if marker_mask is not None:
            mask_sum = marker_mask.sum().clamp(min=1)
            mean_b1 = (b1_scale * marker_mask).sum() / mask_sum
            b1_scale = b1_scale / mean_b1.clamp(min=1e-6)
            b1_scale = b1_scale * marker_mask

        return b1_scale


# ======================================================================
# 9. SSFP Banding B0 Estimator
# ======================================================================


class SSFPBandingB0Estimator(nn.Module):
    r"""Estimate B0 gradient from bSSFP banding artefact spacing.

    Balanced SSFP exhibits dark bands at off-resonance frequencies:

    .. math::

        f_{\text{null}} = \frac{n}{TR}, \quad n = \pm 1, \pm 2, \ldots

    The physical spacing of these bands across the known marker geometry
    acts as a ruler for the spatial B0 gradient:

    .. math::

        \frac{\partial B_0}{\partial r} = \frac{1}{TR \cdot \Delta r_{\text{bands}}}

    HONESTY / SCOPE: this is a **coarse proxy**, not a quantitative field fit. It
    counts hard threshold crossings (a fixed 0.3·mean cut) of a width-averaged
    marker profile and returns a single scalar gradient broadcast over the whole
    marker — it is NOT spatially resolved and is sensitive to the threshold,
    noise, and partial bands. Use it as a rough banding-severity indicator, not
    as a calibrated Hz/pixel measurement. (Fixed 2026-07: divide the transition
    count by 2 to get the band count, and normalise by the marker extent along
    PE rather than the full image height.)

    Args:
        tr_s: Repetition time (seconds).
    """

    def __init__(self, tr_s: float = 0.005) -> None:
        super().__init__()
        self.tr_s = tr_s

    def forward(
        self,
        ssfp_image: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate B0 gradient from banding spacing within marker.

        Args:
            ssfp_image: Magnitude bSSFP image ``[B, 1, H, W]``.
            marker_mask: Marker region mask ``[1, 1, H, W]``.

        Returns:
            B0 gradient estimate ``[B, 1, H, W]`` in Hz/pixel.

        forward method for SSFPBandingB0Estimator.

        Executes PyTorch tensor operations.

        Args:
            ssfp_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, _, H, W = ssfp_image.shape

        # Extract marker profile along PE (vertical) direction
        masked = ssfp_image * marker_mask

        # Find dark bands via signal minima along each column
        # Average across width for robust estimate
        profile = masked.sum(dim=-1) / (marker_mask.sum(dim=-1) + 1e-6)  # [B, 1, H]

        # Detect minima: where signal dips below threshold.
        # NOTE: 0.3 is a hardcoded, uncalibrated threshold — this whole estimator
        # is a COARSE proxy (see the class docstring), not a quantitative fit.
        threshold = profile.mean(dim=-1, keepdim=True) * 0.3
        is_band = (profile < threshold).float()

        # Count dark bands, then convert to a B0 gradient. A bSSFP dark band
        # recurs every Δf = 1/TR in off-resonance, so N bands span ≈ N/TR Hz.
        # Each band contributes TWO threshold crossings (entering + leaving), so
        # n_bands = transitions / 2 (a prior version used the raw transition
        # count, ~2× the band count). The frequency range spreads over the
        # MARKER extent along PE, not the full image height H (a prior version
        # divided by H, diluting the gradient for small markers).
        band_diff = is_band[:, :, 1:] - is_band[:, :, :-1]
        transitions = band_diff.abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        n_bands = transitions / 2.0
        # Marker extent in rows: number of PE rows the marker actually covers.
        marker_extent = (
            (marker_mask.sum(dim=-1) > 0).float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        )  # [1, 1, 1]
        b0_grad = n_bands / (self.tr_s * marker_extent)  # Hz/pixel (coarse)

        return b0_grad.unsqueeze(-1).expand(B, 1, H, W) * marker_mask


# ======================================================================
# 10. STEAM vs Spin Echo B1+ Estimator
# ======================================================================


class STEAMSpinEchoB1Estimator(nn.Module):
    r"""B1+ estimation from stimulated echo / spin echo amplitude ratio.

    The ratio of a stimulated echo (STEAM) to a spin echo (SE) is
    sensitive to the actual flip angle:

    .. math::

        \frac{S_{\text{STE}}}{S_{\text{SE}}} =
            \frac{\sin(\alpha) \cdot \sin^2(\alpha/2)}{
                  \sin(\alpha)} =
            \sin^2(\alpha/2)

    For a nominal 90° pulse, :math:`S_{\text{STE}}/S_{\text{SE}} = 0.5`.
    Deviations directly measure B1+ scaling, free of B0 phase wraps.

    Args:
        nominal_flip_deg: Nominal flip angle (degrees).
    """

    def __init__(self, nominal_flip_deg: float = 90.0) -> None:
        super().__init__()
        self.nominal_alpha = math.radians(nominal_flip_deg)

    def forward(
        self,
        signal_steam: torch.Tensor,
        signal_se: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute B1+ scale from STEAM/SE ratio.

        Args:
            signal_steam: Stimulated echo magnitude ``[B, 1, H, W]``.
            signal_se: Spin echo magnitude ``[B, 1, H, W]``.
            marker_mask: Optional spatial mask.

        Returns:
            B1+ scale map ``[B, 1, H, W]`` (1.0 = perfect calibration).

        forward method for STEAMSpinEchoB1Estimator.

        Executes PyTorch tensor operations.

        Args:
            signal_steam (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            signal_se (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        ratio = signal_steam / (signal_se + 1e-8)
        ratio = ratio.clamp(0.0, 1.0)

        # ratio = sin²(α_actual/2) → α_actual = 2·arcsin(√ratio)
        alpha_actual = 2.0 * torch.asin(torch.sqrt(ratio))

        b1_scale = alpha_actual / self.nominal_alpha

        if marker_mask is not None:
            b1_scale = b1_scale * marker_mask

        return b1_scale


# ======================================================================
# 11. K0 Phase-Drift Tracker (Temporal B0 via k-space centre)
# ======================================================================


class K0PhaseDriftTracker(nn.Module):
    r"""Track temporal B0 drift from k-space centre phase of synthetic marker.

    In the synthetic forward model, time-varying B0 drift is added as a
    global phase scalar.  The k-space centre point :math:`k_0` of the
    marker captures this drift directly:

    .. math::

        \Delta B_0(t) = \frac{\angle(k_0(t)) - \angle(k_0(0))}
                             {2\pi \cdot TE}

    Unlike :class:`LarmorDriftTracker` (which uses spectral peak shift),
    this method operates directly in k-space without Fourier analysis,
    making it faster and more robust for simulated data.

    Args:
        te: Echo time in seconds.
    """

    def __init__(self, te: float = 0.01) -> None:
        super().__init__()
        self.te = te

    def forward(
        self,
        kspace_timeseries: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract B0 drift from k-space centre phase over time.

        Args:
            kspace_timeseries: Complex k-space ``[T, 1, H, W]`` per timepoint.
            marker_mask: Optional spatial mask ``[1, 1, H, W]`` — if provided,
                uses spatially-masked k-space centre instead of global.

        Returns:
            B0 drift ``[T]`` in Hz relative to first timepoint.

        forward method for K0PhaseDriftTracker.

        Executes PyTorch tensor operations.

        Args:
            kspace_timeseries (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        T = kspace_timeseries.shape[0]

        if marker_mask is not None:
            # Mask in image space → restricted k-space centre
            images = torch.fft.ifft2(kspace_timeseries, dim=(-2, -1))
            masked = images * marker_mask
            k0 = torch.fft.fft2(masked, dim=(-2, -1))[:, :, 0, 0]  # DC component
        else:
            # Direct k-space centre
            H, W = kspace_timeseries.shape[-2:]
            k0 = kspace_timeseries[:, :, H // 2, W // 2]

        # Phase evolution relative to t=0
        phase_ref = torch.angle(k0[0:1])
        phase_all = torch.angle(k0)
        delta_phase = phase_all - phase_ref  # [T, 1]

        # B0 = Δφ / (2π·TE)
        b0_drift = delta_phase.squeeze() / (2.0 * math.pi * self.te)

        return b0_drift


# ======================================================================
# 12. Phase Unwrapping Seed (Marker-Guided)
# ======================================================================


class PhaseUnwrappingSeed(nn.Module):
    r"""Use synthetic marker as trusted seed for spatial phase unwrapping.

    When simulating severe B0 fields, phase values wrap beyond
    :math:`[-\pi, \pi]`.  The synthetic marker has known continuous
    boundaries, providing an absolute phase reference to seed the
    unwrapping algorithm:

    .. math::

        \phi_{\text{unwrapped}}(\mathbf{r}) =
            \phi_{\text{wrapped}}(\mathbf{r}) + 2\pi n(\mathbf{r})

    where :math:`n(\mathbf{r})` is determined by flood-filling outward
    from the marker seed region.

    The marker's known absolute B0 value provides the correct
    :math:`2\pi` count at its location.

    Args:
        known_b0_at_marker: Expected B0 value at the marker centre (Hz).
        te: Echo time in seconds.
    """

    def __init__(
        self,
        known_b0_at_marker: float = 0.0,
        te: float = 0.01,
    ) -> None:
        super().__init__()
        self.known_b0 = known_b0_at_marker
        self.te = te

    def forward(
        self,
        wrapped_phase: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Unwrap phase using marker as absolute seed.

        Args:
            wrapped_phase: Wrapped phase map ``[B, 1, H, W]`` in radians.
            marker_mask: Binary mask for marker region ``[1, 1, H, W]``.

        Returns:
            Unwrapped phase ``[B, 1, H, W]`` in radians.

        forward method for PhaseUnwrappingSeed.

        Executes PyTorch tensor operations.

        Args:
            wrapped_phase (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = wrapped_phase.shape

        # True phase at marker location from known B0
        true_marker_phase = 2.0 * math.pi * self.known_b0 * self.te

        # Measured phase at marker
        marker_phase = (wrapped_phase * marker_mask).sum(dim=(-2, -1))
        marker_area = marker_mask.sum().clamp(min=1)
        mean_marker_phase = marker_phase / marker_area  # [B, 1]

        # How many 2π wraps offset?
        delta = true_marker_phase - mean_marker_phase.view(B, 1, 1, 1)
        n_wraps = torch.round(delta / (2.0 * math.pi))
        global_offset = n_wraps * 2.0 * math.pi

        # Apply global correction (simple model — assumes single wrap region)
        unwrapped = wrapped_phase + global_offset

        # Iterative local unwrapping via neighbour consistency
        # For each pixel, compare to neighbours; if diff > π, add 2π
        for _ in range(3):  # 3 passes for convergence
            padded = torch.nn.functional.pad(unwrapped, (1, 1, 1, 1), mode="replicate")
            neighbours = torch.stack(
                [
                    padded[:, :, :-2, 1:-1],  # up
                    padded[:, :, 2:, 1:-1],  # down
                    padded[:, :, 1:-1, :-2],  # left
                    padded[:, :, 1:-1, 2:],  # right
                ],
                dim=0,
            )
            mean_neighbour = neighbours.mean(dim=0)

            diff = unwrapped - mean_neighbour
            unwrapped = unwrapped - torch.round(diff / (2.0 * math.pi)) * 2.0 * math.pi

        return unwrapped


__all__ = [
    "BlochSiegertB1Estimator",
    "K0PhaseDriftTracker",
    "LarmorDriftTracker",
    "MarkerB0FromDisplacement",
    "MarkerB0FromDualEcho",
    "MarkerB1ReceiveInversion",
    "MarkerB1TransmitAFI",
    "MarkerB1TransmitDoubleAngle",
    "PhaseUnwrappingSeed",
    "SSFPBandingB0Estimator",
    "STEAMSpinEchoB1Estimator",
    "SusceptibilityB0Solver",
]
