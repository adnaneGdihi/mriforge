"""
Differentiable SPGR Bloch Equation Layer.

This module implements the forward MRI physics model for quantitative mapping.
Given tissue parameters (T1, T2, PD) and sequence parameters (TR, TE, flip angle),
it predicts the steady-state signal magnitude that would be acquired.

Phase-0 extension (PMPS, 2026-05-19): the forward pass now accepts an
``include_concomitant`` flag together with ``gradient_amplitudes_mT_per_m``,
``voxel_size_mm`` and ``field_strength_t``. When the flag is on the
Maxwell concomitant frequency offset from
:func:`maxwell_concomitant_field_hz` is folded into the off-resonance
phase. The correction scales as :math:`1/B_0` and is negligible above
~1.5 T but dominant below ~0.1 T (e.g. Hyperfine 64 mT).
"""

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.maxwell_concomitant import (
    maxwell_concomitant_field_hz,
)


def _resolve_include_concomitant(
    include_concomitant: bool | None,
    field_strength_t: float | None,
    threshold_t: float = 0.5,
) -> bool:
    """Resolve the effective ``include_concomitant`` flag.

    Auto-set to ``True`` whenever the field strength is below
    ``threshold_t`` (0.5 T by default — Hyperfine 64 mT and 0.3 T
    M4Raw fall under this). Explicit user setting wins.
    """
    if include_concomitant is not None:
        return bool(include_concomitant)
    if field_strength_t is None:
        return False
    return field_strength_t < threshold_t


class DifferentiableBlochLayer(nn.Module):
    """
    Differentiable Forward Model for SPGR (Spoiled Gradient Recalled Echo) Sequence.

    Maps quantitative tissue parameters (T1, T2, PD) to MRI signal intensity
    for a given SPGR sequence configuration.

    .. math::

        S(\alpha, TR, TE) = PD \\cdot \\sin(\alpha) \\cdot \frac{1 - e^{-TR/T1}}{1 - \\cos(\alpha)e^{-TR/T1}} \\cdot e^{-TE/T2^*}

    .. mermaid::

        sequenceDiagram
            participant Tissue as Tissue Maps (T1, T2, PD)
            participant Sequence as Sequence Params (TR, TE, alpha)
            participant Field as Field Maps (B0, B1)
            participant Kernel as SPGR Kernel
            participant Signal as MRI Signal (Complex/Mag)

            Sequence->>Kernel: TR, TE, alpha_nominal
            Field->>Kernel: B1 maps (scaling alpha)
            Kernel->>Kernel: alpha_eff = alpha * B1
            Tissue->>Kernel: Local T1, T2*, PD
            Kernel->>Signal: Compute Steady-State Magnitude
            Field->>Signal: Apply B0 Off-Resonance Phase
            Signal-->>Signal: S_complex = S_mag * exp(i * 2pi * f_B0 * TE)
    """

    def __init__(self, sequence_type: str = "SPGR"):
        """
        Args:
            sequence_type: MRI pulse sequence type. Currently 'SPGR' is fully implemented.
        """
        super().__init__()
        self.sequence_type = sequence_type

        if sequence_type != "SPGR":
            raise NotImplementedError(
                f"Sequence type '{sequence_type}' not yet implemented. Currently supporting: SPGR"
            )

    def forward(
        self,
        t1_map: torch.Tensor,
        t2_map: torch.Tensor,
        pd_map: torch.Tensor,
        tr: float,
        te: float,
        flip_angle_rad: torch.Tensor | float,
        b0_map: torch.Tensor | None = None,
        b1_map: torch.Tensor | None = None,
        *,
        include_concomitant: bool | None = None,
        field_strength_t: float | None = None,
        gradient_amplitudes_mT_per_m: tuple[float, float, float] | None = None,
        voxel_size_mm: tuple[float, float, float] | None = None,
        concomitant_auto_threshold_t: float = 0.5,
    ) -> torch.Tensor:
        """
        Simulate steady-state SPGR signal with optional phase and B1 correction.

        Args:
            t1_map (Tensor): Longitudinal relaxation time [B, 1, H, W] in milliseconds.
                             Range: [10, 3000] ms (typical tissue at 3T).
            t2_map (Tensor): Transverse relaxation time [B, 1, H, W] in milliseconds.
                             Range: [1, 500] ms (typical tissue at 3T).
            pd_map (Tensor): Proton density [B, 1, H, W], normalized [0, 1] or scaled.
            tr (float): Repetition time in milliseconds. Typical: 5-20 ms for fast SPGR.
            te (float): Echo time in milliseconds. Typical: 1-5 ms for SPGR.
            flip_angle_rad (float | Tensor): Flip angle in radians.
                                             Scalar or [B, ...] matching map shapes.
            b0_map (Tensor, optional): B0 field inhomogeneity [B, 1, H, W] in Hz.
                                       Range: [-200, 200] Hz (typical at 3T).
                                       If provided, output is complex. If None, output is real.
            b1_map (Tensor, optional): B1 transmit field scaling [B, 1, H, W].
                                       Range: [0.5, 1.5]. Default 1.0 (nominal flip).

        Returns:
            signal (Tensor): Simulated signal [B, 1, H, W].
                            - If b0_map is None: real-valued magnitude
                            - If b0_map provided: complex-valued signal with phase

        Physics:
            SPGR Magnitude:
                S_mag = M0 * sin(α_eff) * (1 - E1) / (1 - cos(α_eff) * E1) * E2

            B0 Phase (off-resonance):
                φ = 2π * f_B0 * TE
                S_complex = S_mag * exp(i * φ)

            B1 Correction:
                α_eff = α_nominal * B1  (effective flip angle)
        """
        eps = 1e-8

        # Ensure T1 and T2 are strictly positive via softplus
        t1_safe = torch.nn.functional.softplus(t1_map) + eps
        t2_safe = torch.nn.functional.softplus(t2_map) + eps

        # Compute relaxation rates
        R1 = 1.0 / t1_safe
        R2_star = 1.0 / t2_safe

        # Precompute decay factors
        E1 = torch.exp(-tr * R1)
        E2 = torch.exp(-te * R2_star)

        # Handle flip angle
        if isinstance(flip_angle_rad, (int, float)):
            flip_angle_rad = torch.tensor(flip_angle_rad, device=t1_map.device, dtype=t1_map.dtype)

        # Apply B1 correction if provided
        if b1_map is not None:
            effective_flip_angle = flip_angle_rad * b1_map
        else:
            effective_flip_angle = flip_angle_rad

        sin_alpha = torch.sin(effective_flip_angle)
        cos_alpha = torch.cos(effective_flip_angle)

        # SPGR equation: S = M0 * sin(α_eff) * (1 - E1) / (1 - cos(α_eff) * E1) * E2
        numerator = pd_map * sin_alpha * (1.0 - E1)
        denominator = 1.0 - cos_alpha * E1 + eps

        magnitude = (numerator / denominator) * E2
        magnitude = torch.clamp(magnitude, min=0.0)

        # Resolve concomitant flag (auto-on below 0.5 T unless user pinned).
        use_concomitant = _resolve_include_concomitant(
            include_concomitant,
            field_strength_t,
            threshold_t=concomitant_auto_threshold_t,
        )

        concomitant_hz: torch.Tensor | None = None
        if use_concomitant:
            if field_strength_t is None:
                raise ValueError(
                    "include_concomitant requires field_strength_t. "
                    "Pass field_strength_t (Tesla) when enabling the "
                    "Maxwell concomitant correction."
                )
            if gradient_amplitudes_mT_per_m is None or voxel_size_mm is None:
                raise ValueError(
                    "include_concomitant requires gradient_amplitudes_mT_per_m "
                    "and voxel_size_mm. The concomitant phase scales with "
                    "G^2 * r^2 / B0 (Bernstein 1998) and cannot be evaluated "
                    "without gradient amplitudes and spatial coordinates."
                )
            # Derive spatial shape from the trailing dims of t1_map.
            spatial_dims = t1_map.shape[-3:]
            if len(spatial_dims) != 3:
                # 2D input — pad with depth=1 so the concomitant grid is 2D.
                spatial_dims = (1,) + tuple(t1_map.shape[-2:])
            concomitant_hz = maxwell_concomitant_field_hz(
                spatial_shape=tuple(spatial_dims),
                voxel_size_mm=voxel_size_mm,
                gradient_amplitudes_mT_per_m=gradient_amplitudes_mT_per_m,
                field_strength_t=field_strength_t,
                device=t1_map.device,
                dtype=t1_map.dtype,
            )
            # Broadcast (D, H, W) → (1, 1, D, H, W) / (1, 1, H, W).
            if t1_map.ndim == 4:
                concomitant_hz = concomitant_hz[0]  # drop singleton depth
                concomitant_hz = concomitant_hz.unsqueeze(0).unsqueeze(0)
            elif t1_map.ndim == 5:
                concomitant_hz = concomitant_hz.unsqueeze(0).unsqueeze(0)

        # Handle B0 phase (off-resonance), optionally with concomitant term.
        if b0_map is not None or concomitant_hz is not None:
            effective_offset_hz: torch.Tensor
            if b0_map is not None and concomitant_hz is not None:
                effective_offset_hz = b0_map + concomitant_hz
            elif b0_map is not None:
                effective_offset_hz = b0_map
            else:
                assert concomitant_hz is not None  # for type checker
                effective_offset_hz = concomitant_hz

            # Phase accumulation: φ = 2π * f_offset * TE (TE ms → s).
            phase_rad = 2.0 * torch.pi * effective_offset_hz * (te * 1e-3)

            # Complex signal: S = |S| * e^(i*φ)
            real_part = magnitude * torch.cos(phase_rad)
            imag_part = magnitude * torch.sin(phase_rad)

            signal = torch.complex(real_part, imag_part)
        else:
            # Real-valued magnitude only (no phase model)
            signal = magnitude

        return signal


# Convenience wrapper for backward compatibility
class SPGRBlochSimulator(DifferentiableBlochLayer):
    """Alias for SPGR-specific simulation.
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{SPGR}(\theta) = PD \frac{\\sin(\alpha)(1 - e^{-TR/T_1})}{1 - \\cos(\alpha)e^{-TR/T_1}} e^{-TE/T_2^*}"""

    def __init__(self):
        """__init__."""
        super().__init__(sequence_type="SPGR")
