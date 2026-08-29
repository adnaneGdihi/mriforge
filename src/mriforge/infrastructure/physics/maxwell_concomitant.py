"""Maxwell concomitant gradient term for ULF MRI.

Maxwell's equations require that any imaging gradient :math:`G`
applied along one axis induces *concomitant* (transverse) gradient
components that are quadratic in spatial coordinates. The
resulting phase accrual

.. math::

    \\phi_{\\text{Maxwell}}(x, y, z) \\approx
        \\frac{(G_x^2 + G_y^2)\\, z^2 + \\tfrac{1}{4}\\, G_z^2 (x^2 + y^2)}
              {2\\, B_0}

is geometrically negligible above ~1.5 T but dominant below
~0.1 T (Hyperfine 64 mT, M4Raw 0.3 T). At a clinical readout it
manifests as off-resonance — i.e. an effective frequency offset
:math:`\\Delta f_\\text{Mx}(x, y, z) = \\gamma \\cdot
\\phi_\\text{Mx}(x, y, z) / (2 \\pi \\cdot t_\\text{ESP})` — that
warps the EPI image along the phase-encode axis exactly as a B0
inhomogeneity does. Composing it with the B0 map and feeding the
sum into the same grid-sample-based unwarp is the simplest
correct treatment.

Reference: Bernstein, M. A., Zhou, X. J., Polzin, J. A., King,
K. F., Ganin, A., Pelc, N. J., & Glover, G. H. (1998).
"Concomitant gradient terms in phase-contrast MR: analysis and
correction." *Magn. Reson. Med.* 39, 300–308.
"""

from __future__ import annotations

import torch

# Gyromagnetic ratio of protons in radians/s/T.
GAMMA_BAR_PROTON_RAD_S_T = 2.675_222_005e8


def maxwell_concomitant_field_hz(
    spatial_shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    gradient_amplitudes_mT_per_m: tuple[float, float, float],
    field_strength_t: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute the Maxwell concomitant frequency-offset map in Hz.

    Args:
        spatial_shape: ``(D, H, W)`` of the volume.
        voxel_size_mm: physical voxel size along ``(D, H, W)`` in
            millimeters. Coordinates are taken about the volume
            centre (isocenter assumed at the centre of the FOV).
        gradient_amplitudes_mT_per_m: amplitudes
            :math:`(G_x, G_y, G_z)` of the imaging gradients during
            the readout, in **mT/m** (typical clinical EPI is
            10-30 mT/m; ULF is often 1-5 mT/m).
        field_strength_t: static field :math:`B_0` in Tesla.
        device, dtype: where / how to allocate the output tensor.

    Returns:
        ``(D, H, W)`` real tensor — the per-voxel concomitant
        frequency offset in Hz. Sign convention matches the
        Bernstein paper.

    Notes:
        At high field this returns values near zero; at ULF
        (``B_0`` ≲ 0.1 T) and large FOVs it can reach hundreds of
        Hz at the edges of the volume.
    """
    if field_strength_t <= 0:
        raise ValueError(f"field_strength_t must be > 0 T, got {field_strength_t}")

    D, H, W = spatial_shape
    dz_mm, dy_mm, dx_mm = voxel_size_mm
    Gx_mT_m, Gy_mT_m, Gz_mT_m = gradient_amplitudes_mT_per_m

    # Convert mT/m → T/m for SI consistency.
    Gx = Gx_mT_m * 1e-3
    Gy = Gy_mT_m * 1e-3
    Gz = Gz_mT_m * 1e-3

    # Spatial coordinate grids in metres, centred on the FOV.
    # Convention: z is the slice axis (D), y is the row axis (H),
    # x is the column axis (W). This matches the data-layer
    # ``(B, C, D, H, W)`` ordering documented in CLAUDE.md.
    z = (torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2.0) * (dz_mm * 1e-3)
    y = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0) * (dy_mm * 1e-3)
    x = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0) * (dx_mm * 1e-3)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # Concomitant magnetic-field magnitude (Bernstein eq. 5):
    #   B_c ≈ ((G_x^2 + G_y^2) z^2 + 1/4 G_z^2 (x^2 + y^2)) / (2 B_0)
    B_c = ((Gx**2 + Gy**2) * zz.pow(2) + 0.25 * Gz**2 * (xx.pow(2) + yy.pow(2))) / (
        2.0 * field_strength_t
    )

    # Convert magnetic field (T) to Larmor frequency offset (Hz).
    # f = gamma_bar * B / (2 pi); gamma_bar is angular here, so divide by 2 pi.
    delta_f_hz = GAMMA_BAR_PROTON_RAD_S_T * B_c / (2.0 * torch.pi)
    return delta_f_hz


__all__ = ["GAMMA_BAR_PROTON_RAD_S_T", "maxwell_concomitant_field_hz"]
