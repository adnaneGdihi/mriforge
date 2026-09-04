"""B0 Field Utilities.

Utilities for applying B0 field effects to MRI data.
"""

import torch


def apply_b0_phase_shift(
    signal: torch.Tensor,
    b0_map: torch.Tensor,
    time_vec: torch.Tensor | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply B0-induced phase accumulation to k-space signal.

    Simulates phase error: exp(-i * 2π * ΔΩ * t)

    .. note::

        This is a **mean-field** model: ``ΔΩ`` is the *spatial mean* of
        ``b0_map``, so the applied phase is a global ramp that depends only on
        the field's mean, NOT its spatial structure. Two different B0 maps with
        the same mean produce identical corruption. A spatially-resolved
        off-resonance operator is NUFFT-style and lives elsewhere; do not use
        this to certify robustness to *spatial* B0 inhomogeneity.

    Args:
        signal: K-space signal [B, 2, N] (real/imag)
        b0_map: B0 map [B, 1, H, W]
        time_vec: Time vector for each k-space point [B, N] (optional)
        scale: Scaling factor for phase shift

    Returns:
        Signal with phase shift applied
    """
    if time_vec is None:
        return signal

    # Per-k-point phase accumulation: φ(n) = 2π · ΔΩ · t(n). We use the
    # spatial mean of ``b0_map`` as ΔΩ (a proper per-k-point sample would
    # be NUFFT-style and lives elsewhere). The previous heuristic that
    # baked in a ``0.01`` factor and ignored ``time_vec`` collapsed to
    # exactly one full rotation (identity) when ``b0_mean × scale = 100``
    # — see ``tests/unit/infrastructure/physics/test_b0_utils.py::
    # test_nonzero_b0_rotates_signal`` for the regression.
    B, _, N = signal.shape

    b0_mean = b0_map.mean(dim=(2, 3))  # [B, 1]
    b0_mean = b0_mean.squeeze(-1)  # [B]

    # time_vec may be [N] (shared across batch) or [B, N].
    if time_vec.dim() == 1:
        time_vec = time_vec.unsqueeze(0).expand(B, -1)
    elif time_vec.dim() == 2 and time_vec.shape[0] != B:
        time_vec = time_vec.expand(B, -1)

    phase = (2.0 * torch.pi) * b0_mean.unsqueeze(-1) * time_vec * scale  # [B, N]

    if signal.shape[1] == 2:  # [B, 2, N]
        cos_p = torch.cos(phase).unsqueeze(1)  # [B, 1, N]
        sin_p = torch.sin(phase).unsqueeze(1)  # [B, 1, N]

        real = signal[:, 0:1] * cos_p - signal[:, 1:2] * sin_p
        imag = signal[:, 0:1] * sin_p + signal[:, 1:2] * cos_p

        return torch.cat([real, imag], dim=1)

    return signal
