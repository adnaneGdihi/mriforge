"""Motion Simulation for MRI.

This module provides tools for simulating physiological motion, including
rigid body motion and elastic respiratory deformations with hysteresis.
"""

import torch
import torch.nn.functional as F


def simulate_elastic_motion_ffd(
    volume: torch.Tensor,
    time_points: torch.Tensor,
    cycle_period: float = 4.0,
    max_displacement: float = 5.0,
    grid_spacing: int = 16,
    seed: int | None = None,
) -> torch.Tensor:
    """Simulates elastic soft-tissue motion using B-Spline Free-Form Deformation (FFD).

    Generates a dense deformation field by upsampling quasi-periodic perturbations
    from a coarse control point grid.

    Args:
        volume: [B, C, D, H, W]
        time_points: [N_shots] time in seconds
        cycle_period: Respiratory cycle length (s)
        max_displacement: Max voxel displacement
        grid_spacing: Spacing of control points (pixels)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Deformed k-space data as a COMPLEX tensor of shape
        ``[N_shots, B, C, D, H, W]`` (``torch.stack`` of per-shot
        ``torch.fft.fftn`` outputs) — NOT a real-stacked ``[..., 2]`` layout.
    """
    B, C, D, H, W = volume.shape
    device = volume.device

    # 1. Respiratory Trace (Hysteresis model)
    omega = 2 * torch.pi / cycle_period
    phase = omega * time_points
    # Asymmetric breathing trace [0, 1]
    trace = torch.cos(phase) ** 2 * torch.exp(-0.2 * torch.sin(phase))

    # 2. Coarse Control Points
    # Grid size for control points (ensure at least 2x2x2 for interpolation)
    cp_D = max(2, D // grid_spacing)
    cp_H = max(2, H // grid_spacing)
    cp_W = max(2, W // grid_spacing)

    # Generate random control point flow field [B, 3, cp_D, cp_H, cp_W]
    # We want coherent motion, e.g. principal component along Superior-Inferior (Z)
    if seed is None:
        seed = 42  # Default seed for reproducibility
    rng = torch.Generator(device=device).manual_seed(seed)

    # Random perturbations
    noise = torch.randn(B, 3, cp_D, cp_H, cp_W, device=device, generator=rng)

    # Smoothen control points (low pass) to ensure physical plausibility
    # or just rely on spline upsampling.

    # Biased motion: Diaphragm moves mostly in Z (dim 0 in flow field? No, dim 2 is Z axis of grid?)
    # flow is [B, 3, ...]. 3 channels: dx, dy, dz.
    # Let's verify grid_sample coordinate system: (x, y, z).
    # dim 0 of flow: x displacement?
    # affine_grid returns (x, y, z). Flow should match.

    # Add strong Z bias (SI direction)
    noise[:, 2, :, :, :] += 2.0  # Bias +Z

    # Normalize to max_displacement
    # Handle potentially empty tensor if dimensions are degenerate (though max(2,...) prevents this)
    if noise.numel() > 0:
        max_val = noise.norm(dim=1, keepdim=True).max()
        if max_val > 0:
            noise = noise / (max_val + 1e-6)
    # Scale in pixels.
    # Convert pixels to normalized coordinates [-1, 1].
    # 2.0 range = D/H/W pixels.
    # Scale factor for each dim.

    # We apply the trace scaling later.

    kspace_data = []

    # Grid for sampling
    base_grid = F.affine_grid(
        torch.eye(3, 4, device=device).unsqueeze(0).repeat(B, 1, 1),
        [B, C, D, H, W],
        align_corners=False,
    )  # [B, D, H, W, 3]

    for t_idx, amp in enumerate(trace):
        # Current coarse flow
        current_cp_flow = noise * amp * max_displacement  # [B, 3, cpD, cpH, cpW] (pixels)

        # Upsample to dense flow field [B, 3, D, H, W] via trilinear interpolation (B-Spline approx)
        dense_flow = F.interpolate(
            current_cp_flow, size=(D, H, W), mode="trilinear", align_corners=False
        )

        # Convert pixel displacement to normalized [-1, 1]
        # flow_x_norm = flow_x_pix / (W/2)
        # flow_y_norm = flow_y_pix / (H/2)
        # flow_z_norm = flow_z_pix / (D/2)

        # dense_flow comes as [dx, dy, dz]?
        # CAUTION: interpolate preserves channel order.
        # grid_sample expects last dim to be (x, y, z).
        # We need to permute dense_flow to [B, D, H, W, 3]
        dense_flow_perm = dense_flow.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        # Apply scaling
        scale = torch.tensor([2.0 / W, 2.0 / H, 2.0 / D], device=device).view(1, 1, 1, 1, 3)
        dense_flow_norm = dense_flow_perm * scale

        # Deform grid
        deformed_grid = base_grid + dense_flow_norm

        # Sample
        deformed_vol = F.grid_sample(
            volume,
            deformed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # FFT3D to k-space
        kspace_shot = torch.fft.fftn(deformed_vol, dim=(-3, -2, -1), norm="ortho")
        kspace_data.append(kspace_shot)

    return torch.stack(kspace_data)


# Alias for backward compatibility / test compliance
apply_elastic_respiratory_motion = simulate_elastic_motion_ffd
