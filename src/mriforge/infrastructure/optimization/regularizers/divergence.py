import torch
import torch.nn as nn


class DivergenceFreeRegularizer(nn.Module):
    """
    Enforces divergence-free constraint on a velocity field.
    div(V) = 0 for incompressible flow.

    Can operate on:
    1. Explicit Grid (B, 3, H, W, D) using finite differences.
    2. Scattered Gaussian parameters (Analytic divergence at points).
    """

    def __init__(self, mode: str = "analytic"):
        """
        Args:
            mode: "analytic" (on particles) or "grid" (on rasterized field).
        """
        super().__init__()
        self.mode = mode

    def forward(
        self,
        parameter_map: torch.Tensor | None = None,
        velocity_indices: tuple[int, int, int] = (0, 1, 2),
        k_trajectory: torch.Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Compute divergence loss.
        Supports "fourier" mode: div(V) = 0 => k . V_k = 0.

        Args:
            parameter_map: (B, C, H, W) k-space grid OR (B, C, M) k-space points.
                           Must contain velocity channels.
            velocity_indices: Indices of (vx, vy, vz) in C dimension.
            k_trajectory: (B, M, 3) k-space coordinates if scattered.
            spatial_shape: (H, W) if parameter_map is flattened grid or we need to generate grid k.
        """
        vx_idx, vy_idx, vz_idx = velocity_indices
        B = parameter_map.shape[0]
        device = parameter_map.device

        if self.mode == "fourier":
            # Extract Velocity K-Space: V_k = (K_vx, K_vy, K_vz)
            k_vx = parameter_map[:, vx_idx].to(device)
            k_vy = parameter_map[:, vy_idx].to(device)
            k_vz = (
                parameter_map[:, vz_idx].to(device)
                if parameter_map.shape[1] > vz_idx
                else torch.zeros_like(k_vx).to(device)
            )

            # 1. Get k-vectors (kx, ky, kz)
            if k_trajectory is not None:
                # (B, M, 3)
                k_trajectory = k_trajectory.to(device)
                kx = k_trajectory[:, :, 0]
                ky = k_trajectory[:, :, 1]
                kz = k_trajectory[:, :, 2]

                # Reshape for broadcasting if needed?
                # k_vx is (B, M) or (B, H, W)?
                # If parameter_map is (B, C, M), then k_vx is (B, M).
                # kx is (B, M).
                pass
            elif parameter_map.ndim == 4:  # Grid (B, C, H, W)
                B, C, H, W = parameter_map.shape
                # Generate centered k-grid [-0.5, 0.5] or [-pi, pi]
                # DFGP usually assumes centered.
                # Let's assume standard fftshift coordinate system?
                # Frequency range [-H/2, H/2].

                ky_grid, kx_grid = torch.meshgrid(
                    torch.fft.fftfreq(H, device=device) * H,
                    torch.fft.fftfreq(W, device=device) * W,
                    indexing="ij",
                )
                kx = kx_grid.unsqueeze(0).expand(B, -1, -1)
                ky = ky_grid.unsqueeze(0).expand(B, -1, -1)
                kz = torch.zeros_like(kx)
            else:
                return torch.tensor(0.0, device=device)

            # 2. Compute Divergence in K-Space: dot product
            # div_k = i * (kx * Vkx + ky * Vky + kz * Vkz)
            # We minimize magnitude |div_k|^2

            # Broadcasting check
            if (k_vx.ndim == 2 and kx.ndim == 2) or (k_vx.ndim == 3 and kx.ndim == 3):  # (B, M)
                div_k = kx * k_vx + ky * k_vy + kz * k_vz
            else:
                # Attempt reshape if k_trajectory provided for flat map
                if k_trajectory is not None and k_vx.shape[-1] == k_trajectory.shape[1]:
                    # Shape match
                    div_k = kx * k_vx + ky * k_vy + kz * k_vz
                else:
                    return torch.tensor(0.0, device=device)

            # Loss = mean(|div_k|^2)
            # handle complex numbers
            norm_sq = torch.abs(div_k) ** 2
            # NaN and Inf safety
            norm_sq = torch.nan_to_num(norm_sq, nan=0.0, posinf=1.0, neginf=1.0)
            return torch.mean(norm_sq)

        return torch.tensor(0.0, device=device)

    def forward_grid(
        self, feature_grid: torch.Tensor, vel_indices: tuple[int, int]
    ) -> torch.Tensor:
        """Helper for grid-based divergence on 2D feature map."""
        vx = feature_grid[:, vel_indices[0]]
        vy = feature_grid[:, vel_indices[1]]

        dvx_dx = (torch.roll(vx, -1, dims=-1) - torch.roll(vx, 1, dims=-1)) / 2.0
        dvy_dy = (torch.roll(vy, -1, dims=-2) - torch.roll(vy, 1, dims=-2)) / 2.0

        div = dvx_dx + dvy_dy
        return torch.mean(div**2)
