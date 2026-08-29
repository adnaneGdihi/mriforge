"""3D Gaussian Primitives with Analytical Fourier Transform.

Implements a differentiable cloud of 3D Gaussians optimized for MRI.
Unlike standard Splatting (rasterization), we use analytical Fourier transforms
to match k-space data directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


class GaussianCloud(nn.Module):
    """Learnable Cloud of 3D Gaussians."""

    def __init__(
        self,
        num_gaussians: int = 1000,
        volume_shape: tuple[int, int, int] = (128, 128, 128),
        init_scale: float = 0.1,
    ):
        """__init__.

        Args:
            num_gaussians (int): Description.
            volume_shape (tuple[int, int, int]): Description.
            init_scale (float): Description.
        """
        super().__init__()
        self.num_gaussians = num_gaussians
        self.volume_shape = volume_shape

        # Learnable Parameters
        # 1. Means (Positions): \mu_k ~ U[0, 1] (or [-1, 1])
        # We work in normalized coords [-1, 1]
        self._xyz = nn.Parameter(torch.rand(num_gaussians, 3) * 2 - 1)

        # 2. Scale (Covariance diagonal/axes): s_k
        # Stored in log space to ensure positivity
        # Initialized to be small (point-like)
        self._scale = nn.Parameter(torch.ones(num_gaussians, 3) * math.log(init_scale))

        # 3. Rotation (Quaternions): q_k (real, i, j, k)
        # Normalized to unit length in forward
        self._rotation = nn.Parameter(torch.rand(num_gaussians, 4))
        # Init as identity rotation roughly
        self._rotation.data[:, 0] = 1.0  # Real part
        self._rotation.data[:, 1:] = 0.0

        # 4. Opacity/Intensity: \alpha_k
        # MRI is complex, but often we reconstruct magnitude images or phase.
        # Usually we assume real-valued proton density map > 0.
        # Stored in logit space (sigmoid)
        self._opacity = nn.Parameter(torch.ones(num_gaussians, 1) * 0.0)  # -> sigmoid(0) = 0.5

        # Activation functions
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid

    @property
    def get_xyz(self) -> Float[Tensor, "N 3"]:
        """get_xyz.

        Returns:
            Float[Tensor, 'N 3']: Description.
        """
        return self._xyz

    @property
    def get_scaling(self) -> Float[Tensor, "N 3"]:
        """get_scaling.

        Returns:
            Float[Tensor, 'N 3']: Description.
        """
        return self.scaling_activation(self._scale)

    @property
    def get_rotation(self) -> Float[Tensor, "N 4"]:
        """get_rotation.

        Returns:
            Float[Tensor, 'N 4']: Description.
        """
        return F.normalize(self._rotation, p=2, dim=-1)

    @property
    def get_opacity(self) -> Float[Tensor, "N 1"]:
        """get_opacity.

        Returns:
            Float[Tensor, 'N 1']: Description.
        """
        return self.opacity_activation(self._opacity)

    def build_covariance(self) -> Float[Tensor, "N 3 3"]:
        """Construct full covariance matrix Sigma = R S S^T R^T."""
        # R from quaternion
        R = self._quaternion_to_matrix(self.get_rotation)

        # S diagonal matrix
        s = self.get_scaling  # [N, 3]
        S = torch.zeros((self.num_gaussians, 3, 3), device=s.device)
        S.diagonal(dim1=-2, dim2=-1).copy_(s)
        # Actually more efficient: Matrix mult is S_mat = diag(s). R @ S_mat
        # (R @ S_mat) @ (R @ S_mat)^T = R S S R^T = R Sigma_diag R^T
        # We can construct Sigma_diag = diag(s^2)

        # Explicit construction
        # Sigma = R @ diag(s**2) @ R.T

        # Vectorized version
        # R: [N, 3, 3]
        # s: [N, 3]

        # R_scaled = R * s.unsqueeze(1) # Broadcast s across rows? No.
        # R_scaled_{ik} = R_{ik} * s_{k} (column scaling)
        R_scaled = R * s.unsqueeze(1)

        Sigma = torch.bmm(R_scaled, R_scaled.transpose(1, 2))
        return Sigma

    def _quaternion_to_matrix(self, q: Float[Tensor, "N 4"]) -> Float[Tensor, "N 3 3"]:
        """Convert quaternion to rotation matrix."""
        # q = (r, x, y, z)
        r, x, y, z = q.unbind(-1)

        # Standard conversion
        # 1 - 2y2 - 2z2   2xy - 2rz      2xz + 2ry
        # 2xy + 2rz       1 - 2x2 - 2z2  2yz - 2rx
        # 2xz - 2ry       2yz + 2rx      1 - 2x2 - 2y2

        R = torch.stack(
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * r * z,
                2 * x * z + 2 * r * y,
                2 * x * y + 2 * r * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * r * x,
                2 * x * z - 2 * r * y,
                2 * y * z + 2 * r * x,
                1 - 2 * x * x - 2 * y * y,
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        return R

    def analytical_fourier_transform(self, k_coords: Float[Tensor, "K 3"]) -> Complex[Tensor, K]:
        """Compute Fourier transform of the Gaussian cloud at specific k-space coordinates.

        G(x) = alpha * exp(-0.5 (x-mu)^T Sigma^-1 (x-mu))
        FT{G}(k) = alpha * |2pi Sigma|^0.5 * exp(-2pi^2 k^T Sigma k - i 2pi k^T mu)
        """
        N = self.num_gaussians
        K = k_coords.shape[0]

        # 1. Get parameters
        mu = self.get_xyz  # [N, 3]
        Sigma = self.build_covariance()  # [N, 3, 3]
        alpha = self.get_opacity.squeeze(1)  # [N]

        # 2. Compute constants
        # |2pi Sigma|^0.5 = (2pi)^1.5 * |Sigma|^0.5
        # |Sigma| = det(R S S^T R^T) = det(S)^2 = (s1 s2 s3)^2
        # |Sigma|^0.5 = s1 s2 s3
        scaling = self.get_scaling  # [N, 3]
        det_sqrt = torch.prod(scaling, dim=1)  # [N]
        norm_const = alpha * (2 * math.pi) ** 1.5 * det_sqrt  # [N]

        # 3. Compute exponent terms
        # This is an N x K interaction.
        # If K (k-space points) is large (e.g. 128^3 ~ 2M), this O(NK) is too big.
        # However, for sparse k-space trajectories or batches, it's feasible.
        # Assuming batching of k_coords outside.

        # k: [K, 3]
        # mu: [N, 3]
        # Sigma: [N, 3, 3]

        # Phase term: -i 2pi k^T mu
        # k^T mu -> [K, N] matrix product
        phase = -2j * math.pi * torch.matmul(k_coords, mu.T)  # [K, N]

        # Decay term: -2pi^2 k^T Sigma k
        # k^T Sigma k. For each n in N, and k in K.
        # Efficient computation:
        # Sigma is symmetric.
        # (k^T R) S S^T (R^T k)
        # Let k' = k @ R [K, N, 3]. But R depends on N.
        # k: [K, 1, 3], R: [1, N, 3, 3]. k @ R -> [K, N, 3]

        # Let's try direct batch matmul if mem allows
        # k_exp = k.unsqueeze(1) # [K, 1, 3]
        # Sigma_exp = Sigma.unsqueeze(0) # [1, N, 3, 3]
        # term = k_exp @ Sigma_exp @ k_exp.transpose(-1, -2) # [K, N, 1, 1]

        # Memory optimization:
        # k^T Sigma k = sum_{ij} k_i Sigma_{ij} k_j
        # = sum_i k_i (sum_j Sigma_{ij} k_j)

        # Or loop over N? No, slow.
        # Loop over K? No, K is batch.
        # This operation is heavy.

        # Alternative: k' = k @ R_n.
        # k_rotated = torch.einsum('ki,nij->knj', k_coords, self._quaternion_to_matrix(self.get_rotation))
        # decay = sum(k_rotated^2 * s^2, dim=-1)

        R = self._quaternion_to_matrix(self.get_rotation)  # [N, 3, 3]
        s = self.get_scaling  # [N, 3]

        # k_rot = k @ R
        # [K, 3] @ [N, 3, 3] -> [K, N, 3]
        # einsum: ki, nij -> knj
        k_rot = torch.einsum("ki,nij->knj", k_coords, R)

        # Weighted norm
        # sum (k_rot_j * s_j)^2
        k_rot_scaled = k_rot * s.unsqueeze(0)  # [K, N, 3]
        decay_arg = torch.sum(k_rot_scaled**2, dim=2)  # [K, N]

        decay = -2 * (math.pi**2) * decay_arg  # [K, N]

        # Combine
        # G(k) = sum_n ( norm_const_n * exp(decay_kn + phase_kn) )

        terms = norm_const.unsqueeze(0) * torch.exp(decay + phase)  # [K, N]

        # Sum over Gaussians
        k_values = torch.sum(terms, dim=1)  # [K]

        return k_values

    def rasterize(self, shape: tuple[int, int, int]) -> Float[Tensor, "1 D H W"]:
        """Rasterize to image space (for visualization/debugging)."""
        D, H, W = shape
        device = self._xyz.device

        # Grid in [-1, 1]
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)

        # Meshgrid
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3)  # [V, 3]

        # Get Gaussian parameters
        mu = self.get_xyz  # [N, 3]
        alpha = self.get_opacity.squeeze(-1)  # [N]
        R = self._quaternion_to_matrix(self.get_rotation)  # [N, 3, 3]
        s = self.get_scaling  # [N, 3]

        # Inverse scale for Sigma^-1 = R @ diag(1/s^2) @ R^T
        s_inv = 1.0 / (s + 1e-8)  # [N, 3]

        density = torch.zeros(coords.shape[0], device=device)

        chunk_size = 10000
        for i in range(0, coords.shape[0], chunk_size):
            chunk = coords[i : i + chunk_size]  # [C, 3]
            C = chunk.shape[0]

            # Compute (x - mu) for all Gaussians: [C, N, 3]
            diff = chunk.unsqueeze(1) - mu.unsqueeze(0)  # [C, N, 3]

            # Rotate difference to Gaussian local frame: diff_rot = (x-mu) @ R
            # einsum: (C, N, 3), (N, 3, 3) -> (C, N, 3)
            diff_rot = torch.einsum("cni,nij->cnj", diff, R)

            # Scale by inverse: diff_scaled = diff_rot * s_inv
            diff_scaled = diff_rot * s_inv.unsqueeze(0)  # [C, N, 3]

            # Mahalanobis distance: ||diff_scaled||^2
            mahal = torch.sum(diff_scaled**2, dim=-1)  # [C, N]

            # Gaussian evaluation: alpha * exp(-0.5 * mahal)
            gauss_vals = alpha.unsqueeze(0) * torch.exp(-0.5 * mahal)  # [C, N]

            # Sum over all Gaussians
            density[i : i + chunk_size] = gauss_vals.sum(dim=-1)

        return density.reshape(1, D, H, W)
