import math

import torch
import torch.nn as nn


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a real-first ``(w, x, y, z)`` quaternion ``[N, 4]`` to ``[N, 3, 3]``.

    Matches the convention used by ``models.gaussian_splatting.gaussian_primitives``
    (``_rotation[:, 0]`` is the real part). The quaternion is normalized first.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r, x, y, z = q.unbind(-1)
    return torch.stack(
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


def differentiable_gaussian_fourier(
    means: torch.Tensor,
    scales: torch.Tensor,
    quaternions: torch.Tensor,
    features: torch.Tensor,
    k_trajectory: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable Fourier-Gaussian Projector (DFGP).
    Direct PyTorch implementation for reliable autograd tracking.
    """
    # Ensure all inputs are on the same device and compatible dtypes
    device = means.device
    dtype = means.dtype

    means = means.to(device)
    scales = scales.to(device).to(dtype)
    k_trajectory = k_trajectory.to(device).to(dtype)
    quaternions = quaternions.to(device).to(dtype)

    # Handle features separately to preserve complex dtype if present
    features = features.to(device)
    if not features.is_complex() and features.dtype != dtype:
        features = features.to(dtype)

    TWO_PI = 2 * math.pi
    TWO_PI_SQ = 2 * (math.pi**2)

    # Phase term (depends only on the Gaussian centres, not orientation).
    if means.dim() == 3:
        # MedGS temporal case: means is [K, N, 3], k_trajectory is [K, 3].
        phase_term = torch.sum(k_trajectory.unsqueeze(1) * means, dim=-1)  # [K, N]
    else:
        # Standard case: means is [N, 3], k_trajectory is [K, 3].
        phase_term = torch.matmul(k_trajectory, means.T)  # [K, N]
    f_exp = features.unsqueeze(0)  # [1, N, C]

    # Anisotropic Gaussian decay WITH orientation. The FT of an oriented Gaussian
    # with covariance C = R diag(s^2) R^T decays as exp(-2pi^2 k^T C k); writing
    # v = k R this is exp(-2pi^2 sum_d (v_d s_d)^2). Previously `quaternions` was
    # accepted but ignored, so every oriented Gaussian collapsed to axis-aligned
    # (decay used only the diagonal scales) — a silent no-op of the orientation
    # knob (pitfall #15). Mirrors models.gaussian_splatting.gaussian_primitives.
    R = _quaternion_to_rotation_matrix(quaternions)  # [N, 3, 3]
    k_rot = torch.einsum("ki,nij->knj", k_trajectory, R)  # [K, N, 3]
    decay_term = ((k_rot * scales.unsqueeze(0)) ** 2).sum(dim=-1)  # [K, N]

    # Complex phase: exp(-i 2pi k.mu) = cos(...) - i sin(...)
    arg = -TWO_PI * phase_term
    complex_phase = torch.complex(torch.cos(arg), torch.sin(arg))

    # Clamp decay_term to avoid excessive values in exp (-TWO_PI_SQ * decay)
    decay_term = torch.clamp(decay_term, max=5.0)
    gaussian_decay = torch.exp(-TWO_PI_SQ * decay_term)

    # Combine
    # weights: [K, N] -> [K, N, 1]
    weights = (complex_phase * gaussian_decay).unsqueeze(-1)

    # Sum over Gaussians: [K, N, 1] * [1, N, C] -> [K, N, C] -> sum(N) -> [K, C]
    k_data = torch.sum(weights * f_exp, dim=1)

    return k_data


class GaussianCloud:
    """Dataclass for Gaussian Cloud parameters."""

    __slots__ = ["features", "means", "opacity", "quaternions", "scales"]

    def __init__(self, means, scales, quaternions, features, opacity):
        """__init__.

        Args:
            means (Any): Description.
            scales (Any): Description.
            quaternions (Any): Description.
            features (Any): Description.
            opacity (Any): Description.
        """
        self.means = means
        self.scales = scales
        self.quaternions = quaternions
        self.features = features
        self.opacity = opacity


class GaussianFourierProjector(nn.Module):
    """GaussianFourierProjector class."""

    def __init__(self):
        """__init__."""
        super().__init__()

    def forward(self, cloud, k_trajectory):
        """forward.

        Args:
            cloud (Any): Description.
            k_trajectory (Any): Description.
        Returns:
            Any: Description.

        forward method for GaussianFourierProjector.

        Executes PyTorch tensor operations.

        Args:
            cloud (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            k_trajectory (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return differentiable_gaussian_fourier(
            cloud.means, cloud.scales, cloud.quaternions, cloud.features, k_trajectory
        )


class TemporalGaussianFourierProjector(nn.Module):
    """Projector for moving Gaussians."""

    def __init__(self):
        """__init__."""
        super().__init__()

    def forward(self, cloud, k_trajectory):
        """forward.

        Args:
            cloud (Any): Description.
            k_trajectory (Any): Description.
        Returns:
            Any: Description.

        forward method for TemporalGaussianFourierProjector.

        Executes PyTorch tensor operations.

        Args:
            cloud (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            k_trajectory (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return differentiable_gaussian_fourier(
            cloud.means[:, 0, :] if cloud.means.dim() == 3 else cloud.means,
            cloud.scales,
            cloud.quaternions,
            cloud.features,
            k_trajectory,
        )


# Alias for backward compatibility
GaussianFourierOp = GaussianFourierProjector
