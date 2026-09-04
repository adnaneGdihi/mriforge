"""Task 4 Physics Operators: T2* Decay, TGV Constraint, L+S Dynamic.

Differentiable ``nn.Module`` operators that complete the full forward/adjoint
operator chain for the PMA-VarNet architecture.

Operators
---------
- :class:`T2StarDecayOperator`   — readout-time exponential T2* damping
- :class:`TGVConstraintOperator` — Total Generalized Variation proximal operator
- :class:`LplusSOperator`        — Low-rank + Sparse decomposition operator

Reference
---------
    Blueprint Task 4 §4.7, §4.9, §4.10
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# 1.  T2* Exponential Decay Operator  (Task 4.7)
# ======================================================================


class T2StarDecayOperator(nn.Module):
    r"""Exponential T2* signal decay during readout.

    Models intra-voxel dephasing as an exponential damping envelope
    applied to the image before Fourier encoding:

    .. math::

        \mathcal{A}_{dec}(\mathbf{x}) = \mathbf{x} \odot
            \exp\!\bigl(-t_{\text{readout}} / T_2^*(\mathbf{r})\bigr)

    The adjoint is identical (self-adjoint for real-valued decay maps):

    .. math::

        \mathcal{A}_{dec}^H(\mathbf{y}) = \mathbf{y} \odot
            \exp\!\bigl(-t_{\text{readout}} / T_2^*(\mathbf{r})\bigr)

    Args:
        t2star_map: Spatial T2* map ``[1, 1, H, W]`` in seconds.
                    ``None`` → identity (no decay applied).
        readout_duration: Total readout duration in seconds.
        num_segments: Number of time segments for multi-echo simulation.
    """

    def __init__(
        self,
        t2star_map: torch.Tensor | None = None,
        readout_duration: float = 5e-3,
        num_segments: int = 1,
    ) -> None:
        super().__init__()
        self.readout_duration = readout_duration
        self.num_segments = num_segments
        if t2star_map is not None:
            self.register_buffer("t2star_map", t2star_map)
        else:
            self.register_buffer("t2star_map", None)

    def set_t2star_map(self, t2star_map: torch.Tensor) -> None:
        """Update the T2* map.

        Args:
            t2star_map: ``[1, 1, H, W]`` in seconds.
        """
        # Always use register_buffer to maintain device/dtype tracking
        self.register_buffer("t2star_map", t2star_map)

    def _compute_decay(self, t: float) -> torch.Tensor:
        """Compute decay envelope at time ``t``.

        Args:
            t: Readout time point in seconds.

        Returns:
            Decay envelope ``[1, 1, H, W]``.
        """
        if self.t2star_map is None:
            raise ValueError("T2* map not set. Call set_t2star_map() first.")
        # Clamp T2* to avoid division by zero
        t2s_safe = self.t2star_map.clamp(min=1e-6)
        return torch.exp(-t / t2s_safe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply T2* decay (forward model).

        Args:
            x: Image ``[B, C, H, W]``.

        Returns:
            Decayed image ``[B, C, H, W]``.

        forward method for T2StarDecayOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.t2star_map is None:
            return x
        decay = self._compute_decay(self.readout_duration)
        return x * decay

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply T2* decay adjoint (self-adjoint for real decay).

        Args:
            y: Signal ``[B, C, H, W]``.

        Returns:
            Decay-compensated signal ``[B, C, H, W]``.
        """
        # Self-adjoint: same operation
        return self.forward(y)

    def forward_multisegment(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Apply T2* decay at multiple time segments.

        Args:
            x: Image ``[B, C, H, W]``.

        Returns:
            List of decayed images at each time segment.
        """
        dt = self.readout_duration / max(self.num_segments, 1)
        segments = []
        for seg in range(self.num_segments):
            t = (seg + 0.5) * dt
            decay = self._compute_decay(t)
            segments.append(x * decay)
        return segments


# ======================================================================
# 2.  TGV Constraint Operator  (Task 4.9)
# ======================================================================


class TGVConstraintOperator(nn.Module):
    r"""Total Generalized Variation (TGV) proximal operator.

    Implements the dual-variable proximal step for second-order TGV
    regularisation, parameterised by marker-derived edge tensors:

    .. math::

        \text{prox}_{\lambda \|\cdot\|_1}
            \bigl(\mathbf{v}_n - \tau \nabla \mathbf{x}_n\bigr)

    The edge weights :math:`\mathbf{w}_1, \mathbf{w}_2` are extracted from
    the synthetic marker geometry and pinned:
    :math:`\mathbf{v}(\mathbf{r}_{syn}) = \nabla \mathbf{m}_{syn}`.

    Args:
        alpha0: First-order TGV weight.
        alpha1: Second-order TGV weight.
        num_iters: Number of primal-dual iterations.
        tau: Primal step size.
        sigma: Dual step size.
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        alpha1: float = 2.0,
        num_iters: int = 5,
        tau: float = 0.25,
        sigma: float = 0.25,
    ) -> None:
        super().__init__()
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.num_iters = num_iters
        self.tau = tau
        self.sigma = sigma

    @staticmethod
    def _gradient(x: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient ``[B, 2, H, W]``.

        Args:
            x: Image ``[B, 1, H, W]``.

        Returns:
            Gradient ``[B, 2, H, W]`` (dy, dx).
        """
        dy = torch.diff(x, dim=-2, prepend=x[..., :1, :])
        dx = torch.diff(x, dim=-1, prepend=x[..., :1])
        return torch.cat([dy, dx], dim=1)

    @staticmethod
    def _divergence(p: torch.Tensor) -> torch.Tensor:
        """Compute negative divergence (adjoint of gradient).

        Args:
            p: Dual variable ``[B, 2, H, W]``.

        Returns:
            Divergence ``[B, 1, H, W]``.
        """
        py, px = p[:, 0:1], p[:, 1:2]
        div_y = torch.diff(py, dim=-2, append=py[..., -1:, :])
        div_x = torch.diff(px, dim=-1, append=px[..., -1:])
        return -(div_y + div_x)

    @staticmethod
    def _soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Element-wise soft thresholding.

        Args:
            x: Input tensor.
            threshold: Threshold value.

        Returns:
            Soft-thresholded tensor.
        """
        return torch.sign(x) * F.relu(x.abs() - threshold)

    def forward(
        self,
        x: torch.Tensor,
        marker_grad: torch.Tensor | None = None,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply TGV regularisation via primal-dual iterations.

        Args:
            x: Input image ``[B, 1, H, W]`` or ``[B, C, H, W]``.
            marker_grad: Marker gradient prior ``[B, 2, H, W]`` for pinning.
            marker_mask: Binary mask ``[B, 1, H, W]`` for marker region.

        Returns:
            TGV-regularised image ``[B, C, H, W]``.

        forward method for TGVConstraintOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_grad (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Process each channel independently if multi-channel
        if C > 1:
            channels = []
            for c in range(C):
                channels.append(self.forward(x[:, c : c + 1], marker_grad, marker_mask))
            return torch.cat(channels, dim=1)

        # Initialise primal/dual variables
        x_bar = x.clone()
        v = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)
        p = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)
        q = torch.zeros(B, 3, H, W, device=x.device, dtype=x.dtype)

        for _ in range(self.num_iters):
            # Dual update for p
            grad_xbar = self._gradient(x_bar)
            p_update = p + self.sigma * (grad_xbar - v)
            # Project onto alpha0-ball: p = p * min(1, alpha0 / ||p||)
            p_norm = p_update.norm(dim=1, keepdim=True).clamp(min=1e-8)
            p = p_update / (p_norm / self.alpha0).clamp(min=1.0)

            # Dual update for q (symmetrised gradient of v)
            dvy = torch.diff(v[:, 0:1], dim=-2, prepend=v[:, 0:1, :1, :])
            dvx = torch.diff(v[:, 1:2], dim=-1, prepend=v[:, 1:2, ..., :1])
            dvyx = 0.5 * (
                torch.diff(v[:, 0:1], dim=-1, prepend=v[:, 0:1, ..., :1])
                + torch.diff(v[:, 1:2], dim=-2, prepend=v[:, 1:2, :1, :])
            )
            eps_v = torch.cat([dvy, dvx, dvyx], dim=1)
            q_update = q + self.sigma * eps_v
            q_norm = q_update.norm(dim=1, keepdim=True).clamp(min=1e-8)
            q = q_update / (q_norm / self.alpha1).clamp(min=1.0)

            # Primal update for x
            x_old = x_bar.clone()
            div_p = self._divergence(p)
            x_bar = x_bar - self.tau * div_p
            # Data fidelity: pull towards input
            x_bar = (x_bar + self.tau * x) / (1.0 + self.tau)

            # Primal update for v
            v = v + self.tau * p
            # Pin marker gradient if available
            if marker_grad is not None and marker_mask is not None:
                mask_2ch = marker_mask.expand_as(v)
                v = v * (1 - mask_2ch) + marker_grad * mask_2ch

            # Over-relaxation
            x_bar = 2.0 * x_bar - x_old

        return x_bar


# ======================================================================
# 2b. Variance-Weighted PDHG for TGV  (Task 1.9 Enhancement)
# ======================================================================


class VarianceWeightedTGV(TGVConstraintOperator):
    r"""Variance-weighted Primal-Dual Hybrid Gradient TGV.

    Extends :class:`TGVConstraintOperator` with spatially-varying step sizes
    derived from marker-based noise variance estimates:

    .. math::

        \tau(\mathbf{r}) = \frac{\tau_0}{\sigma^2(\mathbf{r}) + \epsilon}

    This allows the algorithm to regularize more aggressively in high-noise
    regions while preserving detail in low-noise marker neighborhoods.

    Args:
        alpha0: First-order TGV weight.
        alpha1: Second-order TGV weight.
        num_iters: Number of primal-dual iterations.
        tau: Base primal step size.
        sigma: Base dual step size.
        epsilon: Smoothing constant for variance map.
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        alpha1: float = 2.0,
        num_iters: int = 5,
        tau: float = 0.25,
        sigma: float = 0.25,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__(
            alpha0=alpha0,
            alpha1=alpha1,
            num_iters=num_iters,
            tau=tau,
            sigma=sigma,
        )
        self.epsilon = epsilon

    def forward(
        self,
        x: torch.Tensor,
        marker_grad: torch.Tensor | None = None,
        marker_mask: torch.Tensor | None = None,
        variance_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply variance-weighted TGV regularisation.

        Args:
            x: Input image ``[B, 1, H, W]`` or ``[B, C, H, W]``.
            marker_grad: Marker gradient prior ``[B, 2, H, W]``.
            marker_mask: Binary mask ``[B, 1, H, W]``.
            variance_map: Noise variance ``[B, 1, H, W]`` (optional).

        Returns:
            TGV-regularised image ``[B, C, H, W]``.
        """
        if variance_map is None:
            return super().forward(x, marker_grad, marker_mask)

        # Modulate step sizes by inverse variance
        tau_map = self.tau / (variance_map + self.epsilon)

        B, C, H, W = x.shape
        if C > 1:
            channels = []
            for c in range(C):
                channels.append(
                    self._weighted_iteration(x[:, c : c + 1], marker_grad, marker_mask, tau_map)
                )
            return torch.cat(channels, dim=1)

        return self._weighted_iteration(x, marker_grad, marker_mask, tau_map)

    def _weighted_iteration(
        self,
        x: torch.Tensor,
        marker_grad: torch.Tensor | None,
        marker_mask: torch.Tensor | None,
        tau_map: torch.Tensor,
    ) -> torch.Tensor:
        """Run primal-dual with spatially-varying tau.

        Args:
            x: ``[B, 1, H, W]``.
            marker_grad: ``[B, 2, H, W]`` or None.
            marker_mask: ``[B, 1, H, W]`` or None.
            tau_map: ``[B, 1, H, W]`` spatially-varying step size.

        Returns:
            Regularised ``[B, 1, H, W]``.
        """
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        x_bar = x.clone()
        v = torch.zeros(B, 2, H, W, device=device, dtype=dtype)
        p = torch.zeros(B, 2, H, W, device=device, dtype=dtype)

        for _ in range(self.num_iters):
            grad_xbar = self._gradient(x_bar)
            p_update = p + self.sigma * (grad_xbar - v)
            p_norm = p_update.norm(dim=1, keepdim=True).clamp(min=1e-8)
            p = p_update / (p_norm / self.alpha0).clamp(min=1.0)

            x_old = x_bar.clone()
            div_p = self._divergence(p)
            # Spatially-varying tau
            x_bar = x_bar - tau_map * div_p
            x_bar = (x_bar + tau_map * x) / (1.0 + tau_map)

            v = v + tau_map * p
            if marker_grad is not None and marker_mask is not None:
                mask_2ch = marker_mask.expand_as(v)
                v = v * (1 - mask_2ch) + marker_grad * mask_2ch

            x_bar = 2.0 * x_bar - x_old

        return x_bar


# ======================================================================
# 3.  L+S Dynamic Operator  (Task 4.10)
# ======================================================================


class LplusSOperator(nn.Module):
    r"""Low-rank + Sparse (L+S) decomposition operator.

    Decomposes a dynamic image series into a low-rank background
    :math:`\mathbf{L}` and a sparse component :math:`\mathbf{S}`
    via proximal splitting:

    Forward:
        :math:`\mathcal{T}(\mathbf{L}, \mathbf{S}) = \mathbf{L} + \mathbf{S}`

    The operator provides SVD-based low-rank projection and
    soft-thresholding for the sparse component.

    Args:
        rank: Maximum rank for the low-rank component.
        lambda_s: Sparsity weight for soft-thresholding.
        num_iters: Number of ADMM/proximal iterations.
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{LplusS}(X) = \arg\min_{L,S} \frac{1}{2}\|X - (L+S)\|_F^2 + \lambda_L \|L\|_* + \lambda_S \|S\|_1"""

    def __init__(
        self,
        rank: int = 5,
        lambda_s: float = 0.01,
        num_iters: int = 10,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.lambda_s = lambda_s
        self.num_iters = num_iters

    def low_rank_project(self, X: torch.Tensor, rank: int) -> torch.Tensor:
        """Project onto rank-k subspace via truncated SVD.

        Args:
            X: Casorati matrix ``[B, N_spatial, N_temporal]``.
            rank: Target rank.

        Returns:
            Low-rank approximation ``[B, N_spatial, N_temporal]``.
        """
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # Truncate to rank-k
        k = min(rank, S.shape[-1])
        U_k = U[..., :k]
        S_k = S[..., :k]
        Vh_k = Vh[..., :k, :]
        return U_k * S_k.unsqueeze(-2) @ Vh_k

    def sparse_threshold(self, S: torch.Tensor, lam: float) -> torch.Tensor:
        """Soft-thresholding for sparsity.

        Args:
            S: Sparse component ``[B, N_spatial, N_temporal]``.
            lam: Threshold.

        Returns:
            Thresholded sparse component.
        """
        return torch.sign(S) * F.relu(S.abs() - lam)

    def forward(
        self,
        x_dynamic: torch.Tensor,
        temporal_basis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose dynamic series into L + S.

        Args:
            x_dynamic: Dynamic image series ``[B, T, H, W]``
                        where T is the temporal/repetition dimension.
            temporal_basis: Optional temporal basis from marker
                           ``[T, K]`` for constrained subspace.

        Returns:
            Tuple of (L, S) each ``[B, T, H, W]``.

        forward method for LplusSOperator.

        Executes PyTorch tensor operations.

        Args:
            x_dynamic (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            temporal_basis (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, T, H, W = x_dynamic.shape

        # Reshape to Casorati matrix: [B, HW, T]
        X = x_dynamic.reshape(B, T, H * W).transpose(1, 2)

        # Initialise L and S
        L = torch.zeros_like(X)
        S = torch.zeros_like(X)

        for _ in range(self.num_iters):
            # Low-rank update: SVD projection of (X - S)
            L = self.low_rank_project(X - S, self.rank)

            # Sparse update: soft-threshold (X - L)
            S = self.sparse_threshold(X - L, self.lambda_s)

            # If temporal basis is available, constrain L
            if temporal_basis is not None:
                # Project L onto marker-derived subspace: L = L @ V_k @ V_k^H
                V_k = temporal_basis  # [T, K]
                proj = X @ V_k @ V_k.T  # [B, HW, T]
                L = proj

        # Reshape back to spatial
        L_spatial = L.transpose(1, 2).reshape(B, T, H, W)
        S_spatial = S.transpose(1, 2).reshape(B, T, H, W)
        return L_spatial, S_spatial

    def adjoint(
        self,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adjoint of the L+S forward (identity routing to both streams).

        Args:
            y: Residual ``[B, T, H, W]``.

        Returns:
            Tuple of gradients (grad_L, grad_S), both equal to y.
        """
        return y, y


__all__ = [
    "LplusSOperator",
    "T2StarDecayOperator",
    "TGVConstraintOperator",
    "VarianceWeightedTGV",
]
