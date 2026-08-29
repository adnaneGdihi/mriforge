"""Adaptive Control for 3D Gaussian Splatting.

Implements densification (split/clone) and pruning strategies based on
gradient magnitude and opacity thresholds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Bool
from torch import Tensor

from .gaussian_primitives import GaussianCloud


class AdaptiveController(nn.Module):
    """Manages densification and pruning of Gaussian Cloud."""

    def __init__(
        self,
        densify_grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        extent: float = 2.0,  # [-1, 1] -> size 2
        percent_dense: float = 0.01,
        split_factor: float = 2.0,
    ):
        """__init__.

        Args:
            densify_grad_threshold (float): Description.
            min_opacity (float): Description.
            extent (float): Description.
            percent_dense (float): Description.
            split_factor (float): Description.
        """
        super().__init__()
        self.densify_grad_threshold = densify_grad_threshold
        self.min_opacity = min_opacity
        self.extent = extent
        self.percent_dense = percent_dense
        self.split_factor = split_factor

        # Accumulators
        self.register_buffer("grad_accum", torch.zeros(1))
        self.register_buffer("denom", torch.zeros(1))
        self.register_buffer("max_radii2D", torch.zeros(1))

    def update_grad_stats(
        self,
        cloud: GaussianCloud,
    ):
        """Accumulate gradients for densification criteria using EMA."""
        # Track gradients on xyz parameters for k-space optimization

        if cloud._xyz.grad is None:
            return

        grads = cloud._xyz.grad.norm(dim=-1)  # [N]
        N = grads.shape[0]

        # Initialize accumulators if needed
        if self.grad_accum.shape[0] != N:
            self.grad_accum = torch.zeros(N, device=grads.device)
            self.denom = torch.zeros(N, device=grads.device)

        # Exponential moving average with decay factor
        ema_decay = 0.9
        self.grad_accum = ema_decay * self.grad_accum + (1 - ema_decay) * grads
        self.denom = ema_decay * self.denom + (1 - ema_decay) * torch.ones_like(grads)

        # Store normalized gradients
        self.curr_grads = self.grad_accum / (self.denom + 1e-10)

    def refine(self, cloud: GaussianCloud, iter_step: int, reset_interval: int = 3000):
        """Perform densification and pruning."""

        # 1. Prune
        self._prune(cloud)

        # 2. Densify
        self._densify(cloud)

        # 3. Opacity Reset (optional, helps escape local minima)
        if iter_step > 0 and iter_step % reset_interval == 0:
            self._reset_opacity(cloud)

    def _prune(self, cloud: GaussianCloud):
        # Remove points with low opacity
        """_prune.

        Args:
            cloud (GaussianCloud): Description.
        Returns:
            Any: Description.
        """
        min_op = self.min_opacity
        start_count = cloud.num_gaussians

        opacities = cloud.get_opacity.squeeze()  # [N]
        valid_mask = opacities > min_op

        # Also prune huge scales (optional)
        scales = cloud.get_scaling.max(dim=1).values  # [N]
        valid_mask = valid_mask & (scales < self.extent * 0.5)

        if valid_mask.sum() < start_count:
            self._filter_cloud(cloud, valid_mask)

    def _densify(self, cloud: GaussianCloud):
        # Identify high gradient regions (under-reconstructed)
        """_densify.

        Args:
            cloud (GaussianCloud): Description.
        Returns:
            Any: Description.
        """
        if not hasattr(self, "curr_grads"):
            return

        grads = self.curr_grads
        high_grad_mask = grads >= self.densify_grad_threshold

        if high_grad_mask.sum() == 0:
            return

        # Decision: Clone or Split?
        # Clone: small gaussians in geometric regions -> clone to fill
        # Split: large gaussians in detailed regions -> split to refine

        scales = cloud.get_scaling.max(dim=1).values  # [N]

        # Threshold for "large" vs "small"
        # Heuristic: > 1% of scene extent
        size_thresh = self.percent_dense * self.extent

        split_mask = high_grad_mask & (scales > size_thresh)
        clone_mask = high_grad_mask & (scales <= size_thresh)

        if split_mask.any():
            self._split_gaussians(cloud, split_mask)

        if clone_mask.any():
            self._clone_gaussians(cloud, clone_mask)

    def _clone_gaussians(self, cloud: GaussianCloud, mask: Bool[Tensor, N]):
        # Simply duplicate
        """_clone_gaussians.

        Args:
            cloud (GaussianCloud): Description.
            mask (Bool[Tensor, N]): Description.
        Returns:
            Any: Description.
        """
        new_xyz = cloud._xyz[mask]
        new_scale = cloud._scale[mask]
        new_rot = cloud._rotation[mask]
        new_op = cloud._opacity[mask]

        self._append_to_cloud(cloud, new_xyz, new_scale, new_rot, new_op)

    def _split_gaussians(self, cloud: GaussianCloud, mask: Bool[Tensor, N]):
        # Split into N=2 smaller gaussians
        # Sample positions from the original gaussian covariance
        """_split_gaussians.

        Args:
            cloud (GaussianCloud): Description.
            mask (Bool[Tensor, N]): Description.
        Returns:
            Any: Description.
        """
        n_splits = 2

        # Original params of parents
        stds = cloud.get_scaling[mask].repeat(n_splits, 1)  # [M*2, 3]
        means = cloud.get_xyz[mask].repeat(n_splits, 1)  # [M*2, 3]

        # Sample perturbations ~ N(0, std)
        samples = torch.randn_like(means) * stds

        # Use rotation matrix to align perturbations
        rots = cloud.get_rotation[mask].repeat(n_splits, 1)  # [M*2, 4]
        R = cloud._quaternion_to_matrix(rots)  # [M*2, 3, 3]
        samples_rot = torch.bmm(R, samples.unsqueeze(-1)).squeeze(-1)

        new_xyz = means + samples_rot

        # Scale reduction
        new_scale_val = cloud.get_scaling[mask].repeat(n_splits, 1) / self.split_factor
        new_scale = torch.log(new_scale_val + 1e-7)

        new_rot = rots
        new_op = cloud._opacity[mask].repeat(n_splits, 1)

        # Update cloud: Keep non-split ones, append new split ones
        keep_mask = ~mask
        self._filter_cloud(cloud, keep_mask)
        self._append_to_cloud(cloud, new_xyz, new_scale, new_rot, new_op)

    def _append_to_cloud(self, cloud, xyz, scale, rot, op):
        """_append_to_cloud.

        Args:
            cloud (Any): Description.
            xyz (Any): Description.
            scale (Any): Description.
            rot (Any): Description.
            op (Any): Description.
        Returns:
            Any: Description.
        """
        cloud._xyz = nn.Parameter(torch.cat([cloud._xyz, xyz], dim=0))
        cloud._scale = nn.Parameter(torch.cat([cloud._scale, scale], dim=0))
        cloud._rotation = nn.Parameter(torch.cat([cloud._rotation, rot], dim=0))
        cloud._opacity = nn.Parameter(torch.cat([cloud._opacity, op], dim=0))
        cloud.num_gaussians = cloud._xyz.shape[0]

    def _filter_cloud(self, cloud, mask):
        """_filter_cloud.

        Args:
            cloud (Any): Description.
            mask (Any): Description.
        Returns:
            Any: Description.
        """
        cloud._xyz = nn.Parameter(cloud._xyz[mask])
        cloud._scale = nn.Parameter(cloud._scale[mask])
        cloud._rotation = nn.Parameter(cloud._rotation[mask])
        cloud._opacity = nn.Parameter(cloud._opacity[mask])
        cloud.num_gaussians = cloud._xyz.shape[0]

        if hasattr(self, "curr_grads") and self.curr_grads.shape[0] == mask.shape[0]:
            self.curr_grads = self.curr_grads[mask]

    def _reset_opacity(self, cloud):
        # Pruning aggressively can clear space.
        # Resetting opacity to lower value allows re-learning?
        # Or inverse? Standard 3DGS resets opacities to max 0.01 to prune floaters.

        # Inverse sigmoid(0.01) -> logit
        """_reset_opacity.

        Args:
            cloud (Any): Description.
        Returns:
            Any: Description.
        """
        inv_sigmoid = lambda x: torch.log(torch.tensor(x) / (1 - torch.tensor(x)))
        reset_val = inv_sigmoid(0.01).to(cloud._opacity.device)

        current_op = cloud.get_opacity
        mask = current_op > 0.01  # Reset those that are visible?
        # Usually checking implementation:
        # "Reset opacity of all Gaussians to 0.01"

        # cloud._opacity.data.fill_(reset_val)
        # But we want to preserve gradients? typically performed at end of step.
        new_op = torch.min(cloud._opacity.data, torch.ones_like(cloud._opacity.data) * reset_val)
        cloud._opacity.data = new_op
