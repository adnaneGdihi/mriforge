"""Graph Distance Calculator for Physics-Aware MRI.

Modular k-NN distance calculation with multiple metrics:
- 'euclidean': Standard geometric distance (isotropic)
- 'polar': (r, θ) space for Radial MRI geometry
- 'time_weighted': Adds acquisition time dimension for B0 correction
- 'cosine': Angular similarity for feature alignment

References:
- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds" TOG 2019
"""

from typing import Literal

import torch
import torch.nn.functional as F

DistanceMetric = Literal["euclidean", "polar", "time_weighted", "cosine"]


class GraphDistanceCalculator:
    """Compute k-Nearest Neighbors using physics-aware distance metrics."""

    @staticmethod
    def calculate(
        x: torch.Tensor,
        k: int,
        metric: DistanceMetric = "euclidean",
        time_vec: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Compute k-NN indices using specified metric.

        Args:
            x: Input coordinates [B, C, N] (usually trajectory kx, ky)
            k: Number of neighbors
            metric: Distance metric to use
            time_vec: Readout time [B, 1, N] for 'time_weighted'
            alpha: Time weighting factor for 'time_weighted'

        Returns:
            KNN indices [B, N, k]
        """
        batch_size, channels, num_points = x.size()
        k = min(k, num_points)  # Safety cap

        # Pre-process coordinates based on metric
        x_search = GraphDistanceCalculator._preprocess(x, metric, time_vec, alpha)

        # Compute pairwise distances
        pairwise_dist = GraphDistanceCalculator._pairwise_distance(x_search, metric)

        # Find k-NN (smallest distances)
        # topk on negative distance gives smallest actual distances
        _, idx = pairwise_dist.topk(k=k, dim=-1, largest=True)

        return idx

    @staticmethod
    def _preprocess(
        x: torch.Tensor,
        metric: str,
        time_vec: torch.Tensor | None,
        alpha: float,
    ) -> torch.Tensor:
        """Transform coordinates based on metric."""

        if metric == "polar":
            # Convert (kx, ky) -> (r, theta)
            # x: [B, 2, N] where channel 0=kx, 1=ky
            if x.shape[1] >= 2:
                kx, ky = x[:, 0, :], x[:, 1, :]
            else:
                # Single channel: assume kx only
                kx = x[:, 0, :]
                ky = torch.zeros_like(kx)

            r = torch.sqrt(kx**2 + ky**2 + 1e-8)
            theta = torch.atan2(ky, kx)

            # Normalize theta to comparable range with r
            # r typically in [0, π], theta in [-π, π]
            # Scale theta to [0, 1] for balance
            theta_norm = (theta + torch.pi) / (2 * torch.pi)
            r_norm = r / (r.max() + 1e-8)

            return torch.stack([r_norm, theta_norm], dim=1)

        elif metric == "time_weighted":
            if time_vec is None:
                raise ValueError(
                    "Metric 'time_weighted' requires 'time_vec' input. "
                    "Pass readout time from trajectory generator."
                )

            # Normalize time to [0, 1]
            t = time_vec.squeeze(1) if time_vec.dim() == 3 else time_vec
            t_norm = t / (t.max() + 1e-8)

            # Concatenate scaled time to spatial coords
            # x: [B, 2, N], t_norm: [B, N] -> [B, 3, N]
            return torch.cat([x, t_norm.unsqueeze(1) * alpha], dim=1)

        # Default: use raw coordinates
        return x

    @staticmethod
    def _pairwise_distance(
        x: torch.Tensor,
        metric: str,
    ) -> torch.Tensor:
        """Compute pairwise distance matrix.

        Returns NEGATIVE squared distance so topk(largest) gives nearest.
        """
        if metric == "cosine":
            # Cosine similarity: 1 - cos(angle)
            x_norm = F.normalize(x, p=2, dim=1)  # [B, C, N]

            # [B, N, N] = [B, N, C] @ [B, C, N]
            similarity = torch.bmm(x_norm.transpose(1, 2), x_norm)

            # Convert to negative distance (larger = closer)
            return similarity

        else:
            # Euclidean: ||A - B||^2 = ||A||^2 + ||B||^2 - 2*<A,B>
            # x: [B, C, N]

            # Inner product: [B, N, N]
            inner = torch.bmm(x.transpose(1, 2), x)

            # Squared norms: [B, 1, N]
            xx = torch.sum(x**2, dim=1, keepdim=True)

            # Squared distance: [B, N, N]
            # dist = xx + xx^T - 2*inner
            dist_sq = xx.transpose(1, 2) + xx - 2 * inner

            # Return negative so topk largest = smallest actual distance
            return -dist_sq

    @staticmethod
    def get_edge_features(
        x: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Gather neighbor features for EdgeConv.

        Args:
            x: Node features [B, C, N]
            idx: KNN indices [B, N, k]

        Returns:
            Edge features [B, C, N, k]
        """
        B, C, N = x.shape
        _, _, k = idx.shape

        # Expand x for gathering: [B, C, N*k]
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        idx_flat = (idx + idx_base).view(B, -1)  # [B, N*k]

        # Gather: [B, C, N*k]
        x_flat = x.reshape(B * N, C).T  # [C, B*N]

        # Alternative: use index_select per batch
        neighbors = []
        for b in range(B):
            # idx[b]: [N, k]
            neighbor_idx = idx[b].reshape(-1)  # [N*k]
            neighbor_feat = x[b, :, neighbor_idx]  # [C, N*k]
            neighbors.append(neighbor_feat.view(C, N, k))

        return torch.stack(neighbors, dim=0)  # [B, C, N, k]


def knn_graph(
    x: torch.Tensor,
    k: int,
    metric: DistanceMetric = "euclidean",
    time_vec: torch.Tensor | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Convenience function for k-NN graph construction.

    Args:
        x: Coordinates [B, C, N]
        k: Number of neighbors
        metric: Distance metric
        time_vec: Readout time [B, 1, N]
        alpha: Time weighting

    Returns:
        KNN indices [B, N, k]
    """
    return GraphDistanceCalculator.calculate(x, k, metric, time_vec, alpha)


__all__ = [
    "DistanceMetric",
    "GraphDistanceCalculator",
    "knn_graph",
]
