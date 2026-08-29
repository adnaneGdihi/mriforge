import torch
import torch.nn as nn


class KNNGraphRegularizer(nn.Module):
    """
    K-Nearest Neighbor (KNN) Graph Regularizer for Gaussian Splatting.

    Prevents "needle" artifacts and enforces local smoothness by penalizing
    large deviations in Gaussian attributes (scale, rotation, opacity, color)
    among spatial neighbors.
    """

    def __init__(self, k: int = 5, distance_metric: str = "euclidean"):
        """
        Args:
            k: Number of nearest neighbors.
            distance_metric: Metric for modifying spatial distance.
        """
        super().__init__()
        self.k = k
        self.distance_metric = distance_metric

    def _compute_knn(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k-nearest neighbors for a set of points.

        Args:
            positions: (B, N, 3) or (N, 3) tensor of spatial coordinates.

        Returns:
            indices: (B, N, k) indices of neighbors.
            distances: (B, N, k) distances to neighbors.
        """
        # If batch dimension is missing, add it
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)

        B, N, D = positions.shape

        # Simple brute force for prototyping (O(N^2)).
        # For large N (~1M), this will OOM.
        # But for MRI latent splats (e.g. 64x64 or 128x128 grid = ~16k), it's feasible.
        # Optimization: Use torch.cdist which is optimized.

        # Calculate pairwise distances
        # (B, N, N)
        dist_matrix = torch.cdist(positions, positions, p=2)

        # Get top k+1 (including self)
        # We want smallest distances
        values, indices = torch.topk(dist_matrix, k=self.k + 1, dim=-1, largest=False)

        # Remove self (which is index 0 usually, but let's be safe and take 1:)
        return indices[:, :, 1:], values[:, :, 1:]

    def forward(
        self,
        positions: torch.Tensor,
        attributes: torch.Tensor,
        attribute_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute regularization loss.

        Args:
            positions: (B, N, 3) spatial positions (xyz).
            attributes: (B, N, F) features/attributes to regularize (e.g. scales, rotations).
            attribute_weights: (F,) weight per attribute dimension.

        Returns:
            loss: Scalar regularization loss.
        """
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if attributes.ndim == 2:
            attributes = attributes.unsqueeze(0)

        B, N, _ = positions.shape

        # 1. Find Neighbors
        neighbor_indices, _ = self._compute_knn(positions)  # (B, N, k)

        # 2. Gather neighbor attributes
        # batch index gather is tricky.
        # Flatten batch for gather?
        # Construct flat indices: b*N + idx

        flat_attributes = attributes.contiguous().view(B * N, -1)  # (B*N, F)
        flat_indices = neighbor_indices + (torch.arange(B, device=positions.device) * N).view(
            B, 1, 1
        )
        flat_indices = flat_indices.view(-1)  # (B*N*k)

        neighbor_attrs = flat_attributes[flat_indices].view(B, N, self.k, -1)  # (B, N, k, F)

        # 3. Compute variance/deviation from neighbors
        # Local Consistency Loss: || attr_i - mean(attr_neighbors) ||^2
        # Or pairwise: sum || attr_i - attr_j ||^2

        # Option A: Deviation from local mean
        local_mean = neighbor_attrs.mean(dim=2)  # (B, N, F)
        deviation = (attributes - local_mean).pow(2)  # (B, N, F)

        if attribute_weights is not None:
            deviation = deviation * attribute_weights.view(1, 1, -1)

        loss = deviation.mean()

        return loss
