import torch
import torch.nn as nn

from spectramr.models.registry import register_model


@register_model(name="latent_aligner", training_mode="adaptation")
class LatentAligner(nn.Module):
    """
    Computes statistical alignment (MMD) between local latent batches and
    global statistics, without sharing raw data.
    """

    def __init__(self, kernel_type="rbf"):
        """__init__.

        Args:
            kernel_type (Any): Description.
        """
        super().__init__()
        self.kernel_type = kernel_type

    def gaussian_kernel(self, x, y, sigma=1.0):
        """gaussian_kernel.

        Args:
            x (Any): Description.
            y (Any): Description.
            sigma (Any): Description.
        Returns:
            Any: Description.
        """
        beta = 1.0 / (2.0 * sigma)
        dist = torch.cdist(x, y) ** 2
        return torch.exp(-beta * dist)

    def mmd_loss(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Maximum Mean Discrepancy (MMD) Loss.
        Args:
            source_features: [B, D]
            target_features: [B, D]
        """
        xx = self.gaussian_kernel(source_features, source_features)
        yy = self.gaussian_kernel(target_features, target_features)
        xy = self.gaussian_kernel(source_features, target_features)

        return xx.mean() + yy.mean() - 2 * xy.mean()

    def feature_matching_loss(
        self, local_features: torch.Tensor, global_statistics: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Aligns local features to global statistics (Mean/Covariance).
        Used when global samples are not available, only stats.
        Args:
            local_features: [B, D]
            global_statistics: {'mean': [D], 'cov': [D, D]} (optional)
        """
        local_mean = local_features.mean(dim=0)
        global_mean = global_statistics["mean"]

        # L2 distance on means
        loss_mean = torch.norm(local_mean - global_mean)

        loss = loss_mean

        if "cov" in global_statistics:
            # Align covariance (optional, expensive)
            # Center features
            centered = local_features - local_mean
            local_cov = (centered.T @ centered) / (local_features.shape[0] - 1)
            loss_cov = torch.norm(local_cov - global_statistics["cov"])
            loss += loss_cov.item()

        return loss


class FederatedServer:
    """
    Aggregates latent statistics from multiple clients.
    """

    def __init__(self, feature_dim: int):
        """__init__.

        Args:
            feature_dim (int): Description.
        """
        self.feature_dim = feature_dim
        self.global_stats = {"mean": torch.zeros(feature_dim), "count": 0}

    def update_statistics(self, client_stats: list[dict[str, torch.Tensor]]):
        """
        Aggregates means from clients.
        Weight by batch size (n) if provided, else average.
        """
        total_n = 0
        agg_mean = torch.zeros(self.feature_dim)

        for stats in client_stats:
            n = stats.get("n", 1)
            mean = stats["mean"]
            agg_mean += mean * n
            total_n += n

        if total_n > 0:
            agg_mean /= total_n

        self.global_stats["mean"] = agg_mean
        self.global_stats["count"] = total_n

    def get_global_statistics(self):
        """get_global_statistics.

        Returns:
            Any: Description.
        """
        return self.global_stats
