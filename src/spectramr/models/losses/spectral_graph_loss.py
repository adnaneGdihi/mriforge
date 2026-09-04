import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="spectral_graph", aliases=["SpectralGraphLoss"])
class SpectralGraphLoss(nn.Module):
    r"""Graph-Laplacian smoothness regulariser (Dirichlet energy).

    L = x^T L x = sum_{i,j} A_ij ||x_i - x_j||^2

    .. math::

        \mathcal{L}_{SpectralGraph} = \lambda \sum_{i,j \in \mathcal{E}} \| f_i - f_j \|_2^2

    Note: despite the "spectral" name this computes the **unnormalised Laplacian
    quadratic form** ``xᵀLx`` over a k-NN graph — there is NO eigendecomposition
    and no normalised Laplacian here (the name is historical / a misnomer). It is
    a genuine graph-smoothness prior, not a spectral (eigenbasis) feature. The
    actual spectral machinery (normalised Laplacian + ``eigh`` + Laplacian
    positional encoding) lives in the model
    :mod:`spectramr.models.topological.spectral_graph`. The k-NN graph is built
    row-wise (directed / not symmetrised), which is still a valid non-negative
    Dirichlet energy since ``||x_i - x_j||²`` is symmetric."""

    def __init__(self, k=8, patch_size=None, lambda_val=1.0):
        """__init__.

        Args:
            k (Any): Description.
            patch_size (Any): Description.
            lambda_val (Any): Description.
        """
        super().__init__()
        self.k = k
        self.patch_size = patch_size
        self.lambda_val = lambda_val

    def knn_graph(self, x, k):
        # x: [B, N, D]
        # dist = (x_i - x_j)^2
        """knn_graph.

        Args:
            x (Any): Description.
            k (Any): Description.
        Returns:
            Any: Description.
        """
        dist = torch.cdist(x, x, p=2)
        _, indices = torch.topk(dist, k=k + 1, largest=False)
        return indices[:, :, 1:]  # [B, N, k]

    def forward(self, pred, target=None):
        # pred: [B, C, H, W] or [B, N, C]
        # We compute graph on patches or pixels?
        # User blueprint says "Construct the graph in k-space patches" for the model.
        # For loss, we can do the same.

        """forward.

        Args:
            pred (Any): Description.
            target (Any): Description.
        Returns:
            Any: Description.

        forward method for SpectralGraphLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if pred.dim() == 4:
            B, C, H, W = pred.shape
            if self.patch_size:
                p = self.patch_size
                # Ensure H, W are divisible by p
                if H % p != 0 or W % p != 0:
                    import torch.nn.functional as F

                    new_h = ((H + p - 1) // p) * p
                    new_w = ((W + p - 1) // p) * p
                    pred = F.interpolate(
                        pred, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                    H, W = new_h, new_w

                x_patches = pred.unfold(2, p, p).unfold(3, p, p)
                h_dim, w_dim = x_patches.shape[2], x_patches.shape[3]
                num_patches = h_dim * w_dim
                x_flat = x_patches.permute(0, 2, 3, 1, 4, 5).reshape(
                    B, num_patches, -1
                )  # [B, N, D]
            else:
                # Flatten spatial dims
                x_flat = pred.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        else:
            # Already node features
            B, N, C = pred.shape
            x_flat = pred

        # [FIX] Handle complex input (cdist only supports float)
        if torch.is_complex(x_flat):
            x_flat = torch.abs(x_flat)

        x = x_flat
        k = self.k
        N = x.shape[1]

        # [FIX] Memory-efficient Dirichlet Energy computation
        # Instead of B x N x N distance matrix (32GiB for N=65536),
        # we process in blocks of rows.

        if N <= 4096:
            # Fast path for small graphs
            dist_sq = torch.cdist(x, x, p=2).pow(2)
            val, _ = torch.topk(dist_sq, k=min(k + 1, N), largest=False, dim=-1)
            dirichlet_energy = val[:, :, 1:].sum() / B
        else:
            # Memory-efficient path for large graphs
            total_energy = 0.0
            block_size = 1024
            for i in range(0, N, block_size):
                end_i = min(i + block_size, N)
                x_block = x[:, i:end_i, :]  # [B, block_size, D]

                # Compute distances from this block to ALL points
                # Still [B, block_size, N] which is much smaller
                dists_sq = torch.cdist(x_block, x, p=2).pow(2)

                # Find k+1 nearest neighbors for these points
                val, _ = torch.topk(dists_sq, k=min(k + 1, N), largest=False, dim=-1)

                # Sum distances (excluding self at index 0)
                total_energy = total_energy + val[:, :, 1:].sum()

            dirichlet_energy = total_energy / B

        return self.lambda_val * dirichlet_energy
