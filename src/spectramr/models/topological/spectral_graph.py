"""Spectral Graph Transformer: Coil Combination via Graph Topologies.

Innovation XIV (SOTA 2.0): Parallel Imaging via Graph Spectral Theory.

Mathematical Foundation:
    - Coil Graph: Adjacency A_ij based on correlation.
    - Laplacian: L = D - A^-1/2 A D^-1/2 (Normalized).
    - LPE: Eigenvectors of L.
    - Combination: w = Transformer(feat || LPE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


class CoilGraph(nn.Module):
    """Computes Laplacian Positional Encodings from Multicoil Data."""

    def __init__(self, top_k_eigen: int = 8):
        """__init__.

        Args:
            top_k_eigen (int): Description.
        """
        super().__init__()
        self.top_k = top_k_eigen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
           x: [B, C, H, W] Multicoil images (magnitude or complex).

        Returns:
           lpe: [B, C, k] Laplacian Positional Encodings.

        forward method for CoilGraph.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # 1. Compute Correlation Matrix
        # Flatten: [B, C, N]
        if x.is_complex():
            x_flat = torch.view_as_real(x).view(B, C, -1)  # Flatten real/imag components together?
            # Or magnitude? Correlation of signal variations.
            # Let's use magnitude for graph construction stability.
            x_mag = x.abs()
            x_flat = x_mag.view(B, C, -1)  # [B, C, N]
        else:
            x_flat = x.view(B, C, -1)

        # Normalize features
        x_norm = F.normalize(x_flat, p=2, dim=-1)

        # Adj: [B, C, C]
        A = torch.bmm(x_norm, x_norm.transpose(1, 2))

        # Pruning? Or strict fully connected weighted graph.
        # Thresholding to remove noise connections?
        # A = torch.where(A > 0.1, A, torch.zeros_like(A))

        # 2. Laplacian
        # Degree matrix D_ii = sum_j A_ij
        D_vec = A.sum(dim=-1)  # [B, C]
        D_inv_sqrt = torch.pow(D_vec + 1e-6, -0.5)
        D_mat = torch.diag_embed(D_inv_sqrt)  # [B, C, C]

        # Normalized Laplacian: I - D^-1/2 A D^-1/2
        I = torch.eye(C, device=x.device).unsqueeze(0)
        DAD = torch.bmm(torch.bmm(D_mat, A), D_mat)
        L = I - DAD

        # 3. Eigendecomposition
        # L is symmetric (if A is symmetric). A is symmetric.
        # torch.linalg.eigh is for Hermitian/Symmetric
        vals, vecs = torch.linalg.eigh(L)
        # vals are sorted ascending.
        # vecs: [B, C, C] columns are eigenvectors.

        # Use first k non-zero (smallest eigenvalues -> low freq geometry)
        # Skip 0 (first one is usually constant for connected graph)
        # But we use top_k lowest.

        k = min(self.top_k, C)
        lpe = vecs[:, :, 1 : k + 1]  # [B, C, k]

        # Loop padding if C < k+1?
        if lpe.shape[2] < self.top_k:
            pad = self.top_k - lpe.shape[2]
            lpe = F.pad(lpe, (0, pad))

        return lpe


@register_model(name="spectral_graph", training_mode="reconstruction")
class SpectralGraphTransformer(nn.Module):
    """Transformer for learning coil combination weights."""

    def __init__(self, in_channels=2, embed_dim=32, num_heads=4, num_layers=2, top_k_eigen=8):
        """__init__.

        Args:
            in_channels (Any): Description.
            embed_dim (Any): Description.
            num_heads (Any): Description.
            num_layers (Any): Description.
            top_k_eigen (Any): Description.
        """
        super().__init__()
        self.coil_graph = CoilGraph(top_k_eigen)

        # Feature embedding: Maps image patch/downsampled to embed_dim
        # We process spatially pooled features for global weights?
        # Or predict spatial maps.
        # Let's predict one weight per coil (Global) for simplicity in this demo,
        # or low-res spatial maps.

        # Let's do Global weights first (Adaptively learned)

        self.feat_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, embed_dim - top_k_eigen),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),  # Weights usually positive? Or complex combination?
            # "Adaptive Combination" usually essentially PCA or matched filter.
            # Let's output magnitude weights.
            # Phase correction is tricky without spatial maps.
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, 2, H, W] (Complex split) or [B, C, H, W] (Magnitude)

        Returns:
            combined: [B, 2, H, W]

        forward method for SpectralGraphTransformer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.ndim == 5:  # [B, C, 2, H, W]
            input_feat = x.view(
                x.shape[0], x.shape[1], -1, x.shape[-2], x.shape[-1]
            )  # [B, C, 2, H, W] -> treat 2 as channel
            # Actually adaptive pool expects [Batch*C, 2, H, W]
            B, C, Ch, H, W = x.shape
            input_flat = x.view(B * C, Ch, H, W)
        else:  # [B, C, H, W] Magnitude
            B, C, H, W = x.shape
            input_flat = x.view(B * C, 1, H, W)
            x = x.unsqueeze(2)  # Make 5D for consistency

        # 1. Embed Features
        feat = self.feat_embed(input_flat)  # [B*C, Emb-K]
        feat = feat.view(B, C, -1)

        # 2. Get LPE
        # Use magnitude for graph
        x_mag = x.norm(dim=2)  # [B, C, H, W]
        lpe = self.coil_graph(x_mag)  # [B, C, K]

        # 3. Concatenate
        token = torch.cat([feat, lpe], dim=-1)  # [B, C, Emb]

        # 4. Transformer
        # Latents: [B, C, Emb]
        latents = self.transformer(token)

        # 5. Predict Weights
        weights = self.head(latents)  # [B, C, 1]

        # Normalize weights? sum to 1?
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        # 6. Combiner
        # x is [B, C, 2, H, W]
        weights_exp = weights.view(B, C, 1, 1, 1)

        combined = (x * weights_exp).sum(dim=1)  # [B, 2, H, W]

        return combined


def create_spectral_graph() -> SpectralGraphTransformer:
    """create_spectral_graph.

    Returns:
        SpectralGraphTransformer: Description.
    """
    return SpectralGraphTransformer()
