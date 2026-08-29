"""SToRM: Smoothness Regularization on Manifolds for Dynamic MRI.

Implements manifold-based regularization for dynamic MRI reconstruction.
The key insight is that temporal frames lie on a low-dimensional manifold,
and we can learn this manifold structure from navigator data.

Reference:
- Poddar et al. "Dynamic MRI Using SmooThness Regularization on Manifolds (SToRM)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_navigator_similarity(
    navigators: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Compute pairwise similarity matrix from navigators.

    Navigators are low-resolution k-space center measurements
    that encode respiratory/cardiac phase.

    Args:
        navigators: [T, D] navigator signals for T frames
        sigma: Gaussian kernel bandwidth

    Returns:
        W: [T, T] affinity matrix
    """
    # Pairwise Euclidean distance
    dist_sq = torch.cdist(navigators, navigators, p=2) ** 2

    # Gaussian kernel
    W = torch.exp(-dist_sq / (2 * sigma**2))

    # Zero diagonal (no self-loops)
    W = W - torch.diag(torch.diag(W))

    return W


def compute_graph_laplacian(W: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """Compute graph Laplacian from affinity matrix.

    Args:
        W: [N, N] affinity/adjacency matrix
        normalized: If True, use normalized Laplacian L = I - D^{-1/2} W D^{-1/2}

    Returns:
        L: [N, N] graph Laplacian
    """
    # Degree matrix
    D = W.sum(dim=1)

    if normalized:
        # Normalized Laplacian
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(D) + 1e-8))
        L = torch.eye(W.shape[0], device=W.device) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        # Unnormalized Laplacian
        L = torch.diag(D) - W

    return L


class ManifoldLaplacian(nn.Module):
    """Learnable Manifold Laplacian for Dynamic MRI.

    Learns the graph structure from navigator or low-resolution data.

    Args:
        num_frames: Number of temporal frames
        nav_dim: Navigator dimension
        sigma: Initial Gaussian kernel width
        learnable_sigma: Whether to learn sigma
    """

    def __init__(
        self,
        num_frames: int,
        nav_dim: int = 64,
        sigma: float = 1.0,
        learnable_sigma: bool = True,
    ):
        """__init__.

        Args:
            num_frames (int): Description.
            nav_dim (int): Description.
            sigma (float): Description.
            learnable_sigma (bool): Description.
        """
        super().__init__()

        self.num_frames = num_frames
        self.nav_dim = nav_dim

        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(sigma).log())
        else:
            self.register_buffer("log_sigma", torch.tensor(sigma).log())

        # Optional navigator encoder
        self.nav_encoder = nn.Sequential(
            nn.Linear(nav_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    @property
    def sigma(self) -> float:
        """sigma.

        Returns:
            float: Description.
        """
        return self.log_sigma.exp()

    def forward(
        self,
        navigators: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Laplacian from navigators.

        Args:
            navigators: [B, T, D] batch of navigator sequences

        Returns:
            L: [B, T, T] batch of Laplacians
            W: [B, T, T] batch of affinity matrices

        forward method for ManifoldLaplacian.

        Executes PyTorch tensor operations.

        Args:
            navigators (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, T, D = navigators.shape

        # Encode navigators
        nav_encoded = self.nav_encoder(navigators)  # [B, T, 64]

        # Compute affinity per batch
        Ws = []
        Ls = []

        for b in range(B):
            W = compute_navigator_similarity(nav_encoded[b], self.sigma.item())
            L = compute_graph_laplacian(W, normalized=True)
            Ws.append(W)
            Ls.append(L)

        return torch.stack(Ls), torch.stack(Ws)


# ``SToRMRegularizer`` was moved to ``mriforge.models.losses.storm_loss``
# during the Phase 5 canonical-home migration (pitfall #12: every
# ``@register_loss`` decorator must live under ``models/losses/``). It is
# re-exported below from this module's package init for any
# pre-existing callers that import it from the legacy path.


class GraphImagePrior(nn.Module):
    """Graph Image Prior (GIP) for Unsupervised Dynamic MRI.

    Combines manifold discovery with deep image prior for
    unsupervised dynamic MRI reconstruction.

    Architecture:
    - Image Recovery: CNN generators for each frame (weight-shared)
    - Manifold Discovery: GCN learns graph structure

    Args:
        in_channels: Number of input channels
        num_frames: Number of temporal frames
        hidden_dim: Hidden dimension for generators
        gcn_layers: Number of GCN layers
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_frames: int = 20,
        hidden_dim: int = 64,
        gcn_layers: int = 3,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_frames (int): Description.
            hidden_dim (int): Description.
            gcn_layers (int): Description.
        """
        super().__init__()

        self.num_frames = num_frames

        # Image generator (Deep Image Prior style)
        # Shared across frames but with frame-specific conditioning
        self.frame_embed = nn.Embedding(num_frames, hidden_dim)

        self.generator = nn.Sequential(
            nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
        )

        # Graph Convolutional Layers for manifold discovery
        self.gcn_layers = nn.ModuleList()
        for _ in range(gcn_layers):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Adjacency learner (from frame features)
        self.adj_learner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def learn_adjacency(
        self,
        frame_features: torch.Tensor,
    ) -> torch.Tensor:
        """Learn adjacency matrix from frame features.

        Args:
            frame_features: [B, T, D] frame-level features

        Returns:
            A: [B, T, T] learned adjacency matrix
        """
        B, T, D = frame_features.shape

        # Pairwise concatenation
        fi = frame_features.unsqueeze(2).expand(-1, -1, T, -1)  # [B, T, T, D]
        fj = frame_features.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, T, D]
        fij = torch.cat([fi, fj], dim=-1)  # [B, T, T, 2D]

        # Learn edge weights
        A = self.adj_learner(fij).squeeze(-1)  # [B, T, T]

        # Symmetrize
        A = (A + A.transpose(1, 2)) / 2

        return A

    def gcn_forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Graph convolution on frame features.

        Args:
            X: [B, T, D] node features
            A: [B, T, T] adjacency matrix

        Returns:
            Updated features
        """
        # Normalize adjacency
        D = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_norm = A / D

        H = X
        for gcn in self.gcn_layers:
            H = torch.bmm(A_norm, H)  # Message passing
            H = gcn(H)  # Transform
            H = F.relu(H)

        return H

    def forward(
        self,
        z: torch.Tensor,
        return_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Generate dynamic sequence with manifold constraints.

        Args:
            z: [B, C, H, W] random noise input (like Deep Image Prior)
            return_graph: Whether to return learned adjacency

        Returns:
            X: [B, T, C, H, W] generated dynamic sequence
            A: [B, T, T] learned adjacency (if return_graph)

        forward method for GraphImagePrior.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_graph (bool): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = z.shape
        device = z.device

        # Get frame embeddings
        frame_idx = torch.arange(self.num_frames, device=device)
        frame_emb = self.frame_embed(frame_idx)  # [T, D]

        # Learn adjacency from embeddings
        frame_emb_batch = frame_emb.unsqueeze(0).expand(B, -1, -1)  # [B, T, D]
        A = self.learn_adjacency(frame_emb_batch)

        # Graph convolution on embeddings
        refined_emb = self.gcn_forward(frame_emb_batch, A)  # [B, T, D]

        # Generate each frame
        frames = []
        for t in range(self.num_frames):
            # Add frame embedding to noise
            emb_t = refined_emb[:, t, :, None, None].expand(-1, -1, H, W)
            z_cond = torch.cat([z, emb_t], dim=1)

            frame = self.generator(z_cond)
            frames.append(frame)

        X = torch.stack(frames, dim=1)  # [B, T, C, H, W]

        if return_graph:
            return X, A
        return X


__all__ = [
    "GraphImagePrior",
    "ManifoldLaplacian",
    "compute_graph_laplacian",
    "compute_navigator_similarity",
]
