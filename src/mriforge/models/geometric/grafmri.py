import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-NN graph from point features.
    Args:
        x: [B, C, N] Point features
        k: Number of nearest neighbors
    Returns:
        idx: [B, N, k] Indices of neighbors
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # [B, N, k]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor = None) -> torch.Tensor:
    """Get edge features."""
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_graph(x, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # [B*N*k, C]
    feature = x.transpose(2, 1).contiguous().view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # x: [B, N, 1, C]
    x = x.transpose(2, 1).contiguous().view(batch_size, num_points, 1, num_dims)
    # feature: [B, N, k, C*2] -> (x_j - x_i, x_i)
    feature = torch.cat((feature - x, x.repeat(1, 1, k, 1)), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()  # [B, 2C, N, k]


class DynamicEdgeConv(nn.Module):
    """DynamicEdgeConv class."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            k (int): Description.
        """
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, N]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DynamicEdgeConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_graph = get_graph_feature(x, k=self.k)  # [B, 2C, N, k]
        x_out = self.conv(x_graph)  # [B, OutC, N, k]
        return x_out.max(dim=-1, keepdim=False)[0]  # [B, OutC, N]


@register_model(name="grafmri", training_mode="reconstruction")
class GraFMRIGenerator(nn.Module, IGenerator):
    """
    GraFMRI: Graph-based Feature MRI Reconstruction.

    Divides the image into patches, treats each patch as a node in a graph,
    and learns non-local relationships for denoising/restoration using Dynamic Graph CNNs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 4,
        hidden_dim: int = 64,
        k: int = 20,
        layers: int = 3,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            patch_size (int): Description.
            hidden_dim (int): Description.
            k (int): Description.
            layers (int): Description.
        """
        super().__init__()
        self.patch_size = patch_size
        self.k = k

        # Patch vector dim = C * P * P
        self.input_dim = in_channels * patch_size * patch_size
        self.out_dim = out_channels * patch_size * patch_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        # GNN Layers
        self.gnn_layers = nn.ModuleList()
        curr_dim = hidden_dim
        for _ in range(layers):
            self.gnn_layers.append(DynamicEdgeConv(curr_dim, curr_dim, k=k))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(curr_dim, self.out_dim),
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "GraFMRI"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]

        forward method for GraFMRIGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        P = self.patch_size

        # 1. Unfold to patches
        # [B, C*P*P, L], L = (H//P)*(W//P) if divisible and no overlap (simplified)
        # Using unfold with stride=patch_size (non-overlapping)
        # For reconstruction, overlapping patches + fold average is better, but start simple.

        # [B, C, H, W] -> [B, C*P*P, N_patches]
        patches = F.unfold(x, kernel_size=P, stride=P)
        B, D, N = patches.shape

        # 2. Graph Processing
        # Treat patches as point cloud [B, D, N]

        # Embedding: [B, D, N] -> [B, Hidden, N]
        # Linear expects [B, N, D]
        x_feat = patches.transpose(1, 2).reshape(B * N, D)
        x_feat = self.encoder(x_feat)
        x_feat = x_feat.view(B, N, -1).transpose(1, 2)  # [B, Hidden, N]

        # Dynamic Edge Convs (recomputes graph each layer)
        for layer in self.gnn_layers:
            x_feat = layer(x_feat) + x_feat  # Residual connection

        # 3. Decode
        # [B, Hidden, N] -> [B, OutD, N]
        x_out = x_feat.transpose(1, 2).reshape(B * N, -1)
        x_out = self.decoder(x_out)
        x_out = x_out.view(B, N, -1).transpose(1, 2)  # [B, OutD, N]

        # 4. Fold back
        # [B, OutD, N] -> [B, C_out, H, W]
        # output_size should match x if C_out matches C_in logic
        out = F.fold(x_out, output_size=(H, W), kernel_size=P, stride=P)

        return out

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from input tensor x."""
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())
