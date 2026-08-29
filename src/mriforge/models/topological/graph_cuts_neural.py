import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model
from mriforge.models.topological.graph_kan import GraphKANLayer

# Differentiable Graph Module (DGM) / Vision GNN Logic


class GraphConstruct(nn.Module):
    """GraphConstruct class."""

    def __init__(self, k=8, distance="euclidean"):
        """__init__.

        Args:
            k (Any): Description.
            distance (Any): Description.
        """
        super().__init__()
        self.k = k
        self.distance = distance

    def forward(self, x):
        # x: [B, N, D]
        # Compute pairwise distance
        # Return k-NN edge index
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for GraphConstruct.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.distance == "euclidean":
            dist = torch.cdist(x, x)
        elif self.distance == "cosine":
            x_norm = F.normalize(x, dim=-1)
            dist = 1 - torch.matmul(x_norm, x_norm.transpose(1, 2))
        else:
            raise ValueError(
                f"GraphConstruct: unknown distance {self.distance!r}; "
                "expected 'euclidean' or 'cosine'"
            )

        _, indices = torch.topk(dist, k=self.k + 1, largest=False)
        return indices[:, :, 1:]  # [B, N, k] Exclude self


class DifferentiableGraphModule(nn.Module):
    """
    Learns to construct the graph dynamically and perform message passing.
    """

    def __init__(self, in_dim, out_dim, k=8):
        """__init__.

        Args:
            in_dim (Any): Description.
            out_dim (Any): Description.
            k (Any): Description.
        """
        super().__init__()
        self.k = k
        self.graph_construct = GraphConstruct(k=k)
        # Using a simple MLP or KAN for edge features / update?
        # Reusing GraphKANLayer logic which uses KAN for message passing.
        self.gnn = GraphKANLayer(in_dim, out_dim, k=k)

    def forward(self, x):
        # x: [B, N, D]
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for DifferentiableGraphModule.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        edge_index = self.graph_construct(x)
        # We need positions for GraphKANLayer, but here we treat features as positions for graph construction
        # and also as features for message passing.
        # GraphKANLayer expects (h, pos). Let's pass x as both.
        out = self.gnn(x, x, edge_index=edge_index)
        return out


@register_model(
    name="graph_cuts_neural",
    training_mode="reconstruction",
    # Measured, not inferred from the name (#1106). ``forward`` opens with
    # ``B, C, H, W = x.shape``, so a 5-D input raises on the unpack.
    spatial_dims=(2,),
    # The class docstring says "on K-Space Patches", but nothing in the forward is
    # k-space specific -- it patchifies, embeds with a ``Linear``, message-passes
    # and unpatchifies, with no FFT on any path. So the domain is whatever the arm
    # feeds, and both arms feed image: the inprogress arm declares
    # input_domain/output_domain: image over ``coils.processing_mode: rss_image``
    # (IFFT inside the dataset pipeline, so ``dataset_type: kspace`` describes the
    # FILES), and the active sibling is ``dataset_type: image`` outright.
    input_domain="image",
    output_domain="image",
    # Measured: complex input raises "mat1 and mat2 must have the same dtype" at
    # the embedding ``Linear``. Real/imag-interleaved channels are the route.
    accepts_complex=False,
    requires_paired_data=True,
)
class GraphCutsNeural(nn.Module):
    """
    Graph-based MRI Reconstruction using Differentiable Graph Modules on K-Space Patches.
    Mimics 'Graph Cuts' via differentiable energy minimization on a learned graph.
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        features=[32, 64, 128],
        patch_size=16,
        k=8,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            features (Any): Description.
            patch_size (Any): Description.
            k (Any): Description.
        """
        super().__init__()
        self.patch_size = patch_size
        self.k = k
        self.embedding = nn.Linear(in_channels * patch_size * patch_size, features[0])

        layers = []
        in_dim = features[0]
        for f in features:
            layers.append(DifferentiableGraphModule(in_dim, f, k=k))
            in_dim = f

        self.backbone = nn.Sequential(*layers)
        self.readout = nn.Linear(features[-1], out_channels * patch_size * patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for GraphCutsNeural.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        # Patchify
        p = self.patch_size
        x_patches = x.unfold(2, p, p).unfold(3, p, p)
        # [B, C, H/p, W/p, p, p]
        h_dim, w_dim = x_patches.shape[2], x_patches.shape[3]
        num_patches = h_dim * w_dim

        x_flat = x_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, -1)
        # [B, N, C*p*p]

        # Graph Processing
        h = self.embedding(x_flat)
        h = self.backbone(h)
        out_flat = self.readout(h)

        # Unpatchify
        out_patches = out_flat.view(B, h_dim, w_dim, C, p, p)
        out_patches = out_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out_patches.view(B, C, H, W)

        if self.training:
            # Return features/graph for spectral loss calculation?
            # Or just output. Config enables lambda_graph_consistency/spectral_kspace.
            return out
        return out
