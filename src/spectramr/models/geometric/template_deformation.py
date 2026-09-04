"""Template Deformation Module using Graph Convolutional Networks (GCN).

This module implements PrIntMesh-style deformation.
It deforms a spherical or anatomical template mesh to match target anatomy
based on image features.
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from spectramr.models.registry import register_model


class GraphConv(nn.Module):
    """Simple Graph Convolution Layer (Kipf & Welling)."""

    def __init__(self, in_features: int, out_features: int):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(
        self, x: Float[Tensor, "B V Cin"], adj: Float[Tensor, "V V"]
    ) -> Float[Tensor, "B V Cout"]:
        """Forward pass.

        Args:
            x: Node features.
            adj: Normalized adjacency matrix (A_hat).

        forward method for GraphConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            adj (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B V Cout']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # AXW + b
        # Force broadcast of adj over batch
        # x: [B, V, C]
        # adj: [V, V]

        # Support = XW
        support = self.linear(x)

        # Output = A * Support
        # [V, V] @ [B, V, Cout] -> [B, V, Cout]
        # We need to broadcast matrix multiply
        output = torch.matmul(adj, support)

        return output + self.bias


@register_model(name="template_deformer", training_mode="reconstruction")
class TemplateDeformer(nn.Module):
    """Deforms a template mesh based on image features."""

    def __init__(
        self,
        template_vertices: Float[Tensor, "V 3"],
        template_faces: Int[Tensor, "F 3"],
        image_encoder_dims: list[int] = [16, 32, 64],
        hidden_dim: int = 128,
    ):
        """__init__.

        Args:
            template_vertices (Float[Tensor, 'V 3']): Description.
            template_faces (Int[Tensor, 'F 3']): Description.
            image_encoder_dims (list[int]): Description.
            hidden_dim (int): Description.
        """
        super().__init__()
        self.register_buffer("template_vertices", template_vertices)
        self.register_buffer("template_faces", template_faces)

        # Build Adjacency Matrix
        self.register_buffer("adj", self._build_adjacency(template_vertices, template_faces))

        # Image Encoder (Simple 3D CNN)
        layers = []
        in_ch = 1
        for dim in image_encoder_dims:
            layers.append(nn.Conv3d(in_ch, dim, kernel_size=3, padding=1, stride=2))
            layers.append(nn.BatchNorm3d(dim))
            layers.append(nn.ReLU(inplace=True))
            in_ch = dim
        self.encoder = nn.Sequential(*layers)

        # Final encoder mapping to global feature
        self.to_global = nn.AdaptiveAvgPool3d(1)
        self.feature_proj = nn.Linear(image_encoder_dims[-1], hidden_dim)

        # Graph Decoder
        # We concat global feature to every node
        self.gc1 = GraphConv(3 + hidden_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.gc3 = GraphConv(hidden_dim, 3)  # Output: delta coordinates

        self.act = nn.ReLU(inplace=True)

    def _build_adjacency(self, vertices: Tensor, faces: Tensor) -> Tensor:
        """_build_adjacency.

        Args:
            vertices (Tensor): Description.
            faces (Tensor): Description.
        Returns:
            Tensor: Description.
        """
        V = vertices.shape[0]
        adj = torch.eye(V, device=vertices.device)

        # Fill edges
        # faces: [F, 3]
        # Edges: (0,1), (1,2), (2,0)
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)

        # Symmetrize
        # Use sparse tensor construction for memory efficiency in init,
        # but dense for multiplication speed if V is small (<5000).
        # Assuming Med meshes ~2k vertices.

        # Simply fill dense for now
        i = edges[:, 0]
        j = edges[:, 1]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

        # Normalize: D^-0.5 * A * D^-0.5
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        diag_inv = torch.diag(deg_inv_sqrt)

        adj_norm = torch.mm(torch.mm(diag_inv, adj), diag_inv)
        return adj_norm

    def forward(self, image: Float[Tensor, "B 1 D H W"]) -> Float[Tensor, "B V 3"]:
        """Predict deformed mesh vertices.

        forward method for TemplateDeformer.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B V 3']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = image.shape[0]
        V = self.template_vertices.shape[0]

        # 1. Encode Image
        features = self.encoder(image)
        global_feat = self.to_global(features).view(B, -1)
        global_feat = self.feature_proj(global_feat)  # [B, Hidden]

        # 2. Prepare Graph Input
        # Expand template to batch
        # [B, V, 3]
        x = self.template_vertices.unsqueeze(0).expand(B, -1, -1)

        # Expand global feature to nodes
        # [B, V, Hidden]
        feat_expanded = global_feat.unsqueeze(1).expand(-1, V, -1)

        # Concat: [B, V, 3+Hidden]
        x_in = torch.cat([x, feat_expanded], dim=-1)

        # 3. Graph Convolution
        h = self.act(self.gc1(x_in, self.adj))
        h = self.act(self.gc2(h, self.adj))

        # Predict Displacement
        delta = self.gc3(h, self.adj)

        # Constraint to small deformations?
        # Usually standard GCN is fine.

        return x + delta
