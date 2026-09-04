"""TorchIO-compatible Graph Encoding Transform.

Converts grid-based k-space to graph representation for manifold experiments.
Used by GraphUNetGenerator (Experiment 42).
"""

import torch
import torchio as tio

from spectramr.data.transforms.registry import register_transform


@register_transform(
    "graph_encoding",
    produces=("graph_nodes", "graph_h", "graph_w", "graph_k_neighbors"),
    requires=("kspace",),
)
class GraphEncodingTransform(tio.Transform):
    """
    TorchIO Transform that converts grid k-space to graph node features.

    For manifold experiments like Exp 42 (Graph Diffusion), the model
    expects graph tuples (nodes with coordinates) rather than grid tensors.

    This transform:
    1. Flattens the spatial grid (H, W) -> N nodes
    2. Generates normalized k-space coordinates (kx, ky) for each node
    3. Computes density compensation function (DCF)
    4. Outputs node features: [Re, Im, Kx, Ky, DCF]

    Args:
        k_neighbors: Number of k-nearest neighbors for graph construction
        max_nodes: Maximum number of nodes (subsamples if exceeded)
        normalize_coords: Whether to normalize coordinates to [-1, 1]
    """

    def __init__(
        self,
        k_neighbors: int = 8,
        max_nodes: int = 4096,
        normalize_coords: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            k_neighbors (int): Description.
            max_nodes (int): Description.
            normalize_coords (bool): Description.
        """
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors
        self.max_nodes = max_nodes
        self.normalize_coords = normalize_coords

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply graph encoding to kspace image in subject."""
        # Find kspace image
        if "kspace" not in subject:
            return subject  # No kspace to encode

        kspace_image = subject["kspace"]
        tensor = kspace_image.data  # (C, H, W, D)

        # Handle 4D TorchIO format: (C, H, W, D)
        C, H, W, D = tensor.shape

        # Process each slice (depth dimension)
        encoded_slices = []
        for d in range(D):
            slice_2d = tensor[:, :, :, d]  # (C, H, W)
            nodes = self._encode_slice(slice_2d)  # (N, 5)
            encoded_slices.append(nodes)

        # Stack slices: (D, N, 5)
        if len(encoded_slices) > 1:
            graph_data = torch.stack(encoded_slices, dim=0)
        else:
            graph_data = encoded_slices[0].unsqueeze(0)  # (1, N, 5)

        # Store graph data in subject metadata (not as image since shape differs)
        subject["graph_nodes"] = graph_data
        subject["graph_k_neighbors"] = self.k_neighbors
        subject["graph_h"] = H
        subject["graph_w"] = W

        return subject

    def _encode_slice(self, slice_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a 2D k-space slice to graph nodes.

        Args:
            slice_tensor: (C, H, W) k-space slice

        Returns:
            nodes: (N, 5) node features [Re, Im, Kx, Ky, DCF]
        """
        C, H, W = slice_tensor.shape
        N = H * W
        device = slice_tensor.device

        # 1. Flatten signal: (C, H, W) -> (N, C)
        signal = slice_tensor.permute(1, 2, 0).reshape(N, C)

        # Ensure 2 channels (Re, Im)
        if C == 1:
            signal = torch.cat([signal, torch.zeros_like(signal)], dim=-1)
        elif C > 2:
            signal = signal[:, :2]

        # 2. Generate coordinates
        if self.normalize_coords:
            kx = torch.linspace(-1, 1, W, device=device)
            ky = torch.linspace(-1, 1, H, device=device)
        else:
            kx = torch.arange(W, device=device).float()
            ky = torch.arange(H, device=device).float()

        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(N, 2)

        # 3. DCF (uniform for now)
        dcf = torch.ones(N, 1, device=device)

        # 4. Concatenate: [Re, Im, Kx, Ky, DCF]
        nodes = torch.cat([signal, coords, dcf], dim=-1)  # (N, 5)

        # 5. Subsample if too many nodes
        if self.max_nodes < N:
            indices = torch.randperm(N)[: self.max_nodes]
            nodes = nodes[indices]

        return nodes
