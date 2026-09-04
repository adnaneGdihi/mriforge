import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


class TensorVM(nn.Module):
    """
    Vector-Matrix (VM) decomposition for TensoRF-style representation.
    Represents a 3D tensor as a sum of outer products of vectors and matrices.
    """

    def __init__(self, dim: list[int], rank: int = 16, init_scale: float = 0.1):
        """__init__.

        Args:
            dim (list[int]): Description.
            rank (int): Description.
            init_scale (float): Description.
        """
        super().__init__()
        self.dim = dim  # [D, H, W]
        self.rank = rank

        # Define shapes for VM decomposition
        D, H, W = dim
        self.shapes = {
            "mat_xy": (1, rank, H, W),
            "mat_xz": (1, rank, D, W),
            "mat_yz": (1, rank, D, H),
            "vec_z": (1, rank, D, 1),
            "vec_y": (1, rank, H, 1),
            "vec_x": (1, rank, W, 1),
        }

    def get_output_shapes(self) -> dict:
        """Return the shapes of parameters required by this VM decomposition."""
        return self.shapes

    @staticmethod
    def compute_density(
        xyz_sampled: torch.Tensor,
        mat_xy: torch.Tensor,
        mat_xz: torch.Tensor,
        mat_yz: torch.Tensor,
        vec_z: torch.Tensor,
        vec_y: torch.Tensor,
        vec_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute density/intensity at sampled coordinates using VM decomposition.

        Args:
            xyz_sampled: [B, 3] coordinates in [-1, 1]
            mat_xy: [1, R, H, W]
            mat_xz: [1, R, D, W]
            mat_yz: [1, R, D, H]
            vec_z:  [1, R, D, 1]
            vec_y:  [1, R, 1, H]
            vec_x:  [1, R, 1, 1] (Wait, vectors correspond to the orthogonal dimension)

            Actually for VM:
            V(x,y,z) = sum_r ( Matrix_XY_r(x,y) * Vector_Z_r(z) ) + ...
        """
        # Coordinate normalization from [-1, 1] to [0, 1] equivalent for grid_sample?
        # grid_sample uses [-1, 1].

        # Project coordinates to planes
        # xyz_sampled is [B, 3] -> (x, y, z)
        # We need (x,y) for XY plane, (x,z) for XZ, (y,z) for YZ.

        # Reshape for grid sample: [1, 1, B, 2]
        B = xyz_sampled.shape[0]
        pts = xyz_sampled.view(1, 1, B, 3)

        # XY plane (use x, y)
        coord_xy = pts[..., [0, 1]]
        # XZ plane (use x, z)
        coord_xz = pts[..., [0, 2]]
        # YZ plane (use y, z)
        coord_yz = pts[..., [1, 2]]

        # Sample Matrices
        # mat_xy: [1, R, H, W]
        feat_xy = F.grid_sample(mat_xy, coord_xy, align_corners=True).view(-1, B)  # [R, B]
        feat_xz = F.grid_sample(mat_xz, coord_xz, align_corners=True).view(-1, B)
        feat_yz = F.grid_sample(mat_yz, coord_yz, align_corners=True).view(-1, B)

        # Sample Vectors
        # Vectors are 1D. We can treat them as 1xWx1 or similar or just grid sample 1D?
        # F.grid_sample is 4D/5D. Let's strictly use 4D input [1, R, 1, D] and sample.

        # z coordinate for vec_z
        coord_z = torch.stack(
            [torch.zeros_like(pts[..., 2]), pts[..., 2]], dim=-1
        )  # [1,1,B,2] (dummy y)
        feat_z = F.grid_sample(vec_z, coord_z, align_corners=True).view(-1, B)

        coord_y = torch.stack([torch.zeros_like(pts[..., 1]), pts[..., 1]], dim=-1)
        feat_y = F.grid_sample(vec_y, coord_y, align_corners=True).view(-1, B)

        coord_x = torch.stack([torch.zeros_like(pts[..., 0]), pts[..., 0]], dim=-1)
        feat_x = F.grid_sample(vec_x, coord_x, align_corners=True).view(-1, B)

        # VM Product
        # [R, B] * [R, B] -> [R, B] -> sum over R -> [B]
        res_xy = (feat_xy * feat_z).sum(dim=0)
        res_xz = (feat_xz * feat_y).sum(dim=0)
        res_yz = (feat_yz * feat_x).sum(dim=0)

        density = res_xy + res_xz + res_yz
        return density  # [B]


class GridGenerator(nn.Module):
    """
    MLP that generates the parameters for the TensorVM grid.
    Feature Grid = G(z; theta)
    """

    def __init__(self, output_shapes: dict, latent_dim: int = 64, hidden_dim: int = 128):
        """__init__.

        Args:
            output_shapes (dict): Description.
            latent_dim (int): Description.
            hidden_dim (int): Description.
        """
        super().__init__()
        self.output_shapes = output_shapes
        self.latent_dim = latent_dim

        # Calculate total parameters needed
        self.total_params = 0
        self.param_splits = []
        self.param_names = []

        # Order: mat_xy, mat_xz, mat_yz, vec_z, vec_y, vec_x
        keys = ["mat_xy", "mat_xz", "mat_yz", "vec_z", "vec_y", "vec_x"]
        for k in keys:
            shape = output_shapes[k]
            size = int(np.prod(shape))
            self.param_splits.append(size)
            self.param_names.append(k)
            self.total_params += size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_params),
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [1, latent_dim] noise vector
        Returns:
            dict of tensors matching output_shapes

        forward method for GridGenerator.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        raw_out = self.net(z)  # [1, total_params]

        # Split
        parts = torch.split(raw_out, self.param_splits, dim=1)

        outputs = {}
        for name, part, output_shape in zip(
            self.param_names,
            parts,
            [self.output_shapes[k] for k in self.param_names],
            strict=False,
        ):
            outputs[name] = part.view(*output_shape)

        return outputs


@register_model(name="zerorf", training_mode="reconstruction")
class ZeroRFReconstructor(nn.Module):
    """ZeroRF-style implicit neural representation for MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        volume_shape: tuple[int, int, int] | None = None,
        rank: int = 16,
        latent_dim: int = 64,
        **kwargs,
    ):
        """Initialize ZeroRF reconstructor.

        Args:
            in_channels: Input channels (for factory compatibility).
            out_channels: Output channels (for factory compatibility).
            volume_shape: 3D volume shape (D, H, W). Defaults to (1, 256, 256).
            rank: TensorVM rank.
            latent_dim: Latent code dimension.
        """
        super().__init__()
        if volume_shape is None:
            volume_shape = (1, 256, 256)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.volume_shape = volume_shape
        D, H, W = volume_shape

        # Initialize TensorVM definition
        self.vm = TensorVM(dim=[D, H, W], rank=rank)
        self.shapes = self.vm.get_output_shapes()

        self.generator = GridGenerator(self.shapes, latent_dim=latent_dim)
        self.latent_dim = latent_dim

        # Learnable latent code
        self.z_code = nn.Parameter(torch.randn(1, latent_dim))

    def forward(self, x: torch.Tensor | None = None, num_points: int = 10000, **kwargs):
        """Forward pass.

        Args:
            x: Optional input tensor (B, C, H, W). If provided, output matches spatial dims.
            num_points: Number of points to sample (unused when x is provided).

        Returns:
            Reconstructed volume or image.
        """
        # Generate grid parameters
        params = self.generator(self.z_code)
        vol = self.reconstruct_volume(params)

        if x is not None and x.ndim == 4:
            # Reshape to match input spatial dims: (B, out_channels, H, W)
            B, C, H, W = x.shape
            # reconstruct_volume returns [D, H, W]; pick the middle depth slice so
            # D>1 volumes do not break the [1,1,H,W] reshape (D==1 is unchanged).
            vol_2d = vol[vol.shape[0] // 2].reshape(1, 1, vol.shape[-2], vol.shape[-1])
            if vol_2d.shape[-2:] != (H, W):
                vol_2d = F.interpolate(vol_2d, size=(H, W), mode="bilinear", align_corners=False)
            return vol_2d.expand(B, self.out_channels, H, W)

        return vol

    def reconstruct_volume(self, params: dict) -> torch.Tensor:
        """
        Reconstruct the full dense volume from parameters.
        """
        D, H, W = self.volume_shape

        # Create full coordinate grid
        # meshgrid 'ij' indexing: z, y, x
        lz = torch.linspace(-1, 1, D)
        ly = torch.linspace(-1, 1, H)
        lx = torch.linspace(-1, 1, W)

        grid_z, grid_y, grid_x = torch.meshgrid(lz, ly, lx, indexing="ij")

        # Stack: [D, H, W, 3] -> (x, y, z) for grid_sample usually expects x,y,z order (W, H, D)
        # grid_sample coordinates: (x, y, z).
        # Our grid is z, y, x.
        xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [D, H, W, 3]

        # Flatten for batch processing
        xyz_flat = xyz.reshape(-1, 3).to(self.z_code.device)

        # Compute density in chunks to save memory if needed,
        # but for demo we try full batch
        density = TensorVM.compute_density(
            xyz_flat,
            params["mat_xy"],
            params["mat_xz"],
            params["mat_yz"],
            params["vec_z"],
            params["vec_y"],
            params["vec_x"],
        )

        return density.reshape(D, H, W)
