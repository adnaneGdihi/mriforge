"""Differentiable Slicing Module using Generalized Winding Numbers.

This module allows for differentiable voxelization and slicing of 3D meshes.
It computes the occupancy of a point in space relative to a mesh using the
Generalized Winding Number (GWN), which is an analytic, differentiable
measure of "insideness".

Reference:
    Jacobson et al., "Robust Inside-Outside Segmentation using Generalized Winding Numbers", 2013.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spectramr.models.registry import register_model


@register_model(name="differentiable_slicer", training_mode="reconstruction")
class DifferentiableSlicer(nn.Module):
    """Computes soft occupancy of a mesh on a query grid (slice)."""

    def __init__(self, sigma: float = 1.0):
        """__init__.

        Args:
            sigma (float): Description.
        """
        super().__init__()
        self.sigma = sigma  # Smoothing parameter for the sigmoid (if used for binary prob)

    def forward(
        self,
        vertices: Float[Tensor, "B V 3"],
        faces: Int[Tensor, "F 3"],
        query_points: Float[Tensor, "B H W 3"],
    ) -> Float[Tensor, "B H W"]:
        """Compute occupancy at query points.

        Args:
            vertices: Mesh vertices.
            faces: Mesh faces (indices). Assumed constant topology for the batch.
            query_points: 3D coordinates of the slice plane.

        Returns:
            Occupancy values [0, 1] (soft).

        forward method for DifferentiableSlicer.

        Executes PyTorch tensor operations.

        Args:
            vertices (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            faces (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            query_points (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B H W']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Calculate Winding Number
        # W(p) = 1/4pi * sum_i Omega_i(p)
        # where Omega_i is solid angle of triangle i subtended at p.

        # We need to broadcast carefully.
        # B: Batch
        # Q: Number of query points (H*W)
        # F: Number of faces

        B, H, W, _ = query_points.shape
        F_count = faces.shape[0]

        # Flatten query points: [B, Q, 3]
        flat_queries = query_points.reshape(B, -1, 3)

        # Gather triangle vertices: [B, F, 3, 3]
        # faces is [F, 3], vertices is [B, V, 3]
        # We expand faces to match batch
        batch_faces = faces.expand(B, F_count, 3)

        # Gather operations using advanced indexing is tricky with batch dim.
        # Simple loop over batch or optimized gather.
        # Let's use loop for readability and because B is usually small (1-4) in med imaging.

        occupancies = []

        for b in range(B):
            # Vertices for this batch item: [V, 3]
            v = vertices[b]
            # Triangle vertices: [F, 3, 3]
            tris = v[faces.long()]

            # Points: [Q, 3]
            q = flat_queries[b]

            # Compute Winding Number
            wn = self.compute_winding_number(tris, q)
            occupancies.append(wn)

        occupancy = torch.stack(occupancies, dim=0)
        return occupancy.reshape(B, H, W)

    def compute_winding_number(
        self, tris: Float[Tensor, "F 3 3"], points: Float[Tensor, "Q 3"]
    ) -> Float[Tensor, Q]:
        """Compute generalized winding number for points relative to triangles.

        Uses the arctan formula for solid angle.
        Omega = 2 * atan2( det(a,b,c), al*bl*cl + (a.b)cl + (b.c)al + (c.a)bl )
        where a, b, c are vectors from point to triangle vertices.
        """
        # A, B, C: [F, 3]
        A = tris[:, 0, :]
        B = tris[:, 1, :]
        C = tris[:, 2, :]

        # We need pairwise vectors between ALL points and ALL triangles.
        # Cost: O(Q * F). Can be large.
        # For a slice 256x256 = 65k points. F ~ 5k faces.
        # 300M ops. Doable on GPU.

        # Expand dims for broadcasting:
        # P: [Q, 1, 3]
        P = points.unsqueeze(1)

        # Vectors from point P to vertices A, B, C
        # a, b, c: [Q, F, 3]
        a = A.unsqueeze(0) - P
        b = B.unsqueeze(0) - P
        c = C.unsqueeze(0) - P

        # Lengths
        al = torch.norm(a, dim=-1)
        bl = torch.norm(b, dim=-1)
        cl = torch.norm(c, dim=-1)

        # Dot products
        # [Q, F]
        ab = torch.sum(a * b, dim=-1)
        bc = torch.sum(b * c, dim=-1)
        ca = torch.sum(c * a, dim=-1)

        # Determinant (Scalar Triple Product) -> Signed Volume
        # det = (a x b) . c
        det = torch.sum(torch.cross(a, b, dim=-1) * c, dim=-1)

        # Denominator for atan2
        denom = al * bl * cl + ab * cl + bc * al + ca * bl

        # Solid angle per triangle
        omega = 2.0 * torch.atan2(det, denom)

        # Sum over faces and normalize
        # [Q]
        total_omega = torch.sum(omega, dim=1)
        winding_number = total_omega / (4 * torch.pi)

        # Winding number should be ~1 inside, ~0 outside.
        # It's continuous/approximate for open meshes, exact for closed.

        # Taking abs because orientation might flip sign,
        # though strictly it implies inside/outside direction.
        # For segmentation mask we usually just want existence.
        return torch.abs(winding_number)

    def grid_from_slice_params(
        self,
        center: Tensor,
        normal: Tensor,
        scale: Tensor,
        res_h: int,
        res_w: int,
        device: torch.device,
    ) -> Tensor:
        """Utility to generate query points for a slice plane.

        Args:
            center: [B, 3]
            normal: [B, 3] normalized normal vector
            scale: [B, 2] (scale_h, scale_w) spacing

        Returns:
            grid: [B, H, W, 3]
        """
        # Construct Basis
        # Arbitrary consistent tangent basis from normal
        # ... (simplified for now, implementing generic basis construction) # IMPL
        B = center.shape[0]

        # Create normalized basis
        n = F.normalize(normal, dim=1)

        # Find arbitrary tangent t1
        # If n is close to z-axis (0,0,1), use x-axis. Else z-axis.
        mask = torch.abs(n[:, 2]) > 0.99
        t1_cand = torch.zeros_like(n)
        t1_cand[mask, 0] = 1.0  # x-axis
        t1_cand[~mask, 2] = 1.0  # z-axis

        # Project t1 onto plane and normalize
        # t1 = t1 - (t1 . n) * n
        dot = torch.sum(t1_cand * n, dim=1, keepdim=True)
        t1 = t1_cand - dot * n
        t1 = F.normalize(t1, dim=1)

        # t2 = n x t1
        t2 = torch.cross(n, t1, dim=1)

        # Grid generation
        # [-H/2, H/2]
        ys = torch.linspace(-res_h / 2, res_h / 2, res_h, device=device)
        xs = torch.linspace(-res_w / 2, res_w / 2, res_w, device=device)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # [B, H, W, 1] * [B, 1, 1, 3]
        # plane_p = center + dy * scale_h * t1 + dx * scale_w * t2

        # Expand dims [B, H, W, 1]
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)

        # Vectors [B, 1, 1, 3]
        c = center.view(B, 1, 1, 3)
        v1 = t1.view(B, 1, 1, 3) * scale[:, 0].view(B, 1, 1, 1)
        v2 = t2.view(B, 1, 1, 3) * scale[:, 1].view(B, 1, 1, 1)

        grid_3d = c + grid_y * v1 + grid_x * v2
        return grid_3d
