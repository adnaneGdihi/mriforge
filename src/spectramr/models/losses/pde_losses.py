import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="helmholtz_pde", aliases=["HelmholtzLoss", "pde"])
class HelmholtzPDELoss(nn.Module):
    r"""Computes the Helmholtz Wave Equation residual loss via analytical 2nd-order derivatives."""

    def __init__(self, k_sq: float = 0.0, **kwargs):
        """__init__.

        Args:
            k_sq (float): Description.
        """
        super().__init__()
        self.k_sq = k_sq

    def forward(self, arg1, arg2=None, coords=None, **kwargs):
        """
        Args:
            arg1: S_real (legacy) or complex pred [B, C, H, W] (CNN).
            arg2: S_imag (legacy) or target [B, C, H, W] (CNN).
            coords: Spatial coordinates for explicit autograd (N, 2).

        Returns:
            Scalar tensor containing the Mean Squared PDE residual.

        forward method for HelmholtzPDELoss.

        Executes PyTorch tensor operations.

        Args:
            arg1 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            arg2 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            coords (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # If arg1 is 4D grid tensor, we compute finite difference Laplacian
        if arg1.dim() == 4:
            pred = arg1
            B, C, H, W = pred.shape

            kernel = torch.tensor(
                [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
                device=pred.device,
                dtype=pred.real.dtype if pred.is_complex() else pred.dtype,
            )

            if pred.is_complex():
                S_real = pred.real
                S_imag = pred.imag
            else:
                S_real = pred
                S_imag = torch.zeros_like(pred)

            total_pde_res = 0.0
            for c in range(C):
                sr = S_real[:, c : c + 1, :, :]
                si = S_imag[:, c : c + 1, :, :]

                lap_r = torch.nn.functional.conv2d(sr, kernel, padding=1)
                lap_i = torch.nn.functional.conv2d(si, kernel, padding=1)

                # Helmholtz Residual: \nabla^2 S + k^2 S = 0
                res_r = lap_r + self.k_sq * sr
                res_i = lap_i + self.k_sq * si

                total_pde_res += torch.mean(res_r**2 + res_i**2)

            return total_pde_res / C

        else:
            # Legacy SIREN path
            S_real = arg1
            S_imag = arg2
            if coords is None:
                coords = kwargs.get("coords")

            assert coords is not None and coords.requires_grad, (
                "coords with requires_grad=True required for autograd PDE"
            )

            num_coils = S_real.shape[1]
            total_pde_res = 0.0

            for c in range(num_coils):
                sr = S_real[..., c].unsqueeze(-1)
                si = S_imag[..., c].unsqueeze(-1)

                grad_r = torch.autograd.grad(
                    sr, coords, grad_outputs=torch.ones_like(sr), create_graph=True
                )[0]
                grad_i = torch.autograd.grad(
                    si, coords, grad_outputs=torch.ones_like(si), create_graph=True
                )[0]

                grad_r_x = grad_r[..., 0].unsqueeze(-1).contiguous()
                grad_r_y = grad_r[..., 1].unsqueeze(-1).contiguous()

                d2r_dx2 = (
                    torch.autograd.grad(
                        grad_r_x,
                        coords,
                        grad_outputs=torch.ones_like(grad_r_x),
                        create_graph=True,
                    )[0][..., 0]
                    .unsqueeze(-1)
                    .contiguous()
                )

                d2r_dy2 = (
                    torch.autograd.grad(
                        grad_r_y,
                        coords,
                        grad_outputs=torch.ones_like(grad_r_y),
                        create_graph=True,
                    )[0][..., 1]
                    .unsqueeze(-1)
                    .contiguous()
                )

                grad_i_x = grad_i[..., 0].unsqueeze(-1).contiguous()
                grad_i_y = grad_i[..., 1].unsqueeze(-1).contiguous()

                d2i_dx2 = (
                    torch.autograd.grad(
                        grad_i_x,
                        coords,
                        grad_outputs=torch.ones_like(grad_i_x),
                        create_graph=True,
                    )[0][..., 0]
                    .unsqueeze(-1)
                    .contiguous()
                )

                d2i_dy2 = (
                    torch.autograd.grad(
                        grad_i_y,
                        coords,
                        grad_outputs=torch.ones_like(grad_i_y),
                        create_graph=True,
                    )[0][..., 1]
                    .unsqueeze(-1)
                    .contiguous()
                )

                laplacian_r = d2r_dx2 + d2r_dy2
                laplacian_i = d2i_dx2 + d2i_dy2

                res_r = laplacian_r + self.k_sq * sr
                res_i = laplacian_i + self.k_sq * si

                total_pde_res += torch.mean(res_r**2 + res_i**2)

            return total_pde_res / num_coils
