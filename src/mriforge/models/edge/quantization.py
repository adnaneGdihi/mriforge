"""
Quantization Modules for Edge MRI (Q-KAN).

Implements 1-bit/2-bit quantization for KAN spline coefficients
using Straight-Through Estimators (STE).
"""

import torch

from mriforge.models.registry import register_model

from ..spectral.fourier_kan import FourierKANLayer


class SplineQuantizer(torch.autograd.Function):
    """
    Straight-Through Estimator for Quantization.
    Forward: Quantize
    Backward: Identify (Gradient pass-through)
    """

    @staticmethod
    def forward(ctx, input, bits=2):
        """forward.

        Args:
            ctx (Any): Description.
            input (Any): Description.
            bits (Any): Description.
        Returns:
            Any: Description.
        """
        ctx.save_for_backward(input)

        if bits == 1:
            # Binary {-1, 1}
            return input.sign()
        elif bits == 2:
            # Ternary {-1, 0, 1} or customized 2-bit
            # Simple uniform quantization to [-1, 0, 1] assuming input in [-1, 1]
            # Scale to [-1.5, 1.5] then round
            # Round(x) -> {-1, 0, 1}
            return input.clamp(-1, 1).mul(1.0).round()

        else:
            raise ValueError(
                f"Unsupported quantization bits={bits}; only 1 and 2 are implemented "
                "in SplineQuantizer."
            )

    @staticmethod
    def backward(ctx, grad_output):
        # Straight Through Estimator: return grad_output directly
        """backward.

        Args:
            ctx (Any): Description.
            grad_output (Any): Description.
        Returns:
            Any: Description.
        """
        (input,) = ctx.saved_tensors
        return grad_output, None


@register_model(name="q_kan", training_mode="reconstruction")
class QKANLayer(FourierKANLayer):
    """
    Quantized Fourier KAN Layer.
    Wraps standard FourierKANLayer but enforces quantized weights during forward.
    """

    def __init__(self, in_features, out_features, grid_size=300, add_bias=True, bits=2):
        """__init__.

        Args:
            in_features (Any): Description.
            out_features (Any): Description.
            grid_size (Any): Description.
            add_bias (Any): Description.
            bits (Any): Description.
        """
        super().__init__(in_features, out_features, grid_size, add_bias)
        self.bits = bits
        if bits not in (1, 2):
            raise ValueError(
                f"QKANLayer received unsupported bits={bits}; only 1 and 2 are "
                "implemented (SplineQuantizer). Set bits to 1 or 2."
            )

    def forward(self, x):
        # 1. Quantize weights on the fly
        # We explicitly quantize fourier_coeffs
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        q_coeffs = SplineQuantizer.apply(self.fourier_coeffs, self.bits)

        # 2. Compute Fourier features
        xshp = x.shape
        outshape = xshp[0:-1] + (self.out_features,)
        x = torch.reshape(x, (-1, self.in_features))

        k = torch.reshape(
            torch.arange(1, self.grid_size + 1, device=x.device),
            (1, 1, 1, self.grid_size),
        )
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # 3. Apply Quantized Weights
        # Shape: [1, outdim, inputdim, gridsize]
        # q_coeffs comes from fouriercoeffs which is [2, outdim, inputdim, gridsize]
        # We split into cos/sin coeffs

        # y = Sum( coeff_cos * cos + coeff_sin * sin )

        # fourier_coeffs shape [out, in, 2 * grid] -> [o, i, 2g]
        # basis construction logic in base class is:
        # basis = cat(sin, cos) -> [N, In, 2K]
        # y = einsum("nib,oib->no", basis, q_coeffs)

        # BUT here in this file, we had different logic:
        # y = torch.einsum("dbik,xik->xdb", q_coeffs, torch.stack([c, s], dim=0))
        # This implementation seems to assume q_coeffs follows [2, out, in, grid] shape, which conflicts with base class [out, in, 2*grid].

        # I should stick to the base class logic but use quantized coeffs!

        # Re-implement base class logic:
        x_flat = x  # [N, In]
        k = torch.arange(1, self.grid_size + 1, device=x.device).float().view(1, 1, -1)
        x_expanded = x_flat.unsqueeze(-1)
        angles = x_expanded * k * 3.141592653589793

        basis = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N, In, 2K]

        # use q_coeffs instead of self.fourier_coeffs
        # q_coeffs is quant(self.fourier_coeffs) -> [Out, In, 2K]

        kan_out = torch.einsum("nib,oib->no", basis, q_coeffs)

        y = kan_out

        if self.base_linear.bias is not None:
            y += self.base_linear(x_flat)  # Add base linear output too?
            # Wait, base class adds base_linear(x) + kan_out.
        else:
            y += self.base_linear(x_flat)

        # The base class `forward` does: out = base_out + kan_out
        # So I should do the same.

        out = self.base_linear(x_flat) + kan_out

        return torch.reshape(out, outshape)


def create_q_kan(**kwargs):
    """create_q_kan.

    Returns:
        Any: Description.
    """
    return QKANLayer(**kwargs)
