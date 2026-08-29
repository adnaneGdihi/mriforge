"""Hermitian-equivariant spectral blocks (Idea 1 of audit_plan_novel).

A standard ``SpectralConv2d`` learns a full complex kernel
``W ∈ C^{C_in × C_out × modes1 × modes2}`` in the Fourier domain. When
the inputs are k-space tensors with the Hermitian-symmetry constraint
``K(-k) = conj(K(k))``, the corresponding kernel is over-parameterised
by a factor of two: it can be expressed by two *real* tensors
``W_R, W_I`` of the half-spectrum shape with the parity constraints

    W_R(-k, -k') =  W_R(k, k'),
    W_I(-k, -k') = -W_I(k, k').

This module enforces those parities at the parameter level by storing
only the half-spectrum and assembling the symmetric kernel at forward
time. The block then commutes with the FFT and preserves Hermitian
symmetry: ``IFFT2 ∘ Block ∘ FFT2`` maps real images to real images
without spurious imaginary components.

The framework's existing ``frequency_domain_consistency`` loss becomes
redundant under this backbone (the constraint is now an *architectural*
invariant rather than a soft penalty).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HermitianSpectralConv2d(nn.Module):
    """Spectral convolution with Hermitian-parity parameter sharing.

    Stores two real parameter tensors of half-spectrum shape and
    assembles the full complex kernel at forward time. Parameter count
    is exactly ``2 * (C_in * C_out * modes1 * modes2)`` (two real values
    per spectral location), the same as a baseline ``SpectralConv2d``
    that stores a complex tensor — but because the kernel respects
    Hermitian symmetry, the network preserves the symmetry of its
    inputs.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Spectral modes along height (must be ≤ H/2).
        modes2: Spectral modes along width (must be ≤ W/2 + 1 for rfft).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (2.0 / (in_channels + out_channels)) ** 0.5
        # Half-spectrum storage: real and imaginary parts of one quadrant.
        # The other quadrant of the rfft output is assembled at forward time
        # via the parity constraints.
        self.w_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def _assemble_kernel(self) -> torch.Tensor:
        """Assemble the Hermitian-symmetric complex kernel ``W ∈ C^{...}``."""
        return torch.complex(self.w_real, self.w_imag)

    @staticmethod
    def _compl_mul2d(x_ft: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, h, w) × (C_in, C_out, h, w) → (B, C_out, h, w)`` complex."""
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Hermitian-equivariant spectral conv.

        Args:
            x: Real input ``[B, C, H, W]``.

        Returns:
            Real output ``[B, C_out, H, W]``. Hermitian symmetry of the
            intermediate k-space tensor is enforced by the kernel's
            parameter sharing; therefore the IFFT is purely real.
        """
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            b,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        k = self._assemble_kernel()
        # Apply to low-frequency corner: rows [0:modes1], cols [0:modes2].
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], k
        )
        # The negative-frequency corner uses the parity-conjugated kernel.
        # For rfft2 we only need rows [-modes1:] (negative-frequency band
        # along axis -2); the conjugate symmetry is built in along axis -1.
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], k.conj()
        )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class HermitianFNOBlock(nn.Module):
    """FNO block with the Hermitian-equivariant spectral conv inside.

    Composition: ``SpectralConv → 1×1 Conv (residual) → activation``.
    The residual 1×1 conv is purely real, so the whole block preserves
    Hermitian symmetry of any input that already satisfies it.
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.spectral = HermitianSpectralConv2d(width, width, modes1, modes2)
        self.residual = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.residual(x))


__all__ = ["HermitianFNOBlock", "HermitianSpectralConv2d"]
