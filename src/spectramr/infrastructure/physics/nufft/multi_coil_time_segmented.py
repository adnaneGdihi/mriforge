import torch
import torch.nn as nn

from .time_segmented_nufft import TimeSegmentedNUFFT


class MultiCoilTimeSegmentedNUFFT(nn.Module):
    r"""
    Multi-Coil wrapper for Time-Segmented Multi-Frequency Interpolation (MFI) NUFFT.

    This operator applies the single-coil TimeSegmentedNUFFT operator across an
    array of coil sensitivities.

    Mathematical Model:
        Forward: s_c(t) = A(S_c * m)
        Adjoint: m^H = \sum_c S_c^* * A^H(s_c)
    """

    def __init__(self, image_shape: tuple = (256, 256), num_time_segments: int = 6):
        """__init__.

        Args:
            image_shape (tuple): Description.
            num_time_segments (int): Description.
        """
        super().__init__()
        self.nufft_op = TimeSegmentedNUFFT(
            image_shape=image_shape, num_time_segments=num_time_segments
        )
        self.image_shape = image_shape

    def forward(
        self,
        image: torch.Tensor,
        coil_sensitivities: torch.Tensor,
        b0_map: torch.Tensor,
        trajectory: torch.Tensor,
        gnl_displacement: torch.Tensor | None = None,
        readout_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Forward Operator A_c(m).

        Args:
            image: True magnetization m(r) [B, 1, H, W] complex
            coil_sensitivities: S_c(r) [B, C, H, W] complex
            b0_map: Off-resonance map \Delta \omega(r) [B, 1, H, W]
            trajectory: Base k-space trajectory [B, 2, K]
            gnl_displacement: \Delta r_{GNL} [B, 2, H, W] to warp trajectory
            readout_times: [K] readout timings for B0

        Returns:
            Simulated multi-coil k-space measurements s_c(t) [B, C, K] complex

        forward method for MultiCoilTimeSegmentedNUFFT.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coil_sensitivities (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b0_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            gnl_displacement (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            readout_times (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = image.device
        B, C, H, W = coil_sensitivities.shape
        K = trajectory.shape[-1]

        simulated_kspace = torch.zeros(B, C, K, dtype=torch.complex64, device=device)

        # Apply S_c * m
        # m is [B, 1, H, W], S_c is [B, C, H, W] -> [B, C, H, W]
        coil_images = image * coil_sensitivities

        # Process each coil
        for c in range(C):
            coil_img_c = coil_images[:, c : c + 1, :, :]  # [B, 1, H, W]
            kspace_c = self.nufft_op.forward(
                image=coil_img_c,
                b0_map=b0_map,
                trajectory=trajectory,
                gnl_displacement=gnl_displacement,
                readout_times=readout_times,
            )  # [B, 1, K]
            simulated_kspace[:, c : c + 1, :] = kspace_c

        return simulated_kspace

    def adjoint(
        self,
        kspace: torch.Tensor,
        coil_sensitivities: torch.Tensor,
        b0_map: torch.Tensor,
        trajectory: torch.Tensor,
        gnl_displacement: torch.Tensor | None = None,
        readout_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Adjoint Operator m^H = \sum_c S_c^* * A^H(s_c).

        Args:
            kspace: s_c(t) [B, C, K] complex
            coil_sensitivities: S_c(r) [B, C, H, W] complex
            b0_map: Off-resonance map \Delta \omega(r) [B, 1, H, W]
            trajectory: Base k-space trajectory [B, 2, K]

        Returns:
            Reconstructed image m^H [B, 1, H, W] complex
        """
        device = kspace.device
        B, C, K = kspace.shape

        reconstructed_image = torch.zeros(
            B, 1, *self.image_shape, dtype=torch.complex64, device=device
        )

        for c in range(C):
            kspace_c = kspace[:, c : c + 1, :]  # [B, 1, K]
            # Adjoint A^H(s_c) -> [B, 1, H, W]
            img_c = self.nufft_op.adjoint(
                kspace=kspace_c,
                b0_map=b0_map,
                trajectory=trajectory,
                gnl_displacement=gnl_displacement,
                readout_times=readout_times,
            )
            # S_c^* * A^H(s_c)
            sens_conj = coil_sensitivities[:, c : c + 1, :, :].conj()
            reconstructed_image += img_c * sens_conj

        return reconstructed_image
