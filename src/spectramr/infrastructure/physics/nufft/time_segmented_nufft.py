import torch
import torch.nn as nn

try:
    import torchkbnufft as tkbn

    TORCH_KBNUFFT_AVAILABLE = True
except ImportError:
    TORCH_KBNUFFT_AVAILABLE = False


class TimeSegmentedNUFFT(nn.Module):
    """
    Time-Segmented Multi-Frequency Interpolation (MFI) for B0 off-resonance.

    Approximates the continuous, time-varying phase evolution of B0 inhomogeneities
    using L discrete time bins (tau_l) and temporal interpolation weights (a_l(t)),
    evaluated via TorchKBNufft. The forward/adjoint form a true adjoint pair.

    .. note::

        Gradient-Non-Linearity (GNL) fusion is **not implemented**. A rigorous
        GNL model warps the encoding by a spatially-varying displacement, which
        is non-trivial for a standard NUFFT and cannot be folded into a plain
        trajectory warp; a naive image-domain warp would also break the
        forward/adjoint pair. Passing ``gnl_displacement`` therefore raises
        ``NotImplementedError`` (CLAUDE.md #9/#16) rather than being silently
        ignored, which is what the previous ``pass`` did.
    """

    def __init__(self, image_shape: tuple = (256, 256), num_time_segments: int = 6):
        """__init__.

        Args:
            image_shape (tuple): Description.
            num_time_segments (int): Description.
        """
        super().__init__()
        if not TORCH_KBNUFFT_AVAILABLE:
            raise ImportError("TorchKBNufft is required but not installed.")

        if num_time_segments < 1:
            raise ValueError(f"num_time_segments must be >= 1, got {num_time_segments}")

        self.image_shape = image_shape
        self.L = num_time_segments

        # NUFFT Operator
        self.nufft_ob = tkbn.KbNufft(im_size=image_shape)
        self.adjnufft_ob = tkbn.KbNufftAdjoint(im_size=image_shape)

    def _compute_interpolation_weights(self, readout_times: torch.Tensor, tau_bins: torch.Tensor):
        """
        Computes temporal interpolation weights a_l(t) for B0 phase approximations.
        Args:
            readout_times: [N_samples] continuous readout sequence
            tau_bins: [L] discrete time bins
        """
        # Hanning-window based interpolation or min-max optimization.
        # Simplified linear interpolation for prototype.
        # L == 1 (single time bin / no B0 segmentation): every readout sample
        # is fully assigned to the one bin -> constant unit weights [1, N].
        if tau_bins.shape[0] == 1:
            return torch.ones(
                1,
                readout_times.shape[0],
                device=readout_times.device,
                dtype=readout_times.dtype,
            )
        distances = torch.abs(readout_times.unsqueeze(1) - tau_bins.unsqueeze(0))  # [N, L]
        weights = torch.nn.functional.relu(1.0 - (distances / (tau_bins[1] - tau_bins[0] + 1e-8)))
        # Normalize
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        return weights.T  # [L, N]

    def forward(
        self,
        image: torch.Tensor,
        b0_map: torch.Tensor,
        trajectory: torch.Tensor,
        gnl_displacement: torch.Tensor | None = None,
        readout_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Forward Operator A(m).

        Args:
            image: True magnetization m(r) [B, 1, H, W] complex
            b0_map: Off-resonance map \Delta \omega(r) [B, 1, H, W]
            trajectory: Base k-space trajectory [B, 2, K]
            gnl_displacement: \Delta r_{GNL} [B, 2, H, W] to warp trajectory
            readout_times: [K] readout timings for B0

        Returns:
            Simulated k-space measurements s(t) [B, 1, K] complex

        forward method for TimeSegmentedNUFFT.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b0_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            gnl_displacement (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            readout_times (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if gnl_displacement is not None:
            raise NotImplementedError(
                "gnl_displacement is not implemented for TimeSegmentedNUFFT: GNL "
                "encoding warp cannot be folded into a standard NUFFT trajectory, "
                "and an image-domain warp would break the forward/adjoint pair. "
                "Pass gnl_displacement=None (B0 time-segmentation only)."
            )
        device = image.device
        B, _, H, W = image.shape
        K = trajectory.shape[-1]

        if readout_times is None:
            # Assume constant generic readout
            readout_times = torch.linspace(1e-3, 30e-3, K, device=device)

        tau_bins = torch.linspace(readout_times.min(), readout_times.max(), self.L, device=device)
        a_lt = self._compute_interpolation_weights(readout_times, tau_bins)  # [L, K]

        simulated_kspace = torch.zeros(B, 1, K, dtype=torch.complex64, device=device)

        # O(L * N log N) unrolled evaluation
        for seg in range(self.L):
            tau = tau_bins[seg]
            temporal_weights = a_lt[seg].view(1, 1, K).to(device)

            # 1. Apply B0 phase accumulation for this specific time bin
            # m(r) * e^{-i \Delta \omega(r) \tau_l}
            phase_shift = torch.exp(-1j * b0_map * tau)
            m_shifted = image * phase_shift

            # 2. Non-Uniform FFT
            # 3. Multiply by temporal interpolation weight 'a_l(t)'
            # NOTE: For rigorous GNL, the trajectory must incorporate the spatial warp
            # which is non-trivial for standard TKBN.
            k_bin = self.nufft_ob(m_shifted, trajectory)
            simulated_kspace += temporal_weights * k_bin

        return simulated_kspace

    def adjoint(
        self,
        kspace: torch.Tensor,
        b0_map: torch.Tensor,
        trajectory: torch.Tensor,
        gnl_displacement: torch.Tensor | None = None,
        readout_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Adjoint Operator A^H(s).
        """
        if gnl_displacement is not None:
            raise NotImplementedError(
                "gnl_displacement is not implemented for TimeSegmentedNUFFT.adjoint; "
                "pass gnl_displacement=None (B0 time-segmentation only)."
            )
        device = kspace.device
        B, _, K = kspace.shape

        if readout_times is None:
            readout_times = torch.linspace(1e-3, 30e-3, K, device=device)

        tau_bins = torch.linspace(readout_times.min(), readout_times.max(), self.L, device=device)
        a_lt = self._compute_interpolation_weights(readout_times, tau_bins)  # [L, K]

        reconstructed_image = torch.zeros(
            B, 1, *self.image_shape, dtype=torch.complex64, device=device
        )

        for seg in range(self.L):
            tau = tau_bins[seg]
            # Conjugate of interpolator a_l*(t)
            temporal_weights_conj = a_lt[seg].view(1, 1, K).conj().to(device)

            # S(t) * a_l*(t)
            weighted_kspace = kspace * temporal_weights_conj

            # Adjoint NUFFT
            img_bin = self.adjnufft_ob(weighted_kspace, trajectory)

            # Positive conjugate phase accumulation: e^{+i \Delta \omega(r) \tau_l}
            phase_shift_conj = torch.exp(1j * b0_map * tau)
            reconstructed_image += img_bin * phase_shift_conj

        return reconstructed_image
