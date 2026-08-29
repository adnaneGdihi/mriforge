r"""Iterative Image Reconstruction and B₀ Mapping Framework.

Implements the joint iterative reconstruction scheme from Koolstra et al.
(2021). Three approaches are supported:

1. **FFT** — Standard Fourier reconstruction + phase-difference B₀ mapping.
2. **Iterative CPR** — Conjugate Phase Reconstruction alternated with
   B₀ re-estimation (3 iterations).
3. **Iterative MB** — Model-Based reconstruction alternated with
   B₀ re-estimation (3 iterations), optionally combined with
   Compressed Sensing.

The iterative loop:

.. code-block:: text

    ΔB₀ ← 0  (initial guess)
    for k = 1 … N_iter:
        1. Reconstruct images using CPR or MB (given current ΔB₀)
        2. Compute phase difference between reconstructed images
        3. Estimate ΔB₀ from phase difference (Tikhonov regularised)
        4. Fit ΔB₀ to 2nd-order spherical harmonics (smooth + extrapolate)

Reference:
    Koolstra K et al., MAGMA (2021) 34:631–642, §2.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import torch

from mriforge.infrastructure.physics.b0_mapping import (
    estimate_b0_tikhonov,
    map_b0,
    phase_difference,
)
from mriforge.infrastructure.physics.conjugate_phase import ConjugatePhaseReconstructor
from mriforge.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
from mriforge.infrastructure.physics.fft_ops import ifft2c
from mriforge.infrastructure.physics.optimizers.split_bregman import SplitBregmanSolver
from mriforge.infrastructure.physics.spherical_harmonics import fit_spherical_harmonics

logger = logging.getLogger(__name__)


class ReconMethod(Enum):
    """Available reconstruction methods."""

    FFT = auto()
    CPR = auto()
    MB = auto()
    MB_CS = auto()


@dataclass
class ReconParams:
    """Parameters for MB reconstruction (Split Bregman).

    Default values correspond to the paper's simulation settings.
    """

    mu: float = 1.0
    lambda_tv: float = 1e-10
    sb_outer_iters: int = 2
    sb_inner_iters: int = 1
    cg_tolerance: float = 1e-2

    # CS-specific
    cs_acceleration: int = 1  # 1 = fully sampled
    cs_center_fraction: float = 0.08

    @classmethod
    def for_simulation(cls) -> ReconParams:
        """Parameters for simulated data (paper §2.5)."""
        return cls(mu=1.0, lambda_tv=1e-10)

    @classmethod
    def for_phantom(cls) -> ReconParams:
        """Parameters for phantom measurements (paper §2.5)."""
        return cls(mu=1.0, lambda_tv=5e-9)

    @classmethod
    def for_invivo(cls) -> ReconParams:
        """Parameters for in vivo measurements (paper §2.5)."""
        return cls(mu=1.0, lambda_tv=8e-11)

    @classmethod
    def for_cs_simulation(cls) -> ReconParams:
        """Parameters for MB+CS simulation (paper §2.5.1)."""
        return cls(
            mu=2e-10,
            lambda_tv=2e-25,
            sb_outer_iters=3,
            cs_acceleration=2,
        )

    @classmethod
    def for_cs_phantom(cls) -> ReconParams:
        """Parameters for MB+CS phantom (paper §2.5.1)."""
        return cls(
            mu=2e-4,
            lambda_tv=2e-12,
            sb_outer_iters=3,
            cs_acceleration=2,
        )

    @classmethod
    def for_cs_invivo(cls) -> ReconParams:
        """Parameters for MB+CS in vivo (paper §2.5.1)."""
        return cls(
            mu=2.5e-4,
            lambda_tv=2.5e-13,
            sb_outer_iters=3,
            cs_acceleration=2,
        )


@dataclass
class IterativeReconResult:
    """Result of iterative reconstruction."""

    image: torch.Tensor  # [1, 1, H, W] complex
    b0_map: torch.Tensor  # [1, 1, H, W] Hz
    b0_maps_history: list[torch.Tensor]  # Per-iteration B₀ maps
    images_history: list[torch.Tensor]  # Per-iteration images


class IterativeReconstructor:
    r"""Joint iterative image reconstruction and B₀ mapping.

    Orchestrates the alternating optimisation between image
    reconstruction and field map estimation.

    Args:
        method: Reconstruction method (:class:`ReconMethod`).
        n_iterations: Number of outer iterations (default 3).
        t_shift: Readout gradient time-shift in seconds.
        tikhonov_gamma: Tikhonov regularisation parameter for B₀ mapping.
        sh_order: Spherical harmonic order for B₀ smoothing (default 2).
        fov_m: Field of view in metres.
        bw_hz: Readout bandwidth in Hz.
        recon_params: Parameters for MB reconstruction.
        cpr_freq_range: Frequency range for CPR multifrequency interpolation.
        cpr_freq_step: Frequency step for CPR.
        b0_relaxation: Relaxation factor for B₀ updates (0–1). A value of
            1.0 uses the new estimate directly; 0.5 averages old and new.
            Damping prevents oscillation between even/odd iterations.
    """

    def __init__(
        self,
        method: ReconMethod | str = ReconMethod.CPR,
        n_iterations: int = 3,
        t_shift: float = 100e-6,
        tikhonov_gamma: float = 2e-10,
        sh_order: int = 2,
        fov_m: float = 0.225,
        bw_hz: float = 20_000.0,
        recon_params: ReconParams | None = None,
        cpr_freq_range: tuple[float, float] = (-4000.0, 4000.0),
        cpr_freq_step: float = 3.0,
        b0_relaxation: float = 0.5,
    ) -> None:
        if isinstance(method, str):
            method = ReconMethod[method.upper()]

        self.method = method
        self.n_iter = n_iterations
        self.t_shift = t_shift
        self.gamma = tikhonov_gamma
        self.sh_order = sh_order
        self.fov = fov_m
        self.bw = bw_hz
        self.recon_params = recon_params or ReconParams.for_simulation()
        self.b0_relax = b0_relaxation

        # Build CPR engine
        self._cpr = ConjugatePhaseReconstructor(
            freq_range_hz=cpr_freq_range,
            freq_step_hz=cpr_freq_step,
            bw_hz=bw_hz,
        )

    def _fft_recon(self, kspace: torch.Tensor) -> torch.Tensor:
        """Standard FFT reconstruction."""
        return ifft2c(kspace)

    def _cpr_recon(
        self,
        kspace: torch.Tensor,
        b0_map: torch.Tensor,
        t_shift: float,
    ) -> torch.Tensor:
        """CPR reconstruction using current B₀ estimate."""
        return self._cpr.reconstruct(kspace, b0_map, t_shift=t_shift)

    def _mb_recon(
        self,
        kspace: torch.Tensor,
        b0_map: torch.Tensor,
        t_shift: float,
        sampling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Model-based reconstruction using Split Bregman."""
        H, W = b0_map.squeeze().shape
        n_ro = kspace.shape[-1]
        n_pe = kspace.shape[-2]

        enc = EncodingMatrixBuilder(
            n_readout=n_ro,
            n_phase=n_pe,
            fov_m=self.fov,
            bw_hz=self.bw,
        )

        # Per-phase-encode-line undersampling mask R (diagonal, self-adjoint).
        # The CS forward model is R·F, so the mask must multiply the forward
        # output, the adjoint input, AND the measured data — otherwise MB_CS
        # silently solves the fully-sampled problem (the mask was previously
        # accepted but never applied, collapsing MB_CS to MB).
        mask = sampling_mask.reshape(-1, 1).to(kspace.dtype) if sampling_mask is not None else None

        # Define forward/adjoint operators
        def fwd(m: torch.Tensor) -> torch.Tensor:
            k = enc.forward(m, b0_map, t_shift=t_shift)
            return k if mask is None else k * mask

        def adj(y: torch.Tensor) -> torch.Tensor:
            if mask is not None:
                y = y * mask
            return enc.adjoint(y, b0_map, t_shift=t_shift)

        solver = SplitBregmanSolver(
            forward_op=fwd,
            adjoint_op=adj,
            mu=self.recon_params.mu,
            lambda_tv=self.recon_params.lambda_tv,
            outer_iterations=self.recon_params.sb_outer_iters,
            inner_iterations=self.recon_params.sb_inner_iters,
            cg_tolerance=self.recon_params.cg_tolerance,
        )

        measured = kspace if mask is None else kspace * mask
        return solver.solve(measured, image_shape=(H, W))

    def _generate_vd_mask(
        self,
        n_pe: int,
        acceleration: int,
        center_fraction: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate variable-density Cartesian undersampling mask.

        Args:
            n_pe: Number of phase-encoding lines.
            acceleration: Acceleration factor.
            center_fraction: Fraction of centre lines to always sample.
            device: Target device.

        Returns:
            Binary mask [n_pe].
        """
        mask = torch.zeros(n_pe, device=device)

        # Always sample centre
        center_lines = max(1, int(n_pe * center_fraction))
        center_start = n_pe // 2 - center_lines // 2
        mask[center_start : center_start + center_lines] = 1.0

        # Variable-density sampling for outer lines
        remaining = n_pe - center_lines
        n_outer = remaining // acceleration

        # Probability proportional to 1/|distance from centre|
        outer_indices = list(range(center_start)) + list(range(center_start + center_lines, n_pe))
        if len(outer_indices) > n_outer:
            # Weighted random selection
            dists = torch.tensor(
                [abs(i - n_pe // 2) for i in outer_indices],
                dtype=torch.float32,
                device=device,
            )
            probs = 1.0 / (dists + 1.0)
            probs = probs / probs.sum()

            selected = torch.multinomial(probs, n_outer, replacement=False)
            for idx in selected:
                mask[outer_indices[idx.item()]] = 1.0
        else:
            for idx in outer_indices:
                mask[idx] = 1.0

        return mask

    def reconstruct(
        self,
        kspace_unshifted: torch.Tensor,
        kspace_shifted: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> IterativeReconResult:
        """Run the full iterative reconstruction pipeline.

        Args:
            kspace_unshifted: K-space from unshifted acquisition [1, 1, N_pe, N_ro].
            kspace_shifted: K-space from shifted acquisition [1, 1, N_pe, N_ro].
            mask: Optional object mask [1, 1, H, W] for B₀ fitting.

        Returns:
            :class:`IterativeReconResult` with final image, B₀ map, and history.
        """
        device = kspace_unshifted.device
        N_pe = kspace_unshifted.shape[-2]
        N_ro = kspace_unshifted.shape[-1]

        # ── Approach 1: FFT baseline ────────────────────────────────
        if self.method == ReconMethod.FFT:
            img_unshifted = self._fft_recon(kspace_unshifted)
            img_shifted = self._fft_recon(kspace_shifted)

            b0_map = map_b0(
                img_unshifted,
                img_shifted,
                t_shift=self.t_shift,
                gamma=self.gamma,
                mask=mask,
            )

            return IterativeReconResult(
                image=img_unshifted,
                b0_map=b0_map,
                b0_maps_history=[b0_map],
                images_history=[img_unshifted],
            )

        # ── Approaches 2/3: Iterative CPR or MB ────────────────────
        # Initial guess: ΔB₀ = 0
        b0_map = torch.zeros(1, 1, N_pe, N_ro, device=device)

        b0_history: list[torch.Tensor] = []
        img_history: list[torch.Tensor] = []

        # Optional CS mask
        cs_mask: torch.Tensor | None = None
        if self.method == ReconMethod.MB_CS and self.recon_params.cs_acceleration > 1:
            cs_mask = self._generate_vd_mask(
                N_pe,
                self.recon_params.cs_acceleration,
                self.recon_params.cs_center_fraction,
                device,
            )
            logger.info(
                "MB+CS: %d / %d PE lines sampled (R=%d)",
                int(cs_mask.sum().item()),
                N_pe,
                self.recon_params.cs_acceleration,
            )

        for iteration in range(self.n_iter):
            logger.info("Iteration %d / %d", iteration + 1, self.n_iter)

            # ── Step 1: Image reconstruction ──────────────────────
            if self.method == ReconMethod.CPR:
                img_unshifted = self._cpr_recon(kspace_unshifted, b0_map, t_shift=0.0)
                img_shifted = self._cpr_recon(kspace_shifted, b0_map, t_shift=self.t_shift)

            elif self.method in (ReconMethod.MB, ReconMethod.MB_CS):
                img_unshifted = self._mb_recon(
                    kspace_unshifted,
                    b0_map,
                    t_shift=0.0,
                    sampling_mask=cs_mask,
                )
                img_shifted = self._mb_recon(
                    kspace_shifted,
                    b0_map,
                    t_shift=self.t_shift,
                    sampling_mask=cs_mask,
                )

            else:
                raise ValueError(f"Unknown method: {self.method}")

            img_history.append(img_unshifted.clone())

            # ── Step 2: B₀ mapping from reconstructed images ─────
            phi = phase_difference(img_unshifted, img_shifted)
            b0_new = estimate_b0_tikhonov(
                phi,
                t_shift=self.t_shift,
                gamma=self.gamma,
                mask=mask,
            )

            # ── Step 3: Fit to spherical harmonics ───────────────
            b0_new, _ = fit_spherical_harmonics(
                b0_new,
                max_order=self.sh_order,
                mask=mask,
            )

            # ── Step 4: Relaxed B₀ update ────────────────────────
            # Blend new estimate with previous to prevent oscillation
            b0_map = self.b0_relax * b0_new + (1.0 - self.b0_relax) * b0_map

            b0_history.append(b0_map.clone())

        return IterativeReconResult(
            image=img_unshifted,
            b0_map=b0_map,
            b0_maps_history=b0_history,
            images_history=img_history,
        )


__all__ = [
    "IterativeReconResult",
    "IterativeReconstructor",
    "ReconMethod",
    "ReconParams",
]
