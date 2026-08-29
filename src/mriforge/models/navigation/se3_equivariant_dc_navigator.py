r"""SE(3)-equivariant DC-navigator reconstruction model (B-3 of the VF campaign).

Pipeline (per ``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md`` Idea 3):

.. code-block:: text

    y (multi-coil k-space)
        --> extract DC / central k-space line  (endogenous navigator)
        --> SE3LieAlgebraMambaBlock  x  n_layers
        --> motion latent  xi(t) in se(3)
        --> rigid-body warp head  -->  x_corrected (image domain)

The navigator is *endogenous*: it is read directly from the central
phase-encode line of the acquired k-space (the part most sensitive to bulk
motion phase), so no external tracking hardware is needed.

The model registers under ``training_mode="reconstruction"`` and consumes the
real-interleaved ``[B, 2C, H, W]`` tensor that ``BaseTrainingStrategy`` produces
(it converts complex inputs before calling ``forward``). It returns a
real-interleaved ``[B, 2C, H, W]`` reconstruction so the standard image /
complex / kspace losses can be applied downstream.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

from mriforge.models.blocks.se3_lie_algebra_mamba import (
    DEFAULT_BIO_HARMONIC_FREQS,
    SE3LieAlgebraMambaBlock,
)
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


def _rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """Build a batch of 2x2 in-plane rotation matrices from angles ``theta``.

    Args:
        theta: ``[B]`` rotation angles (radians).

    Returns:
        ``[B, 2, 2]`` rotation matrices.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


@register_model(
    name="se3_equivariant_dc_navigator",
    training_mode="reconstruction",
    spatial_dims=(2,),
    # Endogenous navigator reads k-space; VF / motion strategies hand the
    # backbone an image-domain real-interleaved tensor, so both on-disk
    # domains are valid (mirrors hyper_mamba_unet's declaration).
    input_domain=("kspace", "image"),
    output_domain=("image", "kspace"),
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=True,
)
class SE3EquivariantDCNavigator(nn.Module):
    r"""Endogenous DC-navigator + SE(3)-equivariant Lie-algebra Mamba stack.

    Args:
        in_channels: Real-interleaved input channels (``2 * n_coils``).
        out_channels: Real-interleaved output channels. Defaults to
            ``in_channels``.
        n_mamba_layers: Number of stacked :class:`SE3LieAlgebraMambaBlock`.
        d_state: SSM hidden-state dimension per block.
        d_conv: Depthwise conv kernel length inside each scalar Mamba.
        dc_navigator_sample_rate_hz: Navigator temporal sampling rate (Hz);
            forwarded to the bio-harmonic init / Nyquist check.
        bio_harmonic_freqs_hz: Physiological resonance frequencies (Hz).
        group: ``"SO3"`` or ``"SE3"``; forwarded to every block.
        nav_hidden: Hidden width of the navigator encoder MLP.

    Notes:
        **Warp restriction (documented simplification).** The motion latent
        :math:`\xi(t)\in\mathfrak{se}(3)` is decoded by the equivariant block,
        but the *spatial* correction applied to the 2D slice uses only the
        in-plane SE(2) part — an in-plane rotation :math:`\theta=\bar\omega_z`
        and translation :math:`(\bar v_x, \bar v_y)` (the temporally-averaged
        bulk motion). Through-plane components are estimated and exposed (for
        the equivariance-defect loss / metrics) but not applied to the single
        slice. This is the physically correct restriction for 2D MRI motion
        correction; the block itself remains fully SO(3)-equivariant.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int | None = None,
        n_mamba_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        dc_navigator_sample_rate_hz: float = 100.0,
        bio_harmonic_freqs_hz: tuple[float, ...] = DEFAULT_BIO_HARMONIC_FREQS,
        group: Literal["SO3", "SE3"] = "SE3",
        nav_hidden: int = 64,
    ) -> None:
        super().__init__()

        if n_mamba_layers < 1:
            raise ValueError("n_mamba_layers must be >= 1.")
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels else int(in_channels)
        self.group = group
        self._dim = SE3LieAlgebraMambaBlock._DIM  # 6

        # ------------------------------------------------------------------
        # Navigator encoder: central k-space column -> per-row se(3) sequence.
        # The DC navigator extracts the central phase-encode column from the
        # multi-coil k-space (the column carries the most motion-sensitive
        # phase). Each k-space row (frequency-encode index) becomes a temporal
        # navigator sample t = 0..H-1. A small per-row MLP lifts the per-row
        # complex coil vector (real-interleaved) to a 6-D se(3) coordinate.
        # ------------------------------------------------------------------
        self.nav_encoder = nn.Sequential(
            nn.Linear(self.in_channels, nav_hidden),
            nn.SiLU(),
            nn.Linear(nav_hidden, nav_hidden),
            nn.SiLU(),
            nn.Linear(nav_hidden, self._dim),
        )

        # Stack of equivariant Lie-algebra Mamba blocks.
        self.mamba_layers = nn.ModuleList(
            [
                SE3LieAlgebraMambaBlock(
                    d_state=d_state,
                    d_conv=d_conv,
                    bio_harmonic_freqs=tuple(bio_harmonic_freqs_hz),
                    sample_rate_hz=dc_navigator_sample_rate_hz,
                    group=group,
                )
                for _ in range(n_mamba_layers)
            ]
        )

        # Residual refinement head on the image (applied after the rigid warp)
        # to absorb non-rigid residual artefacts. Kept lightweight so the
        # rigid warp carries the bulk of the correction.
        self.refine = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, self.out_channels, kernel_size=3, padding=1),
        )
        # Zero-init the last refine conv so the model starts as a pure warp.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

        logger.info(
            "[SE3EquivariantDCNavigator] init: in=%d out=%d layers=%d group=%s "
            "fs=%.1fHz harmonics=%s",
            self.in_channels,
            self.out_channels,
            n_mamba_layers,
            group,
            dc_navigator_sample_rate_hz,
            tuple(bio_harmonic_freqs_hz),
        )

    # ----------------------------------------------------------------------
    # Navigator
    # ----------------------------------------------------------------------
    def extract_dc_navigator(self, kspace_ri: torch.Tensor) -> torch.Tensor:
        """Extract the central k-space column as a per-row navigator sequence.

        Args:
            kspace_ri: ``[B, 2C, H, W]`` real-interleaved k-space.

        Returns:
            ``[B, H, 2C]`` per-row navigator features (W collapsed to centre).
        """
        w = kspace_ri.shape[-1]
        centre = w // 2
        # central column -> [B, 2C, H]
        column = kspace_ri[:, :, :, centre]
        # to per-row token sequence [B, H, 2C]
        return column.transpose(1, 2).contiguous()

    def estimate_motion_latent(self, kspace_ri: torch.Tensor) -> torch.Tensor:
        r"""Run the navigator + equivariant Mamba stack to a se(3) trajectory.

        Args:
            kspace_ri: ``[B, 2C, H, W]`` real-interleaved k-space.

        Returns:
            ``[B, H, 6]`` motion latent :math:`\xi(t)`.
        """
        nav = self.extract_dc_navigator(kspace_ri)  # [B, H, 2C]
        xi = self.nav_encoder(nav)  # [B, H, 6]
        for layer in self.mamba_layers:
            # Residual connection keeps the stack stable and the
            # equivariance exact (sum of equivariant maps is equivariant).
            xi = xi + layer(xi)
        return xi

    @staticmethod
    def _warp_image(
        img_ri: torch.Tensor,
        theta: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """Apply an in-plane SE(2) rigid warp to a real-interleaved image.

        Args:
            img_ri: ``[B, 2C, H, W]`` real-interleaved image.
            theta: ``[B]`` in-plane rotation angle (radians).
            trans: ``[B, 2]`` normalised in-plane translation in
                ``grid_sample`` coordinates ([-1, 1] across the FOV).

        Returns:
            ``[B, 2C, H, W]`` warped image.
        """
        rot = _rotation_matrix_2d(theta)  # [B, 2, 2]
        # affine_grid theta is [B, 2, 3] = [R | t].
        affine = torch.cat([rot, trans.unsqueeze(-1)], dim=-1)  # [B, 2, 3]
        grid = torch.nn.functional.affine_grid(affine, list(img_ri.shape), align_corners=False)
        return torch.nn.functional.grid_sample(
            img_ri, grid, align_corners=False, padding_mode="zeros"
        )

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        kspace_measured: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Reconstruct a motion-corrected image from corrupted input.

        Args:
            x: ``[B, 2C, H, W]`` real-interleaved corrupted image (the strategy
                hands an image-domain tensor here; complex inputs are converted
                upstream by ``BaseTrainingStrategy``).
            kspace_measured: optional ``[B, 2C, H, W]`` real-interleaved
                measured k-space used for the endogenous navigator. When absent
                we derive it from ``x`` via a centered FFT.
            **kwargs: ignored extra forward kwargs (mask, sensitivity_maps, …).

        Returns:
            ``[B, out_channels, H, W]`` real-interleaved reconstruction.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, 2C, H, W]; got {tuple(x.shape)}.")

        # Endogenous navigator source: prefer measured k-space, else FFT(x).
        ksp = kspace_measured if kspace_measured is not None else self._image_to_kspace_ri(x)

        # 1. Motion latent xi(t) in se(3).
        xi = self.estimate_motion_latent(ksp)  # [B, H, 6]

        # 2. Reduce the trajectory to a bulk SE(2) correction (temporal mean of
        #    the in-plane components). omega_z (index 2) = in-plane rotation;
        #    v_x, v_y (indices 3, 4) = in-plane translation.
        xi_mean = xi.mean(dim=1)  # [B, 6]
        theta = xi_mean[:, 2]  # in-plane rotation angle
        # Bound the translation to the FOV via tanh (grid coords in [-1, 1]).
        trans = torch.tanh(xi_mean[:, 3:5])  # [B, 2]

        # 3. Apply the inverse rigid warp to undo the estimated bulk motion.
        warped = self._warp_image(x, -theta, -trans)

        # 4. Lightweight residual refinement (zero-init → starts as pure warp).
        out = warped[:, : self.out_channels] + self.refine(warped)
        return out

    @staticmethod
    def _image_to_kspace_ri(img_ri: torch.Tensor) -> torch.Tensor:
        """FFT a real-interleaved image to real-interleaved k-space.

        Uses the physics SSOT ``fft2c`` (centered, ortho-normalised). Treats
        consecutive channel pairs as (real, imag) coil components.
        """
        from mriforge.infrastructure.physics.fft_ops import fft2c

        b, c, h, w = img_ri.shape
        if c % 2 == 0:
            cplx = torch.complex(img_ri[:, 0::2], img_ri[:, 1::2])
        else:
            cplx = img_ri.to(torch.complex64)
        ksp = fft2c(cplx)
        # back to real-interleaved
        out = torch.empty(b, ksp.shape[1] * 2, h, w, device=img_ri.device, dtype=img_ri.dtype)
        out[:, 0::2] = ksp.real.to(img_ri.dtype)
        out[:, 1::2] = ksp.imag.to(img_ri.dtype)
        return out


__all__ = ["SE3EquivariantDCNavigator"]
