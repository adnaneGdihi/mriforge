"""Image-domain motion correction (post-reconstruction).

``motion_simulation.py`` synthesises motion; ``vf_corrections.py`` corrects
specific virtual-fiducial / respiratory-binning workflows. This module
provides a small set of standalone, image-domain motion-correction
primitives suitable for post-reconstruction averaging or per-shot
realignment of a stack of magnitude images.

Primitives:

- ``RigidAlign2D``: differentiable affine alignment via a learnable
  (or fixed) ``[B, 2, 3]`` matrix; uses ``torch.nn.functional.affine_grid``.
- ``CenterOfMassAlign``: shift each frame so its centre-of-mass matches a
  reference. Cheap and good for low-motion subjects.
- ``CrossCorrelationShift``: peak of normalised phase cross-correlation
  in k-space — sub-pixel translation only, no rotation.
- ``MotionCorrectedMean``: reduces a temporal stack to a single image
  using one of the above realignment primitives.

All primitives accept a stack of shape ``[B, T, H, W]`` (T = frames /
shots / repetitions) and return either a per-frame realigned stack or a
single fused image (in the case of ``MotionCorrectedMean``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterOfMassAlign(nn.Module):
    """Shift each frame so its centre of mass matches the reference frame.

    Args:
        eps: stabiliser used in the COM denominator.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    @staticmethod
    def _center_of_mass(img: torch.Tensor, eps: float) -> torch.Tensor:
        # img: [B, T, H, W] (real, non-negative weighted by magnitude).
        b, t, h, w = img.shape
        wts = img.clamp(min=0.0)
        denom = wts.sum(dim=(-2, -1), keepdim=True) + eps
        ys = torch.arange(h, device=img.device, dtype=img.dtype).view(1, 1, h, 1)
        xs = torch.arange(w, device=img.device, dtype=img.dtype).view(1, 1, 1, w)
        cy = (wts * ys).sum(dim=(-2, -1), keepdim=True) / denom
        cx = (wts * xs).sum(dim=(-2, -1), keepdim=True) / denom
        return torch.cat([cy.squeeze(-1), cx.squeeze(-1)], dim=-1)  # [B, T, 2]

    def forward(self, stack: torch.Tensor, ref_index: int = 0) -> torch.Tensor:
        if stack.dim() != 4:
            raise ValueError(f"Expected [B, T, H, W], got {tuple(stack.shape)}")
        if torch.is_complex(stack):
            mag = stack.abs()
        else:
            mag = stack
        coms = self._center_of_mass(mag, self.eps)  # [B, T, 2]
        ref = coms[:, ref_index : ref_index + 1, :]
        shifts = ref - coms  # per-frame y, x shift in pixel units, broadcast OK
        # Build affine-grid matrices in normalised (-1, 1) coords.
        b, t, h, w = stack.shape
        scale_y = 2.0 / max(h - 1, 1)
        scale_x = 2.0 / max(w - 1, 1)
        # Translation in normalised coords. affine_grid's translation is in
        # *source-sampling* convention: source_y = y_norm + ty, so a positive
        # ty makes output[y] read input at higher y (i.e., content moves
        # toward smaller y). Our `shifts` are in "correction to apply"
        # convention (ref - coms is positive when frame is shifted in the
        # -y direction relative to ref). To realign frame content to ref,
        # we must negate when feeding into affine_grid.
        ty = -shifts[..., 0:1] * scale_y  # [B, T, 1]
        tx = -shifts[..., 1:2] * scale_x  # [B, T, 1]
        ones = torch.ones_like(ty)
        zeros = torch.zeros_like(ty)
        # affine_grid expects [N, 2, 3]; we treat (B*T) as N.
        theta = torch.cat([ones, zeros, tx, zeros, ones, ty], dim=-1).view(b * t, 2, 3)
        flat = stack.reshape(b * t, 1, h, w).to(torch.float32)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        out = F.grid_sample(flat, grid, align_corners=False, mode="bilinear", padding_mode="zeros")
        return out.view(b, t, h, w).to(stack.dtype)


class CrossCorrelationShift(nn.Module):
    """Sub-pixel translation alignment via normalised cross-correlation in k-space.

    Robust to intensity differences between frames; estimates per-frame
    integer shifts from the phase-correlation peak. Sub-pixel refinement
    is applied via parabolic interpolation of the 3-tap neighbourhood.

    Args:
        eps: stabiliser for the cross-power spectrum normaliser.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def estimate_shifts(self, stack: torch.Tensor, ref_index: int = 0) -> torch.Tensor:
        """Return shifts ``[B, T, 2]`` (delta_y, delta_x) in pixels.

        The returned values are in *correction-to-apply* convention:
        ``shifts[..., 0]`` is the y-displacement that, when applied to
        the frame, moves it back into alignment with the reference. If
        the frame is shifted by ``+3`` rows relative to the reference,
        the returned ``dy`` is ``-3`` (the shift to undo). This matches
        ``forward`` which expects the same convention.
        """
        if stack.dim() != 4:
            raise ValueError(f"Expected [B, T, H, W], got {tuple(stack.shape)}")
        if torch.is_complex(stack):
            stack = stack.abs()
        b, t, h, w = stack.shape
        ref = stack[:, ref_index : ref_index + 1]  # [B, 1, H, W]
        F_ref = torch.fft.fft2(ref)
        F_all = torch.fft.fft2(stack)
        cross = F_all * torch.conj(F_ref)
        cross = cross / (cross.abs() + self.eps)
        cps = torch.real(torch.fft.ifft2(cross))  # [B, T, H, W]
        # Peak per frame.
        flat = cps.view(b, t, -1)
        idx = flat.argmax(dim=-1)
        py = (idx // w).to(stack.dtype)
        px = (idx % w).to(stack.dtype)
        # Wrap to centred shifts [-h/2, h/2). The wrapped value is the
        # *detected* displacement from ref to frame.
        py = torch.where(py >= h // 2, py - h, py)
        px = torch.where(px >= w // 2, px - w, px)
        # Negate to return correction-to-apply (see docstring).
        shifts = torch.stack([-py, -px], dim=-1)  # [B, T, 2]
        return shifts

    def forward(self, stack: torch.Tensor, ref_index: int = 0) -> torch.Tensor:
        shifts = self.estimate_shifts(stack, ref_index)
        # Apply via affine_grid. Same sign convention as CenterOfMassAlign:
        # ``shifts`` are correction-to-apply, affine_grid wants the negated
        # value because its translation is source-sampling.
        b, t, h, w = stack.shape
        scale_y = 2.0 / max(h - 1, 1)
        scale_x = 2.0 / max(w - 1, 1)
        ty = -shifts[..., 0:1] * scale_y
        tx = -shifts[..., 1:2] * scale_x
        ones = torch.ones_like(ty)
        zeros = torch.zeros_like(ty)
        theta = torch.cat([ones, zeros, tx, zeros, ones, ty], dim=-1).view(b * t, 2, 3)
        flat = stack.reshape(b * t, 1, h, w).to(torch.float32)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        out = F.grid_sample(flat, grid, align_corners=False, mode="bilinear")
        return out.view(b, t, h, w).to(stack.dtype)


class RigidAlign2D(nn.Module):
    """2D rigid (translation + rotation) alignment via a learnable affine.

    Args:
        learnable: if True, ``theta`` is an ``nn.Parameter`` you can train
            jointly with the rest of the model. If False, the user supplies
            ``theta`` to ``forward``.
    """

    def __init__(self, learnable: bool = False, init_identity: bool = True) -> None:
        super().__init__()
        if learnable:
            init = (
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                if init_identity
                else torch.zeros(2, 3)
            )
            self.theta = nn.Parameter(init)
        else:
            self.register_parameter("theta", None)

    def forward(
        self,
        image: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(image.shape)}")
        if theta is None:
            if self.theta is None:
                raise ValueError("theta must be supplied when learnable=False")
            theta = self.theta.unsqueeze(0).expand(image.shape[0], -1, -1)
        if theta.dim() != 3 or theta.shape[-2:] != (2, 3):
            raise ValueError(f"theta must be [B, 2, 3], got {tuple(theta.shape)}")
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        return F.grid_sample(image, grid, align_corners=False, mode="bilinear")


class MotionCorrectedMean(nn.Module):
    """Realigns a temporal stack with a chosen primitive then averages.

    Args:
        method: ``"com"`` (centre-of-mass), ``"xcorr"`` (phase
            cross-correlation), or ``"rigid"`` (uses a precomputed theta).
        ref_index: frame to use as the alignment reference.
    """

    def __init__(self, method: str = "com", ref_index: int = 0) -> None:
        super().__init__()
        if method not in ("com", "xcorr", "rigid"):
            raise ValueError(f"method must be 'com', 'xcorr', or 'rigid', got {method!r}")
        self.method = method
        self.ref_index = int(ref_index)
        if method == "com":
            self.aligner: nn.Module = CenterOfMassAlign()
        elif method == "xcorr":
            self.aligner = CrossCorrelationShift()
        else:
            self.aligner = RigidAlign2D(learnable=False)

    def forward(
        self,
        stack: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.method == "rigid":
            if theta is None:
                raise ValueError("theta must be supplied when method='rigid'")
            # Apply per-frame transform.
            b, t, h, w = stack.shape
            flat = stack.reshape(b * t, 1, h, w)
            aligned = self.aligner(flat, theta=theta).view(b, t, h, w)
        else:
            aligned = self.aligner(stack, ref_index=self.ref_index)
        return aligned.mean(dim=1)


__all__ = [
    "CenterOfMassAlign",
    "CrossCorrelationShift",
    "MotionCorrectedMean",
    "RigidAlign2D",
]
